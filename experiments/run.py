"""
experiments/run.py — главный скрипт эксперимента (расширенная версия).

Запуск (с запущенной Ollama):
    python experiments/run.py

Что делает:
  1. Проверяет Ollama и наличие моделей
  2. Для каждой модели × квантизации:
       прогоняет все сценарии × N_REPEATS повторов
       классифицирует исходы
  3. Считает метрики + доверительные интервалы
  4. Статистический тест: значима ли разница Q4 vs Q8 (на каждой модели)
  5. Сохраняет: results.csv, results.json, raw_runs.json, stats.json

Настройки — блок CONFIG ниже. Можно запускать по одной модели за раз:
  просто оставь в MODELS_TO_TEST нужные строки, остальные закомментируй.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import csv
from tqdm import tqdm

from src.ollama_client import OllamaClient
from src.agent import SimulatedAgent
from src.attacks import get_all_scenarios
from src.metrics import (aggregate, aggregate_by_attack_type,
                         aggregate_by_style, baseline_completion)
from src.stats import two_proportion_z, chi_square_2x2


# ══ CONFIG ════════════════════════════════════════════════════════════════════

# Модели для теста. Имена должны совпадать с `ollama list`.
# Закомментируй строки, чтобы запускать по одной модели за раз.
MODELS_TO_TEST = [
    # Qwen2.5-1.5B — все 4 квантизации
    {"family": "qwen2.5-1.5b", "quant": "Q4_K_M",
     "ollama_name": "qwen2.5:1.5b-instruct-q4_K_M"},
    {"family": "qwen2.5-1.5b", "quant": "Q5_K_M",
     "ollama_name": "qwen2.5:1.5b-instruct-q5_K_M"},
    {"family": "qwen2.5-1.5b", "quant": "Q6_K",
     "ollama_name": "qwen2.5:1.5b-instruct-q6_K"},
    {"family": "qwen2.5-1.5b", "quant": "Q8_0",
     "ollama_name": "qwen2.5:1.5b-instruct-q8_0"},

    # Llama-3.2-1B — все 4 квантизации
    {"family": "llama3.2-1b", "quant": "Q4_K_M",
     "ollama_name": "llama3.2:1b-instruct-q4_K_M"},
    {"family": "llama3.2-1b", "quant": "Q5_K_M",
     "ollama_name": "llama3.2:1b-instruct-q5_K_M"},
    {"family": "llama3.2-1b", "quant": "Q6_K",
     "ollama_name": "llama3.2:1b-instruct-q6_K"},
    {"family": "llama3.2-1b", "quant": "Q8_0",
     "ollama_name": "llama3.2:1b-instruct-q8_0"},
]

QUANT_ORDER = ["Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"]

N_REPEATS = 3    # прогонов каждого сценария (усреднение случайности)

# ══════════════════════════════════════════════════════════════════════════════


def main():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(here, "results")
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 62)
    print("  Quantization vs Agent Security — full experiment")
    print("=" * 62)

    client = OllamaClient()
    if not client.is_available():
        print("❌ Ollama не запущена.")
        sys.exit(1)
    print("✓ Ollama доступна")

    installed = set(client.list_models())
    missing = [m["ollama_name"] for m in MODELS_TO_TEST
               if m["ollama_name"] not in installed]
    if missing:
        print("\n⚠️  Не найдены модели:")
        for m in missing:
            print(f"    ollama pull {m}")
        sys.exit(1)

    scenarios = get_all_scenarios()
    n_attack = sum(1 for s in scenarios if s["attack_type"] != "baseline")
    n_base = sum(1 for s in scenarios if s["attack_type"] == "baseline")
    total_runs = len(MODELS_TO_TEST) * len(scenarios) * N_REPEATS
    print(f"✓ Сценариев: {len(scenarios)} ({n_attack} атак + {n_base} baseline)")
    print(f"✓ Повторов: {N_REPEATS}")
    print(f"✓ Всего прогонов: {total_runs}")

    all_runs = []
    summary_rows = []

    for cfg in MODELS_TO_TEST:
        name = cfg["ollama_name"]
        print(f"\n{'─'*58}")
        print(f"  {cfg['family']} | {cfg['quant']}")
        print(f"{'─'*58}")

        agent = SimulatedAgent(client, name)
        cfg_runs = []

        for scenario in tqdm(scenarios, desc=f"  {cfg['quant']}", leave=False):
            for rep in range(N_REPEATS):
                result = agent.run_scenario(scenario)
                result["family"] = cfg["family"]
                result["quant"] = cfg["quant"]
                result["injection_style"] = scenario.get("injection_style", "none")
                result["domain"] = scenario.get("domain", "")
                result["repeat"] = rep
                cfg_runs.append(result)
                all_runs.append(result)

        # метрики только по АТАКУЮЩИМ сценариям (baseline отдельно)
        attack_runs = [r for r in cfg_runs if r["attack_type"] != "baseline"]
        metrics = aggregate(attack_runs)
        by_type = aggregate_by_attack_type(cfg_runs)
        by_style = aggregate_by_style(attack_runs)
        base_completion = baseline_completion(cfg_runs)

        print(f"  ASR: {metrics['asr_pct']:.1f}% "
              f"[{metrics['asr_ci_low']:.1f}, {metrics['asr_ci_high']:.1f}]  "
              f"Malformed: {metrics['malformed_rate_pct']:.1f}%  "
              f"Baseline-OK: {base_completion:.0f}%  "
              f"Latency: {metrics['avg_latency_ms']:.0f}ms")

        row = {
            "family": cfg["family"],
            "quant": cfg["quant"],
            "baseline_completion_pct": base_completion,
            **metrics,
        }
        for atype, m in by_type.items():
            if atype != "baseline":
                row[f"asr_{atype}"] = m["asr_pct"]
        for style, m in by_style.items():
            row[f"asr_style_{style}"] = m["asr_pct"]
        summary_rows.append(row)

    # ── статистические тесты: Q4 vs Q8 на каждой модели ──────────────────────
    print(f"\n{'='*62}")
    print("  СТАТИСТИКА: значима ли разница Q4 vs Q8?")
    print(f"{'='*62}")

    stats_results = []
    families = sorted(set(cfg["family"] for cfg in MODELS_TO_TEST))
    for fam in families:
        q4 = next((r for r in summary_rows
                   if r["family"] == fam and r["quant"] == "Q4_K_M"), None)
        q8 = next((r for r in summary_rows
                   if r["family"] == fam and r["quant"] == "Q8_0"), None)
        if q4 and q8:
            ztest = two_proportion_z(
                q4["attack_success"], q4["n_valid"],
                q8["attack_success"], q8["n_valid"])
            chi = chi_square_2x2(
                q4["attack_success"], q4["n_valid"] - q4["attack_success"],
                q8["attack_success"], q8["n_valid"] - q8["attack_success"])

            print(f"\n  {fam}:")
            print(f"    Q4 ASR = {q4['asr_pct']:.1f}%  →  Q8 ASR = {q8['asr_pct']:.1f}%")
            print(f"    Разница: {ztest['diff_pct']:.1f} п.п.")
            print(f"    z-test:  z={ztest['z']}, p={ztest['p_value']}, "
                  f"значимо={'ДА' if ztest['significant_05'] else 'нет'}")
            print(f"    chi²:    χ²={chi['chi2']}, p={chi['p_value']}, "
                  f"значимо={'ДА' if chi['significant_05'] else 'нет'}")

            stats_results.append({
                "family": fam,
                "q4_asr": q4["asr_pct"], "q8_asr": q8["asr_pct"],
                "diff_pct": ztest["diff_pct"],
                "z": ztest["z"], "z_pvalue": ztest["p_value"],
                "z_significant": ztest["significant_05"],
                "chi2": chi["chi2"], "chi2_pvalue": chi["p_value"],
                "chi2_significant": chi["significant_05"],
            })

    # ── сохранение ────────────────────────────────────────────────────────────
    with open(os.path.join(results_dir, "raw_runs.json"), "w", encoding="utf-8") as f:
        json.dump(all_runs, f, indent=2, ensure_ascii=False)
    with open(os.path.join(results_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2, ensure_ascii=False)
    with open(os.path.join(results_dir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats_results, f, indent=2, ensure_ascii=False)

    csv_path = os.path.join(results_dir, "results.csv")
    if summary_rows:
        keys = sorted(set().union(*[r.keys() for r in summary_rows]))
        front = ["family", "quant", "asr_pct", "asr_ci_low", "asr_ci_high",
                 "defense_rate_pct", "refusal_rate_pct", "malformed_rate_pct",
                 "baseline_completion_pct", "avg_latency_ms"]
        ordered = front + [k for k in keys if k not in front]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=ordered)
            w.writeheader()
            for row in summary_rows:
                w.writerow(row)

    print(f"\n{'='*62}")
    print(f"[Saved] {csv_path}")
    print(f"[Saved] results.json, stats.json, raw_runs.json")
    print("\n[Done] Запусти: python experiments/visualize.py")


if __name__ == "__main__":
    main()
