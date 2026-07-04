"""
experiments/run.py — финальный эксперимент (публикационное качество).

Дизайн:
  temp = 0.7            реальный режим агентов (не детерминизм)
  seeds = [42,123,456]  независимые прогоны для воспроизводимости
  Полная матрица моделей × квантизаций
  Метрики агрегируются ПО SEEDS (mean ± std, честные CI)

Проверяемые гипотезы:
  H1: квантизация НЕ влияет на устойчивость к injection
  H2: стиль инъекции — доминирующий фактор
  H3: устойчивость зависит от размера модели
  H4: разные семейства имеют разные слабости

Запуск (Ollama должна быть запущена):
    python experiments/run.py

Настройки — блок CONFIG. Финальный прогон長, лучше на ночь.
Убедись, что ноутбук на питании и не уходит в сон.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import csv
from collections import defaultdict
from tqdm import tqdm

from src.ollama_client import OllamaClient
from src.agent import SimulatedAgent
from src.attacks import get_all_scenarios
from src.metrics import (aggregate, aggregate_by_attack_type,
                         aggregate_by_style, baseline_completion)
from src.stats import (two_proportion_z, chi_square_2x2, cohens_h,
                       pairwise_comparison, aggregate_over_seeds)


# ══ CONFIG ════════════════════════════════════════════════════════════════════

# Финальный набор моделей. Имена ДОЛЖНЫ совпадать с `ollama list`.
# family — для группировки; size_b — размер в млрд параметров (для H3);
# arch — семейство архитектуры (для H4).
MODELS_TO_TEST = [
    # ── Qwen2.5-0.5B: крайние кванты (size-ось) ──
    {"family": "qwen2.5-0.5b", "size_b": 0.5, "arch": "qwen", "quant": "Q4_K_M",
     "ollama_name": "qwen2.5:0.5b-instruct-q4_K_M"},
    {"family": "qwen2.5-0.5b", "size_b": 0.5, "arch": "qwen", "quant": "Q8_0",
     "ollama_name": "qwen2.5:0.5b-instruct-q8_0"},

    # ── Qwen2.5-1.5B: полная кривая квантизации (H1) ──
    {"family": "qwen2.5-1.5b", "size_b": 1.5, "arch": "qwen", "quant": "Q4_K_M",
     "ollama_name": "qwen2.5:1.5b-instruct-q4_K_M"},
    {"family": "qwen2.5-1.5b", "size_b": 1.5, "arch": "qwen", "quant": "Q5_K_M",
     "ollama_name": "qwen2.5:1.5b-instruct-q5_K_M"},
    {"family": "qwen2.5-1.5b", "size_b": 1.5, "arch": "qwen", "quant": "Q6_K",
     "ollama_name": "qwen2.5:1.5b-instruct-q6_K"},
    {"family": "qwen2.5-1.5b", "size_b": 1.5, "arch": "qwen", "quant": "Q8_0",
     "ollama_name": "qwen2.5:1.5b-instruct-q8_0"},

    # ── Qwen2.5-3B: крайние кванты (size-ось) ──
    {"family": "qwen2.5-3b", "size_b": 3.0, "arch": "qwen", "quant": "Q4_K_M",
     "ollama_name": "qwen2.5:3b-instruct-q4_K_M"},
    {"family": "qwen2.5-3b", "size_b": 3.0, "arch": "qwen", "quant": "Q8_0",
     "ollama_name": "qwen2.5:3b-instruct-q8_0"},

    # ── Llama-3.2-1B: полная кривая квантизации (H1, arch-ось) ──
    {"family": "llama3.2-1b", "size_b": 1.0, "arch": "llama", "quant": "Q4_K_M",
     "ollama_name": "llama3.2:1b-instruct-q4_K_M"},
    {"family": "llama3.2-1b", "size_b": 1.0, "arch": "llama", "quant": "Q5_K_M",
     "ollama_name": "llama3.2:1b-instruct-q5_K_M"},
    {"family": "llama3.2-1b", "size_b": 1.0, "arch": "llama", "quant": "Q6_K",
     "ollama_name": "llama3.2:1b-instruct-q6_K"},
    {"family": "llama3.2-1b", "size_b": 1.0, "arch": "llama", "quant": "Q8_0",
     "ollama_name": "llama3.2:1b-instruct-q8_0"},

    # ── Gemma2-2B: один квант (arch-ось; Q8 не докачался — сетевая ошибка) ──
    {"family": "gemma2-2b", "size_b": 2.0, "arch": "gemma", "quant": "Q4_K_M",
     "ollama_name": "gemma2:2b-instruct-q4_K_M"},

    # ── Phi-3-mini: один вариант (arch-ось) ──
    {"family": "phi3-mini", "size_b": 3.8, "arch": "phi", "quant": "default",
     "ollama_name": "phi3:mini"},
]

QUANT_ORDER = ["Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"]

SEEDS = [42, 123, 456]     # независимые прогоны для воспроизводимости
TEMPERATURE = 0.7          # реальный режим агентов

# ══════════════════════════════════════════════════════════════════════════════


def main():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(here, "results")
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 64)
    print("  Quantization vs Agent Security — FINAL experiment")
    print(f"  temp={TEMPERATURE}, seeds={SEEDS}")
    print("=" * 64)

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
    total = len(MODELS_TO_TEST) * len(scenarios) * len(SEEDS)
    print(f"✓ Моделей: {len(MODELS_TO_TEST)} | Сценариев: {len(scenarios)} "
          f"({n_attack} атак) | Seeds: {len(SEEDS)}")
    print(f"✓ Всего прогонов: {total}")

    all_runs = []
    # структура: per_config[config_key][seed] = список прогонов
    summary_rows = []

    for cfg in MODELS_TO_TEST:
        name = cfg["ollama_name"]
        print(f"\n{'─'*60}")
        print(f"  {cfg['family']} | {cfg['quant']} (size={cfg['size_b']}B, {cfg['arch']})")
        print(f"{'─'*60}")

        agent = SimulatedAgent(client, name)

        # прогоны по каждому seed отдельно (для агрегации по seeds)
        per_seed_runs = {seed: [] for seed in SEEDS}

        for seed in SEEDS:
            for scenario in tqdm(scenarios,
                                 desc=f"  {cfg['quant']} seed={seed}", leave=False):
                result = agent.run_scenario(scenario, seed=seed,
                                            temperature=TEMPERATURE)
                result.update({
                    "family": cfg["family"], "quant": cfg["quant"],
                    "size_b": cfg["size_b"], "arch": cfg["arch"],
                    "injection_style": scenario.get("injection_style", "none"),
                    "domain": scenario.get("domain", ""), "seed": seed,
                })
                per_seed_runs[seed].append(result)
                all_runs.append(result)

        # ── агрегация: считаем ASR отдельно по каждому seed, потом усредняем ──
        per_seed_asr = []
        for seed in SEEDS:
            attack_runs = [r for r in per_seed_runs[seed]
                           if r["attack_type"] != "baseline"]
            m = aggregate(attack_runs, with_ci=False)
            per_seed_asr.append(m["asr_pct"])

        seed_agg = aggregate_over_seeds(per_seed_asr)

        # общая агрегация (все seeds вместе) для разбивок
        all_attack = [r for r in all_runs
                      if r["family"] == cfg["family"] and r["quant"] == cfg["quant"]
                      and r["attack_type"] != "baseline"]
        metrics = aggregate(all_attack)
        by_type = aggregate_by_attack_type(
            [r for r in all_runs if r["family"] == cfg["family"]
             and r["quant"] == cfg["quant"]])
        by_style = aggregate_by_style(all_attack)
        base_comp = baseline_completion(
            [r for r in all_runs if r["family"] == cfg["family"]
             and r["quant"] == cfg["quant"]])

        print(f"  ASR (по seeds): {seed_agg['mean']:.1f}% ± {seed_agg['std']:.1f} "
              f"[{seed_agg['ci_low']:.1f}, {seed_agg['ci_high']:.1f}]  "
              f"Malformed: {metrics['malformed_rate_pct']:.1f}%  "
              f"Baseline-OK: {base_comp:.0f}%")

        row = {
            "family": cfg["family"], "quant": cfg["quant"],
            "size_b": cfg["size_b"], "arch": cfg["arch"],
            "asr_mean": seed_agg["mean"], "asr_std": seed_agg["std"],
            "asr_ci_low": seed_agg["ci_low"], "asr_ci_high": seed_agg["ci_high"],
            "asr_seed_min": seed_agg["min"], "asr_seed_max": seed_agg["max"],
            "malformed_rate_pct": metrics["malformed_rate_pct"],
            "baseline_completion_pct": base_comp,
            "avg_latency_ms": metrics["avg_latency_ms"],
            "n_valid": metrics["n_valid"], "attack_success": metrics["attack_success"],
        }
        for atype, mm in by_type.items():
            if atype != "baseline":
                row[f"asr_{atype}"] = mm["asr_pct"]
        for style, mm in by_style.items():
            row[f"asr_style_{style}"] = mm["asr_pct"]
        summary_rows.append(row)

    # ══ АНАЛИЗ ГИПОТЕЗ ═══════════════════════════════════════════════════════
    analysis = {}

    # ── H1: влияет ли квантизация? (Q4 vs Q8 на каждой модели) ──
    print(f"\n{'='*64}")
    print("  H1: влияет ли квантизация? (Q4 vs Q8)")
    print(f"{'='*64}")
    h1 = []
    families = sorted(set(c["family"] for c in MODELS_TO_TEST))
    for fam in families:
        q4 = next((r for r in summary_rows if r["family"]==fam and r["quant"]=="Q4_K_M"), None)
        q8 = next((r for r in summary_rows if r["family"]==fam and r["quant"]=="Q8_0"), None)
        if q4 and q8:
            zt = two_proportion_z(q4["attack_success"], q4["n_valid"],
                                  q8["attack_success"], q8["n_valid"])
            h = cohens_h(q4["attack_success"]/q4["n_valid"] if q4["n_valid"] else 0,
                         q8["attack_success"]/q8["n_valid"] if q8["n_valid"] else 0)
            sig = "ДА" if zt["significant_05"] else "нет"
            print(f"  {fam:<16} Q4={q4['asr_mean']:.0f}% Q8={q8['asr_mean']:.0f}% "
                  f"p={zt['p_value']:.3f} значимо={sig} h={h['h']}({h['magnitude']})")
            h1.append({"family": fam, "q4_asr": q4["asr_mean"], "q8_asr": q8["asr_mean"],
                       "p_value": zt["p_value"], "significant": zt["significant_05"],
                       "effect_h": h["h"], "effect_mag": h["magnitude"]})
    analysis["H1_quantization"] = h1

    # ── H2: стиль инъекции — доминирующий фактор? ──
    print(f"\n{'='*64}")
    print("  H2: стиль инъекции — попарные сравнения (все данные)")
    print(f"{'='*64}")
    style_groups = defaultdict(lambda: {"success": 0, "n": 0})
    for r in all_runs:
        if r["attack_type"] == "baseline":
            continue
        st = r["injection_style"]
        if r["outcome"] in ("attack_success", "defense_hold", "refused"):
            style_groups[st]["n"] += 1
            if r["outcome"] == "attack_success":
                style_groups[st]["success"] += 1
    pw = pairwise_comparison(dict(style_groups))
    print(f"  {pw['n_comparisons']} пар, значимых после Холма: "
          f"{pw['n_significant_after_correction']}")
    # покажем ASR по стилям
    for st, g in sorted(style_groups.items(),
                        key=lambda x: -x[1]["success"]/max(x[1]["n"],1)):
        asr = 100*g["success"]/g["n"] if g["n"] else 0
        print(f"    {st:<12} ASR={asr:.1f}% (n={g['n']})")
    analysis["H2_style"] = pw

    # ── H3: влияет ли размер? (Qwen 0.5/1.5/3B на одном кванте) ──
    print(f"\n{'='*64}")
    print("  H3: влияет ли размер модели? (Qwen, Q8)")
    print(f"{'='*64}")
    h3 = []
    for size in [0.5, 1.5, 3.0]:
        row = next((r for r in summary_rows
                    if r["arch"]=="qwen" and r["size_b"]==size and r["quant"]=="Q8_0"), None)
        if row:
            print(f"  Qwen-{size}B (Q8): ASR={row['asr_mean']:.1f}%")
            h3.append({"size_b": size, "asr": row["asr_mean"]})
    analysis["H3_size"] = h3

    # ── H4: разные семейства — разные слабости? (roleplay) ──
    print(f"\n{'='*64}")
    print("  H4: архитектурные различия (ASR по стилю roleplay)")
    print(f"{'='*64}")
    h4 = []
    for arch in sorted(set(c["arch"] for c in MODELS_TO_TEST)):
        arch_runs = [r for r in all_runs if r["arch"]==arch
                     and r["injection_style"]=="roleplay"
                     and r["outcome"] in ("attack_success","defense_hold","refused")]
        if arch_runs:
            succ = sum(1 for r in arch_runs if r["outcome"]=="attack_success")
            asr = 100*succ/len(arch_runs)
            print(f"  {arch:<8} roleplay ASR={asr:.1f}% (n={len(arch_runs)})")
            h4.append({"arch": arch, "roleplay_asr": round(asr,1), "n": len(arch_runs)})
    analysis["H4_arch"] = h4

    # ── сохранение ────────────────────────────────────────────────────────────
    with open(os.path.join(results_dir, "raw_runs.json"), "w", encoding="utf-8") as f:
        json.dump(all_runs, f, indent=2, ensure_ascii=False)
    with open(os.path.join(results_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2, ensure_ascii=False)
    with open(os.path.join(results_dir, "analysis.json"), "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)

    csv_path = os.path.join(results_dir, "results.csv")
    if summary_rows:
        keys = sorted(set().union(*[r.keys() for r in summary_rows]))
        front = ["family", "arch", "size_b", "quant", "asr_mean", "asr_std",
                 "asr_ci_low", "asr_ci_high", "malformed_rate_pct",
                 "baseline_completion_pct", "avg_latency_ms"]
        ordered = front + [k for k in keys if k not in front]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=ordered)
            w.writeheader()
            for row in summary_rows:
                w.writerow(row)

    print(f"\n{'='*64}")
    print(f"[Saved] {csv_path}")
    print(f"[Saved] results.json, analysis.json, raw_runs.json")
    print("\n[Done] Запусти: python experiments/visualize.py")


if __name__ == "__main__":
    main()
