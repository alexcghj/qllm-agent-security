"""
experiments/run.py — главный скрипт эксперимента.

Запуск (с запущенной Ollama):
    python experiments/run.py

Логика:
  1. Проверяем доступность Ollama и наличие моделей
  2. Для каждой квантизации (Q4/Q5/Q6/Q8):
       для каждого сценария (direct_harm + data_stealing):
         прогоняем агента, классифицируем исход
  3. Агрегируем метрики (ASR, malformed rate, latency)
  4. Сохраняем results/results.csv и results/results.json
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import csv
from tqdm import tqdm

from src.ollama_client import OllamaClient
from src.agent import SimulatedAgent
from src.attacks import get_all_scenarios
from src.metrics import aggregate, aggregate_by_attack_type


# ── конфигурация: какие модели/квантизации тестируем ──────────────────────────

# имена должны точно совпадать с тем, что в `ollama list`
MODELS_TO_TEST = [
    {"family": "qwen2.5-1.5b", "quant": "Q4_K_M",
     "ollama_name": "qwen2.5:1.5b-instruct-q4_K_M"},
    {"family": "qwen2.5-1.5b", "quant": "Q5_K_M",
     "ollama_name": "qwen2.5:1.5b-instruct-q5_K_M"},
    {"family": "qwen2.5-1.5b", "quant": "Q6_K",
     "ollama_name": "qwen2.5:1.5b-instruct-q6_K"},
    {"family": "qwen2.5-1.5b", "quant": "Q8_0",
     "ollama_name": "qwen2.5:1.5b-instruct-q8_0"},
]

# порядок квантизаций для оси X графика (от сильного сжатия к слабому)
QUANT_ORDER = ["Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"]

N_REPEATS = 1   # сколько раз прогнать каждый сценарий
                # (temp=0 → детерминированно, 1 достаточно;
                #  можно >1 если хочешь усреднить случайность)


def main():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(here, "results")
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 60)
    print("  Quantization vs Agent Security")
    print("=" * 60)

    # ── проверка Ollama ───────────────────────────────────────────────────────
    client = OllamaClient()
    if not client.is_available():
        print("❌ Ollama не запущена. Запусти приложение Ollama или `ollama serve`.")
        sys.exit(1)
    print("✓ Ollama доступна")

    installed = set(client.list_models())
    print(f"✓ Установлено моделей: {len(installed)}")

    # проверяем, что нужные модели есть
    missing = [m["ollama_name"] for m in MODELS_TO_TEST
               if m["ollama_name"] not in installed]
    if missing:
        print("\n⚠️  Не найдены модели:")
        for m in missing:
            print(f"    ollama pull {m}")
        print("\nСкачай их и запусти снова.")
        sys.exit(1)

    scenarios = get_all_scenarios()
    print(f"✓ Сценариев: {len(scenarios)} "
          f"({sum(1 for s in scenarios if s['attack_type']=='direct_harm')} direct_harm, "
          f"{sum(1 for s in scenarios if s['attack_type']=='data_stealing')} data_stealing)")

    # ── прогон ────────────────────────────────────────────────────────────────
    all_runs = []          # сырые результаты каждого прогона
    summary_rows = []      # агрегаты по конфигурациям

    for cfg in MODELS_TO_TEST:
        name = cfg["ollama_name"]
        print(f"\n{'─'*55}")
        print(f"  {cfg['family']} | {cfg['quant']}")
        print(f"{'─'*55}")

        agent = SimulatedAgent(client, name)
        cfg_runs = []

        for scenario in tqdm(scenarios, desc=f"  {cfg['quant']}", leave=False):
            for rep in range(N_REPEATS):
                result = agent.run_scenario(scenario)
                result["family"] = cfg["family"]
                result["quant"] = cfg["quant"]
                result["repeat"] = rep
                cfg_runs.append(result)
                all_runs.append(result)

        # агрегируем по этой конфигурации
        metrics = aggregate(cfg_runs)
        by_type = aggregate_by_attack_type(cfg_runs)

        print(f"  ASR: {metrics['asr_pct']:.1f}%  "
              f"Defense: {metrics['defense_rate_pct']:.1f}%  "
              f"Refused: {metrics['refusal_rate_pct']:.1f}%  "
              f"Malformed: {metrics['malformed_rate_pct']:.1f}%  "
              f"Latency: {metrics['avg_latency_ms']:.0f}ms")

        row = {
            "family": cfg["family"],
            "quant": cfg["quant"],
            **metrics,
        }
        # добавляем разбивку по типам атак
        for atype, m in by_type.items():
            row[f"asr_{atype}"] = m["asr_pct"]
            row[f"malformed_{atype}"] = m["malformed_rate_pct"]
        summary_rows.append(row)

    # ── сохранение ────────────────────────────────────────────────────────────
    # сырые прогоны
    raw_path = os.path.join(results_dir, "raw_runs.json")
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(all_runs, f, indent=2, ensure_ascii=False)

    # сводка
    summary_path = os.path.join(results_dir, "results.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2, ensure_ascii=False)

    # CSV
    csv_path = os.path.join(results_dir, "results.csv")
    if summary_rows:
        keys = sorted(set().union(*[r.keys() for r in summary_rows]))
        # ставим важные колонки вперёд
        front = ["family", "quant", "asr_pct", "defense_rate_pct",
                 "refusal_rate_pct", "malformed_rate_pct", "avg_latency_ms"]
        ordered = front + [k for k in keys if k not in front]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=ordered)
            w.writeheader()
            for row in summary_rows:
                w.writerow(row)

    print(f"\n{'='*60}")
    print(f"[Saved] {csv_path}")
    print(f"[Saved] {summary_path}")
    print(f"[Saved] {raw_path}")

    # ── итоговая таблица ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  ИТОГ: ASR по уровням квантизации")
    print(f"{'='*60}")
    print(f"  {'Quant':<10} {'ASR':>8} {'Malformed':>12} {'Latency':>10}")
    print("  " + "-" * 42)
    for q in QUANT_ORDER:
        row = next((r for r in summary_rows if r["quant"] == q), None)
        if row:
            print(f"  {q:<10} {row['asr_pct']:>7.1f}% "
                  f"{row['malformed_rate_pct']:>11.1f}% "
                  f"{row['avg_latency_ms']:>9.0f}ms")

    print("\n[Done] Запусти: python experiments/visualize.py")


if __name__ == "__main__":
    main()
