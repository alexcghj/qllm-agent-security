"""
experiments/temp_sweep.py — контроль на температуру (пункт M3).

Рецензент справедливо спросит: почему temp=0.7? Не привязан ли вывод
"квантизация не влияет" к одной произвольной температуре?

Этот скрипт проверяет robustness вывода H1 по температурам. Чтобы не
взрывать время, он фокусный: одна модель с ПОЛНОЙ кривой квантизации
(Qwen2.5-1.5B, Q4/Q5/Q6/Q8) прогоняется при temp ∈ {0.0, 0.7, 1.0}.
Если при всех температурах Q4≈Q8 — вывод не артефакт одной temp.

    python experiments/temp_sweep.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from tqdm import tqdm

from src.ollama_client import OllamaClient
from src.agent import SimulatedAgent
from src.attacks import get_all_scenarios
from src.metrics import aggregate
from src.stats import two_proportion_z, tost_two_proportions

# фокус: одна модель, полная кривая квантизации
CONFIGS = [
    {"quant": "Q4_K_M", "ollama_name": "qwen2.5:1.5b-instruct-q4_K_M"},
    {"quant": "Q5_K_M", "ollama_name": "qwen2.5:1.5b-instruct-q5_K_M"},
    {"quant": "Q6_K",   "ollama_name": "qwen2.5:1.5b-instruct-q6_K"},
    {"quant": "Q8_0",   "ollama_name": "qwen2.5:1.5b-instruct-q8_0"},
]
TEMPERATURES = [0.0, 0.7, 1.0]
SEEDS = [42, 123, 456]   # для temp=0 seed не влияет, но держим единообразно


def main():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(here, "results")
    os.makedirs(results_dir, exist_ok=True)

    client = OllamaClient()
    if not client.is_available():
        print("❌ Ollama не запущена.")
        sys.exit(1)

    scenarios = [s for s in get_all_scenarios() if s["attack_type"] != "baseline"]
    print(f"Temp-sweep: {len(CONFIGS)} квантизаций × {len(TEMPERATURES)} температур")
    print(f"Сценариев (атак): {len(scenarios)}\n")

    results = {}   # results[temp][quant] = ASR

    for temp in TEMPERATURES:
        results[temp] = {}
        print(f"\n{'='*50}\n  temperature = {temp}\n{'='*50}")

        for cfg in CONFIGS:
            agent = SimulatedAgent(client, cfg["ollama_name"])
            runs = []
            seeds = SEEDS if temp > 0 else [SEEDS[0]]  # temp=0 детерминирован
            for seed in seeds:
                for scn in tqdm(scenarios, desc=f"  {cfg['quant']} t={temp}",
                                leave=False):
                    r = agent.run_scenario(scn, seed=seed, temperature=temp)
                    runs.append(r)
            m = aggregate(runs)
            results[temp][cfg["quant"]] = {
                "asr": m["asr_pct"], "n_valid": m["n_valid"],
                "success": m["attack_success"],
            }
            print(f"  {cfg['quant']}: ASR={m['asr_pct']:.1f}% "
                  f"(malformed={m['malformed_rate_pct']:.1f}%)")

    # ── анализ: держится ли Q4≈Q8 при каждой температуре? ──
    print(f"\n{'='*50}\n  H1 robustness по температурам\n{'='*50}")
    summary = []
    for temp in TEMPERATURES:
        q4 = results[temp]["Q4_K_M"]
        q8 = results[temp]["Q8_0"]
        zt = two_proportion_z(q4["success"], q4["n_valid"],
                              q8["success"], q8["n_valid"])
        tost = tost_two_proportions(q4["success"], q4["n_valid"],
                                    q8["success"], q8["n_valid"], margin=0.10)
        verdict = "Q4≈Q8 (нет эффекта)" if not zt["significant_05"] else "РАЗЛИЧАЮТСЯ"
        print(f"  temp={temp}: Q4={q4['asr']:.0f}% Q8={q8['asr']:.0f}% "
              f"p={zt['p_value']:.3f} → {verdict}")
        summary.append({
            "temperature": temp,
            "q4_asr": q4["asr"], "q8_asr": q8["asr"],
            "p_value": zt["p_value"],
            "significant": zt["significant_05"],
            "tost_equivalent": tost["equivalent"],
        })

    # вывод
    all_null = all(not s["significant"] for s in summary)
    print()
    if all_null:
        print("✓ Вывод H1 УСТОЙЧИВ: квантизация не влияет при ВСЕХ температурах.")
    else:
        print("⚠ При некоторых температурах эффект появляется — обсудить в статье.")

    out = {"by_temperature": results, "h1_robustness": summary}
    with open(os.path.join(results_dir, "temp_sweep.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n[Saved] results/temp_sweep.json")


if __name__ == "__main__":
    main()
