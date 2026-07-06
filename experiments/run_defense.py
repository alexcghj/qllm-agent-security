"""
experiments/run_defense.py — сравнение защит weak vs hardened (пункт C2).

Отвечает на критику "соломенного чучела": вдруг высокий ASR — артефакт
намеренно слабой защиты? Прогоняем те же сценарии с УСИЛЕННОЙ защитой
(spotlighting + разделители недоверенных данных) и смотрим:
  1. снижает ли hardened-защита ASR (значит атака реальна, а не тривиальна)
  2. держится ли вывод "квантизация не влияет" и при hardened-защите

    python experiments/run_defense.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from tqdm import tqdm

from src.ollama_client import OllamaClient
from src.agent import SimulatedAgent
from src.attacks import get_all_scenarios
from src.metrics import aggregate
from src.stats import two_proportion_z

# фокус: 2 семейства × Q4/Q8 × 2 защиты
CONFIGS = [
    {"family": "qwen2.5-1.5b", "quant": "Q4_K_M",
     "ollama_name": "qwen2.5:1.5b-instruct-q4_K_M"},
    {"family": "qwen2.5-1.5b", "quant": "Q8_0",
     "ollama_name": "qwen2.5:1.5b-instruct-q8_0"},
    {"family": "llama3.2-1b", "quant": "Q4_K_M",
     "ollama_name": "llama3.2:1b-instruct-q4_K_M"},
    {"family": "llama3.2-1b", "quant": "Q8_0",
     "ollama_name": "llama3.2:1b-instruct-q8_0"},
]
SEEDS = [42, 123, 456]
TEMPERATURE = 0.7


def main():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    client = OllamaClient()
    if not client.is_available():
        print("❌ Ollama не запущена.")
        sys.exit(1)

    scenarios = [s for s in get_all_scenarios() if s["attack_type"] != "baseline"]
    print(f"Defense comparison: {len(CONFIGS)} конфиг × 2 защиты (weak/hardened)")
    print(f"Сценариев: {len(scenarios)}, seeds: {len(SEEDS)}\n")

    summary = []
    for defense in ["weak", "hardened"]:
        print(f"\n{'='*50}\n  defense = {defense}\n{'='*50}")
        for cfg in CONFIGS:
            agent = SimulatedAgent(client, cfg["ollama_name"], defense=defense)
            runs = []
            for seed in SEEDS:
                for scn in tqdm(scenarios, desc=f"  {cfg['quant']} {defense}",
                                leave=False):
                    r = agent.run_scenario(scn, seed=seed, temperature=TEMPERATURE)
                    runs.append(r)
            m = aggregate(runs)
            print(f"  {cfg['family']} {cfg['quant']}: ASR={m['asr_pct']:.1f}%")
            summary.append({"family": cfg["family"], "quant": cfg["quant"],
                            "defense": defense, "asr": m["asr_pct"],
                            "n_valid": m["n_valid"],
                            "success": m["attack_success"]})

    # ── анализ ──
    print(f"\n{'='*50}\n  Эффект защиты и robustness H1\n{'='*50}")
    # 1. снижает ли hardened ASR?
    for cfg in CONFIGS:
        w = next(r for r in summary if r["family"]==cfg["family"]
                 and r["quant"]==cfg["quant"] and r["defense"]=="weak")
        h = next(r for r in summary if r["family"]==cfg["family"]
                 and r["quant"]==cfg["quant"] and r["defense"]=="hardened")
        zt = two_proportion_z(w["success"], w["n_valid"],
                              h["success"], h["n_valid"])
        drop = w["asr"] - h["asr"]
        print(f"  {cfg['family']} {cfg['quant']}: weak={w['asr']:.0f}% → "
              f"hardened={h['asr']:.0f}% (Δ={drop:+.0f}пп, p={zt['p_value']:.3f})")

    # 2. держится ли Q4≈Q8 при hardened?
    print()
    for fam in sorted(set(c["family"] for c in CONFIGS)):
        q4 = next(r for r in summary if r["family"]==fam
                  and r["quant"]=="Q4_K_M" and r["defense"]=="hardened")
        q8 = next(r for r in summary if r["family"]==fam
                  and r["quant"]=="Q8_0" and r["defense"]=="hardened")
        zt = two_proportion_z(q4["success"], q4["n_valid"],
                              q8["success"], q8["n_valid"])
        v = "Q4≈Q8 (держится)" if not zt["significant_05"] else "различаются"
        print(f"  [hardened] {fam}: Q4={q4['asr']:.0f}% Q8={q8['asr']:.0f}% "
              f"p={zt['p_value']:.3f} → {v}")

    with open(os.path.join(here, "results", "defense_results.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n[Saved] results/defense_results.json")


if __name__ == "__main__":
    main()
