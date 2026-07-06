"""
experiments/run_injecagent.py — прогон на бенчмарке InjecAgent (пункты C3+C1).

Проверяет внешнюю валидность: держатся ли наши выводы на ПРИЗНАННОМ
бенчмарке, а не только на наших сценариях. Прогоняет подвыборку InjecAgent
через тот же agent + четырёхисходный классификатор.

Требует склонированный InjecAgent рядом с проектом:
    git clone https://github.com/uiuc-kang-lab/InjecAgent.git

    python experiments/run_injecagent.py [путь_к_InjecAgent]
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from tqdm import tqdm

from src.ollama_client import OllamaClient
from src.agent import SimulatedAgent
from src.injecagent_adapter import load_injecagent_scenarios
from src.metrics import aggregate
from src.stats import two_proportion_z, tost_two_proportions

# те же модели с полной кривой квантизации — для проверки H1 на внешнем бенчмарке
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
MAX_PER_TYPE = 60   # подвыборка: 60 direct_harm + 60 data_stealing = 120


def main():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ia_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(here), "InjecAgent")
    if not os.path.isdir(os.path.join(ia_dir, "data")):
        print(f"❌ InjecAgent не найден в {ia_dir}")
        print("   git clone https://github.com/uiuc-kang-lab/InjecAgent.git")
        sys.exit(1)

    client = OllamaClient()
    if not client.is_available():
        print("❌ Ollama не запущена.")
        sys.exit(1)

    scenarios = load_injecagent_scenarios(ia_dir, setting="base",
                                          max_per_type=MAX_PER_TYPE)
    print(f"InjecAgent: {len(scenarios)} сценариев (base setting)")
    print(f"Конфигураций: {len(CONFIGS)}, seeds: {len(SEEDS)}\n")

    summary = []
    for cfg in CONFIGS:
        agent = SimulatedAgent(client, cfg["ollama_name"])
        runs = []
        for seed in SEEDS:
            for scn in tqdm(scenarios, desc=f"  {cfg['family']} {cfg['quant']}",
                            leave=False):
                r = agent.run_scenario(scn, seed=seed, temperature=TEMPERATURE)
                runs.append(r)
        m = aggregate(runs)
        print(f"  {cfg['family']} {cfg['quant']}: ASR={m['asr_pct']:.1f}% "
              f"(malformed={m['malformed_rate_pct']:.1f}%, n={m['n_valid']})")
        summary.append({"family": cfg["family"], "quant": cfg["quant"],
                        "asr": m["asr_pct"], "n_valid": m["n_valid"],
                        "success": m["attack_success"],
                        "malformed_pct": m["malformed_rate_pct"]})

    # H1 на InjecAgent: Q4 vs Q8 по каждому семейству
    print(f"\n{'='*50}\n  H1 на InjecAgent (внешняя валидность)\n{'='*50}")
    for fam in sorted(set(c["family"] for c in CONFIGS)):
        q4 = next((r for r in summary if r["family"]==fam and r["quant"]=="Q4_K_M"), None)
        q8 = next((r for r in summary if r["family"]==fam and r["quant"]=="Q8_0"), None)
        if q4 and q8:
            zt = two_proportion_z(q4["success"], q4["n_valid"],
                                  q8["success"], q8["n_valid"])
            v = "Q4≈Q8" if not zt["significant_05"] else "РАЗЛИЧАЮТСЯ"
            print(f"  {fam}: Q4={q4['asr']:.0f}% Q8={q8['asr']:.0f}% "
                  f"p={zt['p_value']:.3f} → {v}")

    with open(os.path.join(here, "results", "injecagent_results.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n[Saved] results/injecagent_results.json")


if __name__ == "__main__":
    main()
