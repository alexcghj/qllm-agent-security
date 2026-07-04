"""
experiments/preflight.py — быстрая проверка ГОТОВНОСТИ перед ночным прогоном.

Запусти ЭТО перед полным run.py, чтобы за ~2 минуты убедиться:
  1. Ollama запущена
  2. Все модели из MODELS_TO_TEST установлены
  3. Каждая модель реально отвечает и формат парсится
  4. Классификатор работает на всех моделях

Экономит ночь: ловит битые имена, проблемы формата, зависшие модели
ДО того, как запустишь многочасовой эксперимент.

    python experiments/preflight.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ollama_client import OllamaClient
from src.agent import SimulatedAgent
from src.attacks import get_all_scenarios

# импортируем список моделей из run.py
from experiments.run import MODELS_TO_TEST, SEEDS, TEMPERATURE


def main():
    print("=" * 60)
    print("  PREFLIGHT — проверка готовности к эксперименту")
    print("=" * 60)

    # 1. Ollama
    client = OllamaClient()
    if not client.is_available():
        print("❌ Ollama не запущена. Запусти приложение Ollama.")
        sys.exit(1)
    print("✓ Ollama доступна")

    # 2. Модели установлены
    installed = set(client.list_models())
    missing = [m["ollama_name"] for m in MODELS_TO_TEST
               if m["ollama_name"] not in installed]
    if missing:
        print(f"\n❌ Не найдено моделей: {len(missing)}")
        for m in missing:
            print(f"    ollama pull {m}")
        sys.exit(1)
    print(f"✓ Все {len(MODELS_TO_TEST)} моделей установлены")

    # 3+4. Каждая модель отвечает и формат парсится
    print(f"\nПроверяю ответ каждой модели (temp={TEMPERATURE})...")
    scenarios = get_all_scenarios()
    test_scenario = next(s for s in scenarios if s["attack_type"] != "baseline")

    problems = []
    for cfg in MODELS_TO_TEST:
        name = cfg["ollama_name"]
        agent = SimulatedAgent(client, name)
        result = agent.run_scenario(test_scenario, seed=SEEDS[0],
                                    temperature=TEMPERATURE)
        outcome = result["outcome"]
        lat = result["latency_ms"]

        # проверяем, что модель дала осмысленный исход, не сплошной malformed
        status = "✓" if outcome != "malformed" else "⚠ MALFORMED"
        if outcome == "malformed":
            problems.append(name)
        print(f"  {status}  {cfg['family']:<16}{cfg['quant']:<9} "
              f"outcome={outcome:<15} latency={lat:.0f}ms")

    # оценка времени
    print(f"\n{'─'*60}")
    n_runs = len(MODELS_TO_TEST) * len(scenarios) * len(SEEDS)
    # грубая оценка по средней латентности первого прогона
    print(f"Всего прогонов в полном эксперименте: {n_runs}")
    print(f"(модели × сценарии × seeds = "
          f"{len(MODELS_TO_TEST)} × {len(scenarios)} × {len(SEEDS)})")

    if problems:
        print(f"\n⚠️  {len(problems)} моделей дали MALFORMED на тесте.")
        print("    Это может быть ОК (разово при temp>0) или проблема формата.")
        print("    Если модель СТАБИЛЬНО malformed — она плохо держит JSON.")
        print("    Реши: убрать её или оставить (malformed — тоже результат).")
    else:
        print("\n✓ Все модели отвечают и формат парсится.")

    print("\n[Preflight OK] Можно запускать: python experiments/run.py")


if __name__ == "__main__":
    main()
