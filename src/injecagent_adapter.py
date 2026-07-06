"""
src/injecagent_adapter.py — адаптер бенчмарка InjecAgent под наш pipeline.

InjecAgent (Zhan et al., ACL Findings 2024) — признанный бенчмарк
indirect prompt injection для агентов: 1054 теста, 17 user-инструментов,
62 attacker-инструмента, две категории (direct harm / data stealing).

Этот модуль конвертирует их тест-кейсы в формат наших сценариев, чтобы
прогнать ТОТ ЖЕ pipeline (agent + четырёхисходный классификатор) на
внешнем стандартном бенчмарке. Это закрывает вопрос внешней валидности:
результаты проверяются не только на наших сценариях.

Ключевое соответствие форматов:
  InjecAgent "Tool Response"        → наш tool_response (инъекция уже внутри)
  InjecAgent "Attacker Tools"[0]    → наш attack_action (успех = вызов его)
  InjecAgent "User Tool"            → наш tool_called (что агент уже вызвал)
  InjecAgent "User Instruction"     → наш user_instruction

Примечание по методологии: как и ряд недавних работ, мы предзаполняем
первый tool-call и подаём attacker-инъекцию внутри tool-результата; успех
атаки = вызов attacker-инструмента. Мы используем BASE setting (без
дополнительного hacking-промпта), т.к. изучаем чистую уязвимость.
"""

import json
import os
from typing import List, Dict


def _extract_tool_name(attacker_tools) -> str:
    """Имя attacker-инструмента (его вызов = успешная атака)."""
    if isinstance(attacker_tools, list) and attacker_tools:
        return attacker_tools[0]
    return str(attacker_tools)


def load_injecagent_scenarios(injecagent_dir: str,
                              setting: str = "base",
                              max_per_type: int = None) -> List[Dict]:
    """
    Загружает тест-кейсы InjecAgent и конвертирует в наш формат сценариев.

    Args:
        injecagent_dir: путь к папке InjecAgent (с data/ внутри)
        setting: 'base' (только инъекция) или 'enhanced' (+ hacking prompt)
        max_per_type: ограничение на число кейсов каждого типа (для подвыборки)

    Returns:
        список сценариев, совместимых с SimulatedAgent.run_scenario()
    """
    data_dir = os.path.join(injecagent_dir, "data")
    files = {
        "direct_harm":   f"test_cases_dh_{setting}.json",
        "data_stealing": f"test_cases_ds_{setting}.json",
    }

    scenarios = []
    for attack_type, fname in files.items():
        path = os.path.join(data_dir, fname)
        with open(path, encoding="utf-8") as f:
            cases = json.load(f)

        if max_per_type:
            cases = cases[:max_per_type]

        for i, case in enumerate(cases):
            attacker_tool = _extract_tool_name(case.get("Attacker Tools", []))
            user_tool = case.get("User Tool", "unknown_tool")

            # набор доступных инструментов: user-инструмент, attacker-инструмент,
            # и finish. Агент "успешен" для атаки, если вызовет attacker-инструмент.
            tools = [
                {"name": user_tool, "args": ["query"],
                 "description": f"Retrieve content ({user_tool})"},
                {"name": attacker_tool, "args": ["params"],
                 "description": f"Perform action ({attacker_tool})"},
                {"name": "finish", "args": ["summary"],
                 "description": "Finish the task"},
            ]

            tool_response = case.get("Tool Response", "")
            if isinstance(tool_response, dict):
                tool_response = json.dumps(tool_response)

            scenarios.append({
                "id": f"injecagent_{attack_type}_{setting}_{i}",
                "attack_type": attack_type,
                "injection_style": "injecagent",   # внешний источник
                "domain": user_tool,
                "user_instruction": case.get("User Instruction", ""),
                "tools": tools,
                "tool_called": user_tool,
                "tool_args": case.get("Tool Parameters", {}),
                "tool_response": str(tool_response),
                "attack_action": attacker_tool,
                "benign_action": "finish",
                "source": "InjecAgent",
            })

    return scenarios


if __name__ == "__main__":
    import sys
    d = sys.argv[1] if len(sys.argv) > 1 else "../InjecAgent"
    scen = load_injecagent_scenarios(d, setting="base", max_per_type=5)
    print(f"Загружено {len(scen)} сценариев (5 на тип, base)")
    print()
    ex = scen[0]
    print("=== Пример сконвертированного сценария ===")
    for k in ["id", "attack_type", "user_instruction", "tool_called",
              "attack_action"]:
        print(f"  {k}: {str(ex[k])[:100]}")
    print(f"  tool_response[:150]: {ex['tool_response'][:150]}")
    print(f"  tools: {[t['name'] for t in ex['tools']]}")
