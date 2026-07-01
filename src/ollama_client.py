"""
src/ollama_client.py — тонкая обёртка над Ollama REST API.

Ollama поднимает локальный сервер на http://localhost:11434.
Мы используем эндпоинт /api/chat с stream=False — получаем
весь ответ разом плюс метрики времени (total_duration и т.д.).

Документация: https://docs.ollama.com/api/chat
"""

import requests
from typing import List, Dict, Optional


class OllamaClient:
    """
    Минимальный клиент Ollama для эксперимента.

    Возможности:
      - chat(): отправить диалог, получить ответ + метрики
      - list_models(): какие модели реально установлены
      - is_available(): запущен ли сервер
    """

    def __init__(self, base_url: str = "http://localhost:11434",
                 timeout: int = 300):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout   # секунд; на CPU ответы небыстрые

    # ── проверка доступности ──────────────────────────────────────────────────

    def is_available(self) -> bool:
        """True если сервер Ollama отвечает."""
        try:
            r = requests.get(f"{self.base_url}/api/version", timeout=5)
            return r.status_code == 200
        except requests.RequestException:
            return False

    def list_models(self) -> List[str]:
        """Список установленных моделей (как в `ollama list`)."""
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=10)
            r.raise_for_status()
            data = r.json()
            return [m["name"] for m in data.get("models", [])]
        except requests.RequestException as e:
            print(f"[OllamaClient] Ошибка получения списка моделей: {e}")
            return []

    # ── основной вызов ────────────────────────────────────────────────────────

    def chat(self,
             model: str,
             messages: List[Dict[str, str]],
             temperature: float = 0.0,
             num_predict: int = 512,
             tools: Optional[List[Dict]] = None) -> Dict:
        """
        Отправляет диалог модели, возвращает ответ и метрики.

        Args:
            model:       имя модели (напр. 'qwen2.5:1.5b-instruct-q4_K_M')
            messages:    список {'role': ..., 'content': ...}
            temperature: 0.0 = детерминированно (важно для воспроизводимости!)
            num_predict: макс. токенов в ответе
            tools:       опционально — описания инструментов для tool-calling

        Returns:
            {
              'content':       str   — текст ответа,
              'tool_calls':    list  — вызовы инструментов (если были),
              'latency_ms':    float — время генерации,
              'eval_count':    int   — сколько токенов сгенерировано,
              'prompt_tokens': int   — токенов в промпте,
              'error':         str | None
            }
        """
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,                      # весь ответ разом
            "options": {
                "temperature": temperature,
                "num_predict": num_predict,
            },
        }
        if tools:
            payload["tools"] = tools

        try:
            r = requests.post(f"{self.base_url}/api/chat",
                              json=payload, timeout=self.timeout)
            r.raise_for_status()
            data = r.json()

            message = data.get("message", {})
            # total_duration приходит в наносекундах → переводим в мс
            total_ns = data.get("total_duration", 0)
            latency_ms = total_ns / 1e6 if total_ns else 0.0

            return {
                "content":       message.get("content", ""),
                "tool_calls":    message.get("tool_calls", []),
                "latency_ms":    round(latency_ms, 2),
                "eval_count":    data.get("eval_count", 0),
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "error":         None,
            }

        except requests.RequestException as e:
            return {
                "content":       "",
                "tool_calls":    [],
                "latency_ms":    0.0,
                "eval_count":    0,
                "prompt_tokens": 0,
                "error":         str(e),
            }


# ── быстрая проверка при прямом запуске ───────────────────────────────────────

if __name__ == "__main__":
    client = OllamaClient()

    if not client.is_available():
        print("❌ Ollama не запущена. Запусти `ollama serve` или приложение.")
    else:
        print("✓ Ollama доступна")
        models = client.list_models()
        print(f"✓ Установлено моделей: {len(models)}")
        for m in models:
            print(f"    {m}")

        # пробный запрос
        if models:
            print("\nПробный запрос...")
            resp = client.chat(
                model=models[0],
                messages=[{"role": "user", "content": "Say 'OK' and nothing else."}],
                num_predict=10,
            )
            if resp["error"]:
                print(f"  Ошибка: {resp['error']}")
            else:
                print(f"  Ответ: {resp['content'].strip()}")
                print(f"  Время: {resp['latency_ms']:.0f} ms")
