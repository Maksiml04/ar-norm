"""
Модуль LLM-анализа для AI-нормконтролера.
Использует OpenRouter API (через библиотеку openai или requests) для проверки текста.
"""
import os
import json
import logging
from typing import List, Dict, Any, Optional
import requests

from .logging_config import get_logger

logger = get_logger(__name__)


class LLMAnalyzer:
    """
    Класс для анализа текста с использованием LLM через OpenRouter API.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Инициализация анализатора.
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")

        # ИСПРАВЛЕНО: Убран пробел в конце URL, используется официальный адрес
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"

        if not self.api_key:
            logger.warning("API ключ не предоставлен. LLM-анализ будет недоступен.")
        else:
            logger.info("LLM-анализатор инициализирован")

        # ИСПРАВЛЕНО: Используем проверенную бесплатную модель
        self.model = "qwen/qwen3-next-80b-a3b-instruct:free"
        logger.info(f"Используемая модель: {self.model}")

    def analyze_chunk(self, chunk_text: str, chunk_type: str, rules: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Анализирует чанк текста на наличие нарушений правил.
        """
        if not self.api_key:
            return {
                "has_violation": False,
                "violations": [],
                "is_correct": True,
                "confidence": 0.0,
                "warning": "LLM-анализ недоступен (нет API ключа)"
            }

        try:
            prompt = self._build_prompt(chunk_text, chunk_type, rules)
            response_data = self._call_llm_api(prompt)
            result = self._parse_response(response_data, chunk_text, rules)

            logger.info(f"Анализ завершен: найдено нарушений={result.get('has_violation', False)}")
            return result

        except Exception as e:
            logger.error(f"Ошибка LLM-анализа: {e}", exc_info=True)
            return {
                "has_violation": False,
                "violations": [],
                "is_correct": True,
                "confidence": 0.0,
                "error": f"Ошибка анализа: {str(e)}"
            }

    def _build_prompt(self, chunk_text: str, chunk_type: str, rules: List[Dict[str, Any]]) -> str:
        """Строит промпт для LLM."""
        rules_text = "\n".join([
            f"- [{r.get('gost_id', 'N/A')}] {r.get('text', '')[:200]}..."
            for r in rules
        ])

        prompt = f"""Ты эксперт по нормоконтролю инженерных документов по ГОСТ.
Твоя задача: проверить фрагмент документа на соответствие правилам.
Ты строгий эксперт по ГОСТ. Отвечай ТОЛЬКО валидным JSON объектом. Не используй markdown блоки ```json ... ```.
Тип фрагмента: {chunk_type}
Текст фрагмента: "{chunk_text}"

Применимые правила ГОСТ:
{rules_text}

Инструкция:
1. Проанализируй текст фрагмента.
2. Сравни с каждым правилом.
3. Если есть нарушение - опиши его.
4. Верни ответ ТОЛЬКО в формате JSON (без markdown разметки):
{{
    "has_violation": true/false,
    "violations": [
        {{
            "rule_id": "номер правила",
            "description": "описание нарушения",
            "severity": "high/medium/low"
        }}
    ],
    "is_correct": true/false,
    "confidence": 0.0-1.0,
    "comment": "общий комментарий"
}}
"""
        return prompt

    def _call_llm_api(self, prompt: str) -> Dict[str, Any]:
        """
        Вызывает API OpenRouter напрямую через requests.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "AI Normkontroler"
        }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 600
        }

        logger.debug(f"Отправка запроса к LLM (модель={self.model})")

        response = requests.post(self.api_url, headers=headers, json=payload, timeout=45)

        if response.status_code != 200:
            logger.error(f"API вернул ошибку {response.status_code}: {response.text}")
            raise Exception(f"API Error {response.status_code}: {response.text}")

        data = response.json()

        if "choices" not in data or len(data["choices"]) == 0:
            raise Exception("Пустой ответ от API")

        content = data["choices"][0]["message"]["content"]
        logger.debug(f"Получен ответ от LLM: {content[:150]}...")

        return self._extract_json(content)

    def _extract_json(self, text: str) -> Dict[str, Any]:
        """Извлекает JSON из ответа, убирая лишние символы."""
        text = text.strip()

        # Удаляем маркеры markdown, если модель их добавила
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]

        text = text.strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            logger.warning(f"Не удалось распарсить JSON напрямую, попытка очистки: {e}")
            # Попытка найти JSON внутри текста
            start = text.find("{")
            end = text.rfind("}") + 1
            if start != -1 and end != -1:
                try:
                    return json.loads(text[start:end])
                except:
                    pass
            raise ValueError("Не удалось извлечь валидный JSON из ответа модели")

    def _parse_response(self, response: Dict[str, Any], chunk_text: str, rules: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Валидирует и дополняет ответ."""
        result = response.copy()

        if "has_violation" not in result:
            result["has_violation"] = len(result.get("violations", [])) > 0
        if "is_correct" not in result:
            result["is_correct"] = not result["has_violation"]
        if "confidence" not in result:
            result["confidence"] = 0.5

        return result