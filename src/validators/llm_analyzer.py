# src/validators/llm_analyzer.py
from __future__ import annotations
import asyncio
import logging
import os
import re
import json
from typing import Optional, List

from openai import AsyncOpenAI, AuthenticationError, BadRequestError

from src.schemas import (
    ChunkType,
    DocumentChunk,
    ValidationResult,
    ValidatorType,
    Violation,  # ← ДОБАВЛЕНО: был забыт импорт!
    Severity,  # ← ДОБАВЛЕНО: нужен для парсинга
)
from src.rules_db import RULES_DB, get_rules_for_chunk

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """Ты — строгий нормоконтролёр технической документации по ГОСТ 2.105-95.
Проверяй СТРОГО по предоставленным правилам. Отвечай ТОЛЬКО валидным JSON-массивом.
Формат: [{"rule_id": "str", "violation_type": "str", "explanation": "str", "severity": "minor|major"}]"""


class LLMAnalyzer:
    def __init__(
            self,
            api_key: Optional[str] = None,
            model: str = "deepseek/deepseek-chat",
            base_url: str = "https://openrouter.ai/api/v1",
            max_concurrency: int = 5,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.max_concurrency = max_concurrency

        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY", "")
        if not self.api_key:
            logger.warning("⚠️ LLMAnalyzer: API key missing — LLM disabled")
            self.client = None
        else:
            self.client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                default_headers={
                    "HTTP-Referer": "http://localhost:8000",
                    "X-Title": "AI-Normcontrol"
                }
            )
            logger.info(f"✅ LLM client: model={model}, base_url={base_url}")

    def _get_llm_rules_for_chunk(self, chunk_type: ChunkType) -> list[dict]:
        """Возвращает только LLM-правила для данного типа блока."""
        # ← УДАЛЕНО: дубликат метода, оставили один рабочий вариант
        return get_rules_for_chunk(chunk_type, validator_type="llm")

    async def analyze_batch_async(self, chunks: list[DocumentChunk]) -> list[ValidationResult]:
        """Async версия для вызова из FastAPI."""
        if not chunks or not self.client:
            return [self._make_empty_result(c.chunk_id, c.location) for c in chunks]

        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def bounded(chunk: DocumentChunk) -> ValidationResult:
            async with semaphore:
                return await self._analyze_one(chunk)

        results = await asyncio.gather(*(bounded(c) for c in chunks), return_exceptions=True)
        # Фильтруем исключения из gather
        return [r if isinstance(r, ValidationResult) else self._make_empty_result("unknown", error=str(r)) for r in
                results]

    def _make_empty_result(self, chunk_id: str, location: Optional[dict] = None,
                           error: Optional[str] = None) -> ValidationResult:
        """Вспомогательный метод: всегда валидный пустой результат."""
        return ValidationResult(
            chunk_id=chunk_id,
            validator=ValidatorType.LLM,
            has_violation=False,  # ← ОБЯЗАТЕЛЬНОЕ ПОЛЕ
            violations=[],  # ← ОБЯЗАТЕЛЬНОЕ ПОЛЕ
            confidence=0.0,  # ← ОБЯЗАТЕЛЬНОЕ ПОЛЕ
            location=location,
            error=error
        )

    async def _analyze_one(self, chunk: DocumentChunk) -> ValidationResult:
        if not self.client:
            return self._make_empty_result(chunk.chunk_id, chunk.location, error="LLM client not initialized")

        rules = self._get_llm_rules_for_chunk(chunk.chunk_type)
        if not rules:
            return self._make_empty_result(chunk.chunk_id, chunk.location)

        rules_text = "\n".join(f"[{r['id']}] {r['description']}" for r in rules)
        user_message = (
            f"ПРАВИЛА:\n{rules_text}\n\n"
            f"ТИП БЛОКА: {chunk.chunk_type.value}\n\n"
            f"ТЕКСТ:\n{chunk.text[:1500]}\n\n"
            f"ОТВЕТЬ ТОЛЬКО JSON-МАССИВОМ реальных нарушений. Если нарушений нет — верни []. "
            f"Поле 'severity' должно быть: 'minor', 'major' или 'critical'. НЕ используй 'none'."
            f"НИКОГДА не используй значение 'none' для поля severity. "
            f"Допустимые значения: 'minor', 'major', 'critical'."
        )

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.0,
                response_format={"type": "json_object"}
            )

            content = response.choices[0].message.content or "[]"
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if not json_match:
                return self._make_empty_result(chunk.chunk_id, chunk.location, error="No JSON array in LLM response")

            raw_data = json.loads(json_match.group(0))

            # 🔑 Конвертация с защитой от 'none' и неизвестных severity
            violations: List[Violation] = []
            for item in raw_data:
                if not isinstance(item, dict) or not item.get("violation_type"):
                    continue
                severity_raw = item.get("severity", "minor")
                explanation = item.get("explanation", "")

                # Пропускаем «псевдо-нарушения»
                if severity_raw == "none" or "нарушений нет" in explanation.lower() or "no violation" in explanation.lower():
                    continue

                try:
                    severity = Severity(severity_raw)
                except ValueError:
                    logger.warning(f"Unknown severity '{severity_raw}' → default MINOR")
                    severity = Severity.MINOR

                violations.append(Violation(
                    rule_id=item.get("rule_id", "LLM_GENERAL"),
                    violation_type=item["violation_type"],
                    explanation=explanation,
                    severity=severity
                ))

            return ValidationResult(
                chunk_id=chunk.chunk_id,
                validator=ValidatorType.LLM,
                has_violation=bool(violations),
                violations=violations,
                confidence=0.85 if violations else 0.0,
                location=chunk.location
            )

        except AuthenticationError as e:
            logger.error(f"🔐 LLM Auth Error: {e}")
            return self._make_empty_result(chunk.chunk_id, chunk.location, error=f"Auth error: {str(e)}")
        except Exception as e:
            logger.error(f"LLM error: {type(e).__name__}: {e}")
            return self._make_empty_result(chunk.chunk_id, chunk.location, error=str(e))