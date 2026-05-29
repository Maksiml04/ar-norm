from __future__ import annotations
import logging
import os
from dataclasses import dataclass
from typing import Optional

from src.schemas import ChunkType, DocumentChunk, ValidationResult, ValidatorType
from src.validators.rule_based import RuleBasedValidator
from src.validators.llm_analyzer import LLMAnalyzer
from src.rules_db import RULES_DB

logger = logging.getLogger(__name__)


@dataclass
class PipelineStats:
    total_chunks: int = 0
    rb_violations: int = 0
    rb_clean: int = 0
    llm_sent: int = 0
    llm_violations: int = 0
    llm_skipped: int = 0
    total_violations: int = 0


class NormcontrolPipeline:
    def __init__(
            self,
            api_key: Optional[str] = None,
            model: str = "deepseek/deepseek-v4-flash",
            base_url: Optional[str] = "https://openrouter.ai/api/v1"
    ):
        self.rule_based = RuleBasedValidator()

        # 🔑 FIX 1: Автоматически берём ключ из .env, если не передан явно
        self._api_key = api_key or os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.llm = LLMAnalyzer(api_key=self._api_key, model=model, base_url=base_url) if self._api_key else None

        # 📖 FIX 2: Безопасное чтение chunk_types из базы (защита от KeyError)
        self._llm_eligible_types = set()
        for r in RULES_DB.values():
            if r.get("validator_type") == "llm" and "chunk_types" in r:
                self._llm_eligible_types.update(r["chunk_types"])

        logger.info(f"🔧 Pipeline init: LLM={'ENABLED' if self.llm else 'DISABLED'} | "
                    f"Eligible types: {len(self._llm_eligible_types)} | "
                    f"API key: {'SET' if self._api_key else 'MISSING'}")

    async def process_document_async(self, chunks: list[DocumentChunk]) -> tuple[list[ValidationResult], PipelineStats]:
        stats = PipelineStats(total_chunks=len(chunks))
        rb_results: dict[str, ValidationResult] = {}
        needs_llm: list[DocumentChunk] = []

        # 1️⃣ Rule-Based (всегда запускаем)
        for chunk in chunks:
            # 🔑 FIX 3: Убрали continue. RB валидируем всегда, LLM фильтруем позже
            rb_result = self.rule_based.validate(chunk)
            rb_results[chunk.chunk_id] = rb_result

            if rb_result.has_violation:
                stats.rb_violations += 1
            else:
                stats.rb_clean += 1

            # Фильтр для LLM
            if (self.llm and
                    chunk.chunk_type in self._llm_eligible_types and
                    len(chunk.text.strip()) >= 10):
                needs_llm.append(chunk)

        stats.llm_sent = len(needs_llm)

        # 2️⃣ LLM Batch (асинхронно)
        llm_results: dict[str, ValidationResult] = {}
        if needs_llm:
            logger.info(f"🤖 Отправляю {len(needs_llm)} чанков в LLM...")
            llm_batch = await self.llm.analyze_batch_async(needs_llm)
            for res in llm_batch:
                llm_results[res.chunk_id] = res
                if res.has_violation:
                    stats.llm_violations += 1

        # 3️⃣ Мерж результатов
        final_results: list[ValidationResult] = []
        for chunk in chunks:
            rb = rb_results.get(chunk.chunk_id)
            llm = llm_results.get(chunk.chunk_id)
            if not rb:
                continue

            all_violations = list(rb.violations)
            llm_found_something = bool(llm and llm.has_violation)

            # 🔑 FIX 4: Ручной мерж вместо несуществующего rb.merge(llm)
            if llm_found_something:
                existing_keys = {(v.rule_id, v.explanation[:30]) for v in all_violations}
                for v in llm.violations:
                    key = (v.rule_id, v.explanation[:30])
                    if key not in existing_keys:
                        all_violations.append(v)
                        existing_keys.add(key)

            # 🔑 FIX 5: Возвращаем ВСЕ чанки, даже без нарушений
            final_results.append(ValidationResult(
                chunk_id=rb.chunk_id,
                validator=ValidatorType.HYBRID if llm_found_something else rb.validator,
                has_violation=bool(all_violations),
                violations=all_violations,
                confidence=0.9 if llm_found_something else 1.0,
                location=rb.location,
            ))

        stats.total_violations = sum(1 for r in final_results if r.has_violation)
        logger.info(f"📊 Pipeline done: {stats.total_violations} violations | LLM sent: {stats.llm_sent}")
        return final_results, stats