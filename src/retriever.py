"""
Детерминированный поиск правил ГОСТ на основе data.json.
Использует прямое сопоставление типов блоков и правил без векторного поиска.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from .logging_config import get_logger

logger = get_logger(__name__)


class GOSTRetriever:
    """
    Детерминированный поиск правил ГОСТ по типу блока.
    Использует предопределённые соответствия из data.json.
    """

    def __init__(
        self,
        block_rule_registry: dict[str, list[str]],
        rules_db: dict[str, dict[str, Any]],
    ) -> None:
        self.block_rule_registry = block_rule_registry
        self.rules_db = rules_db
        logger.info(
            f"GOSTRetriever инициализирован. "
            f"Правил в базе: {len(rules_db)}, типов блоков: {len(block_rule_registry)}"
        )

    @classmethod
    def load(cls, data_path: str = "src/data.json") -> "GOSTRetriever":
        """
        Загружает retriever из JSON-файла с детерминированными правилами.
        """
        path = Path(data_path)
        if not path.exists():
            raise FileNotFoundError(f"Файл с правилами не найден: {data_path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        block_rule_registry = data.get("block_rule_registry", {})
        rules_db = data.get("rules_db", {})

        logger.info(f"Загружено {len(rules_db)} правил для {len(block_rule_registry)} типов блоков")

        return cls(block_rule_registry=block_rule_registry, rules_db=rules_db)

    def search(
        self,
        chunk_text: str,
        chunk_type: str = "TEXT_BODY",
        context_query: str = "",
        top_k: int = 5,
        search_pool: int = 30,
    ) -> list[dict[str, Any]]:
        """
        Ищет правила для данного типа блока.
        Возвращает top_k наиболее релевантных правил.
        """
        if not self.rules_db or not self.block_rule_registry:
            logger.warning("Retriever не готов: база правил или реестр пусты")
            return []

        # Нормализуем тип блока (приводим к верхнему регистру)
        chunk_type_upper = chunk_type.upper()
        
        # Получаем список ID правил для данного типа блока
        rule_ids = self.block_rule_registry.get(chunk_type_upper, [])
        
        # Если для типа блока нет правил, пробуем TEXT_BODY как fallback
        if not rule_ids:
            logger.warning(f"Нет правил для типа блока '{chunk_type}', используем TEXT_BODY")
            rule_ids = self.block_rule_registry.get("TEXT_BODY", [])

        # Собираем правила из базы
        candidates: list[dict[str, Any]] = []
        for rule_id in rule_ids:
            if rule_id in self.rules_db:
                rule = self.rules_db[rule_id].copy()
                # Добавляем фиктивный score для совместимости интерфейса
                rule["_score"] = 1.0
                rule["_type_bonus"] = 0.0
                candidates.append(rule)

        # Сортируем (пока все с одинаковым score, можно расширить логику)
        candidates.sort(key=lambda r: r.get("_score", 0), reverse=True)
        result = candidates[:top_k]

        logger.info(
            f"search [{chunk_type}] → {len(result)} правил. "
            f"IDs: {[r['id'] for r in result]}"
        )
        return result

    def search_batch(
        self,
        chunks: list[dict[str, Any]],
        top_k: int = 5,
        search_pool: int = 30,
    ) -> list[list[dict[str, Any]]]:
        """
        Батч-поиск для списка чанков.
        """
        if not chunks:
            return []

        results: list[list[dict[str, Any]]] = []
        for chunk in chunks:
            chunk_type = chunk.get("chunk_type", "TEXT_BODY")
            chunk_text = chunk.get("text", "")
            result = self.search(
                chunk_text=chunk_text,
                chunk_type=chunk_type,
                top_k=top_k,
                search_pool=search_pool,
            )
            results.append(result)

        return results
