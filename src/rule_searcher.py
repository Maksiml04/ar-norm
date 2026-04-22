"""
Модуль поиска релевантных правил ГОСТ для AI-нормконтролера.
Использует векторный индекс FAISS для семантического поиска.
"""
import logging
from typing import List, Dict, Any, Optional
import numpy as np

from .logging_config import get_logger

logger = get_logger(__name__)


class GOSTRuleSearcher:
    """
    Класс для поиска релевантных правил ГОСТ.

    Пример использования:
        searcher = GOSTRuleSearcher(index, rules, model)
        results = searcher.search("требования к шрифту", chunk_type="text", top_k=3)
    """

    def __init__(self, index, rules: List[Dict[str, Any]], model):
        """
        Инициализация поисковика правил.

        Args:
            index: Индекс FAISS
            rules: Список правил ГОСТ (словари с текстом и метаданными)
            model: Модель для генерации эмбеддингов
        """
        self.index = index
        self.rules = rules
        self.model = model
        logger.info(f"Инициализирован поисковик правил: {len(rules)} правил в индексе")

    def search(self, query: str, chunk_type: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Выполняет поиск релевантных правил для данного запроса.

        Args:
            query: Текст запроса (например, содержимое чанка)
            chunk_type: Тип чанка (text, section_header, table и т.д.)
            top_k: Количество возвращаемых результатов

        Returns:
            Список словарей с правилами и метаданными
        """
        try:
            if self.index.ntotal == 0:
                logger.warning("Индекс пуст, поиск невозможен")
                return []

            # Генерируем эмбеддинг запроса
            logger.debug(f"Генерация эмбеддинга для запроса: '{query[:50]}...'")
            query_embedding = self.model.encode([query], convert_to_numpy=True)

            # Поиск ближайших соседей в индексе
            logger.debug(f"Поиск {top_k} ближайших правил в индексе")
            distances, indices = self.index.search(query_embedding, k=top_k)

            # Формируем результаты
            results = []
            for i, idx in enumerate(indices[0]):
                if idx == -1:  # FAISS возвращает -1 если не хватает элементов
                    continue

                rule = self.rules[idx].copy()
                rule["distance"] = float(distances[0][i])
                rule["relevance_score"] = 1.0 / (1.0 + rule["distance"])  # Простая конверсия дистанции в скор
                rule["rank"] = i + 1

                results.append(rule)

            logger.debug(f"Найдено {len(results)} релевантных правил")
            return results

        except Exception as e:
            logger.error(f"Ошибка при поиске правил: {e}", exc_info=True)
            return []

    def search_by_ids(self, ids: List[int]) -> List[Dict[str, Any]]:
        """
        Получает правила по их ID.

        Args:
            ids: Список ID правил

        Returns:
            Список правил
        """
        results = []
        for idx in ids:
            if 0 <= idx < len(self.rules):
                results.append(self.rules[idx].copy())
            else:
                logger.warning(f"Правило с индексом {idx} не найдено")
        return results