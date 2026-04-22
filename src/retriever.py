# src/retriever.py
import faiss
import numpy as np
import pickle
from typing import List, Dict, Any, Optional
from .logging_config import get_logger

logger = get_logger(__name__)

class GOSTRetriever:
    def __init__(self, index: faiss.IndexFlatL2, rules: List[Dict[str, Any]], embedding_func):
        self.index = index
        self.rules = rules
        self.embedding_func = embedding_func
        
        # Создаем мапу типов для быстрого поиска
        # Группируем индексы правил по их chunk_type
        self.type_map = {}
        for i, rule in enumerate(rules):
            r_type = rule.get('chunk_type', 'text')
            if r_type not in self.type_map:
                self.type_map[r_type] = []
            self.type_map[r_type].append(i)
            
        logger.info(f"Retriever инициализирован. Загружено правил: {len(rules)}")
        logger.info(f"Карта типов правил: { {k: len(v) for k, v in self.type_map.items()} }")

    def search(self, query: str, chunk_type: str = "text", top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Ищет релевантные правила с учетом типа чанка.
        """
        if not self.rules or self.index is None:
            return []

        try:
            # 1. Получаем вектор запроса
            query_vector = self.embedding_func(query)
            if query_vector is None:
                logger.warning("Не удалось получить эмбеддинг для запроса")
                return []
            
            query_vector = np.array([query_vector]).astype('float32')

            # 2. Определяем, какие индексы правил нам доступны для этого типа
            # Логика: берем правила строго своего типа + немного общих (если нужно)
            # Для надежности пока берем только свой тип, чтобы не было мусора
            candidate_indices = self.type_map.get(chunk_type, [])
            
            # Fallback: если для конкретного типа ничего нет (например, 'subsection_header'),
            # пробуем взять родительский тип 'section_header' или общий 'text'
            if not candidate_indices:
                logger.warning(f"Нет правил для типа '{chunk_type}'. Пробуем fallback на 'text'.")
                candidate_indices = self.type_map.get('text', [])
            
            if not candidate_indices:
                logger.error("Вообще нет правил для поиска!")
                return []

            logger.debug(f"Поиск для типа '{chunk_type}'. Кандидатов в базе: {len(candidate_indices)}")

            # 3. Поиск по ВСЕМУ индексу, но потом отфильтруем результаты
            # (FAISS не умеет искать сразу по подмножеству индексов без копирования, 
            # поэтому ищем топ-К, а потом чистим)
            D, I = self.index.search(query_vector, k=20) # Берем с запасом

            results = []
            scores = D[0]
            indices = I[0]

            for score, idx in zip(scores, indices):
                if idx == -1: continue # Конец списка
                
                rule = self.rules[idx]
                rule_type = rule.get('chunk_type', 'text')

                # ФИЛЬТРАЦИЯ: Оставляем только если тип совпадает
                # Или если правило универсальное (можно добавить метку 'universal' в правила)
                if rule_type == chunk_type or rule_type == 'text': 
                    results.append(rule)
                    logger.debug(f"Найдено правило: {rule['gost_id']} ({rule_type}) - Score: {score:.4f}")
                
                if len(results) >= top_k:
                    break

            if not results:
                # Если строгая фильтрация убрала всё, берем просто топ из кандидатов этого типа
                logger.warning(f"Строгая фильтрация дала 0 результатов. Берем лучшие из доступных для типа {chunk_type}")
                for idx in candidate_indices:
                    if len(results) >= top_k: break
                    results.append(self.rules[idx])

            logger.info(f"Итог поиска для '{chunk_type}': найдено {len(results)} правил. IDs: {[r['gost_id'] for r in results]}")
            return results

        except Exception as e:
            logger.error(f"Ошибка поиска: {e}", exc_info=True)
            return []