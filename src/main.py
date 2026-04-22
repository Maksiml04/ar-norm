"""
Основной модуль системы AI Нормконтролер.
Управляет загрузкой правил, поиском (FAISS) и LLM-анализом.
"""
import os
import pickle
from typing import List, Dict, Any, Optional
from pathlib import Path

# Импорты библиотек
try:
    import faiss
except ImportError:
    faiss = None  # Обработаем ошибку позже, если индекс действительно нужен

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# Локальные импорты
from src.logging_config import get_logger
from src.llm_analyzer import LLMAnalyzer

logger = get_logger(__name__)


class AINormkontroler:
    """Основной класс системы нормоконтроля."""

    def __init__(self, rules: List[Dict[str, Any]], analyzer: Optional[LLMAnalyzer] = None):
        """
        Инициализация системы.

        Args:
            rules: Список загруженных правил ГОСТ.
            analyzer: Экземпляр LLM анализатора (опционально).
        """
        self.rules = rules
        self.analyzer = analyzer
        self.index = None  # Инициализируем пустым индексом

        logger.info(f"AINormkontroler инициализирован. Загружено правил: {len(rules)}")
        if self.analyzer:
            logger.info("LLM анализатор подключен.")
        else:
            logger.warning("LLM анализатор НЕ подключен (режим только поиска).")

    @classmethod
    def load_from_index(cls, index_path: str, meta_path: str, api_key: Optional[str] = None) -> "AINormkontroler":
        """
        Загружает систему из сохраненных файлов индекса и метаданных.

        Args:
            index_path: Путь к файлу индекса FAISS (.index).
            meta_path: Путь к файлу метаданных (.pkl).
            api_key: API ключ для LLM.

        Returns:
            Экземпляр AINormkontroler.
        """
        if faiss is None:
            raise ImportError("Библиотека faiss не установлена. Выполните: pip install faiss-cpu")

        logger.info(f"Загрузка индекса правил из {index_path}")

        # 1. Загрузка FAISS индекса
        try:
            index = faiss.read_index(index_path)
            logger.info("Индекс FAISS успешно загружен.")
        except Exception as e:
            logger.error(f"Ошибка чтения индекса FAISS: {e}")
            raise

        # 2. Загрузка метаданных (правил)
        rules = []
        try:
            with open(meta_path, 'rb') as f:
                meta_data = pickle.load(f)

            # Обработка устаревшего формата (список)
            if isinstance(meta_data, list):
                logger.warning("⚠️ Обнаружен устаревший формат мета-файла (список). Попытка конвертации...")
                # Предполагаем, что список содержит словари правил напрямую или имеет структуру
                # Если это просто список правил, оставляем как есть
                if len(meta_data) > 0 and isinstance(meta_data[0], dict):
                    rules = meta_data
                else:
                    # Если структура сложная,可能需要 дополнительная логика, но пока берем как есть
                    rules = meta_data
            elif isinstance(meta_data, dict):
                # Новый формат (словарь с ключом 'rules')
                rules = meta_data.get('rules', [])
                if not rules and 'data' in meta_data:
                    rules = meta_data['data']  # Альтернативный ключ

            if not rules:
                logger.warning("Правила не найдены в файле метаданных.")
            else:
                logger.info(f"Успешно загружено {len(rules)} правил из метаданных.")

        except Exception as e:
            logger.error(f"Ошибка загрузки метаданных: {e}", exc_info=True)
            raise

        # 3. Инициализация LLM анализатора
        analyzer = None
        if api_key:
            try:
                analyzer = LLMAnalyzer(api_key=api_key)
                logger.info("✅ LLM анализатор инициализирован.")
            except Exception as e:
                logger.error(f"Не удалось инициализировать LLM анализатор: {e}")
                # Не прерываем запуск, система будет работать в режиме поиска без LLM
        else:
            logger.warning("API ключ не предоставлен. LLM анализатор не будет инициализирован.")

        # 4. Создание экземпляра класса
        # Передаем только rules и analyzer, так как index устанавливается внутрь объекта
        instance = cls(rules=rules, analyzer=analyzer)
        instance.index = index  # Сохраняем индекс внутри экземпляра

        return instance

    def search_rules(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Поиск релевантных правил по запросу.

        Args:
            query: Текст запроса.
            top_k: Количество возвращаемых результатов.

        Returns:
            Список найденных правил.
        """
        if self.index is None or not self.rules:
            logger.warning("Поиск невозможен: индекс или правила не загружены.")
            return []

        # Здесь должна быть логика векторизации запроса и поиска в FAISS
        # Для примера заглушка, если нет кода векторизации (embedding)
        # В реальной системе здесь нужен код получения эмбеддинга запроса
        # и вызова self.index.search(...)

        logger.warning(f"Метод search_rules вызван, но логика поиска через FAISS требует реализации эмбеддингов.")

        # Временная заглушка: возврат первых попавшихся правил (удалить после реализации поиска)
        return self.rules[:top_k]

    def analyze_chunk(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """
        Анализирует чанк документа.

        Args:
            chunk: Словарь с данными чанка (text, type, id, location).

        Returns:
            Результат анализа.
        """
        if not self.analyzer:
            return {
                "chunk_id": chunk.get("id", "unknown"),
                "has_violation": False,
                "violations": [],
                "is_correct": True,
                "confidence": 0.0,
                "warning": "LLM анализатор не доступен."
            }

        text = chunk.get("text", "")
        chunk_type = chunk.get("chunk_type", "text")

        # 1. Поиск релевантных правил
        # Если поиск через FAISS еще не реализован полноценно, передаем все правила или часть
        # В будущем заменить на реальный поиск: relevant_rules = self.search_rules(text, top_k=5)
        relevant_rules = self.rules[:5]  # Временная заглушка

        # 2. Вызов LLM анализатора
        try:
            result = self.analyzer.analyze_chunk(
                chunk_text=text,
                chunk_type=chunk_type,
                rules=relevant_rules
            )

            # Добавляем метаданные чанка в результат
            result["chunk_id"] = chunk.get("id", "unknown")
            result["location"] = chunk.get("location", {})

            return result

        except Exception as e:
            logger.error(f"Ошибка анализа чанка {chunk.get('id')}: {e}", exc_info=True)
            return {
                "chunk_id": chunk.get("id", "unknown"),
                "has_violation": False,
                "violations": [],
                "is_correct": True,
                "confidence": 0.0,
                "error": str(e)
            }