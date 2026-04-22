"""
Модуль построения векторного индекса для AI-нормконтролера.
Создает индекс FAISS из правил ГОСТ для семантического поиска.
"""
import os
import json
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Any

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

from .logging_config import get_logger

logger = get_logger(__name__)


class GOSTIndexBuilder:
    """
    Класс для построения индекса правил ГОСТ.

    Пример использования:
        builder = GOSTIndexBuilder()
        builder.load_rules("data/gost_2_105_95_rules.json")
        builder.build_index()
        builder.save("data")
    """

    def __init__(self, model_name: str = "cointegrated/LaBSE-en-ru", device: str = "cpu"):
        """
        Инициализация построителя.

        Args:
            model_name: Название модели для эмбеддингов
            device: Устройство для вычислений (cpu/cuda)
        """
        self.model_name = model_name
        self.device = device
        self.rules: List[Dict[str, Any]] = []
        self.model = None
        self.index = None

        logger.info(f"Инициализация построителя индекса (model={model_name}, device={device})")

    def load_rules(self, rules_path: str):
        """
        Загружает правила из JSON файла.

        Args:
            rules_path: Путь к JSON файлу
        """
        try:
            logger.info(f"Загрузка правил из: {rules_path}")
            with open(rules_path, 'r', encoding='utf-8') as f:
                self.rules = json.load(f)

            logger.info(f"Загружено {len(self.rules)} правил")

            if not self.rules:
                raise ValueError("Список правил пуст")

        except FileNotFoundError:
            logger.error(f"Файл с правилами не найден: {rules_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Ошибка парсинга JSON: {e}")
            raise

    def build_index(self):
        """
        Строит векторный индекс FAISS.
        """
        if not self.rules:
            raise ValueError("Сначала загрузите правила через load_rules()")

        try:
            # Загрузка модели
            logger.info(f"Загрузка модели: {self.model_name}")
            self.model = SentenceTransformer(self.model_name, device=self.device)

            # Подготовка текстов для эмбеддингов
            # Объединяем номер правила и текст для лучшего поиска
            texts = [
                f"{r.get('gost_id', '')} {r.get('text', '')}"
                for r in self.rules
            ]

            logger.info(f"Генерация эмбеддингов для {len(texts)} правил...")

            # Генерация эмбеддингов батчами (чтобы не переполнить память)
            embeddings = self.model.encode(
                texts,
                batch_size=32,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True
            )

            dimension = embeddings.shape[1]
            logger.info(f"Размерность эмбеддингов: {dimension}")

            # Создание индекса FAISS (Inner Product для нормализованных векторов = косинусное сходство)
            logger.info("Создание индекса FAISS...")
            self.index = faiss.IndexFlatIP(dimension)
            self.index.add(embeddings)

            logger.info(f"Индекс построен! Добавлено {self.index.ntotal} правил")

        except Exception as e:
            logger.error(f"Ошибка при построении индекса: {e}", exc_info=True)
            raise

    def save(self, save_dir: str):
        """
        Сохраняет индекс и метаданные на диск.

        Args:
            save_dir: Директория для сохранения
        """
        if self.index is None:
            raise ValueError("Сначала постройте индекс через build_index()")

        try:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)

            index_path = save_path / "gost.index"
            meta_path = save_path / "gost_rules_meta.pkl"

            logger.info(f"Сохранение индекса в: {index_path}")
            faiss.write_index(self.index, str(index_path))

            logger.info(f"Сохранение метаданных в: {meta_path}")
            with open(meta_path, 'wb') as f:
                pickle.dump(self.rules, f)

            logger.info("Сохранение завершено успешно")

        except Exception as e:
            logger.error(f"Ошибка при сохранении: {e}", exc_info=True)
            raise


def main():
    """Точка входа для построения индекса через консоль."""
    # Определяем пути
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "data"
    rules_path = data_dir / "gost_2_105_95_rules.json"

    # Проверка наличия файла с правилами
    if not rules_path.exists():
        logger.error(f"Файл с правилами не найден: {rules_path}")
        logger.error("Убедитесь, что файл gost_2_105_95_rules.json существует в папке data/")
        return

    try:
        # Инициализация и построение
        builder = GOSTIndexBuilder(device="cpu")  # Можно изменить на "cuda" если есть GPU
        builder.load_rules(str(rules_path))
        builder.build_index()
        builder.save(str(data_dir))

        logger.info("=" * 50)
        logger.info("✅ ИНДЕКС УСПЕШНО ПОСТРОЕН И СОХРАНЕН")
        logger.info(f"📂 Путь к данным: {data_dir}")
        logger.info("=" * 50)

    except Exception as e:
        logger.error(f"❌ Критическая ошибка: {e}")
        raise


if __name__ == "__main__":
    main()