"""
Модуль конфигурации для AI-нормконтролера.
Загружает настройки из переменных окружения и файлов.
"""
import os
from pathlib import Path
from typing import Optional


class Config:
    """Класс конфигурации проекта."""

    def __init__(self):
        # Пути
        self.base_dir = Path(__file__).resolve().parent.parent
        self.data_dir = self.base_dir / "data"
        self.logs_dir = self.base_dir / "logs"

        # Настройки модели эмбеддингов
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "cointegrated/LaBSE-en-ru")
        self.device = os.getenv("DEVICE", "cpu")  # cpu или cuda

        # Настройки разбиения текста (чанкинг)
        self.chunk_size = int(os.getenv("CHUNK_SIZE", "500"))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "50"))

        # Настройки LLM
        self.openai_api_key = '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'
        self.llm_model = os.getenv("LLM_MODEL", "deepseek/deepseek-chat")

        # Пути к файлам индекса
        self.index_path = self.data_dir / "gost.index"
        self.rules_meta_path = self.data_dir / "gost_rules_meta.pkl"
        self.rules_json_path = self.data_dir / "gost_2_105_95_rules.json"

        # Логирование
        self.log_level = os.getenv("LOG_LEVEL", "INFO")

        # Создание директорий при инициализации
        self.data_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)


# Глобальный экземпляр конфигурации
config = Config()