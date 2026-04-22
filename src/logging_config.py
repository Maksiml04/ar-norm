"""
Модуль настройки логирования для AI-нормконтролера.
Обеспечивает вывод логов в консоль и файлы.
"""
import sys
import logging
from pathlib import Path
from logging.handlers import RotatingFileHandler
from datetime import datetime
import os

# Цвета для консоли (ANSI)
COLORS = {
    'DEBUG': '\033[94m',  # Синий
    'INFO': '\033[92m',  # Зеленый
    'WARNING': '\033[93m',  # Желтый
    'ERROR': '\033[91m',  # Красный
    'CRITICAL': '\033[95m',  # Фиолетовый
    'RESET': '\033[0m'  # Сброс
}


class ColoredFormatter(logging.Formatter):
    """Форматтер с цветным выводом для консоли."""

    def format(self, record):
        log_color = COLORS.get(record.levelname, COLORS['RESET'])
        record.levelname = f"{log_color}{record.levelname}{COLORS['RESET']}"
        return super().format(record)


def get_logger(name: str = "ai_normkontroler") -> logging.Logger:
    """
    Создает или получает логгер с настройками для проекта.

    Args:
        name: Имя логгера

    Returns:
        Настроенный экземпляр logging.Logger
    """
    logger = logging.getLogger(name)

    # Если уже настроен - возвращаем
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    # Определяем уровень логирования из env или по умолчанию INFO
    log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)

    # Форматы
    file_format = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(funcName)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    console_format = ColoredFormatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )

    # Директория для логов
    log_dir = Path(__file__).parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d")

    # 1. Консольный обработчик
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # 2. Общий файл логов (все уровни от DEBUG)
    general_log_file = log_dir / f"app_{timestamp}.log"
    file_handler = RotatingFileHandler(
        general_log_file,
        maxBytes=5 * 1024 * 1024,  # 5 MB
        backupCount=3,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)

    # 3. Файл только для ошибок (ERROR и выше)
    error_log_file = log_dir / f"errors_{timestamp}.log"
    error_handler = RotatingFileHandler(
        error_log_file,
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding='utf-8'
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(file_format)
    logger.addHandler(error_handler)

    # Логирование старта
    logger.info("=" * 80)
    logger.info("AI НОРМКОНТРОЛЕР - ЗАПУСК СИСТЕМЫ ЛОГИРОВАНИЯ")
    logger.info("=" * 80)
    logger.info(f"Окружение: {'Local' if not os.getenv('DOCKER_ENV') else 'Docker'}")
    logger.info(f"Уровень логирования: {log_level_str}")
    logger.info(f"Логирование в файл: Включено")
    logger.info(f"Директория логов: {log_dir.resolve()}")
    logger.info(f"PID процесса: {os.getpid()}")
    logger.info("-" * 80)

    return logger