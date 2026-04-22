"""
AI Normkontroler - Система автоматического нормоконтроля.
"""
from .main import AINormkontroler
from .logging_config import get_logger

__version__ = "1.0.0"
__all__ = ["AINormkontroler", "get_logger"]