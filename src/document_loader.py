"""
Модуль загрузки документов для AI-нормконтролера.
Поддерживает PDF, TXT, DOCX.
"""
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader

from .logging_config import get_logger

logger = get_logger(__name__)


class DocumentLoader:
    """Класс для загрузки документов из различных источников."""

    def __init__(self, documents_path: Path):
        self.documents_path = documents_path

    def load_documents(self) -> List[Document]:
        """
        Загружает все документы из указанной директории.

        Returns:
            Список объектов Document
        """
        if not self.documents_path.exists():
            logger.warning(f"Директория документов не найдена: {self.documents_path}")
            return []

        all_documents = []

        # Загрузка PDF
        pdf_files = list(self.documents_path.glob("*.pdf"))
        if pdf_files:
            logger.info(f"Найдено {len(pdf_files)} PDF файлов")
            for pdf_file in pdf_files:
                try:
                    loader = PyPDFLoader(str(pdf_file))
                    docs = loader.load()
                    all_documents.extend(docs)
                    logger.debug(f"Загружен PDF: {pdf_file.name} ({len(docs)} страниц)")
                except Exception as e:
                    logger.error(f"Ошибка загрузки PDF {pdf_file.name}: {e}")

        # Загрузка TXT
        txt_files = list(self.documents_path.glob("*.txt"))
        if txt_files:
            logger.info(f"Найдено {len(txt_files)} TXT файлов")
            for txt_file in txt_files:
                try:
                    loader = TextLoader(str(txt_file), encoding="utf-8")
                    docs = loader.load()
                    all_documents.extend(docs)
                    logger.debug(f"Загружен TXT: {txt_file.name}")
                except Exception as e:
                    logger.error(f"Ошибка загрузки TXT {txt_file.name}: {e}")

        logger.info(f"Всего загружено документов: {len(all_documents)}")
        return all_documents

    def load_single_file(self, file_path: Path) -> List[Document]:
        """Загружает один конкретный файл."""
        if not file_path.exists():
            logger.error(f"Файл не найден: {file_path}")
            return []

        try:
            if file_path.suffix.lower() == ".pdf":
                loader = PyPDFLoader(str(file_path))
            elif file_path.suffix.lower() == ".txt":
                loader = TextLoader(str(file_path), encoding="utf-8")
            else:
                logger.warning(f"Неподдерживаемый формат файла: {file_path.suffix}")
                return []

            return loader.load()
        except Exception as e:
            logger.error(f"Ошибка загрузки файла {file_path.name}: {e}", exc_info=True)
            return []