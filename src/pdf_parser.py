"""
Улучшенный PDF Parser на основе pdfplumber для AI-нормконтролера.
Обеспечивает качественное извлечение текста с сохранением структуры документа.
Поддерживает умное разбиение на чанки с учетом семантики (заголовки, рисунки, таблицы).
"""
import re
import json
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional, Any
from pathlib import Path

try:
    import pdfplumber
except ImportError:
    pdfplumber = None
    raise ImportError("Установите pdfplumber: pip install pdfplumber")

# Используем абсолютный импорт для совместимости
try:
    from src.logging_config import get_logger
except ImportError:
    from logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class DocumentChunk:
    """Структура чанка документа с метаданными"""
    chunk_id: str
    chunk_type: str
    text: str
    location: Dict = field(default_factory=dict)
    metadata: Dict = field(default_factory=dict)
    context_query: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Конвертирует чанк в словарь для совместимости с анализатором"""
        return {
            "id": self.chunk_id,
            "text": self.text,
            "chunk_type": self.chunk_type,
            "location": self.location,
            "metadata": self.metadata,
            "context_query": self.context_query
        }


class PDFChunker:
    """
    Умный чанкер PDF документов на основе pdfplumber.

    Особенности:
    - Качественное извлечение текста с сохранением структуры
    - Очистка артефактов (разорванные слова, лишние пробелы)
    - Семантическое разбиение на чанки (титульник, содержание, разделы, рисунки, таблицы)
    - Добавление метаданных для каждого чанка
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Инициализация чанкера.

        Args:
            chunk_size: Размер чанка в символах
            chunk_overlap: Перекрытие между чанками
        """
        if pdfplumber is None:
            raise ImportError("Библиотека pdfplumber не установлена")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Контекстные запросы для разных типов чанков
        self.context_queries = {
            "title_page": "оформление титульного листа отчета ГОСТ 7.32 основная надпись форма 2а",
            "toc": "оформление содержания оглавление нумерация страниц ГОСТ 7.32",
            "section_header": "нумерация разделов подразделов заголовки оформление ГОСТ 2.105",
            "text": "абзацный отступ шрифт размер интервалы перечисления формулы ГОСТ 2.105",
            "table_ref": "оформление таблиц нумерация заголовки граф ссылки на таблицы ГОСТ 2.105",
            "figure_ref": "оформление рисунков подписи нумерация ссылки на рисунки ГОСТ 2.105",
            "bibliography": "оформление списка литературы библиографическое описание ГОСТ 7.32 ГОСТ Р 7.0.5"
        }

        logger.info(f"PDFChunker инициализирован (chunk_size={chunk_size}, overlap={chunk_overlap})")

    def chunk_pdf(self, pdf_path: str) -> List[DocumentChunk]:
        """
        Загружает PDF файл и разбивает на чанки.

        Args:
            pdf_path: Путь к PDF файлу

        Returns:
            Список чанков DocumentChunk
        """
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"Файл не найден: {pdf_path}")

        logger.info(f"Начало обработки PDF: {path.name}")
        chunks = []
        chunk_id = 0

        try:
            with pdfplumber.open(str(path)) as pdf:
                logger.info(f"Открыт PDF: {len(pdf.pages)} страниц")

                for page_num, page in enumerate(pdf.pages, 1):
                    # Извлекаем текст с максимальной точностью
                    page_text = page.extract_text()

                    if not page_text or not page_text.strip():
                        logger.debug(f"Страница {page_num} пустая, пропускаем")
                        continue

                    # Очищаем текст от артефактов
                    cleaned_text = self._clean_text(page_text)

                    # Обрабатываем страницу
                    page_chunks = self._process_page(cleaned_text, page_num, chunk_id)
                    chunks.extend(page_chunks)
                    chunk_id += len(page_chunks)

                logger.info(f"Из PDF '{path.name}' извлечено {len(chunks)} чанков")
                return chunks

        except Exception as e:
            logger.error(f"Ошибка при чтении PDF {path.name}: {e}", exc_info=True)
            raise

    def _clean_text(self, text: str) -> str:
        """
        Очищает текст от артефактов извлечения PDF.

        Args:
            text: Исходный текст

        Returns:
            Очищенный текст
        """
        # Удаляем лишние пробелы внутри слов (частая проблема PDF)
        text = re.sub(r'(\w)\s+(\w)', r'\1\2', text)

        # Исправляем разорванные переносом строки слова (если слово разорвано в конце строки)
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)

        # Заменяем множественные пробелы на один
        text = re.sub(r'\s{2,}', ' ', text)

        # Удаляем пробелы перед знаками препинания
        text = re.sub(r'\s+([.,;:!?])', r'\1', text)

        # Нормализуем переносы строк
        text = re.sub(r'\n\s*\n', '\n\n', text)

        return text.strip()

    def _process_page(self, page_text: str, page_num: int, start_id: int) -> List[DocumentChunk]:
        """
        Обрабатывает одну страницу и разбивает на чанки.

        Args:
            page_text: Текст страницы
            page_num: Номер страницы
            start_id: Начальный ID чанка

        Returns:
            Список чанков
        """
        lines = [line.strip() for line in page_text.split('\n') if line.strip()]
        if not lines:
            return []

        # Определяем тип страницы
        page_type = self._detect_page_type(page_text, lines)

        # Титульная страница, содержание и библиография обрабатываются целиком
        if page_type in ["title_page", "toc", "bibliography"]:
            chunk = self._make_chunk(
                chunk_type=page_type,
                text=page_text,
                page=page_num,
                chunk_id=start_id
            )
            return [chunk]

        # Обычная страница с содержанием - разбиваем на логические блоки
        return self._split_content_page(lines, page_num, start_id)

    def _detect_page_type(self, page_text: str, lines: List[str]) -> str:
        """
        Определяет тип страницы по содержимому.

        Args:
            page_text: Полный текст страницы
            lines: Список строк

        Returns:
            Тип страницы
        """
        text_lower = page_text.lower()

        # Титульный лист
        if ('отчет по производственной практике' in text_lower or
            'отчет по учебной практике' in text_lower) and 'факультет' in text_lower:
            return "title_page"

        # Содержание
        if 'содержание' in text_lower and any(re.search(r'\.+\s*\d+', line) for line in lines):
            return "toc"

        # Библиография
        if ('список использованных источников' in text_lower or
            'литература' in text_lower or
            'библиографический список' in text_lower):
            return "bibliography"

        return "content"

    def _split_content_page(self, lines: List[str], page_num: int, start_id: int) -> List[DocumentChunk]:
        """
        Разбивает страницу с содержанием на логические блоки.

        Args:
            lines: Список строк
            page_num: Номер страницы
            start_id: Начальный ID чанка

        Returns:
            Список чанков
        """
        chunks = []
        current_block = []
        chunk_id = start_id

        for line in lines:
            # Пропускаем номера страниц
            if re.match(r'^\s*\d+\s*$', line):
                continue

            # Рисунок
            if re.match(r'^[Рр](исунок|ис\.?)?\s*\d+', line):
                if current_block:
                    chunks.append(self._make_text_chunk('\n'.join(current_block), page_num, chunk_id))
                    chunk_id += 1
                    current_block = []

                chunks.append(self._make_chunk(
                    chunk_type="figure_ref",
                    text=line,
                    page=page_num,
                    chunk_id=chunk_id,
                    metadata={"figure_number": self._extract_num(line)}
                ))
                chunk_id += 1

            # Таблица
            elif re.match(r'^[Тт](аблица|аб\.?)?\s*\d+', line):
                if current_block:
                    chunks.append(self._make_text_chunk('\n'.join(current_block), page_num, chunk_id))
                    chunk_id += 1
                    current_block = []

                chunks.append(self._make_chunk(
                    chunk_type="table_ref",
                    text=line,
                    page=page_num,
                    chunk_id=chunk_id,
                    metadata={"table_number": self._extract_num(line)}
                ))
                chunk_id += 1

            # Заголовок раздела
            elif re.match(r'^\d+(?:\.\d+)+\s+.+$', line):
                if current_block:
                    chunks.append(self._make_text_chunk('\n'.join(current_block), page_num, chunk_id))
                    chunk_id += 1
                    current_block = []

                match = re.match(r'^(\d+(?:\.\d+)+)\s+(.+)$', line)
                if match:
                    chunks.append(self._make_chunk(
                        chunk_type="section_header",
                        text=line,
                        page=page_num,
                        chunk_id=chunk_id,
                        metadata={
                            "section_number": match.group(1),
                            "section_title": match.group(2)
                        }
                    ))
                    chunk_id += 1

            # Обычный текст
            else:
                current_block.append(line)

        # Добавляем последний блок
        if current_block:
            chunks.append(self._make_text_chunk('\n'.join(current_block), page_num, chunk_id))

        return chunks

    def _extract_num(self, text: str) -> Optional[str]:
        """Извлекает номер из текста (для рисунков, таблиц)"""
        match = re.search(r'\d+', text)
        return match.group() if match else None

    def _make_chunk(self, chunk_type: str, text: str, page: int, chunk_id: int,
                    metadata: Dict = None) -> DocumentChunk:
        """Создает чанк с метаданными"""
        return DocumentChunk(
            chunk_id=f"{chunk_id:04d}",
            chunk_type=chunk_type,
            text=text[:3000],  # Ограничиваем размер
            location={"page": page},
            metadata=metadata or {},
            context_query=self.context_queries.get(chunk_type, "правила оформления технической документации ГОСТ")
        )

    def _make_text_chunk(self, text: str, page: int, chunk_id: int) -> DocumentChunk:
        """Создает текстовый чанк с расширенными метаданными"""
        meta = {
            "word_count": len(text.split()),
            "has_formulas": bool(re.search(r'\[?\d+\]?|\([^)]+\)', text)),
            "has_lists": bool(re.search(r'^[–•\-)\d\.]\s+', text, re.MULTILINE)),
            "has_gost_ref": bool(re.search(r'ГОСТ\s+\d+', text)),
            "has_requirements": bool(re.search(r'(должен|следует|требуется|не допускается)', text, re.IGNORECASE))
        }
        return self._make_chunk("text", text, page, chunk_id, metadata=meta)

    def save_chunks(self, chunks: List[DocumentChunk], output_path: str):
        """
        Сохраняет чанки в JSON файл.

        Args:
            chunks: Список чанков
            output_path: Путь для сохранения
        """
        from collections import Counter

        types = Counter(c.chunk_type for c in chunks)
        data = {
            "total_chunks": len(chunks),
            "types": dict(types),
            "chunks": [c.to_dict() for c in chunks]
        }

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"✅ Сохранено {len(chunks)} чанков в {output_path}")
        logger.info(f"📊 Типы чанков: {dict(types)}")