"""
Улучшенный PDF Parser (Версия 2.2).
Исправления: надежное определение типов страниц и чанков, детальное логирование.
"""
import re
import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from pathlib import Path
from collections import Counter

try:
    import pdfplumber
except ImportError:
    raise ImportError("Установите pdfplumber: pip install pdfplumber")

try:
    from src.logging_config import get_logger
except ImportError:
    from logging_config import get_logger

logger = get_logger(__name__)

@dataclass
class DocumentChunk:
    chunk_id: str
    chunk_type: str
    text: str
    location: Dict = field(default_factory=dict)
    metadata: Dict = field(default_factory=dict)
    context_query: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.chunk_id,
            "text": self.text,
            "chunk_type": self.chunk_type,
            "location": self.location,
            "metadata": self.metadata,
            "context_query": self.context_query
        }

class PDFChunker:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.context_queries = {
            "title_page": "титульный лист отчет практика ГОСТ 7.32",
            "toc": "содержание оглавление номера страниц ГОСТ 7.32",
            "section_header": "заголовок раздела подраздела номер ГОСТ 2.105",
            "text": "текст абзац опечатки пробелы переносы ГОСТ 2.105",
            "figure_ref": "рисунок подпись номер ГОСТ 2.105",
            "table_ref": "таблица заголовок номер ГОСТ 2.105",
            "bibliography": "список литературы источники ГОСТ 7.32"
        }
        logger.info(f"PDFChunker initialized")

    def chunk_pdf(self, pdf_path: str) -> List[DocumentChunk]:
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"Файл не найден: {pdf_path}")

        logger.info(f"📄 Обработка PDF: {path.name}")
        chunks = []
        chunk_id = 0

        with pdfplumber.open(str(path)) as pdf:
            logger.info(f"Открыт PDF: {len(pdf.pages)} страниц")

            for page_num, page in enumerate(pdf.pages, 1):
                raw_text = page.extract_text()
                if not raw_text or not raw_text.strip():
                    continue

                # Очистка текста (минимальная, чтобы не ломать структуру)
                cleaned_text = self._clean_text(raw_text)
                lines = [l.strip() for l in cleaned_text.split('\n') if l.strip()]

                # 1. Определяем тип страницы
                page_type = self._detect_page_type(cleaned_text, lines, page_num)
                logger.debug(f"Стр. {page_num}: тип страницы = {page_type}")

                if page_type == "title_page":
                    chunk = self._make_chunk("title_page", cleaned_text, page_num, chunk_id)
                    chunks.append(chunk)
                    chunk_id += 1
                
                elif page_type == "toc":
                    chunk = self._make_chunk("toc", cleaned_text, page_num, chunk_id)
                    chunks.append(chunk)
                    chunk_id += 1

                elif page_type == "bibliography":
                    chunk = self._make_chunk("bibliography", cleaned_text, page_num, chunk_id)
                    chunks.append(chunk)
                    chunk_id += 1
                
                else:
                    # Обычная страница: разбиваем на блоки
                    page_chunks = self._split_content_page(lines, page_num, chunk_id)
                    chunks.extend(page_chunks)
                    chunk_id += len(page_chunks)

        # Статистика
        stats = Counter(c.chunk_type for c in chunks)
        logger.info(f"✅ Готово. Всего чанков: {len(chunks)}. Распределение: {dict(stats)}")
        return chunks

    def _clean_text(self, text: str) -> str:
        # 1. Убираем лишние пробелы между буквами внутри слов (артефакт PDF)
        # Эвристика: если между двумя кириллическими буквами один пробел, склеиваем.
        # Но осторожно, чтобы не склеить слова.
        # Более безопасный вариант: склеиваем, если слово разорвано явно странно (например, "бла годаря")
        # Пока оставим простую нормализацию множественных пробелов.
        text = re.sub(r'\s{2,}', ' ', text) # Множественные пробелы -> один
        
        # Убираем пробелы перед точками/запятыми
        text = re.sub(r'\s+([.,;:!?])', r'\1', text)
        
        # Склеиваем разорванные переносом строки слова (дефис в конце строки)
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
        
        return text.strip()

    def _detect_page_type(self, text: str, lines: List[str], page_num: int) -> str:
        text_lower = text.lower()

        # Титульник (обычно стр 1, есть ключевые слова)
        if page_num <= 2:
            if ('отчет по' in text_lower and 'практике' in text_lower) or \
               ('факультет' in text_lower and 'кафедра' in text_lower):
                return "title_page"

        # Содержание (много точек и номеров страниц)
        # Ищем строки вида: "Название ........... 12"
        toc_lines = 0
        for line in lines:
            if re.search(r'\.{5,}\s*\d+\s*$', line):
                toc_lines += 1
        
        if toc_lines >= 3: # Если хотя бы 3 строки с точками и номерами
            return "toc"

        # Библиография
        if 'список использованных источников' in text_lower or \
           'список литературы' in text_lower:
            return "bibliography"

        return "content"

    def _split_content_page(self, lines: List[str], page_num: int, start_id: int) -> List[DocumentChunk]:
        chunks = []
        current_block = []
        chunk_id = start_id

        # Паттерны
        # Заголовок с номером: "1.2 Название"
        re_section = re.compile(r'^(\d+(?:\.\d+)*)\s+(.+)$')
        # Рисунок: "Рисунок 1 – ..."
        re_figure = re.compile(r'^[Рр](исунок|ис\.?)?\s*\d+', re.IGNORECASE)
        # Таблица: "Таблица 1 – ..."
        re_table = re.compile(r'^[Тт](аблица|аб\.?)?\s*\d+', re.IGNORECASE)
        # Заголовок без номера (Введение, Заключение) - если строка короткая и одна из известных
        re_standalone = re.compile(r'^(Введение|Заключение|Приложение)\s*$', re.IGNORECASE)

        for line in lines:
            # Пропуск номеров страниц
            if re.match(r'^\d{1,3}$', line.strip()):
                continue

            is_header = False
            
            # Проверка на рисунок
            if re_figure.match(line):
                is_header = True
                if current_block:
                    chunks.append(self._make_text_chunk('\n'.join(current_block), page_num, chunk_id))
                    chunk_id += 1
                    current_block = []
                chunks.append(self._make_chunk("figure_ref", line, page_num, chunk_id, {"raw": line}))
                chunk_id += 1
                continue

            # Проверка на таблицу
            if re_table.match(line):
                is_header = True
                if current_block:
                    chunks.append(self._make_text_chunk('\n'.join(current_block), page_num, chunk_id))
                    chunk_id += 1
                    current_block = []
                chunks.append(self._make_chunk("table_ref", line, page_num, chunk_id, {"raw": line}))
                chunk_id += 1
                continue

            # Проверка на заголовок раздела
            match_sec = re_section.match(line)
            if match_sec:
                is_header = True
                if current_block:
                    chunks.append(self._make_text_chunk('\n'.join(current_block), page_num, chunk_id))
                    chunk_id += 1
                    current_block = []
                
                num, title = match_sec.groups()
                chunks.append(self._make_chunk(
                    "section_header", 
                    line, 
                    page_num, 
                    chunk_id, 
                    {"section_number": num, "title": title}
                ))
                chunk_id += 1
                continue
            
            # Проверка на одиночный заголовок
            if re_standalone.match(line):
                is_header = True
                if current_block:
                    chunks.append(self._make_text_chunk('\n'.join(current_block), page_num, chunk_id))
                    chunk_id += 1
                    current_block = []
                chunks.append(self._make_chunk("section_header", line, page_num, chunk_id, {"title": line}))
                chunk_id += 1
                continue

            # Обычный текст
            current_block.append(line)

        if current_block:
            chunks.append(self._make_text_chunk('\n'.join(current_block), page_num, chunk_id))

        return chunks

    def _make_chunk(self, ctype: str, text: str, page: int, cid: int, meta: Dict = None) -> DocumentChunk:
        return DocumentChunk(
            chunk_id=f"p{page}_c{cid}",
            chunk_type=ctype,
            text=text[:3000],
            location={"page": page},
            metadata=meta or {},
            context_query=self.context_queries.get(ctype, "ГОСТ")
        )

    def _make_text_chunk(self, text: str, page: int, cid: int) -> DocumentChunk:
        meta = {
            "words": len(text.split()),
            "has_lists": bool(re.search(r'^[-•\d)]', text, re.MULTILINE))
        }
        return self._make_chunk("text", text, page, cid, meta)