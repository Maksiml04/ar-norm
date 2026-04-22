# src/chunker_pdfplumber.py
import re
import json
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional
from pathlib import Path
import pdfplumber


@dataclass
class DocumentChunk:
    chunk_id: str
    chunk_type: str
    text: str
    location: Dict = field(default_factory=dict)
    metadata: Dict = field(default_factory=dict)
    context_query: str = ""

    def to_dict(self):
        return asdict(self)


class ReportChunker:
    def __init__(self):
        self.context_queries = {
            "title_page": "оформление титульного листа отчета ГОСТ 7.32 основная надпись форма 2а",
            "toc": "оформление содержания оглавление нумерация страниц ГОСТ 7.32",
            "section_header": "нумерация разделов подразделов заголовки оформление ГОСТ 2.105",
            "text": "абзацный отступ шрифт размер интервалы перечисления формулы ГОСТ 2.105",
            "table_ref": "оформление таблиц нумерация заголовки граф ссылки на таблицы ГОСТ 2.105",
            "figure_ref": "оформление рисунков подписи нумерация ссылки на рисунки ГОСТ 2.105",
            "bibliography": "оформление списка литературы библиографическое описание ГОСТ 7.32 ГОСТ Р 7.0.5"
        }

    def chunk_pdf(self, pdf_path: str) -> List[DocumentChunk]:
        chunks = []
        chunk_id = 0

        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                # Извлечение текста с настройками для лучшей точности
                page_text = page.extract_text(x_tolerance=2, y_tolerance=2)
                if not page_text:
                    continue

                # Предварительная очистка от артефактов переноса строк внутри слов
                # Часто в PDF слова разрываются дефисом и переносом строки
                page_text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', page_text)

                page_chunks = self._process_page(page_text, page_num, chunk_id)
                chunks.extend(page_chunks)
                chunk_id += len(page_chunks)
        return chunks

    def _process_page(self, page_text: str, page_num: int, start_id: int) -> List[DocumentChunk]:
        lines = [line.strip() for line in page_text.split('\n') if line.strip()]
        if not lines:
            return []

        page_type = self._detect_page_type(page_text, lines)
        if page_type in ["title_page", "toc", "bibliography"]:
            return [self._make_chunk(page_type, page_text, page_num, start_id)]

        return self._split_content_page(lines, page_num, start_id)

    def _detect_page_type(self, page_text: str, lines: List[str]) -> str:
        text_lower = page_text.lower()
        if 'отчет по производственной практике' in text_lower and 'факультет' in text_lower:
            return "title_page"
        if 'содержание' in text_lower and any(re.search(r'\.+\s*\d+', line) for line in lines):
            return "toc"
        if 'список использованных источников' in text_lower or 'литература' in text_lower:
            return "bibliography"
        return "content"

    def _split_content_page(self, lines: List[str], page_num: int, start_id: int) -> List[DocumentChunk]:
        chunks = []
        current_block = []
        chunk_id = start_id

        for line in lines:
            if re.match(r'^\s*\d+\s*$', line):  # номер страницы
                continue

            # Рисунок
            if re.match(r'^[Рр](исунок|ис\.)?\s*\d+', line):
                if current_block:
                    chunks.append(self._make_text_chunk('\n'.join(current_block), page_num, chunk_id))
                    chunk_id += 1
                    current_block = []
                chunks.append(self._make_chunk("figure_ref", line, page_num, chunk_id,
                                               metadata={"figure_number": self._extract_num(line)}))
                chunk_id += 1

            # Таблица
            elif re.match(r'^[Тт](аблица|аб\.)?\s*\d+', line):
                if current_block:
                    chunks.append(self._make_text_chunk('\n'.join(current_block), page_num, chunk_id))
                    chunk_id += 1
                    current_block = []
                chunks.append(self._make_chunk("table_ref", line, page_num, chunk_id,
                                               metadata={"table_number": self._extract_num(line)}))
                chunk_id += 1

            # Заголовок раздела
            elif re.match(r'^\d+(?:\.\d+)+\s+.+$', line):
                if current_block:
                    chunks.append(self._make_text_chunk('\n'.join(current_block), page_num, chunk_id))
                    chunk_id += 1
                    current_block = []
                match = re.match(r'^(\d+(?:\.\d+)+)\s+(.+)$', line)
                if match:
                    chunks.append(self._make_chunk("section_header", line, page_num, chunk_id,
                                                   metadata={"section_number": match.group(1),
                                                             "section_title": match.group(2)}))
                    chunk_id += 1

            # Текст
            else:
                current_block.append(line)

        if current_block:
            chunks.append(self._make_text_chunk('\n'.join(current_block), page_num, chunk_id))
        return chunks

    def _extract_num(self, text: str) -> Optional[str]:
        match = re.search(r'\d+', text)
        return match.group() if match else None

    def _make_chunk(self, chunk_type: str, text: str, page: int, chunk_id: int,
                    metadata: Dict = None) -> DocumentChunk:
        # Дополнительная очистка текста перед сохранением
        clean_text = re.sub(r'\s+', ' ', text).strip()
        return DocumentChunk(
            chunk_id=f"{chunk_id:04d}",
            chunk_type=chunk_type,
            text=clean_text[:3000],
            location={"page": page},
            metadata=metadata or {},
            context_query=self.context_queries.get(chunk_type, "правила оформления технической документации ГОСТ")
        )

    def _make_text_chunk(self, text: str, page: int, chunk_id: int) -> DocumentChunk:
        clean_text = re.sub(r'\s+', ' ', text).strip()
        meta = {
            "word_count": len(clean_text.split()),
            "has_formulas": bool(re.search(r'\[?\d+\]?|\([^)]+\)', clean_text)),
            "has_lists": bool(re.search(r'^[–•\-)\d\.]\s+', clean_text, re.MULTILINE)),
            "has_gost_ref": bool(re.search(r'ГОСТ\s+\d+', clean_text))
        }
        return self._make_chunk("text", clean_text, page, chunk_id, metadata=meta)