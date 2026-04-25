"""
PDF Parser v4.0 (PyMuPDF Edition) — с постобработкой по координатам
Особенности:
- Извлечение строк с координатами (y0, y1)
- Группировка строк в абзацы по вертикальным отступам внутри страницы
- Сохранение координат в текстовых чанках
- Умная склейка чанков по вертикальному разрыву в merge_text_chunks
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import fitz  # PyMuPDF
except ImportError:
    raise ImportError("Требуется установка: pip install pymupdf")

try:
    from src.logging_config import get_logger
except ImportError:
    from logging_config import get_logger

logger = get_logger(__name__)


# ─── Константы ────────────────────────────────────────────────────────────────

_STAMP_TOKENS = frozenset(["Инв.№", "Подп.", "Взам.", "Копировал", "Формат", "Лист"])
_STAMP_Y_THRESHOLD = 0.85

_RE_NUMBERED_HEADER = re.compile(r'^([А-ЯЁA-Z]\.)?(\d+(?:\.\d+)*)\s+([А-ЯЁA-Z][А-ЯЁa-zё\s]{2,})$')
_RE_FIGURE = re.compile(r'^Рисун?о?к\.?\s*\d+', re.IGNORECASE)
_RE_TABLE_LABEL = re.compile(r'^Таблиц?а\.?\s*[\dА-ЯЁ]', re.IGNORECASE)
_RE_PAGE_NUMBER = re.compile(r'^\d{1,3}$')
_RE_ABBREVIATION = re.compile(r'^([А-ЯЁA-Z]{2,10}(?:\s[А-ЯЁA-Z]{1,5})?)\s*[–—-]\s*(.{5,})$', re.MULTILINE)
_RE_FORMULA = re.compile(r'[=≠±≤≥∑∏∫√∞→←↔∧∨¬αβγδεζθικλμνξπρστυφχψω²³⁴ⁿ₁₂₃]')
_RE_HYPHENATED_WORD = re.compile(r'(\w+)-\s*$')

# Порог для склейки (в пунктах) – расстояние между нижней границей одного чанка
# и верхней границей следующего, при котором они считаются одним абзацем.
MERGE_GAP = 12.0


@dataclass
class DocumentChunk:
    chunk_id: str
    chunk_type: str
    text: str
    location: dict = field(default_factory=dict)   # для текстовых чанков: {"page": int, "y0": float, "y1": float}
    metadata: dict = field(default_factory=dict)
    context_query: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.chunk_id,
            "text": self.text,
            "chunk_type": self.chunk_type,
            "location": self.location,
            "metadata": self.metadata,
            "context_query": self.context_query,
        }


class PDFChunker:
    _CONTEXT_QUERIES = {
        "title_page": "титульный лист оформление ГОСТ 7.32 отчет",
        "toc": "содержание оглавление нумерация страниц ГОСТ 7.32",
        "section_header": "заголовок раздела подраздела оформление нумерация ГОСТ 2.105",
        "text": "текст абзац отступ интервал шрифт ГОСТ 2.105 требования",
        "figure_ref": "рисунок подпись нумерация оформление ГОСТ 2.105",
        "table_ref": "таблица заголовок сетка данные оформление ГОСТ 2.105",
        "bibliography": "список литературы источники оформление ГОСТ 7.32",
        "abbreviations": "перечень сокращений обозначений расшифровка ГОСТ 7.32",
    }

    def __init__(self) -> None:
        self.document_abbreviations: dict[str, str] = {}
        self.current_section_context: str = ""
        logger.info("PDFChunker v4.0 (PyMuPDF) инициализирован")

    # ─── Публичный API ────────────────────────────────────────────────────────

    def chunk_pdf(self, pdf_path: str, save_to: str | None = None) -> list[DocumentChunk]:
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"Файл не найден: {pdf_path}")

        logger.info(f"Обработка PDF: {path.name}")
        self.document_abbreviations = {}
        self.current_section_context = ""
        chunks: list[DocumentChunk] = []
        chunk_id = 0

        doc = fitz.open(str(path))
        total_pages = len(doc)
        logger.info(f"Страниц: {total_pages}")

        # Проход 1: сбор сокращений
        for page_num, page in enumerate(doc, 1):
            text = page.get_text("text")
            if self._is_abbreviation_page(text):
                self.document_abbreviations.update(self._extract_abbreviations_from_text(text))

        if self.document_abbreviations:
            logger.info(f"Найдено {len(self.document_abbreviations)} сокращений.")

        # Проход 2: основная разбивка
        for page_num, page in enumerate(doc, 1):
            # Извлекаем все строки страницы с координатами (единый список)
            lines_info = self._extract_lines_with_metadata(page)
            if not lines_info:
                continue

            # Группируем строки в абзацы по вертикальным разрывам
            paragraphs = self._group_lines_into_paragraphs(lines_info, page_num)
            for para in paragraphs:
                # Обрабатываем каждую группу (абзац) – определяем, не является ли она заголовком/таблицей
                para_chunks = self._finalize_paragraph(para, page_num, chunk_id + len(chunks))
                chunks.extend(para_chunks)

        doc.close()

        stats = defaultdict(int)
        for c in chunks:
            stats[c.chunk_type] += 1
        logger.info(f"Готово. Чанков: {len(chunks)}. Распределение: {dict(stats)}")

        if save_to:
            self.save_chunks(chunks, save_to)

        return chunks

    def save_chunks(self, chunks: list[DocumentChunk], output_path: str) -> None:
        data = {
            "total_chunks": len(chunks),
            "chunks": [c.to_dict() for c in chunks]
        }
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Чанки сохранены в {output_path}")

    # ─── Извлечение строк страницы с координатами ────────────────────────────

    def _extract_lines_with_metadata(self, page) -> list[dict]:
        """
        Возвращает список строк страницы с полями:
        - text
        - y0 (верхняя граница)
        - y1 (нижняя граница)
        - is_centered
        - is_bold
        - font_size
        """
        blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
        page_width = page.rect.width
        lines = []

        for block in blocks:
            if block["type"] != 0:  # не текст
                continue
            # Проверка на штамп (нижняя часть страницы)
            if block["bbox"][3] > page.rect.height * _STAMP_Y_THRESHOLD:
                block_text = "".join(span["text"] for line in block["lines"] for span in line["spans"])
                if any(token in block_text for token in _STAMP_TOKENS):
                    continue

            for line in block["lines"]:
                line_text = "".join(span["text"] for span in line["spans"]).strip()
                if not line_text or _RE_PAGE_NUMBER.match(line_text):
                    continue

                bbox = line["bbox"]
                x0, y0, x1, y1 = bbox
                is_centered = abs(x0 - (page_width - x1)) < 25
                fontnames = {span.get("font", "") for span in line["spans"]}
                is_bold = any("Bold" in fn or "Heavy" in fn for fn in fontnames)
                font_size = max(span.get("size", 0) for span in line["spans"])

                lines.append({
                    "text": line_text,
                    "y0": y0,
                    "y1": y1,
                    "is_centered": is_centered,
                    "is_bold": is_bold,
                    "font_size": font_size,
                })
        return lines

    def _group_lines_into_paragraphs(self, lines: list[dict], page_num: int) -> list[list[dict]]:
        """
        Группирует строки в абзацы по вертикальным разрывам.
        Возвращает список абзацев (каждый абзац – список словарей строк).
        """
        if not lines:
            return []

        paragraphs = []
        current_para = [lines[0]]
        for i in range(1, len(lines)):
            prev = lines[i-1]
            curr = lines[i]
            gap = curr["y0"] - prev["y1"]
            # Если расстояние между строками больше порога – новый абзац
            if gap > MERGE_GAP:
                paragraphs.append(current_para)
                current_para = [curr]
            else:
                current_para.append(curr)
        if current_para:
            paragraphs.append(current_para)
        return paragraphs

    # ─── Обработка абзаца (заголовок, рисунок, таблица или обычный текст) ────

    def _finalize_paragraph(self, para_lines: list[dict], page_num: int, local_id: int) -> list[DocumentChunk]:
        """Преобразует группу строк в один или несколько чанков (заголовок, текст, таблица…)."""
        if not para_lines:
            return []

        # Склеиваем текст абзаца
        para_text = " ".join(l["text"] for l in para_lines).strip()
        if not para_text:
            return []

        # Вычисляем общие координаты абзаца
        y0 = min(l["y0"] for l in para_lines)
        y1 = max(l["y1"] for l in para_lines)
        is_centered = any(l["is_centered"] for l in para_lines)  # если хоть одна строка по центру
        is_bold = any(l["is_bold"] for l in para_lines)
        font_size = max(l["font_size"] for l in para_lines)

        # Проверка на номерной заголовок
        header_match = _RE_NUMBERED_HEADER.match(para_text)
        if header_match:
            prefix, number, title = (header_match.group(1) or "", header_match.group(2), header_match.group(3))
            full_number = (prefix + number).strip(".")
            self.current_section_context = f"Раздел {full_number}"
            return [DocumentChunk(
                chunk_id=f"p{page_num}_h{local_id}",
                chunk_type="section_header",
                text=para_text,
                location={"page": page_num, "y0": y0, "y1": y1},
                metadata={
                    "section_number": full_number,
                    "title": title,
                    "is_centered": is_centered,
                    "is_bold": is_bold,
                    "font_size": font_size,
                    "known_abbreviations": self.document_abbreviations,
                },
                context_query=self._CONTEXT_QUERIES["section_header"]
            )]

        # Одиночные заголовки без номера
        if para_text.lower() in ["введение", "заключение", "реферат", "содержание"]:
            self.current_section_context = para_text
            return [DocumentChunk(
                chunk_id=f"p{page_num}_h{local_id}",
                chunk_type="section_header",
                text=para_text,
                location={"page": page_num, "y0": y0, "y1": y1},
                metadata={
                    "title": para_text,
                    "is_centered": is_centered,
                    "is_bold": is_bold,
                    "font_size": font_size,
                    "known_abbreviations": self.document_abbreviations,
                },
                context_query=self._CONTEXT_QUERIES["section_header"]
            )]

        # Рисунок
        if _RE_FIGURE.match(para_text):
            return [DocumentChunk(
                chunk_id=f"p{page_num}_fig{local_id}",
                chunk_type="figure_ref",
                text=para_text,
                location={"page": page_num, "y0": y0, "y1": y1},
                metadata={
                    "is_centered": is_centered,
                    "known_abbreviations": self.document_abbreviations,
                },
                context_query=self._CONTEXT_QUERIES["figure_ref"]
            )]

        # Подпись таблицы
        if _RE_TABLE_LABEL.match(para_text):
            return [DocumentChunk(
                chunk_id=f"p{page_num}_tbl_lbl{local_id}",
                chunk_type="table_ref",
                text=para_text,
                location={"page": page_num, "y0": y0, "y1": y1},
                metadata={
                    "is_centered": is_centered,
                    "known_abbreviations": self.document_abbreviations,
                },
                context_query=self._CONTEXT_QUERIES["table_ref"]
            )]

        # Обычный текст
        if self.current_section_context and not para_text.startswith("["):
            para_text = f"[{self.current_section_context}] {para_text}"

        return [DocumentChunk(
            chunk_id=f"p{page_num}_t{local_id}",
            chunk_type="text",
            text=para_text,
            location={"page": page_num, "y0": y0, "y1": y1},
            metadata={
                "words": len(para_text.split()),
                "is_centered": is_centered,
                "is_bold": is_bold,
                "font_size": font_size,
                "known_abbreviations": self.document_abbreviations,
            },
            context_query=self._CONTEXT_QUERIES["text"]
        )]

    # ─── Постобработка: слияние чанков с учётом координат ────────────────────

    def merge_text_chunks(self, chunks: list[DocumentChunk]) -> list[DocumentChunk]:
        """
        Склеивает соседние текстовые чанки на одной странице,
        если расстояние между ними по вертикали <= MERGE_GAP.
        Заголовки, таблицы, рисунки не трогает.
        """
        if not chunks:
            return chunks

        merged = []
        current_text_chunks = []   # список чанков, которые склеиваются
        # Для хранения объединённых координат
        current_min_y0 = None
        current_max_y1 = None
        current_page = None
        current_text = []

        for chunk in chunks:
            if chunk.chunk_type != "text":
                # Завершаем текущую группу текстовых чанков
                if current_text_chunks:
                    merged.append(self._make_merged_text_chunk(
                        current_text_chunks, current_page, current_min_y0, current_max_y1
                    ))
                    current_text_chunks = []
                    current_min_y0 = None
                    current_max_y1 = None
                    current_page = None
                    current_text = []
                merged.append(chunk)
                continue

            # Текстовый чанк
            page = chunk.location.get("page")
            y0 = chunk.location.get("y0")
            y1 = chunk.location.get("y1")
            if y0 is None or y1 is None:
                # Если нет координат, просто добавляем как есть (старый формат)
                merged.append(chunk)
                continue

            if current_text_chunks:
                prev_y1 = current_max_y1
                gap = y0 - prev_y1
                # Если страница сменилась или разрыв больше порога – завершаем текущий абзац
                if page != current_page or gap > MERGE_GAP:
                    merged.append(self._make_merged_text_chunk(
                        current_text_chunks, current_page, current_min_y0, current_max_y1
                    ))
                    current_text_chunks = []
                    current_min_y0 = None
                    current_max_y1 = None
                    current_page = None
                    current_text = []

            current_text_chunks.append(chunk)
            # Обновляем общий bounding box
            if current_min_y0 is None or y0 < current_min_y0:
                current_min_y0 = y0
            if current_max_y1 is None or y1 > current_max_y1:
                current_max_y1 = y1
            current_page = page

        if current_text_chunks:
            merged.append(self._make_merged_text_chunk(
                current_text_chunks, current_page, current_min_y0, current_max_y1
            ))

        return merged

    def _make_merged_text_chunk(self, chunks: list[DocumentChunk], page: int, y0: float, y1: float) -> DocumentChunk:
        """Создаёт один текстовый чанк из нескольких, склеивая их текст через пробел."""
        full_text = " ".join(c.text for c in chunks).strip()
        # Объединяем метаданные (берём из первого, но можно агрегировать)
        meta = chunks[0].metadata.copy()
        meta["words"] = len(full_text.split())
        meta["merged_from"] = len(chunks)
        return DocumentChunk(
            chunk_id=f"p{page}_merged_{y0}_{y1}",
            chunk_type="text",
            text=full_text,
            location={"page": page, "y0": y0, "y1": y1},
            metadata=meta,
            context_query=self._CONTEXT_QUERIES["text"]
        )

    # ─── Служебные методы (сокращения, таблицы, утилиты) ─────────────────────

    def _is_abbreviation_page(self, text: str) -> bool:
        text_lower = text.lower()
        keywords = ["перечень принятых сокращений", "список сокращений", "обозначения и сокращения"]
        if any(kw in text_lower for kw in keywords):
            return True
        matches = _RE_ABBREVIATION.findall(text)
        return len(matches) >= 3

    def _extract_abbreviations_from_text(self, text: str) -> dict[str, str]:
        result = {}
        for match in _RE_ABBREVIATION.findall(text):
            key, val = match[0].strip(), match[1].strip()
            result[key] = val
        return result

    # Заглушки для таблиц (при необходимости можно включить)
    @staticmethod
    def _table_to_markdown(table) -> str:
        rows = table.extract()
        if not rows:
            return ""
        md_lines = []
        max_cols = max(len(row) for row in rows)
        for i, row in enumerate(rows):
            clean_row = [str(cell).replace("|", "\\|").strip() if cell else "" for cell in row]
            while len(clean_row) < max_cols:
                clean_row.append("")
            md_lines.append("| " + " | ".join(clean_row) + " |")
            if i == 0:
                md_lines.append("| " + " | ".join(["---"] * max_cols) + " |")
        return "\n".join(md_lines)


# ─── Тестовый запуск (пример использования) ───────────────────────────────────
if __name__ == "__main__":
    # Пример использования
    chunker = PDFChunker()
    chunks = chunker.chunk_pdf("src/МФАС.563563.003 ПМ.pdf", save_to="output_raw.json")
    print(f"Сырых чанков: {len(chunks)}")
    merged = chunker.merge_text_chunks(chunks)
    print(f"После слияния: {len(merged)}")
    chunker.save_chunks(merged, "output_merged.json")