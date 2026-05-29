# src/chunker.py
import re
import docx
import pdfplumber
from typing import List, Optional
from src.schemas import ChunkType, DocumentChunk


class DocumentChunker:
    def __init__(self):
        self.chunks: List[DocumentChunk] = []

    def chunk_pdf(self, pdf_path: str) -> List[DocumentChunk]:
        """Извлекает текст из PDF с привязкой к страницам и контекстным объединением."""
        self.chunks = []
        chunk_id = 0

        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            for page_num, page in enumerate(pdf.pages, 1):
                page_text = page.extract_text(x_tolerance=3, y_tolerance=3)
                if not page_text:
                    continue

                lines = [line.strip() for line in page_text.split('\n') if line.strip()]
                page_chunks = self._process_page_lines(lines, page_num, chunk_id, total_pages)
                self.chunks.extend(page_chunks)
                chunk_id += len(page_chunks)

        return self.chunks

    def _is_related_type(self, prev_line: str, curr_line: str) -> bool:
        """
        Определяет, относятся ли две строки к одному логическому блоку.
        Объединяем, если:
        - Обе строки — подписи рисунков/таблиц
        - Обе — элементы перечисления (начинаются с дефиса или а) б) в))
        - Обе — формулы или пояснения к ним ("где ...")
        """
        # Паттерны для разных типов блоков
        figure_pattern = re.compile(r'^[Рр]ис(?:\.|\унок)?\s+\d+', re.IGNORECASE)
        table_pattern = re.compile(r'^[Тт]аблица\s+\d+', re.IGNORECASE)
        list_pattern = re.compile(r'^[-–—]\s+|^[а-яa-z]\)\s+', re.IGNORECASE)
        formula_where = re.compile(r'^[Гг]де(\s|:|$)', re.IGNORECASE)
        note_pattern = re.compile(r'^[Пп]римечание', re.IGNORECASE)

        # Если обе строки — подписи рисунков → объединяем
        if figure_pattern.search(prev_line) and figure_pattern.search(curr_line):
            return True
        # Если обе — заголовки таблиц → объединяем
        if table_pattern.search(prev_line) and table_pattern.search(curr_line):
            return True
        # Если обе — элементы списка → объединяем
        if list_pattern.match(prev_line) and list_pattern.match(curr_line):
            return True
        # Если prev — "где:", а curr — продолжение расшифровки → объединяем
        if formula_where.match(prev_line.strip()) or formula_where.match(curr_line.strip()):
            return True
        # Если обе — примечания → объединяем
        if note_pattern.match(prev_line) and note_pattern.match(curr_line):
            return True

        return False

    def _get_chunk_type_for_line(self, line: str) -> ChunkType:
        """Определяет тип чанка по содержимому строки."""
        line_stripped = line.strip()

        if re.match(r'^\d+(?:\.\d+)*\s+[А-ЯA-Z]', line_stripped):
            return ChunkType.SECTION_HEADER
        elif re.match(r'^[Тт]аблица\s+\d+', line_stripped, re.IGNORECASE):
            return ChunkType.TABLE_TITLE
        elif re.match(r'^[Рр]ис(?:\.|\унок)?\s+\d+', line_stripped, re.IGNORECASE):
            return ChunkType.FIGURE_CAPTION
        elif re.match(r'^[Пп]римечание', line_stripped, re.IGNORECASE):
            return ChunkType.NOTE
        elif re.match(r'^[-–—]\s+|^[а-яa-z]\)\s+', line_stripped):
            return ChunkType.LIST_ITEM
        elif re.match(r'^[Гг]де(\s|:|$)', line_stripped):
            return ChunkType.FORMULA
        elif re.match(r'^[Аа][)]\s+|^1[)]\s+', line_stripped):  # Вложенные списки
            return ChunkType.LIST_ITEM

        return ChunkType.TEXT_BODY

    def _process_page_lines(
            self,
            lines: List[str],
            page_num: int,
            start_id: int,
            page_total: int
    ) -> List[DocumentChunk]:
        """
        Разбивает строки страницы на чанки с контекстным объединением.
        """
        local_chunks = []
        current_block: List[str] = []
        current_type: Optional[ChunkType] = None
        c_id = start_id

        for line in lines:
            # Пропускаем чистые номера страниц (одиночные цифры в строке)
            if re.match(r'^\d+$', line.strip()):
                continue

            line_type = self._get_chunk_type_for_line(line)

            # Если текущая строка НЕ относится к текущему блоку → закрываем блок
            if current_block and current_type and line_type != current_type:
                # Проверяем, можно ли всё-таки объединить (контекстная логика)
                if current_block and not self._is_related_type(current_block[-1], line):
                    # Создаём чанк из накопленного блока
                    local_chunks.append(self._create_chunk(
                        current_type,
                        '\n'.join(current_block),
                        page_num,
                        c_id,
                        page_total
                    ))
                    c_id += 1
                    current_block = []
                    current_type = None

            # Начинаем новый блок или продолжаем текущий
            if not current_block:
                current_type = line_type

            current_block.append(line)

        # Не забываем про последний блок на странице
        if current_block and current_type:
            local_chunks.append(self._create_chunk(
                current_type,
                '\n'.join(current_block),
                page_num,
                c_id,
                page_total
            ))

        return local_chunks

    def _create_chunk(
            self,
            chunk_type: ChunkType,
            text: str,
            page: int,
            chunk_id: int,
            page_total: int
    ) -> DocumentChunk:
        """Создаёт DocumentChunk с нормализованным текстом и метаданными."""
        clean_text = re.sub(r'\s+', ' ', text).strip()
        return DocumentChunk(
            chunk_id=f"{chunk_id:05d}",
            chunk_type=chunk_type,
            text=clean_text,
            location={"page": page, "page_total": page_total},
            metadata={
                "char_length": len(clean_text),
                "format": "pdf",
                "original_lines": text.count('\n') + 1
            }
        )


def _estimate_docx_pages(paragraphs) -> int:
    """Грубая оценка числа страниц: ~28 строк текста = 1 страница А4."""
    total_lines = sum(
        len(p.text.strip().split('\n'))
        for p in paragraphs
        if p.text.strip()
    )
    return max(1, round(total_lines / 28))


def chunk_docx(file_path: str) -> List[DocumentChunk]:
    """Извлекает чанки из DOCX с эмуляцией страниц и контекстным объединением."""
    doc = docx.Document(file_path)
    paragraphs = [p for p in doc.paragraphs if p.text.strip()]
    estimated_total = _estimate_docx_pages(paragraphs)

    chunks = []
    current_block: List[str] = []
    current_type: Optional[ChunkType] = None

    for i, para in enumerate(paragraphs):
        text = para.text.strip()
        if not text:
            continue

        line_type = _get_chunk_type_for_line_docx(text, para)

        # Контекстное объединение (аналогично PDF)
        if current_block and current_type and line_type != current_type:
            if not _is_related_type_docx(current_block[-1], text):
                # Создаём чанк из накопленного блока
                approx_page = _calc_approx_page(i, len(paragraphs), estimated_total)
                chunks.append(_create_docx_chunk(
                    current_type,
                    '\n'.join(current_block),
                    approx_page,
                    estimated_total,
                    f"doc_{i - len(current_block):04d}",
                    para.style.name
                ))
                current_block = []
                current_type = None

        if not current_block:
            current_type = line_type
        current_block.append(text)

    # Последний блок
    if current_block and current_type:
        approx_page = _calc_approx_page(len(paragraphs) - 1, len(paragraphs), estimated_total)
        chunks.append(_create_docx_chunk(
            current_type,
            '\n'.join(current_block),
            approx_page,
            estimated_total,
            f"doc_{len(paragraphs) - len(current_block):04d}",
            para.style.name
        ))

    return chunks


# ─── Вспомогательные функции для DOCX (аналоги методов класса) ──────────────

def _is_related_type_docx(prev_line: str, curr_line: str) -> bool:
    """DOCX-версия _is_related_type (те же правила)."""
    return DocumentChunker()._is_related_type(prev_line, curr_line)


def _get_chunk_type_for_line_docx(text: str, paragraph) -> ChunkType:
    """Определяет тип чанка для DOCX с учётом стилей."""
    # Сначала проверяем стиль (Heading → заголовок)
    if paragraph.style.name.startswith("Heading"):
        return ChunkType.SECTION_HEADER

    # Затем контент-эвристики
    return DocumentChunker()._get_chunk_type_for_line(text)


def _calc_approx_page(current_idx: int, total_items: int, estimated_pages: int) -> int:
    """Распределяет элементы по приблизительным страницам."""
    if estimated_pages <= 1:
        return 1
    return min(estimated_pages, (current_idx * estimated_pages) // total_items + 1)


def _create_docx_chunk(
        chunk_type: ChunkType,
        text: str,
        page: int,
        page_total: int,
        chunk_id: str,
        style_name: str
) -> DocumentChunk:
    """Создаёт DocumentChunk для DOCX."""
    clean_text = re.sub(r'\s+', ' ', text).strip()
    return DocumentChunk(
        chunk_id=chunk_id,
        chunk_type=chunk_type,
        text=clean_text,
        location={
            "page": page,
            "page_total": page_total,
            "estimated": True,
            "source_format": "docx"
        },
        metadata={
            "char_length": len(clean_text),
            "format": "docx",
            "style": style_name,
            "original_lines": text.count('\n') + 1
        }
    )