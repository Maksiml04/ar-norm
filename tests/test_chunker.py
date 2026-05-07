"""
Тесты для ReportChunker (chunker_pdfplumber.py).
Запуск: pytest tests/test_chunker.py -v
"""
import pytest
from unittest.mock import patch, MagicMock
from src.chunker_pdfplumber import ReportChunker, DocumentChunk


@pytest.fixture
def chunker():
    return ReportChunker()


# ── Тесты DocumentChunk ──────────────────────────────────────────────────────

class TestDocumentChunk:
    def test_to_dict_возвращает_словарь(self):
        chunk = DocumentChunk(
            chunk_id="0001",
            chunk_type="text",
            text="текст",
            location={"page": 1},
            metadata={},
            context_query="запрос",
        )
        d = chunk.to_dict()
        assert isinstance(d, dict)
        assert d["chunk_id"] == "0001"
        assert d["text"] == "текст"

    def test_дефолтные_значения(self):
        chunk = DocumentChunk(chunk_id="0001", chunk_type="text", text="текст")
        assert chunk.location == {}
        assert chunk.metadata == {}
        assert chunk.context_query == ""


# ── Тесты _detect_page_type ─────────────────────────────────────────────────

class TestDetectPageType:
    def test_титульный_лист(self, chunker):
        text = "Отчет по производственной практике факультет технический"
        result = chunker._detect_page_type(text, [])
        assert result == "title_page"

    def test_содержание(self, chunker):
        text = "Содержание\n1 Введение.........5"
        lines = ["Содержание", "1 Введение.........5"]
        result = chunker._detect_page_type(text, lines)
        assert result == "toc"

    def test_библиография(self, chunker):
        text = "Список использованных источников\n1. ГОСТ 2.105-95"
        result = chunker._detect_page_type(text, [])
        assert result == "bibliography"

    def test_обычная_страница(self, chunker):
        text = "Обычный текст без специальных признаков"
        result = chunker._detect_page_type(text, [])
        assert result == "content"


# ── Тесты _make_chunk ────────────────────────────────────────────────────────

class TestMakeChunk:
    def test_создаёт_чанк(self, chunker):
        chunk = chunker._make_chunk("text", "текст", 1, 0)
        assert isinstance(chunk, DocumentChunk)

    def test_обрезает_текст_до_3000(self, chunker):
        long_text = "а" * 5000
        chunk = chunker._make_chunk("text", long_text, 1, 0)
        assert len(chunk.text) <= 3000

    def test_нормализует_пробелы(self, chunker):
        chunk = chunker._make_chunk("text", "текст   с   пробелами", 1, 0)
        assert "  " not in chunk.text

    def test_context_query_для_известного_типа(self, chunker):
        chunk = chunker._make_chunk("section_header", "текст", 1, 0)
        assert chunk.context_query != ""
        assert "ГОСТ" in chunk.context_query

    def test_context_query_для_неизвестного_типа(self, chunker):
        chunk = chunker._make_chunk("unknown_type", "текст", 1, 0)
        assert chunk.context_query != ""

    def test_location_содержит_страницу(self, chunker):
        chunk = chunker._make_chunk("text", "текст", 5, 0)
        assert chunk.location["page"] == 5

    def test_chunk_id_форматирован(self, chunker):
        chunk = chunker._make_chunk("text", "текст", 1, 7)
        assert chunk.chunk_id == "0007"


# ── Тесты _make_text_chunk ───────────────────────────────────────────────────

class TestMakeTextChunk:
    def test_создаёт_text_чанк(self, chunker):
        chunk = chunker._make_text_chunk("текст", 1, 0)
        assert chunk.chunk_type == "text"

    def test_метаданные_содержат_word_count(self, chunker):
        chunk = chunker._make_text_chunk("один два три", 1, 0)
        assert "word_count" in chunk.metadata
        assert chunk.metadata["word_count"] == 3

    def test_определяет_ссылку_на_гост(self, chunker):
        chunk = chunker._make_text_chunk("согласно ГОСТ 2.105-95", 1, 0)
        assert chunk.metadata["has_gost_ref"] is True

    def test_нет_ссылки_на_гост(self, chunker):
        chunk = chunker._make_text_chunk("обычный текст без ссылок", 1, 0)
        assert chunk.metadata["has_gost_ref"] is False


# ── Тесты _extract_num ───────────────────────────────────────────────────────

class TestExtractNum:
    def test_извлекает_число(self, chunker):
        assert chunker._extract_num("Таблица 3") == "3"

    def test_нет_числа_возвращает_none(self, chunker):
        assert chunker._extract_num("без числа") is None


# ── Тесты _split_content_page ────────────────────────────────────────────────

class TestSplitContentPage:
    def test_текстовые_строки_объединяются(self, chunker):
        lines = ["строка первая", "строка вторая"]
        chunks = chunker._split_content_page(lines, 1, 0)
        assert any(c.chunk_type == "text" for c in chunks)

    def test_заголовок_раздела_выделяется(self, chunker):
        lines = ["1.1 Введение", "текст раздела"]
        chunks = chunker._split_content_page(lines, 1, 0)
        assert any(c.chunk_type == "section_header" for c in chunks)

    def test_рисунок_выделяется(self, chunker):
        lines = ["текст", "Рисунок 1 — схема"]
        chunks = chunker._split_content_page(lines, 1, 0)
        assert any(c.chunk_type == "figure_ref" for c in chunks)

    def test_таблица_выделяется(self, chunker):
        lines = ["текст", "Таблица 2 — данные"]
        chunks = chunker._split_content_page(lines, 1, 0)
        assert any(c.chunk_type == "table_ref" for c in chunks)

    def test_номер_страницы_пропускается(self, chunker):
        lines = ["42", "нормальный текст"]
        chunks = chunker._split_content_page(lines, 1, 0)
        for c in chunks:
            assert c.text != "42"

    def test_пустой_список_строк(self, chunker):
        chunks = chunker._split_content_page([], 1, 0)
        assert chunks == []


# ── Тесты chunk_pdf ──────────────────────────────────────────────────────────

class TestChunkPdf:
    def test_chunk_pdf_возвращает_список(self, chunker, tmp_path):
        """Тест с мок-pdf."""
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "1.1 Введение\nТекст раздела"

        with patch("src.chunker_pdfplumber.pdfplumber.open") as mock_open:
            mock_pdf = MagicMock()
            mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
            mock_pdf.__exit__ = MagicMock(return_value=False)
            mock_pdf.pages = [mock_page]
            mock_open.return_value = mock_pdf

            result = chunker.chunk_pdf("fake.pdf")
            assert isinstance(result, list)

    def test_пустая_страница_пропускается(self, chunker):
        mock_page = MagicMock()
        mock_page.extract_text.return_value = None

        with patch("src.chunker_pdfplumber.pdfplumber.open") as mock_open:
            mock_pdf = MagicMock()
            mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
            mock_pdf.__exit__ = MagicMock(return_value=False)
            mock_pdf.pages = [mock_page]
            mock_open.return_value = mock_pdf

            result = chunker.chunk_pdf("fake.pdf")
            assert result == []
