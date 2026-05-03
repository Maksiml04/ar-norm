"""
Тесты для GOSTRetriever.

Запуск:
    pytest tests/test_retriever.py -v
"""

import pickle
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.retriever import GOSTRetriever, DEFAULT_CONTEXT_QUERIES, TYPE_MATCH_BONUS


# ── Фикстуры ────────────────────────────────────────────────────────────────

RULES = [
    {
        "id": 31,
        "rule_id": "4.1.2",
        "gost": "ГОСТ 2.105-95",
        "section": "4.1.2",
        "text": "Разделы должны иметь порядковые номера...",
        "chunk_type": "numbering",
        "category": "structure",
        "applies_to": ["section_header", "text"],
    },
    {
        "id": 34,
        "rule_id": "4.1.9",
        "gost": "ГОСТ 2.105-95",
        "section": "4.1.9",
        "text": "Заголовки разделов следует печатать с прописной буквы...",
        "chunk_type": "header",
        "category": "formatting",
        "applies_to": ["section_header"],
    },
    {
        "id": 36,
        "rule_id": "4.2.1",
        "gost": "ГОСТ 2.105-95",
        "section": "4.2.1",
        "text": "Таблицы должны иметь сквозную нумерацию...",
        "chunk_type": "table",
        "category": "tables",
        "applies_to": ["table_ref"],
    },
]


def make_mock_index(n_rules=3, dim=768):
    """Создаёт мок FAISS индекса."""
    index = MagicMock()
    index.ntotal = n_rules
    # Возвращает реальные scores и indices при вызове search
    scores = np.array([[0.9, 0.8, 0.7]], dtype="float32")
    indices = np.array([[0, 1, 2]], dtype="int64")
    index.search.return_value = (scores, indices)
    return index


def make_mock_model():
    """Создаёт мок SentenceTransformer."""
    model = MagicMock()
    # Возвращает нормализованные векторы нужной формы
    model.encode.return_value = np.ones((1, 768), dtype="float32")
    return model


@pytest.fixture
def retriever():
    """Базовый retriever с моками."""
    return GOSTRetriever(
        index=make_mock_index(),
        rules=RULES,
        model=make_mock_model(),
    )


@pytest.fixture
def retriever_no_applies_to():
    """Retriever с правилами без поля applies_to."""
    rules = [{**r} for r in RULES]
    for r in rules:
        r.pop("applies_to", None)
    return GOSTRetriever(
        index=make_mock_index(),
        rules=rules,
        model=make_mock_model(),
    )


# ── Тесты инициализации ──────────────────────────────────────────────────────

class TestInit:
    def test_правила_сохраняются(self, retriever):
        assert len(retriever.rules) == 3

    def test_индекс_сохраняется(self, retriever):
        assert retriever.index is not None

    def test_модель_сохраняется(self, retriever):
        assert retriever.model is not None


# ── Тесты _build_query ───────────────────────────────────────────────────────

class TestBuildQuery:
    def test_использует_дефолтный_контекст(self, retriever):
        query = retriever._build_query("какой-то текст", "text")
        assert DEFAULT_CONTEXT_QUERIES["text"] in query
        assert "какой-то текст" in query

    def test_использует_переданный_context_query(self, retriever):
        query = retriever._build_query("текст", "text", context_query="мой контекст")
        assert "мой контекст" in query

    def test_неизвестный_chunk_type_даёт_дефолт(self, retriever):
        query = retriever._build_query("текст", "unknown_type")
        assert "ГОСТ требования" in query

    def test_обрезает_текст_до_300_символов(self, retriever):
        long_text = "а" * 500
        query = retriever._build_query(long_text, "text")
        # Сниппет не должен превышать 300 символов
        snippet = query.split(" | ", 1)[1]
        assert len(snippet) <= 300

    def test_формат_запроса_с_разделителем(self, retriever):
        query = retriever._build_query("текст документа", "table_ref")
        assert " | " in query


# ── Тесты _soft_type_score ───────────────────────────────────────────────────

class TestSoftTypeScore:
    def test_совпадение_типа_даёт_бонус(self):
        rule = {"applies_to": ["section_header", "text"]}
        bonus = GOSTRetriever._soft_type_score(rule, "section_header")
        assert bonus == TYPE_MATCH_BONUS

    def test_несовпадение_типа_даёт_ноль(self):
        rule = {"applies_to": ["table_ref"]}
        bonus = GOSTRetriever._soft_type_score(rule, "text")
        assert bonus == 0.0

    def test_applies_to_строка_а_не_список(self):
        rule = {"applies_to": "section_header"}
        bonus = GOSTRetriever._soft_type_score(rule, "section_header")
        assert bonus == TYPE_MATCH_BONUS

    def test_нет_applies_to_даёт_ноль(self):
        rule = {}
        bonus = GOSTRetriever._soft_type_score(rule, "text")
        assert bonus == 0.0

    def test_пустой_applies_to_даёт_ноль(self):
        rule = {"applies_to": []}
        bonus = GOSTRetriever._soft_type_score(rule, "text")
        assert bonus == 0.0


# ── Тесты search ─────────────────────────────────────────────────────────────

class TestSearch:
    def test_возвращает_список(self, retriever):
        result = retriever.search("текст документа", chunk_type="text")
        assert isinstance(result, list)

    def test_возвращает_не_больше_top_k(self, retriever):
        result = retriever.search("текст", top_k=2)
        assert len(result) <= 2

    def test_каждый_результат_содержит_score(self, retriever):
        result = retriever.search("текст")
        for r in result:
            assert "_score" in r

    def test_каждый_результат_содержит_type_bonus(self, retriever):
        result = retriever.search("текст")
        for r in result:
            assert "_type_bonus" in r

    def test_результаты_отсортированы_по_убыванию_score(self, retriever):
        result = retriever.search("текст", top_k=3)
        scores = [r["_score"] for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_пустой_retriever_возвращает_пустой_список(self):
        empty = GOSTRetriever(
            index=MagicMock(ntotal=0),
            rules=[],
            model=make_mock_model(),
        )
        result = empty.search("текст")
        assert result == []

    def test_search_вызывает_encode_модели(self, retriever):
        retriever.search("какой-то текст", chunk_type="text")
        retriever.model.encode.assert_called_once()

    def test_search_вызывает_faiss_index_search(self, retriever):
        retriever.search("текст")
        retriever.index.search.assert_called_once()

    def test_правила_обогащаются_полями_из_rules(self, retriever):
        result = retriever.search("текст")
        # Первый результат должен содержать поля из оригинального правила
        assert "rule_id" in result[0]
        assert "gost" in result[0]

    def test_бонус_применяется_при_совпадении_типа(self, retriever):
        # section_header есть в applies_to первого правила
        result = retriever.search("текст раздела", chunk_type="section_header")
        # Хотя бы один результат должен иметь ненулевой бонус
        bonuses = [r["_type_bonus"] for r in result]
        assert any(b > 0 for b in bonuses)


# ── Тесты search_batch ───────────────────────────────────────────────────────

class TestSearchBatch:
    def test_пустой_список_возвращает_пустой_список(self, retriever):
        result = retriever.search_batch([])
        assert result == []

    def test_возвращает_результат_для_каждого_чанка(self, retriever):
        # Настраиваем мок для батч-поиска на 2 чанка
        retriever.model.encode.return_value = np.ones((2, 768), dtype="float32")
        retriever.index.search.return_value = (
            np.array([[0.9, 0.8, 0.7], [0.85, 0.75, 0.65]], dtype="float32"),
            np.array([[0, 1, 2], [0, 1, 2]], dtype="int64"),
        )
        chunks = [
            {"text": "текст 1", "chunk_type": "text"},
            {"text": "текст 2", "chunk_type": "section_header"},
        ]
        result = retriever.search_batch(chunks)
        assert len(result) == 2

    def test_каждый_результат_содержит_score(self, retriever):
        retriever.model.encode.return_value = np.ones((1, 768), dtype="float32")
        chunks = [{"text": "текст", "chunk_type": "text"}]
        result = retriever.search_batch(chunks, top_k=2)
        for r in result[0]:
            assert "_score" in r

    def test_результаты_не_превышают_top_k(self, retriever):
        retriever.model.encode.return_value = np.ones((1, 768), dtype="float32")
        chunks = [{"text": "текст", "chunk_type": "text"}]
        result = retriever.search_batch(chunks, top_k=1)
        assert len(result[0]) <= 1

    def test_чанк_без_полей_не_падает(self, retriever):
        retriever.model.encode.return_value = np.ones((1, 768), dtype="float32")
        chunks = [{}]  # пустой чанк
        result = retriever.search_batch(chunks)
        assert len(result) == 1


# ── Тесты load ───────────────────────────────────────────────────────────────

class TestLoad:
    def test_load_новый_формат_meta(self, tmp_path):
        """Загрузка meta нового формата (dict)."""
        import faiss

        dim = 64
        index = faiss.IndexFlatIP(dim)
        index_path = str(tmp_path / "test.index")
        faiss.write_index(index, index_path)

        meta = {"rules": RULES, "model_name": "cointegrated/LaBSE-en-ru"}
        meta_path = str(tmp_path / "test.pkl")
        with open(meta_path, "wb") as f:
            pickle.dump(meta, f)

        with patch("src.retriever.SentenceTransformer") as mock_st:
            mock_st.return_value = make_mock_model()
            r = GOSTRetriever.load(index_path, meta_path)
            assert len(r.rules) == len(RULES)

    def test_load_старый_формат_meta(self, tmp_path):
        """Загрузка meta старого формата (list)."""
        import faiss

        dim = 64
        index = faiss.IndexFlatIP(dim)
        index_path = str(tmp_path / "test.index")
        faiss.write_index(index, index_path)

        meta_path = str(tmp_path / "test.pkl")
        with open(meta_path, "wb") as f:
            pickle.dump(RULES, f)  # старый формат — просто список

        with patch("src.retriever.SentenceTransformer") as mock_st:
            mock_st.return_value = make_mock_model()
            r = GOSTRetriever.load(index_path, meta_path)
            assert len(r.rules) == len(RULES)
