"""
Тесты для GOSTRuleSearcher.
Запуск: pytest tests/test_rule_searcher.py -v
"""
import pytest
import numpy as np
from unittest.mock import MagicMock
from src.rule_searcher import GOSTRuleSearcher


RULES = [
    {"id": 1, "rule_id": "4.1.2", "gost": "ГОСТ 2.105-95", "text": "Разделы нумеруются арабскими цифрами"},
    {"id": 2, "rule_id": "4.1.9", "gost": "ГОСТ 2.105-95", "text": "Заголовки с прописной буквы"},
    {"id": 3, "rule_id": "4.2.1", "gost": "ГОСТ 2.105-95", "text": "Таблицы имеют сквозную нумерацию"},
]


def make_mock_index(n=3, return_scores=None, return_indices=None):
    index = MagicMock()
    index.ntotal = n
    if return_scores is None:
        return_scores = np.array([[0.9, 0.8, 0.7]], dtype="float32")
    if return_indices is None:
        return_indices = np.array([[0, 1, 2]], dtype="int64")
    index.search.return_value = (return_scores, return_indices)
    return index


def make_mock_model():
    model = MagicMock()
    model.encode.return_value = np.ones((1, 768), dtype="float32")
    return model


@pytest.fixture
def searcher():
    return GOSTRuleSearcher(
        index=make_mock_index(),
        rules=RULES,
        model=make_mock_model(),
    )


# ── Тесты инициализации ──────────────────────────────────────────────────────

class TestInit:
    def test_правила_сохраняются(self, searcher):
        assert len(searcher.rules) == 3

    def test_индекс_сохраняется(self, searcher):
        assert searcher.index is not None


# ── Тесты search ─────────────────────────────────────────────────────────────

class TestSearch:
    def test_возвращает_список(self, searcher):
        result = searcher.search("текст", "text", top_k=3)
        assert isinstance(result, list)

    def test_не_больше_top_k(self, searcher):
        result = searcher.search("текст", "text", top_k=3)
        assert len(result) <= 3

    def test_каждый_результат_содержит_distance(self, searcher):
        result = searcher.search("текст", "text")
        for r in result:
            assert "distance" in r

    def test_каждый_результат_содержит_relevance_score(self, searcher):
        result = searcher.search("текст", "text")
        for r in result:
            assert "relevance_score" in r

    def test_каждый_результат_содержит_rank(self, searcher):
        result = searcher.search("текст", "text")
        for r in result:
            assert "rank" in r

    def test_пустой_индекс_возвращает_пустой_список(self):
        index = MagicMock()
        index.ntotal = 0
        searcher = GOSTRuleSearcher(index=index, rules=[], model=make_mock_model())
        result = searcher.search("текст", "text")
        assert result == []

    def test_faiss_возвращает_минус_1(self):
        index = make_mock_index(
            return_scores=np.array([[0.9, -1.0]], dtype="float32"),
            return_indices=np.array([[0, -1]], dtype="int64"),
        )
        searcher = GOSTRuleSearcher(index=index, rules=RULES, model=make_mock_model())
        result = searcher.search("текст", "text", top_k=5)
        for r in result:
            assert r.get("rank", 0) > 0

    def test_оригинальные_правила_не_изменяются(self, searcher):
        searcher.search("текст", "text")
        assert "distance" not in RULES[0]

    def test_вызывает_encode_модели(self, searcher):
        searcher.search("текст", "text")
        searcher.model.encode.assert_called_once()

    def test_ошибка_возвращает_пустой_список(self, searcher):
        searcher.model.encode.side_effect = Exception("ошибка")
        result = searcher.search("текст", "text")
        assert result == []


# ── Тесты search_by_ids ──────────────────────────────────────────────────────

class TestSearchByIds:
    def test_возвращает_правила_по_индексам(self, searcher):
        result = searcher.search_by_ids([0, 1])
        assert len(result) == 2
        assert result[0]["rule_id"] == "4.1.2"

    def test_несуществующий_индекс_пропускается(self, searcher):
        result = searcher.search_by_ids([0, 999])
        assert len(result) == 1

    def test_пустой_список_возвращает_пустой(self, searcher):
        result = searcher.search_by_ids([])
        assert result == []

    def test_оригинальные_правила_не_изменяются(self, searcher):
        result = searcher.search_by_ids([0])
        result[0]["extra"] = "test"
        assert "extra" not in RULES[0]
