"""
Тесты для LLMAnalyzer.
Запуск: pytest tests/test_llm_analyzer.py -v
"""
import pytest
from unittest.mock import MagicMock, patch
from src.llm_analyzer import LLMAnalyzer, _extract_json


RULES = [
    {"gost_id": "4.1.9", "text": "Заголовки с прописной буквы без точки в конце"},
    {"gost_id": "4.1.2", "text": "Разделы нумеруются арабскими цифрами"},
]

CHUNK = {
    "id": "p1_c1",
    "text": "1.1 введение.",
    "chunk_type": "section_header",
    "location": {"page": 1},
    "metadata": {
        "is_centered": False,
        "is_bold": False,
        "font_size": 14.0,
        "section_number": "1.1",
        "known_abbreviations": {"ЛИАБ": "литий-ионная батарея"},
    },
}

LLM_RESPONSE = '{"has_violation": true, "violations": [{"rule_id": "4.1.9", "violation_type": "точка в конце", "explanation": "есть точка", "severity": "minor"}], "is_correct": false, "confidence": 0.9}'


@pytest.fixture
def analyzer():
    with patch("src.llm_analyzer.OpenAI"):
        return LLMAnalyzer(api_key="test-key")


# ── Тесты _extract_json ──────────────────────────────────────────────────────

class TestExtractJson:
    def test_валидный_json(self):
        raw = '{"has_violation": false, "violations": []}'
        result = _extract_json(raw)
        assert result["has_violation"] is False

    def test_json_с_markdown_фенсами(self):
        raw = '```json\n{"has_violation": true}\n```'
        result = _extract_json(raw)
        assert result["has_violation"] is True

    def test_json_внутри_текста(self):
        raw = 'Вот результат: {"has_violation": false, "violations": []} конец'
        result = _extract_json(raw)
        assert "has_violation" in result

    def test_невалидный_json_возвращает_дефолт(self):
        result = _extract_json("это не json вообще")
        assert result["has_violation"] is False
        assert "error" in result

    def test_пустая_строка_возвращает_дефолт(self):
        result = _extract_json("")
        assert result["has_violation"] is False


# ── Тесты инициализации ──────────────────────────────────────────────────────

class TestInit:
    def test_без_ключа_падает(self):
        with patch("src.llm_analyzer.OpenAI"), \
             patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
                LLMAnalyzer(api_key="")

    def test_с_ключом_создаётся(self):
        with patch("src.llm_analyzer.OpenAI"):
            analyzer = LLMAnalyzer(api_key="test-key")
            assert analyzer.model == "deepseek/deepseek-v3.2"

    def test_кастомная_модель(self):
        with patch("src.llm_analyzer.OpenAI"):
            analyzer = LLMAnalyzer(api_key="test-key", model="my-model")
            assert analyzer.model == "my-model"


# ── Тесты analyze_chunk ──────────────────────────────────────────────────────

class TestAnalyzeChunk:
    def _mock_llm_response(self, analyzer, content):
        mock_response = MagicMock()
        mock_response.choices[0].message.content = content
        analyzer.client.chat.completions.create.return_value = mock_response

    def test_пустой_текст_возвращает_пустой_результат(self, analyzer):
        chunk = {**CHUNK, "text": ""}
        result = analyzer.analyze_chunk(chunk, RULES)
        assert result["has_violation"] is False
        assert result.get("comment") == "Пустой текст"

    def test_без_правил_возвращает_пустой_результат(self, analyzer):
        result = analyzer.analyze_chunk(CHUNK, [])
        assert result["has_violation"] is False
        assert result.get("comment") == "Нет применимых правил"

    def test_возвращает_chunk_id(self, analyzer):
        self._mock_llm_response(analyzer, LLM_RESPONSE)
        result = analyzer.analyze_chunk(CHUNK, RULES)
        assert result["chunk_id"] == "p1_c1"

    def test_возвращает_location(self, analyzer):
        self._mock_llm_response(analyzer, LLM_RESPONSE)
        result = analyzer.analyze_chunk(CHUNK, RULES)
        assert result["location"] == {"page": 1}

    def test_возвращает_has_violation(self, analyzer):
        self._mock_llm_response(analyzer, LLM_RESPONSE)
        result = analyzer.analyze_chunk(CHUNK, RULES)
        assert "has_violation" in result

    def test_ошибка_llm_возвращает_результат_без_падения(self, analyzer):
        analyzer.client.chat.completions.create.side_effect = Exception("API error")
        result = analyzer.analyze_chunk(CHUNK, RULES)
        assert "error" in result
        assert result["has_violation"] is False

    def test_applied_rules_count_в_результате(self, analyzer):
        self._mock_llm_response(analyzer, LLM_RESPONSE)
        result = analyzer.analyze_chunk(CHUNK, RULES)
        assert result["applied_rules_count"] == len(RULES)


# ── Тесты _build_prompt ──────────────────────────────────────────────────────

class TestBuildPrompt:
    def test_возвращает_кортеж_из_двух_строк(self, analyzer):
        system, user = analyzer._build_prompt("текст", "text", {}, RULES)
        assert isinstance(system, str)
        assert isinstance(user, str)

    def test_system_содержит_гост(self, analyzer):
        system, _ = analyzer._build_prompt("текст", "text", {}, RULES)
        assert "ГОСТ" in system

    def test_user_содержит_текст(self, analyzer):
        _, user = analyzer._build_prompt("проверяемый текст", "text", {}, RULES)
        assert "проверяемый текст" in user

    def test_метаданные_is_centered_в_промпте(self, analyzer):
        meta = {"is_centered": True}
        _, user = analyzer._build_prompt("текст", "section_header", meta, RULES)
        assert "центру" in user

    def test_сокращения_в_промпте(self, analyzer):
        meta = {"known_abbreviations": {"ЛИАБ": "литий-ионная батарея"}}
        _, user = analyzer._build_prompt("текст", "text", meta, RULES)
        assert "ЛИАБ" in user

    def test_инструкция_по_типу_section_header(self, analyzer):
        _, user = analyzer._build_prompt("текст", "section_header", {}, RULES)
        assert "ЗАГОЛОВОК" in user

    def test_инструкция_по_типу_table_ref(self, analyzer):
        _, user = analyzer._build_prompt("текст", "table_ref", {}, RULES)
        assert "ТАБЛИЦА" in user or "Таблица" in user
