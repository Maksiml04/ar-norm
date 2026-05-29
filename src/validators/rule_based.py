"""
Детерминированный валидатор ГОСТ 2.105-95.

Принципы:
- Каждый check_* метод отвечает ровно за одно правило.
- Метод получает чанк и возвращает Violation | None.
- Не использует LLM — только regex + логика Python.
- Уверенность (confidence) = 1.0 для всех rule-based результатов.
"""
from __future__ import annotations

import re
from typing import Optional

from src.schemas import (
    ChunkType,
    DocumentChunk,
    Severity,
    ValidationResult,
    ValidatorType,
    Violation,
)


# ─── Вспомогательные константы ────────────────────────────────────────────────

_DASH_VARIANTS = r"[—–-]"
_CORRECT_DASH   = r"[—–]"
_HYPHEN_ONLY    = r"-"


# ─── Валидатор ────────────────────────────────────────────────────────────────

class RuleBasedValidator:
    """
    Детерминированный движок проверок на Python.

    Публичный API:
        validate(chunk) -> ValidationResult
    """

    # В class RuleBasedValidator:
    def validate(self, chunk: DocumentChunk) -> ValidationResult:
        violations: list[Violation] = []
        checks = self._get_checks(chunk.chunk_type)

        for check in checks:
            result = check(chunk)
            if isinstance(result, Violation):
                violations.append(result)
            elif isinstance(result, list):
                violations.extend(result)  # ← Теперь ловим все нарушения в чанке

        return ValidationResult(
            chunk_id=chunk.chunk_id,
            validator=ValidatorType.RULE_BASED,
            has_violation=bool(violations),
            violations=violations,
            confidence=1.0 if violations else 0.0,
            location=chunk.location,
        )

    # ─── Реестр проверок по типу блока (ОБНОВЛЕН) ─────────────────────────────

    def _get_checks(self, chunk_type: ChunkType) -> list:
        mapping = {
            ChunkType.SECTION_HEADER:  [
                self._check_header_no_dot,
                self._check_header_numbering,
                self._check_header_case,
            ],
            ChunkType.APPENDIX_HEADER: [
                self._check_header_no_dot,
                self._check_appendix_letters,  # ← ДОБАВЛЕНО
            ],
            ChunkType.TABLE_TITLE: [
                self._check_table_prefix,
                self._check_table_dash,
            ],
            ChunkType.FIGURE_CAPTION: [
                self._check_figure_prefix,
                self._check_figure_dash,
            ],
            ChunkType.LIST_ITEM: [
                self._check_list_marker,
                self._check_list_item_case,     # ← ДОБАВЛЕНО
            ],
            ChunkType.NOTE: [
                self._check_note_format,
            ],
            ChunkType.FORMULA: [
                self._check_formula_where_colon, # ← ДОБАВЛЕНО
            ],
            ChunkType.TEXT_BODY: [
                self._check_numeric_intervals,   # ← ДОБАВЛЕНО
            ],
            ChunkType.TABLE_BLOCK: [],
        }
        return mapping.get(chunk_type, [])

    # ─── SECTION_HEADER & APPENDIX_HEADER ─────────────────────────────────────

    def _check_header_no_dot(self, chunk: DocumentChunk) -> Optional[Violation]:
        """ГОСТ 2.105-95 п.4.1.9 — заголовок не должен заканчиваться точкой."""
        text = chunk.text.rstrip()
        if text.endswith("."):
            return Violation(
                rule_id        = "GOST2.105-4.1.9_no_dot",
                violation_type = "Точка в конце заголовка",
                explanation    = f"Заголовок заканчивается точкой: «{text[-30:]}»",
                severity       = Severity.MINOR,
            )
        return None

    def _check_header_numbering(self, chunk: DocumentChunk) -> Optional[Violation]:
        """ГОСТ 2.105-95 п.4.1.8 — нумерация заголовка без точки в конце номера."""
        text = chunk.text.strip()
        if re.match(r"^\d", text):
            if not re.match(r"^\d+(?:\.\d+)*\s+\S", text):
                return Violation(
                    rule_id        = "GOST2.105-4.1.8_header_numbering",
                    violation_type = "Неверный формат нумерации заголовка",
                    explanation    = f"Заголовок «{text[:50]}» имеет неверный формат нумерации. Ожидается: «N Название».",
                    severity       = Severity.MINOR,
                )
            if re.match(r"^\d+(?:\.\d+)*\.\s", text):
                return Violation(
                    rule_id        = "GOST2.105-4.1.8_header_numbering",
                    violation_type = "Точка после номера заголовка",
                    explanation    = f"После номера заголовка стоит точка: «{text[:50]}». По п.4.1.8 точка там не ставится.",
                    severity       = Severity.MINOR,
                )
        return None

    def _check_header_case(self, chunk: DocumentChunk) -> Optional[Violation]:
        """ГОСТ 2.105-95 п.4.1.9 — первое слово заголовка с прописной буквы."""
        text = chunk.text.strip()
        without_number = re.sub(r"^\d+(?:\.\d+)*\s+", "", text).lstrip(' "«\'')
        if without_number and without_number[0].islower():
            return Violation(
                rule_id        = "GOST2.105-4.1.9_header_case",
                violation_type = "Заголовок начинается со строчной буквы",
                explanation    = f"Заголовок «{text[:50]}» должен начинаться с прописной буквы.",
                severity       = Severity.MINOR,
            )
        return None

    def _check_appendix_letters(self, chunk: DocumentChunk) -> Optional[Violation]:
        """ГОСТ 2.105-95 п. 4.3.8 — Обозначение приложений исключает буквы Ё, З, Й, О, Ч, Ь, Ы, Ъ."""
        text = chunk.text.strip()
        match = re.search(r"^[Пп]риложение\s+([А-ЯЁа-яё])", text)
        if match:
            letter = match.group(1).upper()
            if letter in "ЁЗЙОЧЬЫЪ":
                return Violation(
                    rule_id        = "GOST2.105-4.3.8_appendix_letter",
                    violation_type = "Недопустимая буква в обозначении приложения",
                    explanation    = f"Приложение обозначено буквой «{letter}». По ГОСТ не допускается использовать буквы Ё, З, Й, О, Ч, Ь, Ы, Ъ.",
                    severity       = Severity.MAJOR,
                )
        return None

    # ─── TABLE_TITLE ─────────────────────────────────────────────────────────

    def _check_table_prefix(self, chunk: DocumentChunk) -> Optional[Violation]:
        """ГОСТ 2.105-95 п.4.4.1 — «Таблица N — Название»."""
        text = chunk.text.strip()
        if not re.match(r"^[Тт]аблица\s+\d+(?:\.\d+)*(?:\s+[—–-].*)?$", text):
            return Violation(
                rule_id        = "GOST2.105-4.4.1_table_prefix",
                violation_type = "Неверный формат заголовка таблицы",
                explanation    = f"Заголовок таблицы «{text[:60]}» не соответствует формату «Таблица N — Название».",
                severity       = Severity.MAJOR,
            )
        return None

    def _check_table_dash(self, chunk: DocumentChunk) -> Optional[Violation]:
        """ГОСТ 2.105-95 п.4.4.1 — после номера таблицы должно стоять тире «—»."""
        text = chunk.text.strip()
        match = re.search(r"^[Тт]аблица\s+\d+(?:\.\d+)*\s*(-)\s*\S", text)
        if match:
            return Violation(
                rule_id        = "GOST2.105-4.4.1_table_dash",
                violation_type = "Дефис вместо тире в заголовке таблицы",
                explanation    = f"В заголовке «{text[:60]}» после номера стоит дефис «-». Должно быть тире «—».",
                severity       = Severity.MINOR,
            )
        return None

    # ─── FIGURE_CAPTION ──────────────────────────────────────────────────────

    def _check_figure_prefix(self, chunk: DocumentChunk) -> list[Violation]:
        violations = []
        # Ищем любые варианты: Рис., рис, Рисунок с номером
        suspicious_re = re.compile(r'[Рр]ис(?:\.|\унок)?\s+\d+', re.IGNORECASE)
        # Правильный ГОСТ-формат
        correct_re = re.compile(r'^[Рр]исунок\s+\d+(?:\.\d+)*(?:\s+[—–-].*)?$')

        for line in chunk.text.split('\n'):
            line = line.strip()
            if not line:
                continue
            if suspicious_re.search(line) and not correct_re.match(line):
                violations.append(Violation(
                    rule_id="GOST2.105-4.3.1_figure_prefix",
                    violation_type="Неверный формат подписи рисунка",
                    explanation=f"Строка «{line}» нарушает п.4.3.1. Требуется: «Рисунок N — Название».",
                    severity=Severity.MAJOR,
                ))
        return violations

    def _check_figure_dash(self, chunk: DocumentChunk) -> Optional[Violation]:
        """Дефис вместо тире в подписи рисунка."""
        text = chunk.text.strip()
        match = re.search(r"^[Рр]исунок\s+\d+(?:\.\d+)*\s*(-)\s*\S", text)
        if match:
            return Violation(
                rule_id        = "GOST2.105-4.3.1_figure_dash",
                violation_type = "Дефис вместо тире в подписи рисунка",
                explanation    = f"В подписи «{text[:60]}» после номера стоит дефис «-». Должно быть тире «—».",
                severity       = Severity.MINOR,
            )
        return None

    # ─── LIST_ITEM ────────────────────────────────────────────────────────────

    def _check_list_marker(self, chunk: DocumentChunk) -> Optional[Violation]:
        """ГОСТ 2.105-95 п.4.1.10 — маркер перечисления: дефис или строчная буква со скобкой."""
        text = chunk.text.strip()
        forbidden = re.match(r"^[•*►→▶●■□◆◇]", text)
        digit_dot = re.match(r"^\d+\.", text)
        if forbidden or digit_dot:
            return Violation(
                rule_id        = "GOST2.105-4.1.10_list_dash",
                violation_type = "Недопустимый маркер перечисления",
                explanation    = f"Элемент перечисления «{text[:50]}» использует недопустимый маркер. Допускается дефис «-» или «а)».",
                severity       = Severity.MINOR,
            )
        return None

    def _check_list_item_case(self, chunk: DocumentChunk) -> Optional[Violation]:
        """ГОСТ 2.105-95 п. 4.1.10 — Элемент списка после дефиса или буквенного маркера должен начинаться со строчной буквы."""
        text = chunk.text.strip()
        clean_text = re.sub(r"^[-–—]\s+|^[а-яa-z]\)\s+", "", text)
        if clean_text and clean_text[0].isupper() and not clean_text.split()[0].isupper():
            return Violation(
                rule_id        = "GOST2.105-4.1.10_list_item_case",
                violation_type = "Элемент списка начинается с заглавной буквы",
                explanation    = f"Текст пункта перечисления «{text[:30]}...» должен начинаться со строчной буквы.",
                severity       = Severity.MINOR,
            )
        return None

    # ─── NOTE ─────────────────────────────────────────────────────────────────

    def _check_note_format(self, chunk: DocumentChunk) -> Optional[Violation]:
        """ГОСТ 2.105-95 п.4.5.1 — «Примечание — текст» или «Примечания: 1. текст»."""
        text = chunk.text.strip()
        single = re.match(r"^Примечание\s+[—–-]\s+\S", text)
        multi  = re.match(r"^Примечания:\s*$|^Примечания:\s+\d+\.", text)
        starts_correct = single or multi

        if text.lower().startswith("примечани") and not starts_correct:
            return Violation(
                rule_id        = "GOST2.105-4.5.1_note_format",
                violation_type = "Неверный формат примечания",
                explanation    = f"Примечание «{text[:60]}» имеет неверный формат. Ожидается: «Примечание — текст».",
                severity       = Severity.MINOR,
            )
        return None

    # ─── FORMULA ──────────────────────────────────────────────────────────────

    def _check_formula_where_colon(self, chunk: DocumentChunk) -> Optional[Violation]:
        """ГОСТ 2.105-95 п. 4.2.1 — В пояснении символов после слова «где» двоеточие не ставится."""
        text = chunk.text.strip()
        if text.startswith("где:"):
            return Violation(
                rule_id        = "GOST2.105-4.2.1_formula_where_colon",
                violation_type = "Двоеточие после слова «где» в формуле",
                explanation    = "По ГОСТ 2.105-95 после слова «где», начинающего расшифровку символов формулы, двоеточие ставить запрещено.",
                severity       = Severity.MINOR,
            )
        return None

    # ─── TEXT_BODY ────────────────────────────────────────────────────────────

    def _check_numeric_intervals(self, chunk: DocumentChunk) -> Optional[Violation]:
        """ГОСТ 2.105-95 п. 4.2.13 — Числовые интервалы физ. величин нельзя оформлять через дефис или тире."""
        text = chunk.text.strip()
        match = re.search(r"\b\d+[-—–]\d+\s+(?:мм|см|кг|г|%|°C|шт|л|м)\b", text)
        if match:
            return Violation(
                rule_id        = "GOST2.105-4.2.13_numeric_interval",
                violation_type = "Неверное оформление интервала величин",
                explanation    = f"Интервал величин «{match.group(0)}» оформлен некорректно. По стандартам следует писать «от ... до ...».",
                severity       = Severity.MINOR,
            )
        return None