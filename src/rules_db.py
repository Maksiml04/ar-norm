# src/rules_db.py
from src.schemas import ChunkType

RULES_DB = {
    # ─── ФОРМАТИРОВАНИЕ (Rule-Based) ────────────────────────────────────────────
    "GOST2.105-4.1.9": {
        "id": "GOST2.105-4.1.9",
        "description": "Заголовки разделов и подразделов не должны заканчиваться точкой.",
        "chunk_types": [ChunkType.SECTION_HEADER, ChunkType.APPENDIX_HEADER],
        "validator_type": "rule_based",
        "category": "formatting"
    },
    "GOST2.105-4.4.1": {
        "id": "GOST2.105-4.4.1",
        "description": "Заголовок таблицы: 'Таблица N — Название'. Без точки, с тире.",
        "chunk_types": [ChunkType.TABLE_TITLE, ChunkType.TABLE_BLOCK],
        "validator_type": "rule_based",
        "category": "formatting"
    },
    "GOST2.105-4.3.1": {
        "id": "GOST2.105-4.3.1",
        "description": "Подпись рисунка: 'Рисунок N — Название'. Не 'Рис.', не 'рис.'.",
        "chunk_types": [ChunkType.FIGURE_CAPTION],
        "validator_type": "rule_based",
        "category": "formatting"
    },
    "GOST2.105-4.1.7": {
        "id": "GOST2.105-4.1.7",
        "description": "Элементы перечисления начинаются со строчной буквы. Нумерация сквозная.",
        "chunk_types": [ChunkType.LIST_ITEM],
        "validator_type": "rule_based",
        "category": "formatting"
    },

    # ─── СТИЛЬ И СЕМАНТИКА (LLM) ────────────────────────────────────────────────
    "GOST2.105-4.2.1": {
        "id": "GOST2.105-4.2.1",
        "description": "Текст должен быть кратким, четким и не допускать различных толкований.",
        "chunk_types": [ChunkType.TEXT_BODY],
        "validator_type": "llm",
        "category": "style"
    },
    "GOST2.105-4.2.3": {
        "id": "GOST2.105-4.2.3",
        "description": "Не допускается применять обороты разговорной речи, техницизмы и канцеляризмы.",
        "chunk_types": [ChunkType.TEXT_BODY],
        "validator_type": "llm",
        "category": "style"
    },
    "GOST2.105-4.2.5": {
        "id": "GOST2.105-4.2.5",
        "description": "Использование личных местоимений (я, мы, мой, наш) недопустимо в научной работе.",
        "chunk_types": [ChunkType.TEXT_BODY],
        "validator_type": "llm",
        "category": "style"
    },
    "GOST_REF_STYLE": {
        "id": "GOST_REF_STYLE",
        "description": "Ссылки на источники должны соответствовать ГОСТ Р 7.0.100. Формат: [1], [2-5].",
        "chunk_types": [ChunkType.TEXT_BODY, ChunkType.NOTE],
        "validator_type": "llm",
        "category": "citation"
    },
    "LLM_GENERAL_GOST_CHECK": {
        "id": "LLM_GENERAL_GOST_CHECK",
        "description": """Проверь текст на соответствие общим принципам ГОСТ 2.105-95:
- Единообразие терминов (не смешивать 'чертёж' / 'схема' / 'эскиз' без необходимости)
- Корректность ссылок на рисунки/таблицы ('см. рисунок 5', а не 'см. рис.5')
- Отсутствие разговорных формулировок в технической документации
- Правильное оформление числовых диапазонов ('от 10 до 20 мм', а не '10-20 мм')
Если найдёшь нарушение — укажи конкретный пункт ГОСТ или напиши 'общее требование'.""",
        "chunk_types": [ChunkType.TEXT_BODY, ChunkType.SECTION_HEADER],
        "validator_type": "llm",
        "category": "general",
        "priority": "low"
    }
}

# 🔍 Вспомогательная функция фильтрации (вместо устаревшего BLOCK_RULE_REGISTRY)
def get_rules_for_chunk(chunk_type: ChunkType, validator_type: str = "all") -> list[dict]:
    """Возвращает правила для типа блока. Если validator_type='llm' или 'rule_based' — фильтрует."""
    rules = RULES_DB.values()
    if validator_type != "all":
        rules = [r for r in rules if r["validator_type"] == validator_type]
    return [r for r in rules if chunk_type in r["chunk_types"]]