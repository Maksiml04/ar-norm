"""

LLM Analyzer v3.0

Ключевые изменения:

  - analyze_chunk теперь принимает полный словарь чанка (включая metadata).

  - Метаданные парсера (is_centered, known_abbreviations, section_number)

    явно передаются в промпт — LLM знает факты, а не гадает.

  - TextCleaner упрощён: новый парсер (extract_words) уже даёт чистый текст.

  - Фильтрация правил перенесена в промпт-инструкцию, а не в Python-логику

    (меньше хрупких эвристик, больше ответственности у LLM).

"""

from __future__ import annotations



import json

import os

import re

from typing import Any



from openai import OpenAI



try:

    from src.logging_config import get_logger

except ImportError:

    from logging_config import get_logger



logger = get_logger(__name__)





# ─── Вспомогательные функции ──────────────────────────────────────────────────



def _extract_json(raw: str) -> dict[str, Any]:

    """Надёжно извлекает JSON из ответа LLM."""

    # Убираем markdown-фенсы

    raw = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()



    # Прямой парсинг

    try:

        return json.loads(raw)

    except json.JSONDecodeError:

        pass



    # Ищем первый {...} блок

    m = re.search(r"\{[\s\S]*\}", raw)

    if m:

        try:

            return json.loads(m.group())

        except json.JSONDecodeError:

            pass



    # Последняя попытка: чистим trailing comma и одиночные кавычки

    raw = re.sub(r",\s*([\]}])", r"\1", raw).replace("'", '"')

    try:

        return json.loads(raw)

    except json.JSONDecodeError as e:

        logger.warning(f"Не удалось распарсить JSON: {e} | raw={raw[:200]}")

        return {

            "has_violation": False,

            "violations": [],

            "is_correct": True,

            "confidence": 0.0,

            "error": f"JSON parse error: {e}",

        }





# ─── Главный класс ────────────────────────────────────────────────────────────



class LLMAnalyzer:

    """

    Анализирует чанк документа на нарушения ГОСТ с помощью LLM.



    Ожидаемый формат входного чанка (dict):

        {

            "id": "p5_c3",

            "text": "...",

            "chunk_type": "section_header",

            "location": {"page": 5},

            "metadata": {

                "is_centered": True,          # из нового парсера

                "is_bold": False,

                "font_size": 14.0,

                "section_number": "3.1",

                "title": "Объект испытаний",

                "known_abbreviations": {       # перечень сокращений документа

                    "ЛИАБ": "литий-ионная аккумуляторная батарея",

                    "НКУ":  "нормальные климатические условия",

                }

            },

            "context_query": "..."

        }

    """



    def __init__(

        self,

        api_key: str | None = None,

        base_url: str = "https://openrouter.ai/api/v1",

        model: str = "deepseek/deepseek-v3.2",

    ) -> None:

        raw_key = (api_key or os.getenv("OPENROUTER_API_KEY", "")).strip()

        if not raw_key:

            raise ValueError("OPENROUTER_API_KEY не задан")



        self.client = OpenAI(api_key=raw_key, base_url=base_url)

        self.model = model

        logger.info(f"LLMAnalyzer инициализирован (model={model})")



    # ── Основной метод ────────────────────────────────────────────────────────



    def analyze_chunk(

        self,

        chunk: dict[str, Any],

        rules: list[dict[str, Any]],

    ) -> dict[str, Any]:

        """

        Анализирует чанк документа.



        Args:

            chunk:  Полный словарь чанка из PDFChunker.to_dict() или AINormkontroler.

            rules:  Список правил из GOSTRetriever.search() (уже с _score).



        Returns:

            Словарь с has_violation, violations, confidence, chunk_id, location.

        """

        chunk_text = chunk.get("text", "").strip()

        chunk_type = chunk.get("chunk_type", "text")

        metadata   = chunk.get("metadata", {})

        chunk_id   = chunk.get("id", "unknown")

        location   = chunk.get("location", {})



        if not chunk_text:

            return self._empty_result(chunk_id, location, comment="Пустой текст")



        if not rules:

            return self._empty_result(chunk_id, location, comment="Нет применимых правил")



        # ── Строим промпт ─────────────────────────────────────────────────────

        system_msg, user_msg = self._build_prompt(chunk_text, chunk_type, metadata, rules)



        logger.debug(f"[{chunk_id}] Запрос к LLM (тип={chunk_type}, правил={len(rules)})")



        try:

            response = self.client.chat.completions.create(

                model=self.model,

                messages=[

                    {"role": "system", "content": system_msg},

                    {"role": "user",   "content": user_msg},

                ],

                temperature=0.1,

                max_tokens=1500,

            )

            raw = response.choices[0].message.content.strip()

            logger.debug(f"[{chunk_id}] LLM raw: {raw[:200]}")



        except Exception as e:

            logger.error(f"[{chunk_id}] Ошибка вызова LLM: {e}", exc_info=True)

            return self._empty_result(chunk_id, location, error=str(e))



        result = _extract_json(raw)

        result.update({

            "chunk_id":            chunk_id,

            "location":            location,

            "chunk_type":          chunk_type,

            "applied_rules_count": len(rules),

        })



        logger.info(

            f"[{chunk_id}] has_violation={result.get('has_violation')} "

            f"violations={len(result.get('violations', []))}"

        )

        return result



    # ── Построение промпта ────────────────────────────────────────────────────



    def _build_prompt(

        self,

        text: str,

        chunk_type: str,

        metadata: dict[str, Any],

        rules: list[dict[str, Any]],

    ) -> tuple[str, str]:

        """

        Формирует system + user сообщения.



        Ключевая идея: всё что знает парсер — явно сообщается LLM.

        LLM не должен гадать о центрировании или расшифровках сокращений.

        """



        # ── Факты о чанке из метаданных парсера ──────────────────────────────

        facts: list[str] = []



        is_centered = metadata.get("is_centered")

        if is_centered is not None:

            facts.append(

                f"• Выравнивание: {'по центру ✓' if is_centered else 'НЕ по центру'}"

            )



        is_bold = metadata.get("is_bold")

        if is_bold is not None:

            facts.append(f"• Шрифт: {'жирный ✓' if is_bold else 'обычный (не жирный)'}")



        font_size = metadata.get("font_size")

        if font_size:

            facts.append(f"• Размер шрифта: {font_size:.1f}pt")



        sec_num = metadata.get("section_number")

        if sec_num:

            facts.append(f"• Номер раздела: {sec_num}")



        # ── Перечень сокращений документа ─────────────────────────────────────

        known_abbr: dict[str, str] = metadata.get("known_abbreviations", {})

        abbr_block = ""

        if known_abbr:

            abbr_lines = "\n".join(

                f"  {abbr} = {definition}"

                for abbr, definition in list(known_abbr.items())[:30]

            )

            abbr_block = (

                "\n\nСОКРАЩЕНИЯ, РАСШИФРОВАННЫЕ В ДОКУМЕНТЕ (не считать нарушением):\n"

                + abbr_lines

            )



        # ── Инструкция по типу чанка ──────────────────────────────────────────

        type_instructions = {

            "section_header": (

                "Это ЗАГОЛОВОК раздела/подраздела. Проверяй правила оформления заголовков: "

                "нумерацию, точку в конце, переносы, написание с заглавной буквы. "

                "НЕ проверяй правила для основного текста (абзацы, отступы)."

            ),

            "text": (

                "Это основной ТЕКСТ. Проверяй: опечатки, неправильные сокращения, "

                "знаки препинания, перечисления, ссылки на рисунки/таблицы. "

                "НЕ применяй правила оформления заголовков."

            ),

            "figure_ref": (

                "Это ПОДПИСЬ К РИСУНКУ. Проверяй: наличие слова «Рисунок», "

                "нумерацию, тире перед названием, точку в конце."

            ),

            "table_ref": (

                "Это ЗАГОЛОВОК ТАБЛИЦЫ. Проверяй: наличие слова «Таблица», "

                "нумерацию, тире перед названием."

            ),

            "toc": (

                "Это СОДЕРЖАНИЕ документа. Проверяй: правильность нумерации разделов, "

                "наличие точечного заполнителя, нумерацию страниц."

            ),

            "title_page": (

                "Это ТИТУЛЬНЫЙ ЛИСТ. Проверяй: наличие обязательных реквизитов, "

                "центрирование ключевых элементов."

            ),

            "bibliography": (

                "Это СПИСОК ЛИТЕРАТУРЫ. Проверяй: оформление источников по ГОСТ."

            ),

        }

        type_hint = type_instructions.get(

            chunk_type,

            f"Тип блока: {chunk_type}. Применяй правила по смыслу."

        )



        # ── Список правил ─────────────────────────────────────────────────────

        rules_text = "\n".join(

            f"[{r.get('gost_id', '?')}] {r.get('text', '')}"

            for r in rules

        )



        # ── facts block ───────────────────────────────────────────────────────

        facts_block = ""

        if facts:

            facts_block = "\n\nИЗВЕСТНЫЕ ФАКТЫ О БЛОКЕ (установлены парсером, НЕ проверять повторно):\n"

            facts_block += "\n".join(facts)



        # ── Сборка ────────────────────────────────────────────────────────────

        system_msg = (

            "Ты эксперт-нормоконтролёр технической документации ГОСТ. "

            "Проверяй ТОЛЬКО нарушения из предоставленного списка правил. "

            "Не выдумывай правила. Отвечай строго JSON без пояснений."

        )



        user_msg = f"""ТИП БЛОКА: {chunk_type}

ИНСТРУКЦИЯ: {type_hint}{facts_block}{abbr_block}



ПРИМЕНЯЕМЫЕ ПРАВИЛА ГОСТ:

{rules_text}



ТЕКСТ ДЛЯ ПРОВЕРКИ:

\"\"\"

{text}

\"\"\"



Верни JSON:

{{

    "has_violation": true/false,

    "violations": [

        {{

            "rule_id": "ID из списка правил",

            "violation_type": "краткое название",

            "explanation": "что именно нарушено и где в тексте",

            "severity": "critical|major|minor"

        }}

    ],

    "is_correct": true/false,

    "confidence": 0.0-1.0

}}"""



        return system_msg, user_msg



    # ── Вспомогательные методы ────────────────────────────────────────────────



    def _empty_result(

        self,

        chunk_id: str,

        location: dict,

        comment: str = "",

        error: str = "",

    ) -> dict[str, Any]:

        result: dict[str, Any] = {

            "chunk_id":      chunk_id,

            "location":      location,

            "has_violation": False,

            "violations":    [],

            "is_correct":    True,

            "confidence":    1.0,

        }

        if comment:

            result["comment"] = comment

        if error:

            result["error"] = error

            result["confidence"] = 0.0

        return result