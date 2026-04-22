"""
Модуль для анализа чанков с помощью LLM (DeepSeek через OpenRouter).
Версия 2.0:
- Очистка текста от артефактов PDF (склеивание разорванных слов).
- Жесткая фильтрация правил по типу чанка.
- Улучшенный промпт с инструкциями игнорирования нерелевантных правил.
- Детальное логирование процесса.
"""
import os
import re
import json
from typing import List, Dict, Any, Optional
from openai import OpenAI

try:
    from src.logging_config import get_logger
except ImportError:
    from logging_config import get_logger

logger = get_logger(__name__)


class TextCleaner:
    """Утилиты для предобработки текста, извлеченного из PDF."""

    @staticmethod
    def fix_broken_words(text: str) -> str:
        """
        Исправляет слова, разорванные пробелом внутри слога (артефакт PDF).
        ВАЖНО: Не склеивать всё подряд, чтобы не потерять структуру текста.

        Эвристика: Склеиваем, если:
        1. Первая часть заканчивается на согласную, вторая начинается с гласной (или наоборот),
           И длина одной из частей очень мала (1-2 буквы), что характерно для обрыва.
        2. ИЛИ это явные случаи из логов: "бла годаря", "являе тся".

        Мы НЕ будем склеивать любые соседние слова, чтобы сохранить возможность
        поиска лишних пробелов и опечаток.
        """

        # Список частых ложных разрывов для принудительного исправления
        common_fixes = {
            "бла годаря": "благодаря",
            "являе тся": "является",
            "со противляемость": "сопротивляемость",
            "лазер ного": "лазерного",
            "из весна я": "известная",  # часто бывает "из вестна я"
            "пов ышенных": "повышенных",
            "при вести": "привести",
            "о бработки": "обработки",
            "х олодной": "холодной",
            "дефо рмации": "деформации",
            "научн ых": "научных",
            "упрочнен ия": "упрочнения",
            "промышленнос ти": "промышленности",
            "такж е": "также",
            "спла вы": "сплавы",
            "м етоды": "методы",
            "п оверхности": "поверхности",
            "возде йствия": "воздействия",
            "фор мы": "формы",
            "достигае т": "достигает",
            "ае ": "ае",  # артефакты конца строк
        }

        cleaned_text = text
        for bad, good in common_fixes.items():
            cleaned_text = cleaned_text.replace(bad, good)

        # Осторожная регулярка: склеиваем только если между буквами 1 пробел
        # и одна из частей очень короткая (1 буква), что явно является ошибкой распознавания.
        # Например: "а б" -> "аб", но "ма шина" -> оставляем "ма шина" (лучше пусть LLM заметит опечатку, чем мы склеим всё)
        # На самом деле, для поиска опечаток лучше вообще не клеить сложные слова автоматически,
        # а довериться LLM, передав текст как есть, но исправив явные "дыры".

        # Паттерн: Буква + Пробел + Буква, где первая часть - 1 символ (часто бывает при разрывах строк)
        # Но это рискованно. Давайте пока оставим только словарные замены выше.
        # Если нужно склеивать больше, добавьте конкретные случаи в словарь common_fixes.

        return cleaned_text

    @staticmethod
    def clean(text: str) -> str:
        """Full cleaning chain."""
        text = TextCleaner.fix_broken_words(text)
        # Убираем двойные пробелы, но оставляем одинарные
        text = re.sub(r'\s{2,}', ' ', text)
        return text.strip()


class LLMAnalyzer:
    """Класс для анализа чанков с помощью LLM DeepSeek через OpenRouter."""

    def __init__(self, api_key: Optional[str] = None, base_url: str = "https://openrouter.ai/api/v1"):
        self.api_key = "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"
        if not self.api_key:
            raise ValueError("API ключ не предоставлен. Установите OPENROUTER_API_KEY.")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=base_url
        )
        self.model = "deepseek/deepseek-v3.2"

        # Категории правил, относящиеся к заголовкам
        self.header_rule_ids = {
            "5.4.1", "5.4.2", "5.4.3", "5.4.4", "5.4.5",
            "5.5.1" # Начало элементов с новой страницы тоже часто относится к структуре/заголовкам
        }

        # Типы чанков, которые считаются заголовками
        self.header_chunk_types = {"section_header", "subsection_header", "header", "title_page"}

    def _filter_rules(self, rules: List[Dict[str, Any]], chunk_type: str) -> List[Dict[str, Any]]:
        """
        Фильтрует правила на основе типа чанка.
        Теперь использует 'белый список' ключевых слов для текста, а не просто блокирует разделы.
        """
        if not rules:
            return []

        # Если это НЕ заголовок, фильтруем правила
        if chunk_type not in self.header_chunk_types:

            # Ключевые слова, которые ПОЗВОЛЕНЫ в обычном тексте (смысловые ошибки, опечатки, стиль)
            allowed_keywords = [
                "опечатк", "описк", "неточност", "разорванн", "перенос",  # Опечатки и переносы
                "сокращен", "слово", "термин",  # Язык
                "шрифт", "размер", "интервал", "черный", "четкий",  # Оформление текста (не заголовка!)
                "абзац", "отступ", "выравнивани", "ширин",  # Структура текста
                "формула", "таблица", "рисунок", "ссылк",  # Упоминания объектов
                "не допускает", "запрещ", "должен", "следует"  # Общие требования
            ]

            # Правила, которые ТОЧНО ЗАПРЕЩЕНЫ для обычного текста (специфика заголовков/страниц)
            forbidden_topics = [
                "заголовок раздела", "заголовок подраздела", "начинаться с заглавной буквы",  # Специфика заголовков
                "точка в конце заголовка", "переносы слов в заголовках",  # Специфика заголовков
                "нумерация страниц", "номер страницы", "титульный лист",  # Специфика страниц
                "новая страница", "начинаться с новой"  # Разрывы страниц
            ]

            filtered_rules = []

            for rule in rules:
                rule_text = rule.get("text", "").lower()
                rule_id = rule.get("gost_id", "")

                # 1. Проверка на явный запрет (если правило про заголовки - убираем)
                is_forbidden = any(topic in rule_text for topic in forbidden_topics)

                # 2. Проверка на разрешение (если есть ключевое слово - оставляем)
                is_allowed = any(keyword in rule_text for keyword in allowed_keywords)

                # Оставляем правило, если оно НЕ запрещено И (разрешено по ключевым словам ИЛИ это общий раздел 4.x)
                if not is_forbidden and (is_allowed or rule_id.startswith("4.")):
                    filtered_rules.append(rule)
                else:
                    # Для отладки: можно логировать, что именно отфильтровалось
                    pass

            removed_count = len(rules) - len(filtered_rules)
            if removed_count > 0:
                logger.debug(f"🚫 Отфильтровано {removed_count} правил (не подходят для типа '{chunk_type}').")

            # Если после фильтрации ничего не осталось, но были найдены правила - значит,
            # возможно, все найденные были про заголовки. Это нормально для текстового чанка,
            # если рядом нет смысловых ошибок.
            if not filtered_rules and rules:
                logger.info(
                    f"⚠️ Все найденные правила ({len(rules)}) отнесены к заголовкам/структуре и исключены для текста.")

            return filtered_rules

        return rules

        # Для заголовков возвращаем всё


    def _extract_json(self, text: str) -> Dict[str, Any]:
        """Извлекает JSON из ответа LLM (с улучшенной обработкой ошибок)."""
        # Удаляем markdown маркеры
        text = re.sub(r'^```json\s*', '', text, flags=re.MULTILINE)
        text = re.sub(r'^```\s*', '', text, flags=re.MULTILINE)
        text = re.sub(r'\s*```$', '', text, flags=re.MULTILINE)
        text = text.strip()

        # Прямой парсинг
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Поиск JSON внутри текста
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        # Попытка исправления частых ошибок
        text_fixed = re.sub(r'"\s*$', '"', text)
        text_fixed = re.sub(r',\s*([\]}])', r'\1', text_fixed)
        text_fixed = text_fixed.replace("'", '"')

        try:
            return json.loads(text_fixed)
        except json.JSONDecodeError as e:
            logger.warning(f"⚠️ Не удалось распарсить JSON: {e}")
            logger.debug(f"Проблемный ответ: {text[:200]}...")

            return {
                "has_violation": False,
                "violations": [],
                "is_correct": True,
                "confidence": 0.0,
                "error": f"JSON Parse Error: {str(e)[:100]}"
            }

    def analyze_chunk(self, chunk_text: str, chunk_type: str, rules: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Анализирует чанк с учетом типа и отфильтрованных правил.
        """
        # 1. Очистка текста
        original_text = chunk_text
        clean_text = TextCleaner.clean(chunk_text)

        if original_text != clean_text:
            logger.debug(f"🧹 Текст очищен от артефактов (было {len(original_text)} символов, стало {len(clean_text)}).")

        # 2. Фильтрация правил
        relevant_rules = self._filter_rules(rules, chunk_type)

        if not relevant_rules:
            logger.info(f"⏭️ Пропуск анализа для типа '{chunk_type}': нет релевантных правил.")
            return {
                "has_violation": False,
                "violations": [],
                "is_correct": True,
                "confidence": 1.0,
                "comment": "Анализ не проводился из-за отсутствия применимых правил для этого типа блока."
            }

        logger.debug(f"📋 Применено правил: {len(relevant_rules)}. IDs: {[r.get('gost_id') for r in relevant_rules]}")

        # 3. Формирование промпта
        rules_text = "\n\n".join([
            f"- Правило {r.get('gost_id')}: {r.get('text')}"
            for r in relevant_rules
        ])

        system_instruction = (
            "Ты эксперт по нормоконтролю документации ГОСТ. Твоя задача — найти нарушения ТОЛЬКО в тех правилах, "
            "которые перечислены ниже.\n\n"
            "ВАЖНОЕ ПРАВИЛО ФИЛЬТРАЦИИ:\n"
            f"Тип текущего блока данных: '{chunk_type}'.\n"
            "- Если тип блока НЕ является заголовком (например, это 'text', 'toc', 'formula'), "
            "то правила оформления заголовков (ГОСТ 5.4.x) к этому блоку НЕ ПРИМЕНЯЮТСЯ. Игнорируй их.\n"
            "- Анализируй только предоставленный список правил. Не выдумывай новые.\n"
            "- Если нарушений нет, верни has_violation: false."
        )

        prompt = f"""
{system_instruction}

СПИСОК ПРОВЕРЯЕМЫХ ПРАВИЛ:
{rules_text}

ТЕКСТ ДЛЯ АНАЛИЗА:
\"\"\"
{clean_text}
\"\"\"

Верни ответ СТРОГО в формате JSON:
{{
    "has_violation": true/false,
    "violations": [
        {{
            "rule_id": "ID правила из списка выше",
            "rule_text": "Текст правила",
            "violation_type": "Краткое название нарушения",
            "explanation": "Почему это нарушение (со ссылкой на текст)",
            "severity": "critical/major/minor"
        }}
    ],
    "is_correct": true/false,
    "confidence": 0.0-1.0
}}
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Ты строгий нормоконтролер. Отвечай только JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1500
            )

            result_text = response.choices[0].message.content.strip()
            logger.info(f"🤖 RAW ответ LLM: {result_text[:150]}...")

            result = self._extract_json(result_text)

            # Добавляем метаданные
            result["chunk_text"] = clean_text
            result["chunk_type"] = chunk_type
            result["applied_rules_count"] = len(relevant_rules)

            has_violation = result.get("has_violation", False)
            logger.info(f"✅ Анализ завершен: найдено нарушений={has_violation}")

            return result

        except Exception as e:
            logger.error(f"❌ Ошибка LLM-анализа: {e}", exc_info=True)
            return {
                "has_violation": False,
                "violations": [],
                "is_correct": True,
                "confidence": 0.0,
                "error": str(e)
            }

    def analyze_batch(self, chunks: List[Dict[str, Any]], search_func) -> List[Dict[str, Any]]:
        """Анализирует пакет чанков."""
        results = []
        for chunk in chunks:
            rules = search_func(
                query=chunk.get("text", ""),
                chunk_type=chunk.get("chunk_type", "text"),
                top_k=5
            )

            analysis = self.analyze_chunk(
                chunk_text=chunk.get("text", ""),
                chunk_type=chunk.get("chunk_type", "text"),
                rules=rules
            )

            analysis["chunk_id"] = chunk.get("id", "")
            analysis["location"] = chunk.get("location", {})
            analysis["applied_rules"] = rules # Сохраняем сами правила для отладки

            results.append(analysis)

        return results