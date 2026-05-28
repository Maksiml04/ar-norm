"""
LLM-Segmented Parser v1.0 — Парсинг PDF и DOCX с использованием LLM для сегментации.

Архитектура:
1. Извлечение текста из PDF/DOCX
2. LLM-сегментация на смысловые блоки (TITLE_PAGE, TOC, SECTION_HEADER, TABLE_BLOCK, TEXT_BODY)
3. Детерминированный подбор правил по типу блока (вместо FAISS)

Особенности:
- Поддержка PDF через pdfplumber
- Поддержка DOCX через python-docx
- LLM-сегментация через OpenRouter API (DeepSeek)
- Детерминированный регистр правил: Тип блока -> Список ID правил
"""

from __future__ import annotations

import json
import re
import os
from pathlib import Path
from typing import Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict

try:
    import pdfplumber
except ImportError:
    raise ImportError("Требуется установка: pip install pdfplumber")

try:
    from docx import Document as DocxDocument
except ImportError:
    raise ImportError("Требуется установка: pip install python-docx")

try:
    from openai import OpenAI
except ImportError:
    raise ImportError("Требуется установка: pip install openai")

try:
    from src.logging_config import get_logger
except ImportError:
    from logging_config import get_logger

logger = get_logger(__name__)


# ─── Регистр правил: Тип блока -> Список ID правил ──────────────────────────

BLOCK_RULE_REGISTRY = {
    "TITLE_PAGE": [
        "GOST2.105-6.9",
        "GOST2.105-6.8",
        "GOST2.304-1.1"
    ],
    "TOC": [
        "GOST2.105-4.1.11",
        "GOST2.105-4.2.3"
    ],
    "SECTION_HEADER": [
        "GOST2.105-4.1.9",
        "GOST2.105-4.1.2",
        "GOST2.105-4.1.10"
    ],
    "TABLE_BLOCK": [
        "GOST2.105-4.4.1",
        "GOST2.105-4.4.4",
        "GOST2.105-4.4.5",
        "GOST2.105-4.4.7",
        "GOST2.105-4.4.9",
        "GOST2.105-4.4.16",
        "GOST2.105-4.4.18",
        "GOST2.105-4.4.22"
    ],
    "STAMP": [
        "GOST2.105-6.1",
        "GOST2.105-6.2"
    ],
    "TEXT_BODY": [
        "GOST2.105-4.2.1",
        "GOST2.105-4.2.3",
        "GOST2.105-4.2.4",
        "GOST2.105-4.2.5",
        "GOST2.105-4.2.7",
        "GOST2.105-4.2.8",
        "GOST2.105-4.2.9",
        "GOST2.105-4.2.10",
        "GOST2.105-4.2.11",
        "GOST2.105-4.2.13",
        "GOST2.105-4.2.14",
        "GOST2.105-4.2.15",
        "GOST2.105-4.2.18",
        "GOST2.105-4.2.21",
        "GOST2.105-4.2.22",
        "GOST2.105-4.3.1",
        "GOST2.105-3.6",
        "GOST2.304-2.2"
    ],
    "FIGURE_REF": [
        "GOST2.105-4.3.1",
        "GOST2.105-4.3.2"
    ],
    "LIST": [
        "GOST2.105-4.1.7",
        "GOST2.105-4.1.8"
    ],
    "FORMULA": [
        "GOST2.105-4.2.15",
        "GOST2.105-4.2.18",
        "GOST2.105-4.2.19"
    ]
}


# ─── База текстов правил ГОСТ ───────────────────────────────────────────────

RULES_DB = {
    "GOST2.105-4.1.2": {
        "id": "GOST2.105-4.1.2",
        "rule_text": """Разделы должны иметь порядковые номера в пределах всего документа, обозначенные арабскими цифрами без точки и записанные с абзацного отступа.""",
        "category": "structure"
    },
    "GOST2.105-4.1.7": {
        "id": "GOST2.105-4.1.7",
        "rule_text": """Перечисления внутри пунктов или подпунктов следует оформлять с дефисом. При необходимости ссылки на перечисление используют строчную букву со скобкой.""",
        "category": "structure"
    },
    "GOST2.105-4.1.8": {
        "id": "GOST2.105-4.1.8",
        "rule_text": """Перечисления с буквенной нумерацией записываются со строчных букв с последующей скобкой.""",
        "category": "structure"
    },
    "GOST2.105-4.1.9": {
        "id": "GOST2.105-4.1.9",
        "rule_text": """Заголовки разделов и подразделов следует печатать с прописной буквы без точки в конце, не подчеркивая. Переносы слов в заголовках не допускаются.""",
        "category": "formatting"
    },
    "GOST2.105-4.1.10": {
        "id": "GOST2.105-4.1.10",
        "rule_text": """Каждый раздел текстового документа рекомендуется начинать с нового листа (страницы).""",
        "category": "structure"
    },
    "GOST2.105-4.1.11": {
        "id": "GOST2.105-4.1.11",
        "rule_text": """Слово «Содержание» записывают в виде заголовка симметрично тексту с прописной буквы. Наименования в содержании записывают строчными буквами, начиная с прописной.""",
        "category": "structure"
    },
    "GOST2.105-4.2.1": {
        "id": "GOST2.105-4.2.1",
        "rule_text": """Текст документа должен быть кратким, четким и не допускать различных толкований. При обязательных требованиях применяют слова: «должен», «следует», «не допускается».""",
        "category": "style"
    },
    "GOST2.105-4.2.3": {
        "id": "GOST2.105-4.2.3",
        "rule_text": """В тексте не допускается применять обороты разговорной речи, техницизмы, профессионализмы, произвольные словообразования и сокращения слов, кроме установленных стандартами.""",
        "category": "terminology"
    },
    "GOST2.105-4.2.4": {
        "id": "GOST2.105-4.2.4",
        "rule_text": """Числовые значения величин с обозначением единиц следует писать цифрами, а числа без обозначения единиц от единицы до девяти — словами.""",
        "category": "style"
    },
    "GOST2.105-4.2.5": {
        "id": "GOST2.105-4.2.5",
        "rule_text": """В тексте не допускается применять математический знак минус перед отрицательными значениями, знак диаметра без цифр, математические знаки (>, <, =) без числовых значений.""",
        "category": "style"
    },
    "GOST2.105-4.2.7": {
        "id": "GOST2.105-4.2.7",
        "rule_text": """Условные буквенные обозначения, изображения или знаки должны соответствовать действующим стандартам. Перед обозначением параметра дают его пояснение.""",
        "category": "references"
    },
    "GOST2.105-4.2.8": {
        "id": "GOST2.105-4.2.8",
        "rule_text": """Следует применять стандартизованные единицы физических величин по ГОСТ 8.417. При необходимости в скобках могут быть указаны единицы ранее применявшихся систем.""",
        "category": "formatting"
    },
    "GOST2.105-4.2.9": {
        "id": "GOST2.105-4.2.9",
        "rule_text": """Числовые значения с обозначением единиц физических величин и единиц счета пишут цифрами, а числа от 1 до 9 без единиц — словами.""",
        "category": "style"
    },
    "GOST2.105-4.2.10": {
        "id": "GOST2.105-4.2.10",
        "rule_text": """Единица физической величины одного и того же параметра в пределах документа должна быть постоянной. Если приводят ряд значений, единицу указывают только после последнего числа.""",
        "category": "formatting"
    },
    "GOST2.105-4.2.11": {
        "id": "GOST2.105-4.2.11",
        "rule_text": """При указании диапазона числовых значений единица физической величины указывается после последнего числового значения диапазона (например, от 1 до 5 мм).""",
        "category": "formatting"
    },
    "GOST2.105-4.2.13": {
        "id": "GOST2.105-4.2.13",
        "rule_text": """Числовые значения указывают с одинаковой степенью точности (выравнивание числа знаков после запятой) для одного наименования изделия.""",
        "category": "style"
    },
    "GOST2.105-4.2.14": {
        "id": "GOST2.105-4.2.14",
        "rule_text": """Дробные числа необходимо приводить в виде десятичных дробей, за исключением размеров в дюймах.""",
        "category": "style"
    },
    "GOST2.105-4.2.15": {
        "id": "GOST2.105-4.2.15",
        "rule_text": """Пояснения символов и числовых коэффициентов к формуле приводят непосредственно под ней, начиная со слова «где» (без двоеточия). Каждое пояснение — с новой строки.""",
        "category": "formatting"
    },
    "GOST2.105-4.2.18": {
        "id": "GOST2.105-4.2.18",
        "rule_text": """Формулы нумеруют сквозной нумерацией арабскими цифрами в круглых скобках справа. Одна формула обозначается (1).""",
        "category": "formatting"
    },
    "GOST2.105-4.2.19": {
        "id": "GOST2.105-4.2.19",
        "rule_text": """Порядок изложения пояснений к формулам должен соответствовать порядку следования символов в формуле.""",
        "category": "formatting"
    },
    "GOST2.105-4.2.21": {
        "id": "GOST2.105-4.2.21",
        "rule_text": """Примечания печатают с прописной буквы с абзацного отступа. Одно примечание: слово «Примечание» и тире, затем текст. Несколько примечаний нумеруют арабскими цифрами.""",
        "category": "formatting"
    },
    "GOST2.105-4.2.22": {
        "id": "GOST2.105-4.2.22",
        "rule_text": """Ссылаться следует на документ в целом или его разделы и приложения. Ссылки на подразделы, пункты, таблицы и иллюстрации допускаются только в пределах данного документа.""",
        "category": "references"
    },
    "GOST2.105-4.3.1": {
        "id": "GOST2.105-4.3.1",
        "rule_text": """Иллюстрации нумеруют арабскими цифрами сквозной нумерацией. Слово «Рисунок» и наименование помещают после пояснительных данных.""",
        "category": "formatting"
    },
    "GOST2.105-4.3.2": {
        "id": "GOST2.105-4.3.2",
        "rule_text": """Иллюстрации, за исключением иллюстраций приложений, следует нумеровать арабскими цифрами сквозной нумерацией.""",
        "category": "formatting"
    },
    "GOST2.105-4.4.1": {
        "id": "GOST2.105-4.4.1",
        "rule_text": """Таблицы нумеруют арабскими цифрами сквозной нумерацией. Название таблицы помещают над ней; при переносе части таблицы название ставят только над первой частью.""",
        "category": "formatting"
    },
    "GOST2.105-4.4.4": {
        "id": "GOST2.105-4.4.4",
        "rule_text": """Заголовки граф таблицы следует печатать с прописной буквы в единственном числе.""",
        "category": "formatting"
    },
    "GOST2.105-4.4.5": {
        "id": "GOST2.105-4.4.5",
        "rule_text": """Таблицу следует располагать в документе непосредственно после текста, в котором она упоминается впервые, или на следующей странице.""",
        "category": "layout"
    },
    "GOST2.105-4.4.7": {
        "id": "GOST2.105-4.4.7",
        "rule_text": """При делении таблицы на части над каждой последующей частью пишут «Продолжение таблицы» с ее номером.""",
        "category": "formatting"
    },
    "GOST2.105-4.4.9": {
        "id": "GOST2.105-4.4.9",
        "rule_text": """Если все показатели таблицы выражены в одной единице физической величины, ее обозначение помещают над таблицей справа.""",
        "category": "units"
    },
    "GOST2.105-4.4.16": {
        "id": "GOST2.105-4.4.16",
        "rule_text": """Повторяющийся в строках графы текст из одного слова заменяют кавычками, из двух и более слов — словами «То же», а затем кавычками.""",
        "category": "style"
    },
    "GOST2.105-4.4.18": {
        "id": "GOST2.105-4.4.18",
        "rule_text": """При отсутствии данных в ячейке таблицы следует ставить прочерк (тире).""",
        "category": "formatting"
    },
    "GOST2.105-4.4.22": {
        "id": "GOST2.105-4.4.22",
        "rule_text": """Цифры в графах таблиц выравнивают по разрядам; количество десятичных знаков должно быть одинаковым для всех значений одного показателя.""",
        "category": "style"
    },
    "GOST2.105-3.6": {
        "id": "GOST2.105-3.6",
        "rule_text": """Расстояние от рамки до границ текста в начале и конце строк должно быть не менее 3 мм, от верхней/нижней строки до рамки — не менее 10 мм.""",
        "category": "paper"
    },
    "GOST2.105-6.1": {
        "id": "GOST2.105-6.1",
        "rule_text": """Основная надпись выполняется по форме 2 и располагается в правом нижнем углу документа.""",
        "category": "layout"
    },
    "GOST2.105-6.2": {
        "id": "GOST2.105-6.2",
        "rule_text": """В основной надписи указывают: наименование изделия, обозначение документа, материал, массу, масштаб, лист и листов.""",
        "category": "layout"
    },
    "GOST2.105-6.8": {
        "id": "GOST2.105-6.8",
        "rule_text": """Лист утверждения оформляется в соответствии с требованиями ГОСТ 2.105.""",
        "category": "layout"
    },
    "GOST2.105-6.9": {
        "id": "GOST2.105-6.9",
        "rule_text": """Титульный лист и лист утверждения выполняют на листах формата А4 по ГОСТ 2.301. Поле 4 — наименование изделия (заглавными буквами).""",
        "category": "layout"
    },
    "GOST2.304-1.1": {
        "id": "GOST2.304-1.1",
        "rule_text": """Размер шрифта — величина, определенная высотой прописных букв в миллиметрах.""",
        "category": "font_parameters"
    },
    "GOST2.304-2.2": {
        "id": "GOST2.304-2.2",
        "rule_text": """Устанавливаются размеры шрифта: 2,5; 3,5; 5; 7; 10; 14; 20; 28; 40. Применение шрифта 1,8 не рекомендуется.""",
        "category": "font_parameters"
    }
}


@dataclass
class DocumentChunk:
    """Чанк документа с метаданными."""
    chunk_id: str
    chunk_type: str
    text: str
    location: dict = field(default_factory=dict)
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


class LLMSegmentedParser:
    """
    Парсер документов с LLM-сегментацией и детерминированным поиском правил.
    
    Поддерживает:
    - PDF файлы (через pdfplumber)
    - DOCX файлы (через python-docx)
    """
    
    SYSTEM_PROMPT = """Ты экспертный парсер технической документации (ЕСКД).
Проанализируй текст страницы и разбей его на смысловые блоки.

Типы блоков:
- TITLE_PAGE: Грифы "УТВЕРЖДАЮ", подписи, год, город, название вуза/организации.
- TOC: "Содержание", список разделов с номерами страниц.
- SECTION_HEADER: Номер и название раздела (напр. "1 Литературный обзор").
- TABLE_BLOCK: Только блоки, содержащие слово 'Таблица' И имеющие структурированные данные в строках/столбцах.
- Рисунок/График/Схема — это НЕ таблица, классифицируй как TEXT_BODY или FIGURE_REF.
- FIGURE_REF: Подписи к рисункам ("Рисунок 1", "Рис. 2").
- FORMULA: Формулы и уравнения.
- LIST: Перечисления с дефисами или буквами.
- TEXT_BODY: Обычный текст абзацев.

Верни JSON массив: [{"type": "...", "content": "текст"}]"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "deepseek/deepseek-chat",
        base_url: str = "https://openrouter.ai/api/v1"
    ) -> None:
        """
        Инициализация парсера.
        
        Args:
            api_key: API ключ OpenRouter (или None для тестового режима)
            model_name: Название модели LLM
            base_url: URL API сервера
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.model_name = model_name
        self.base_url = base_url
        
        self.client = None
        if self.api_key:
            try:
                self.client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url
                )
                logger.info(f"LLM клиент инициализирован: {model_name}")
            except Exception as e:
                logger.warning(f"Не удалось инициализировать LLM клиент: {e}")
        
        self.document_abbreviations: dict[str, str] = {}
        self.current_section_context: str = ""
        
        logger.info("LLMSegmentedParser инициализирован")
    
    def _clean_pdf_text(self, raw_text: str) -> str:
        """Очистка мусора pdfplumber."""
        lines = [l.strip() for l in raw_text.split('\n') if l.strip()]
        return "\n".join(lines)
    
    def _segment_with_llm(self, page_text: str) -> list[dict]:
        """Сегментация текста страницы с помощью LLM."""
        if not self.client:
            # Fallback: возвращаем весь текст как один блок TEXT_BODY
            return [{"type": "TEXT_BODY", "content": page_text}]
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": f"Текст:\n{page_text}"}
                ],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            raw = response.choices[0].message.content
            # Удаляем markdown-обёртку ```json ... ```
            raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw).strip()
            segments = json.loads(raw)
            
            # Нормализация типов блоков: приводим к верхнему регистру и мапим известные варианты
            normalized = []
            for seg in segments:
                b_type = seg.get("type", "TEXT_BODY").upper().strip()
                
                # Маппинг возможных вариантов написания
                type_mapping = {
                    "TEXT": "TEXT_BODY",
                    "TEXTBODY": "TEXT_BODY",
                    "TEXT-BODY": "TEXT_BODY",
                    "PARAGRAPH": "TEXT_BODY",
                    "FIGURE": "FIGURE_REF",
                    "FIG": "FIGURE_REF",
                    "IMAGE": "FIGURE_REF",
                    "RIS": "FIGURE_REF",
                    "RISUNOK": "FIGURE_REF",
                    "ФОРМУЛА": "FORMULA",
                    "EQUATION": "FORMULA",
                    "MATH": "FORMULA",
                    "СПИСОК": "LIST",
                    "ENUM": "LIST",
                    "ITEMIZE": "LIST",
                    "ТАБЛИЦА": "TABLE_BLOCK",
                    "TABL": "TABLE_BLOCK",
                    "TABLE": "TABLE_BLOCK",
                }
                
                b_type = type_mapping.get(b_type, b_type)
                
                # Проверка для TABLE_BLOCK: должен содержать слово "Таблица" или "Table"
                content = seg.get("content", "")
                if b_type == "TABLE_BLOCK":
                    if not any(word in content.upper() for word in ["ТАБЛИЦА", "TABLE", "ПРОДОЛЖЕНИЕ ТАБЛИЦЫ", "ПРОДОЛЖЕНИЕ"]):
                        # Если нет слова "Таблица",降级 до TEXT_BODY
                        b_type = "TEXT_BODY"
                        logger.debug(f"Блок помечен как TABLE_BLOCK, но не содержит ключевого слова. Понижен до TEXT_BODY")
                
                normalized.append({"type": b_type, "content": content})
            
            return normalized
        except Exception as e:
            logger.warning(f"Ошибка LLM-сегментации: {e}")
            return [{"type": "TEXT_BODY", "content": page_text}]
    
    def _get_rules_for_block(self, block_type: str) -> list[dict]:
        """Детерминированный подбор правил по типу блока."""
        type_key = block_type.upper()
        rule_ids = BLOCK_RULE_REGISTRY.get(type_key, BLOCK_RULE_REGISTRY["TEXT_BODY"])
        return [RULES_DB[rid] for rid in rule_ids if rid in RULES_DB]
    
    def parse_pdf(
        self,
        pdf_path: str,
        save_to: Optional[str] = None
    ) -> list[DocumentChunk]:
        """
        Парсинг PDF файла с LLM-сегментацией.
        
        Args:
            pdf_path: Путь к PDF файлу
            save_to: Опциональный путь для сохранения результатов
            
        Returns:
            Список чанков документа
        """
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"Файл не найден: {pdf_path}")
        
        logger.info(f"Обработка PDF: {path.name}")
        chunks: list[DocumentChunk] = []
        chunk_id = 0
        
        with pdfplumber.open(str(path)) as pdf:
            total_pages = len(pdf.pages)
            logger.info(f"Страниц: {total_pages}")
            
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if not text:
                    continue
                
                # Очистка и сегментация
                clean_text = self._clean_pdf_text(text)
                blocks = self._segment_with_llm(clean_text)
                
                for block in blocks:
                    b_type = block.get("type", "TEXT_BODY")
                    content = block.get("content", "")
                    
                    # Фильтр мусора (короткие строки)
                    if len(content.replace(" ", "")) < 10:
                        continue
                    
                    # Попытка извлечь таблицу если это TABLE_BLOCK но content пуст
                    if b_type == "TABLE_BLOCK" and not content:
                        tables = page.extract_tables()
                        if tables:
                            content = "\n".join([str(row) for row in tables[0]])
                    
                    # Логирование предупреждения если тип блока не найден в реестре
                    if b_type not in BLOCK_RULE_REGISTRY:
                        logger.warning(f"Нет правил для типа блока '{b_type}', используем TEXT_BODY")
                    
                    # Создаём чанк
                    chunk = DocumentChunk(
                        chunk_id=f"p{i+1}_{b_type.lower()}_{chunk_id}",
                        chunk_type=b_type.lower(),
                        text=content[:500],  # Ограничиваем длину
                        location={"page": i + 1},
                        metadata={
                            "rules_count": len(self._get_rules_for_block(b_type)),
                            "original_type": b_type,
                        },
                        context_query=f"правила оформления {b_type.lower()} ГОСТ"
                    )
                    chunks.append(chunk)
                    chunk_id += 1
        
        stats = defaultdict(int)
        for c in chunks:
            stats[c.chunk_type] += 1
        logger.info(f"Готово. Чанков: {len(chunks)}. Распределение: {dict(stats)}")
        
        if save_to:
            self.save_chunks(chunks, save_to)
        
        return chunks
    
    def parse_docx(
        self,
        docx_path: str,
        save_to: Optional[str] = None
    ) -> list[DocumentChunk]:
        """
        Парсинг DOCX файла с эвристической сегментацией.
        
        Примечание: Для DOCX LLM-сегментация применяется к каждому параграфу,
        так как структура DOCX уже содержит информацию о стилях.
        
        Args:
            docx_path: Путь к DOCX файлу
            save_to: Опциональный путь для сохранения результатов
            
        Returns:
            Список чанков документа
        """
        path = Path(docx_path)
        if not path.exists():
            raise FileNotFoundError(f"Файл не найден: {docx_path}")
        
        logger.info(f"Обработка DOCX: {path.name}")
        doc = DocxDocument(str(path))
        chunks: list[DocumentChunk] = []
        chunk_id = 0
        
        # Группируем параграфы по страницам (эвристика: ~40 строк на страницу)
        lines_per_page = 40
        current_page = 1
        line_count = 0
        page_blocks: list[str] = []
        
        for para in doc.paragraphs:
            text = para.text.strip()
            if not text:
                continue
            
            # Определяем тип блока по стилю
            style_name = para.style.name if para.style else ""
            block_type = self._infer_block_type_from_style(style_name, text)
            
            page_blocks.append(f"[{block_type}] {text}")
            line_count += 1
            
            # Эвристика перехода на новую страницу
            if line_count >= lines_per_page:
                # Сегментируем накопленные блоки страницы
                page_text = "\n".join(page_blocks)
                blocks = self._segment_with_llm(page_text)
                
                for block in blocks:
                    b_type = block.get("type", "TEXT_BODY")
                    content = block.get("content", "")
                    
                    if len(content.replace(" ", "")) < 10:
                        continue
                    
                    chunk = DocumentChunk(
                        chunk_id=f"p{current_page}_{b_type.lower()}_{chunk_id}",
                        chunk_type=b_type.lower(),
                        text=content[:500],
                        location={"page": current_page},
                        metadata={
                            "rules_count": len(self._get_rules_for_block(b_type)),
                            "original_style": style_name,
                        },
                        context_query=f"правила оформления {b_type.lower()} ГОСТ"
                    )
                    chunks.append(chunk)
                    chunk_id += 1
                
                page_blocks = []
                line_count = 0
                current_page += 1
        
        # Обрабатываем последнюю страницу
        if page_blocks:
            page_text = "\n".join(page_blocks)
            blocks = self._segment_with_llm(page_text)
            
            for block in blocks:
                b_type = block.get("type", "TEXT_BODY")
                content = block.get("content", "")
                
                if len(content.replace(" ", "")) < 10:
                    continue
                
                chunk = DocumentChunk(
                    chunk_id=f"p{current_page}_{b_type.lower()}_{chunk_id}",
                    chunk_type=b_type.lower(),
                    text=content[:500],
                    location={"page": current_page},
                    metadata={
                        "rules_count": len(self._get_rules_for_block(b_type)),
                        "original_style": style_name,
                    },
                    context_query=f"правила оформления {b_type.lower()} ГОСТ"
                )
                chunks.append(chunk)
                chunk_id += 1
        
        stats = defaultdict(int)
        for c in chunks:
            stats[c.chunk_type] += 1
        logger.info(f"Готово. Чанков: {len(chunks)}. Распределение: {dict(stats)}")
        
        if save_to:
            self.save_chunks(chunks, save_to)
        
        return chunks
    
    def _infer_block_type_from_style(self, style_name: str, text: str) -> str:
        """
        Определение типа блока по имени стиля DOCX.
        
        Args:
            style_name: Имя стиля параграфа
            text: Текст параграфа
            
        Returns:
            Тип блока (TITLE_PAGE, TOC, SECTION_HEADER, TEXT_BODY, etc.)
        """
        style_lower = style_name.lower()
        text_lower = text.lower()
        
        # Заголовок документа
        if "title" in style_lower or "заголовок" in style_lower and "1" in style_lower:
            if any(word in text_lower for word in ["утверждаю", "реферат", "отчет"]):
                return "TITLE_PAGE"
            return "SECTION_HEADER"
        
        # Содержание
        if "содержание" in text_lower or "оглавление" in text_lower:
            return "TOC"
        
        # Таблицы
        if "таблица" in text_lower and ":" in text:
            return "TABLE_BLOCK"
        
        # Рисунки
        if "рисунок" in text_lower or "рис." in text_lower:
            return "FIGURE_REF"
        
        # Формулы (эвристика: короткие строки с математическими символами)
        if len(text) < 100 and any(c in text for c in "=∑∫√±"):
            return "FORMULA"
        
        # Перечисления
        if text.startswith("- ") or re.match(r"^[а-яa-z]\)", text_lower):
            return "LIST"
        
        # По умолчанию - основной текст
        return "TEXT_BODY"
    
    def save_chunks(self, chunks: list[DocumentChunk], output_path: str) -> None:
        """Сохранение чанков в JSON файл."""
        data = {
            "total_chunks": len(chunks),
            "chunks": [c.to_dict() for c in chunks]
        }
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Чанки сохранены в {output_path}")
    
    def get_rules_for_chunk(self, chunk: DocumentChunk) -> list[dict]:
        """
        Получение релевантных правил для чанка (детерминированный поиск).
        
        Args:
            chunk: Чанк документа
            
        Returns:
            Список релевантных правил
        """
        return self._get_rules_for_block(chunk.chunk_type)


# ─── Детерминированный ретривер (для совместимости с основным интерфейсом) ───

class DeterministicRetriever:
    """
    Детерминированный ретривер правил вместо FAISS.
    Использует регистр BLOCK_RULE_REGISTRY для подбора правил по типу блока.
    """
    
    def __init__(self) -> None:
        self.rules = list(RULES_DB.values())
        logger.info(f"DeterministicRetriever инициализирован. Правил: {len(self.rules)}")
    
    def search(
        self,
        chunk_text: str,
        chunk_type: str = "text",
        context_query: str = "",
        top_k: int = 5
    ) -> list[dict]:
        """
        Поиск правил по типу блока (детерминированно).
        
        Args:
            chunk_text: Текст чанка (игнорируется, используется только chunk_type)
            chunk_type: Тип чанка
            context_query: Контекстный запрос (игнорируется)
            top_k: Максимальное количество возвращаемых правил
            
        Returns:
            Список релевантных правил
        """
        # Нормализация типа блока: приводим к верхнему регистру и мапим известные варианты
        type_key = chunk_type.upper().strip()
        
        # Маппинг возможных вариантов написания
        type_mapping = {
            "TEXT": "TEXT_BODY",
            "TEXTBODY": "TEXT_BODY",
            "TEXT-BODY": "TEXT_BODY",
            "PARAGRAPH": "TEXT_BODY",
            "FIGURE": "FIGURE_REF",
            "FIG": "FIGURE_REF",
            "IMAGE": "FIGURE_REF",
            "RIS": "FIGURE_REF",
            "RISUNOK": "FIGURE_REF",
            "ФОРМУЛА": "FORMULA",
            "EQUATION": "FORMULA",
            "MATH": "FORMULA",
            "СПИСОК": "LIST",
            "ENUM": "LIST",
            "ITEMIZE": "LIST",
            "ТАБЛИЦА": "TABLE_BLOCK",
            "TABL": "TABLE_BLOCK",
            "TABLE": "TABLE_BLOCK",
        }
        
        type_key = type_mapping.get(type_key, type_key)
        
        # Проверка: если тип блока не найден в реестре, используем TEXT_BODY
        if type_key not in BLOCK_RULE_REGISTRY:
            logger.warning(f"Нет правил для типа блока '{chunk_type}', используем TEXT_BODY")
            type_key = "TEXT_BODY"
        
        rule_ids = BLOCK_RULE_REGISTRY.get(type_key, BLOCK_RULE_REGISTRY["TEXT_BODY"])
        rules = [RULES_DB[rid] for rid in rule_ids if rid in RULES_DB]
        
        # Добавляем score для совместимости с интерфейсом
        for i, rule in enumerate(rules[:top_k]):
            rule["_score"] = 1.0 - (i * 0.1)  # Убывающий скор
        
        logger.info(f"search [{chunk_type}] → {len(rules[:top_k])} правил. IDs: {[r['id'] for r in rules[:top_k]]}")
        return rules[:top_k]
    
    def search_batch(
        self,
        chunks: list[dict],
        top_k: int = 5
    ) -> list[list[dict]]:
        """
        Пакетный поиск правил для списка чанков.
        
        Args:
            chunks: Список чанков (словари с chunk_type)
            top_k: Максимальное количество правил на чанк
            
        Returns:
            Список списков правил
        """
        results = []
        for chunk in chunks:
            chunk_type = chunk.get("chunk_type", "text")
            results.append(self.search("", chunk_type, "", top_k))
        return results
