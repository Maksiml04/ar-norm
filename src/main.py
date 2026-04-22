"""
Основной модуль AI-нормконтролера для анализа инженерных документов.
Объединяет все компоненты системы.
"""
import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from pydantic import BaseModel

from .index_builder import GOSTIndexBuilder
from .rule_searcher import GOSTRuleSearcher
from .llm_analyzer import LLMAnalyzer
from .report_generator import ReportGenerator
from .logging_config import get_logger

logger = get_logger(__name__)


class AnalysisResult(BaseModel):
    """Модель результата анализа для API и внутренней логики."""
    chunk_id: str
    has_violation: bool = False
    violations: List[Dict[str, Any]] = []
    is_correct: bool = True
    confidence: float = 0.0
    location: Optional[Dict[str, Any]] = None
    applied_rules: List[Dict[str, Any]] = []
    error: Optional[str] = None
    warning: Optional[str] = None


class AINormkontroler:
    """
    Основной класс AI-нормконтролера.
    """

    def __init__(self, index, rules: List[Dict[str, Any]], model, api_key: Optional[str] = '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'):
        self.searcher = GOSTRuleSearcher(index=index, rules=rules, model=model)
        self.analyzer = LLMAnalyzer(api_key=api_key) if api_key else None
        self.report_generator = ReportGenerator()
        self.rules = rules
        self.index = index
        self.model = model
        self.api_key = api_key

    @classmethod
    def load_from_index(cls, index_path: str, meta_path: str,
                       model_name: str = "cointegrated/LaBSE-en-ru",
                       device: str = "cpu",
                       api_key: Optional[str] = None):
        from sentence_transformers import SentenceTransformer
        import faiss
        import pickle

        try:
            logger.info(f"Загрузка индекса из: {index_path}")
            index = faiss.read_index(index_path)

            logger.info(f"Загрузка метаданных из: {meta_path}")
            with open(meta_path, 'rb') as f:
                rules = pickle.load(f)

            logger.info(f"Загрузка модели: {model_name} (device={device})")
            model = SentenceTransformer(model_name, device=device)

            logger.info(f"Загружено {index.ntotal} правил")
            return cls(index=index, rules=rules, model=model, api_key=api_key)

        except FileNotFoundError as e:
            logger.error(f"Файл не найден: {e}")
            raise
        except Exception as e:
            logger.error(f"Ошибка при загрузке индекса: {e}", exc_info=True)
            raise

    @classmethod
    def build_new(cls, rules_path: str,
                 model_name: str = "cointegrated/LaBSE-en-ru",
                 device: str = "cpu",
                 api_key: Optional[str] = None,
                 save_dir: Optional[str] = None):
        try:
            logger.info(f"Инициализация построителя индекса: {model_name} (device={device})")
            builder = GOSTIndexBuilder(model_name=model_name, device=device)

            logger.info(f"Загрузка правил из: {rules_path}")
            builder.load_rules(rules_path)

            logger.info("Построение векторного индекса FAISS")
            builder.build_index()

            if save_dir:
                logger.info(f"Сохранение индекса в: {save_dir}")
                builder.save(save_dir)

            logger.info(f"Индекс построен успешно: {builder.index.ntotal} правил")
            return cls(index=builder.index, rules=builder.rules, model=builder.model, api_key=api_key)

        except FileNotFoundError as e:
            logger.error(f"Файл с правилами не найден: {e}")
            raise
        except Exception as e:
            logger.error(f"Ошибка при построении индекса: {e}", exc_info=True)
            raise

    def search_rules(self, query: str, chunk_type: str, top_k: int = 5) -> List[Dict[str, Any]]:
        try:
            logger.debug(f"Поиск правил: query='{query[:100]}...', chunk_type={chunk_type}, top_k={top_k}")
            rules = self.searcher.search(query=query, chunk_type=chunk_type, top_k=top_k)
            logger.debug(f"Найдено {len(rules)} правил")
            return rules
        except Exception as e:
            logger.error(f"Ошибка при поиске правил: {e}", exc_info=True)
            return []

    def analyze_chunk(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        try:
            chunk_id = chunk.get("id", "unknown")
            logger.info(f"Анализ чанка {chunk_id}: тип={chunk.get('chunk_type', 'text')}")

            rules = self.search_rules(
                query=chunk.get("text", ""),
                chunk_type=chunk.get("chunk_type", "text"),
                top_k=5
            )

            if self.analyzer:
                try:
                    logger.debug(f"Вызов LLM-анализатора для чанка {chunk_id}")
                    analysis = self.analyzer.analyze_chunk(
                        chunk_text=chunk.get("text", ""),
                        chunk_type=chunk.get("chunk_type", "text"),
                        rules=rules
                    )
                except Exception as e:
                    logger.error(f"Ошибка LLM-анализа для чанка {chunk_id}: {e}", exc_info=True)
                    analysis = {
                        "has_violation": False,
                        "violations": [],
                        "is_correct": True,
                        "confidence": 0.0,
                        "error": f"LLM анализ не выполнен: {str(e)}"
                    }
            else:
                logger.warning(f"LLM-анализ недоступен для чанка {chunk_id} (API ключ не предоставлен)")
                analysis = {
                    "has_violation": False,
                    "violations": [],
                    "is_correct": True,
                    "confidence": 0.0,
                    "warning": "LLM-анализ недоступен (API ключ не предоставлен)"
                }

            analysis["chunk_id"] = chunk.get("id", "")
            analysis["location"] = chunk.get("location", {})
            analysis["applied_rules"] = rules

            has_violations = analysis.get("has_violation", False)
            logger.info(f"Чанк {chunk_id}: {'нарушения найдены' if has_violations else 'нарушений нет'}")

            return analysis

        except Exception as e:
            logger.error(f"Критическая ошибка при анализе чанка: {e}", exc_info=True)
            return {
                "has_violation": False,
                "violations": [],
                "is_correct": True,
                "confidence": 0.0,
                "error": f"Ошибка анализа: {str(e)}"
            }

    def analyze_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        try:
            total = len(chunks)
            logger.info(f"Начало анализа {total} чанков")

            results = []
            for i, chunk in enumerate(chunks, 1):
                logger.debug(f"Обработка чанка {i}/{total}")
                result = self.analyze_chunk(chunk)
                results.append(result)

            violations_count = sum(1 for r in results if r.get("has_violation", False))
            logger.info(f"Анализ завершён: {violations_count} нарушений из {total} чанков")

            return results

        except Exception as e:
            logger.error(f"Критическая ошибка при анализе чанков: {e}", exc_info=True)
            raise

    def generate_report(self, analysis_results: List[Dict[str, Any]],
                       document_name: str = "") -> Dict[str, Any]:
        try:
            logger.info(f"Генерация отчёта для документа: {document_name or 'без имени'}")
            report = self.report_generator.generate_report(
                analysis_results=analysis_results,
                document_name=document_name
            )

            total_violations = report.get("summary", {}).get("total_violations", 0)
            logger.info(f"Отчёт сгенерирован: {total_violations} нарушений")

            return report

        except Exception as e:
            logger.error(f"Ошибка при генерации отчёта: {e}", exc_info=True)
            raise

    def full_check(self, chunks: List[Dict[str, Any]],
                  document_name: str = "",
                  output_path: Optional[str] = None) -> Dict[str, Any]:
        try:
            logger.info(f"Начало полной проверки {len(chunks)} чанков...")
            results = self.analyze_chunks(chunks)
            report = self.generate_report(results, document_name)
            print(self.report_generator.print_report(report))
            if output_path:
                logger.info(f"Сохранение отчёта в: {output_path}")
                self.report_generator.save_report_json(report, output_path)
            logger.info("Полная проверка завершена успешно")
            return report
        except Exception as e:
            logger.error(f"Критическая ошибка при полной проверке: {e}", exc_info=True)
            raise


def main():
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        logger.warning("OPENROUTER_API_KEY не установлен. LLM-анализ будет недоступен.")

    data_dir = Path(__file__).parent.parent / "data"
    index_path = data_dir / "gost.index"
    meta_path = data_dir / "gost_rules_meta.pkl"
    rules_path = data_dir / "gost_2_105_95_rules.json"

    try:
        if index_path.exists() and meta_path.exists():
            logger.info("Загрузка существующего индекса...")
            normkontroler = AINormkontroler.load_from_index(
                index_path=str(index_path),
                meta_path=str(meta_path),
                api_key=api_key
            )
        else:
            logger.info("Индекс не найден, построение нового...")
            normkontroler = AINormkontroler.build_new(
                rules_path=str(rules_path),
                api_key=api_key,
                save_dir=str(data_dir)
            )

        test_chunks = [
            {
                "id": "test_1",
                "text": "Изделие проходит испытания при температуре 20°C.",
                "chunk_type": "text",
                "location": {"page": 1}
            },
            {
                "id": "test_2",
                "text": "заголовок раздела написан строчными буквами",
                "chunk_type": "section_header",
                "location": {"page": 2}
            }
        ]

        if api_key:
            report = normkontroler.full_check(
                chunks=test_chunks,
                document_name="test_document.pdf"
            )
        else:
            logger.warning("Пропускаем LLM-анализ без API ключа.")
            print("\n⚠️ Для тестирования установите OPENROUTER_API_KEY")

    except Exception as e:
        logger.error(f"Ошибка в main(): {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()