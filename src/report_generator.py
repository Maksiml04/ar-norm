"""

Модуль генерации отчётов для AI-нормконтролера.

Формирует итоговый отчёт о проверке документа.

"""

from __future__ import annotations



import json

from datetime import datetime

from pathlib import Path

from typing import Any



from .logging_config import get_logger



logger = get_logger(__name__)





class ReportGenerator:

    """

    Класс для генерации отчётов по результатам анализа.



    Пример использования:

        generator = ReportGenerator()

        report = generator.generate_report(results, "document.pdf")

        print(generator.print_report(report))

        generator.save_report_json(report, "report.json")

    """



    def generate_report(

        self,

        analysis_results: list[dict[str, Any]],

        document_name: str = "",

    ) -> dict[str, Any]:

        """

        Генерирует структурированный отчёт.



        Args:

            analysis_results: Список результатов анализа чанков

            document_name: Имя проверенного документа



        Returns:

            Словарь с полным отчётом

        """

        try:

            logger.info(f"Генерация отчёта для: {document_name or 'без имени'}")



            total_chunks = len(analysis_results)

            violations_list: list[dict[str, Any]] = []



            for result in analysis_results:

                if result.get("has_violation", False):

                    violations_list.append({

                        "chunk_id": result.get("chunk_id", "unknown"),

                        "location": result.get("location", {}),

                        "text_snippet": result.get("text_snippet", ""),

                        "violations": result.get("violations", []),

                        "applied_rules": result.get("applied_rules", [])[:3],

                        "confidence": result.get("confidence", 0.0)

                    })



            total_violations = len(violations_list)

            correct_chunks = total_chunks - total_violations

            compliance_rate = (

                (correct_chunks / total_chunks * 100) if total_chunks > 0 else 0.0

            )



            report: dict[str, Any] = {

                "metadata": {

                    "document_name": document_name,

                    "check_date": datetime.now().isoformat(),

                    "total_chunks_checked": total_chunks,

                    "generator_version": "1.0.0"

                },

                "summary": {

                    "total_violations": total_violations,

                    "correct_chunks": correct_chunks,

                    "compliance_rate_percent": round(compliance_rate, 2),

                    "status": "PASS" if total_violations == 0 else "FAIL"

                },

                "violations": violations_list,

                "details": analysis_results

            }



            logger.info(

                f"Отчёт сгенерирован: {total_violations} нарушений, "

                f"статус={report['summary']['status']}"

            )

            return report



        except Exception as e:

            logger.error(f"Ошибка при генерации отчёта: {e}", exc_info=True)

            raise



    def print_report(self, report: dict[str, Any]) -> str:

        """

        Форматирует отчёт для вывода в консоль.

        """

        lines: list[str] = []

        lines.append("=" * 80)

        lines.append("ОТЧЁТ AI-НОРМКОНТРОЛЕРА")

        lines.append("=" * 80)



        meta = report.get("metadata", {})

        lines.append(f"Документ: {meta.get('document_name', 'Не указан')}")

        lines.append(f"Дата проверки: {meta.get('check_date', 'Не указана')}")

        lines.append("")



        summary = report.get("summary", {})

        status_icon = "✅" if summary.get("status") == "PASS" else "❌"

        lines.append(f"СТАТУС: {status_icon} {summary.get('status')}")

        lines.append(f"Проверено фрагментов: {summary.get('total_chunks_checked', 0)}")

        lines.append(f"Найдено нарушений: {summary.get('total_violations', 0)}")

        lines.append(f"Процент соответствия: {summary.get('compliance_rate_percent', 0)}%")

        lines.append("")



        violations = report.get("violations", [])

        if violations:

            lines.append("-" * 80)

            lines.append("СПИСОК НАРУШЕНИЙ:")

            lines.append("-" * 80)



            for i, v in enumerate(violations, 1):

                lines.append(f"\n[{i}] Фрагмент ID: {v.get('chunk_id')}")

                loc = v.get('location', {})

                if loc:

                    lines.append(f"    Расположение: стр. {loc.get('page', '?')}")



                for viol in v.get('violations', []):

                    lines.append(

                        f"    ❌ Нарушение: {viol.get('description', 'Нет описания')}"

                    )

                    lines.append(f"       Правило: ГОСТ {viol.get('rule_id', '?')}")

                    lines.append(f"       Критичность: {viol.get('severity', 'unknown')}")

        else:

            lines.append("")

            lines.append("🎉 Нарушений не найдено! Документ соответствует правилам.")



        lines.append("")

        lines.append("=" * 80)



        return "\n".join(lines)



    def save_report_json(

        self, report: dict[str, Any], output_path: str

    ) -> None:

        """Сохраняет отчёт в JSON файл."""

        try:

            path = Path(output_path)

            path.parent.mkdir(parents=True, exist_ok=True)



            with open(path, 'w', encoding='utf-8') as f:

                json.dump(report, f, ensure_ascii=False, indent=2)



            logger.info(f"Отчёт сохранён в: {output_path}")



        except Exception as e:

            logger.error(f"Ошибка сохранения отчёта: {e}", exc_info=True)

            raise