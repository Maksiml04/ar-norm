"""
Тесты для ReportGenerator.
Запуск: pytest tests/test_report_generator.py -v
"""
import json
import pytest
from pathlib import Path
from unittest.mock import patch
from src.report_generator import ReportGenerator


# ── Фикстуры ────────────────────────────────────────────────────────────────

@pytest.fixture
def generator():
    return ReportGenerator()


def make_result(has_violation=False, chunk_id="p1_c1", violations=None):
    return {
        "chunk_id": chunk_id,
        "has_violation": has_violation,
        "location": {"page": 1},
        "text_snippet": "текст фрагмента",
        "violations": violations or [],
        "applied_rules": [{"rule_id": "4.1.2"}],
        "confidence": 0.9,
    }


# ── Тесты generate_report ────────────────────────────────────────────────────

class TestGenerateReport:
    def test_возвращает_словарь(self, generator):
        result = generator.generate_report([], "doc.pdf")
        assert isinstance(result, dict)

    def test_структура_отчёта(self, generator):
        report = generator.generate_report([], "doc.pdf")
        assert "metadata" in report
        assert "summary" in report
        assert "violations" in report
        assert "details" in report

    def test_имя_документа_в_метаданных(self, generator):
        report = generator.generate_report([], "my_doc.pdf")
        assert report["metadata"]["document_name"] == "my_doc.pdf"

    def test_статус_pass_без_нарушений(self, generator):
        results = [make_result(has_violation=False)]
        report = generator.generate_report(results)
        assert report["summary"]["status"] == "PASS"

    def test_статус_fail_с_нарушениями(self, generator):
        results = [make_result(has_violation=True)]
        report = generator.generate_report(results)
        assert report["summary"]["status"] == "FAIL"

    def test_подсчёт_нарушений(self, generator):
        results = [
            make_result(has_violation=True, chunk_id="c1"),
            make_result(has_violation=False, chunk_id="c2"),
            make_result(has_violation=True, chunk_id="c3"),
        ]
        report = generator.generate_report(results)
        assert report["summary"]["total_violations"] == 2

    def test_подсчёт_корректных_чанков(self, generator):
        results = [
            make_result(has_violation=True),
            make_result(has_violation=False),
        ]
        report = generator.generate_report(results)
        assert report["summary"]["correct_chunks"] == 1

    def test_compliance_rate_100_без_нарушений(self, generator):
        results = [make_result(has_violation=False)] * 4
        report = generator.generate_report(results)
        assert report["summary"]["compliance_rate_percent"] == 100.0

    def test_compliance_rate_0_все_нарушения(self, generator):
        results = [make_result(has_violation=True)] * 3
        report = generator.generate_report(results)
        assert report["summary"]["compliance_rate_percent"] == 0.0

    def test_пустой_список_не_падает(self, generator):
        report = generator.generate_report([])
        assert report["summary"]["total_violations"] == 0
        assert report["summary"]["compliance_rate_percent"] == 0.0

    def test_applied_rules_обрезается_до_3(self, generator):
        result = make_result(has_violation=True)
        result["applied_rules"] = [{"rule_id": str(i)} for i in range(10)]
        report = generator.generate_report([result])
        assert len(report["violations"][0]["applied_rules"]) <= 3

    def test_дата_проверки_присутствует(self, generator):
        report = generator.generate_report([])
        assert "check_date" in report["metadata"]
        assert report["metadata"]["check_date"] != ""


# ── Тесты print_report ───────────────────────────────────────────────────────

class TestPrintReport:
    def test_возвращает_строку(self, generator):
        report = generator.generate_report([], "doc.pdf")
        result = generator.print_report(report)
        assert isinstance(result, str)

    def test_содержит_статус_pass(self, generator):
        report = generator.generate_report([make_result(False)])
        text = generator.print_report(report)
        assert "PASS" in text

    def test_содержит_статус_fail(self, generator):
        report = generator.generate_report([make_result(True)])
        text = generator.print_report(report)
        assert "FAIL" in text

    def test_содержит_имя_документа(self, generator):
        report = generator.generate_report([], "test_doc.pdf")
        text = generator.print_report(report)
        assert "test_doc.pdf" in text

    def test_нет_нарушений_сообщение(self, generator):
        report = generator.generate_report([])
        text = generator.print_report(report)
        assert "Нарушений не найдено" in text

    def test_нарушения_выводятся(self, generator):
        result = make_result(has_violation=True, violations=[{
            "description": "Нет точки в конце",
            "rule_id": "4.1.9",
            "severity": "minor"
        }])
        report = generator.generate_report([result])
        text = generator.print_report(report)
        assert "НАРУШЕНИЙ" in text or "Нарушение" in text


# ── Тесты save_report_json ───────────────────────────────────────────────────

class TestSaveReportJson:
    def test_файл_создаётся(self, generator, tmp_path):
        report = generator.generate_report([], "doc.pdf")
        output = str(tmp_path / "report.json")
        generator.save_report_json(report, output)
        assert Path(output).exists()

    def test_файл_валидный_json(self, generator, tmp_path):
        report = generator.generate_report([], "doc.pdf")
        output = str(tmp_path / "report.json")
        generator.save_report_json(report, output)
        with open(output, encoding="utf-8") as f:
            loaded = json.load(f)
        assert loaded["metadata"]["document_name"] == "doc.pdf"

    def test_создаёт_вложенные_директории(self, generator, tmp_path):
        report = generator.generate_report([])
        output = str(tmp_path / "nested" / "dir" / "report.json")
        generator.save_report_json(report, output)
        assert Path(output).exists()
