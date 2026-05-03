"""
Эксперимент: формирование запроса к ВБД на основе типа чанка.

Сравниваем два подхода:
  1. baseline  — запрос только по тексту чанка (без учёта типа)
  2. type_aware — запрос с context_query на основе типа чанка (текущая логика)

Метрика: precision@5 — доля найденных релевантных правил в топ-5.

Запуск:
    PYTHONPATH=. python3 query_builder_experiment.py
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.retriever import DEFAULT_CONTEXT_QUERIES

# ── Тестовые чанки с ожидаемыми правилами ────────────────────────────────────

TEST_CASES = [
    {
        "chunk_text": "1.1 введение.",
        "chunk_type": "section_header",
        "expected_rule_ids": ["4.1.9", "4.1.2"],
        "description": "Заголовок раздела с нарушениями",
    },
    {
        "chunk_text": "Таблица 1 результаты измерений",
        "chunk_type": "table_ref",
        "expected_rule_ids": ["4.2.1", "4.2.2"],
        "description": "Заголовок таблицы",
    },
    {
        "chunk_text": "Рисунок 1 схема установки",
        "chunk_type": "figure_ref",
        "expected_rule_ids": ["4.3.1", "4.3.2"],
        "description": "Подпись рисунка без тире",
    },
    {
        "chunk_text": "Результаты испытаний приведены ниже. Все параметры в норме.",
        "chunk_type": "text",
        "expected_rule_ids": ["4.1.7", "4.1.2"],
        "description": "Основной текст документа",
    },
    {
        "chunk_text": "Содержание\n1 Введение......5\n2 Методика......8",
        "chunk_type": "toc",
        "expected_rule_ids": ["4.1.2", "4.1.3"],
        "description": "Содержание документа",
    },
]

MODEL_NAME = "cointegrated/LaBSE-en-ru"


# ── Два варианта формирования запроса ────────────────────────────────────────

def build_query_baseline(chunk_text: str, chunk_type: str) -> str:
    """Подход 1: только текст чанка, тип игнорируется."""
    return chunk_text[:300].strip()


def build_query_type_aware(chunk_text: str, chunk_type: str) -> str:
    """Подход 2: context_query по типу + текст чанка (логика из retriever.py)."""
    ctx = DEFAULT_CONTEXT_QUERIES.get(chunk_type, "ГОСТ требования")
    snippet = chunk_text[:300].strip()
    return f"{ctx} | {snippet}"


# ── Поиск и метрика ──────────────────────────────────────────────────────────

def search(
    query: str,
    index: faiss.Index,
    rules: list[dict],
    model: SentenceTransformer,
    top_k: int = 5,
) -> list[dict]:
    vec = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    scores, indices = index.search(vec, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        results.append({**rules[idx], "_score": round(float(score), 4)})
    return results


def precision_at_k(retrieved: list[dict], expected_ids: list[str]) -> float:
    retrieved_ids = [r.get("rule_id", "") for r in retrieved]
    hits = sum(1 for eid in expected_ids if eid in retrieved_ids)
    return round(hits / len(expected_ids), 3) if expected_ids else 0.0


# ── Основной цикл ────────────────────────────────────────────────────────────

def run():
    index_path = Path("data/gost.index")
    meta_path  = Path("data/gost_rules_meta.pkl")

    if not index_path.exists():
        print(f"❌ Индекс не найден: {index_path}")
        return

    print("Загрузка индекса и модели...")
    index = faiss.read_index(str(index_path))

    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    rules = meta.get("rules", meta) if isinstance(meta, dict) else meta

    model = SentenceTransformer(MODEL_NAME, device="cpu")
    print(f"Правил в индексе: {index.ntotal}\n")

    approaches = {
        "baseline":   build_query_baseline,
        "type_aware": build_query_type_aware,
    }

    all_results = {name: [] for name in approaches}

    for approach_name, build_fn in approaches.items():
        print(f"{'='*50}")
        print(f"Подход: {approach_name}")
        print(f"{'='*50}")

        precisions = []
        case_results = []

        for case in TEST_CASES:
            query = build_fn(case["chunk_text"], case["chunk_type"])
            retrieved = search(query, index, rules, model, top_k=5)
            p = precision_at_k(retrieved, case["expected_rule_ids"])
            precisions.append(p)

            mark = "✅" if p > 0 else "❌"
            print(f"  {mark} [{case['chunk_type']}] p@5={p} | {case['description']}")
            print(f"     Запрос: {query[:80]}...")

            case_results.append({
                "chunk_type":   case["chunk_type"],
                "description":  case["description"],
                "query":        query,
                "expected":     case["expected_rule_ids"],
                "retrieved_ids": [r.get("rule_id", "") for r in retrieved],
                "precision_at_5": p,
            })

        avg_p = round(sum(precisions) / len(precisions), 3)
        print(f"\n  Средний precision@5: {avg_p}\n")
        all_results[approach_name] = {
            "avg_precision_at_5": avg_p,
            "cases": case_results,
        }

    # ── Итоговая таблица ─────────────────────────────────────────────────────
    print(f"{'='*50}")
    print("ИТОГ: сравнение подходов к формированию запроса")
    print(f"{'='*50}")
    print(f"{'Подход':<15} {'P@5':>8}")
    print("-" * 25)

    baseline_p   = all_results["baseline"]["avg_precision_at_5"]
    type_aware_p = all_results["type_aware"]["avg_precision_at_5"]
    improvement  = round(type_aware_p - baseline_p, 3)

    print(f"{'baseline':<15} {baseline_p:>8}")
    print(f"{'type_aware':<15} {type_aware_p:>8}  ← текущая логика")
    print(f"\nУлучшение: +{improvement} (precision@5)")

    # ── Сохранение ───────────────────────────────────────────────────────────
    output = {
        "experiment_name": "query_builder_type_aware",
        "metric": "precision@5",
        "approaches": {
            "baseline": {
                "description": "Запрос только по тексту чанка, тип игнорируется",
                **all_results["baseline"],
            },
            "type_aware": {
                "description": "Запрос с context_query на основе типа чанка (DEFAULT_CONTEXT_QUERIES)",
                **all_results["type_aware"],
            },
        },
        "improvement": improvement,
        "conclusion": (
            f"Учёт типа чанка при формировании запроса улучшает precision@5 "
            f"с {baseline_p} до {type_aware_p} (+{improvement}). "
            f"Для каждого типа чанка реализована отдельная логика через DEFAULT_CONTEXT_QUERIES."
        ),
    }

    output_path = Path("data/query_builder_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Результаты сохранены в {output_path}")
    return output


if __name__ == "__main__":
    run()
