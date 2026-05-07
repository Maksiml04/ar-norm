"""
Эксперимент 3: стратегии разбиения правил ГОСТ на чанки при построении индекса.

Стратегии:
  1. per_rule    — каждое правило отдельный чанк (текущий подход)
  2. per_section — правила одного раздела объединяются в один чанк
  3. with_overlap — правила с перекрытием (каждый чанк = правило + сосед)

Метрика качества: precision@k — доля релевантных правил в топ-k результатах.

Запуск:
    PYTHONPATH=. python3 rag_experiments_chunking.py
"""

from __future__ import annotations

import json
import pickle
import tempfile
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ── Тестовые запросы с ожидаемыми правилами ──────────────────────────────────

TEST_QUERIES = [
    {
        "query": "заголовок раздела с прописной буквы без точки в конце",
        "expected_rule_ids": ["4.1.9"],
        "chunk_type": "section_header",
    },
    {
        "query": "нумерация разделов арабскими цифрами абзацный отступ",
        "expected_rule_ids": ["4.1.2", "4.1.3"],
        "chunk_type": "section_header",
    },
    {
        "query": "таблица сквозная нумерация заголовок",
        "expected_rule_ids": ["4.2.1", "4.2.2"],
        "chunk_type": "table_ref",
    },
    {
        "query": "рисунок подпись нумерация тире",
        "expected_rule_ids": ["4.3.1", "4.3.2"],
        "chunk_type": "figure_ref",
    },
    {
        "query": "перечисление дефис строчная буква",
        "expected_rule_ids": ["4.1.7"],
        "chunk_type": "text",
    },
]

MODEL_NAME = "cointegrated/LaBSE-en-ru"


# ── Стратегии разбиения ───────────────────────────────────────────────────────

def strategy_per_rule(rules: list[dict]) -> list[dict]:
    """Стратегия 1: каждое правило — отдельный чанк (текущий подход)."""
    chunks = []
    for rule in rules:
        chunks.append({
            "chunk_text": f"{rule.get('rule_id', '')} | {rule.get('text', '')} | {rule.get('gost', '')}",
            "rules": [rule],
            "chunk_id": rule.get("rule_id", str(rule.get("id", ""))),
        })
    return chunks


def strategy_per_section(rules: list[dict]) -> list[dict]:
    """Стратегия 2: правила одного раздела объединяются в один чанк."""
    sections: dict[str, list] = {}
    for rule in rules:
        # Берём первые два уровня номера (4.1.2 → 4.1)
        rule_id = rule.get("rule_id", "")
        parts = rule_id.split(".")
        section_key = ".".join(parts[:2]) if len(parts) >= 2 else rule_id

        if section_key not in sections:
            sections[section_key] = []
        sections[section_key].append(rule)

    chunks = []
    for section_key, section_rules in sections.items():
        combined_text = " | ".join(
            f"{r.get('rule_id', '')} {r.get('text', '')}"
            for r in section_rules
        )
        chunks.append({
            "chunk_text": combined_text,
            "rules": section_rules,
            "chunk_id": f"section_{section_key}",
        })
    return chunks


def strategy_with_overlap(rules: list[dict], overlap: int = 1) -> list[dict]:
    """Стратегия 3: каждое правило + соседние (overlap=1 → берём соседа)."""
    chunks = []
    for i, rule in enumerate(rules):
        # Берём соседей с обеих сторон
        start = max(0, i - overlap)
        end = min(len(rules), i + overlap + 1)
        window = rules[start:end]

        combined_text = " | ".join(
            f"{r.get('rule_id', '')} {r.get('text', '')}"
            for r in window
        )
        chunks.append({
            "chunk_text": combined_text,
            "rules": [rule],  # основное правило — текущее
            "chunk_id": rule.get("rule_id", str(rule.get("id", ""))),
        })
    return chunks


# ── Построение индекса из чанков ─────────────────────────────────────────────

def build_index_from_chunks(
    chunks: list[dict],
    model: SentenceTransformer,
) -> faiss.Index:
    texts = [c["chunk_text"] for c in chunks]
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


# ── Метрика precision@k ───────────────────────────────────────────────────────

def precision_at_k(
    query: str,
    expected_ids: list[str],
    chunks: list[dict],
    index: faiss.Index,
    model: SentenceTransformer,
    k: int = 5,
) -> float:
    query_vec = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    _, indices = index.search(query_vec, k)

    retrieved_rule_ids = []
    for idx in indices[0]:
        if idx < 0:
            continue
        for rule in chunks[idx]["rules"]:
            retrieved_rule_ids.append(rule.get("rule_id", ""))

    hits = sum(1 for eid in expected_ids if eid in retrieved_rule_ids)
    return hits / len(expected_ids) if expected_ids else 0.0


# ── Основной цикл ─────────────────────────────────────────────────────────────

def run_chunking_experiments():
    rules_path = Path("data/gost_2_105_95_rules.json")
    if not rules_path.exists():
        print(f"❌ Файл правил не найден: {rules_path}")
        return

    with open(rules_path, encoding="utf-8") as f:
        rules = json.load(f)

    print(f"Загружено {len(rules)} правил")
    print("Загрузка модели...")
    model = SentenceTransformer(MODEL_NAME, device="cpu")

    strategies = {
        "per_rule":     strategy_per_rule(rules),
        "per_section":  strategy_per_section(rules),
        "with_overlap": strategy_with_overlap(rules, overlap=1),
    }

    all_results = []

    for strategy_name, chunks in strategies.items():
        print(f"\n{'='*50}")
        print(f"Стратегия: {strategy_name} ({len(chunks)} чанков)")
        print(f"{'='*50}")

        index = build_index_from_chunks(chunks, model)

        precisions = []
        for test in TEST_QUERIES:
            p_at_k = precision_at_k(
                query=test["query"],
                expected_ids=test["expected_rule_ids"],
                chunks=chunks,
                index=index,
                model=model,
                k=5,
            )
            precisions.append(p_at_k)
            mark = "✅" if p_at_k > 0 else "❌"
            print(f"  {mark} [{test['chunk_type']}] p@5={p_at_k:.2f} | {test['query'][:50]}")

        avg_precision = round(sum(precisions) / len(precisions), 3)
        print(f"\n  Средний precision@5: {avg_precision}")

        all_results.append({
            "strategy": strategy_name,
            "num_chunks": len(chunks),
            "avg_precision_at_5": avg_precision,
            "per_query": [
                {
                    "query": t["query"],
                    "expected": t["expected_rule_ids"],
                    "precision_at_5": round(p, 3),
                }
                for t, p in zip(TEST_QUERIES, precisions)
            ],
        })

    # ── Итоговая таблица ─────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ — Эксперимент 3: стратегии чанкинга")
    print(f"{'='*50}")
    print(f"{'Стратегия':<20} {'Чанков':>7} {'P@5':>8}")
    print("-" * 40)

    best = max(all_results, key=lambda x: x["avg_precision_at_5"])

    for r in all_results:
        marker = " ← лучшая" if r["strategy"] == best["strategy"] else ""
        print(f"{r['strategy']:<20} {r['num_chunks']:>7} "
              f"{r['avg_precision_at_5']:>8}{marker}")

    # ── Сохранение ───────────────────────────────────────────────────────────
    output = {
        "experiment_name": "chunking_strategy_experiment",
        "model": MODEL_NAME,
        "metric": "precision@5",
        "best_strategy": best["strategy"],
        "best_avg_precision": best["avg_precision_at_5"],
        "all_experiments": all_results,
        "conclusion": (
            f"Лучшая стратегия разбиения: {best['strategy']} "
            f"(precision@5={best['avg_precision_at_5']})"
        ),
    }

    output_path = Path("data/rag_experiment3_chunking.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Результаты сохранены в {output_path}")
    print(f"🏆 Лучшая стратегия: {best['strategy']} "
          f"(precision@5={best['avg_precision_at_5']})")

    return output


if __name__ == "__main__":
    run_chunking_experiments()
