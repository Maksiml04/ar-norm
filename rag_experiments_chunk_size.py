"""
Эксперимент 2: влияние размера чанка (chunk_size) на качество RAG.

Варьируем chunk_size: 300, 500, 800 символов.
Результаты сохраняются отдельно от первого эксперимента.

Запуск:
    PYTHONPATH=. OPENROUTER_API_KEY=... python3 rag_experiments_chunk_size.py
"""

from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from src.main import AINormkontroler

# ── Тестовый документ (длинный текст для разных размеров чанков) ─────────────

FULL_TEXT = """
1.1 введение.
Данный документ описывает результаты испытаний аккумуляторного модуля.
Испытания проводились в соответствии с требованиями ГОСТ 2.105-95.
Все измерения выполнены при нормальных климатических условиях (НКУ).

таблица 1 результаты измерений напряжения
Измерения проводились с интервалом 10 минут на протяжении 2 часов.
Результаты приведены в таблице 1 и на рисунке 1.

2. Методика испытаний
Испытания проводились согласно методике, утверждённой техническим директором.
Погрешность измерений не превышала 0.5%.

Рисунок 1 схема установки для испытаний
Установка включает источник питания, измерительный блок и блок управления.

3. Результаты и выводы
По результатам испытаний установлено соответствие изделия требованиям ТУ.
Все параметры находятся в допустимых пределах.
Изделие рекомендуется к серийному производству.
"""

# ── Параметры экспериментов ──────────────────────────────────────────────────

CHUNK_SIZES = [300, 500, 800]

# Ожидаемые нарушения в тексте
EXPECTED_VIOLATIONS = {
    "введение":     True,   # строчная буква в заголовке
    "таблица 1":    True,   # строчная буква
    "Рисунок 1 схема": True, # нет тире перед названием
    "Методика":     False,  # корректный заголовок
    "Результаты":   False,  # корректный заголовок
}


def split_into_chunks(text: str, chunk_size: int) -> list[dict]:
    """Разбивает текст на чанки заданного размера."""
    lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
    chunks = []
    current = []
    current_len = 0
    chunk_id = 0

    for line in lines:
        if current_len + len(line) > chunk_size and current:
            chunk_text = ' '.join(current)
            chunks.append({
                "id": f"chunk_{chunk_id:03d}",
                "text": chunk_text,
                "chunk_type": detect_type(chunk_text),
                "location": {"page": 1},
                "metadata": {},
            })
            chunk_id += 1
            current = [line]
            current_len = len(line)
        else:
            current.append(line)
            current_len += len(line)

    if current:
        chunk_text = ' '.join(current)
        chunks.append({
            "id": f"chunk_{chunk_id:03d}",
            "text": chunk_text,
            "chunk_type": detect_type(chunk_text),
            "location": {"page": 1},
            "metadata": {},
        })

    return chunks


def detect_type(text: str) -> str:
    """Определяет тип чанка по тексту."""
    if re.match(r'^\d+\.\d+\s+\S', text) or re.match(r'^\d+\.\s+\S', text):
        return "section_header"
    if re.match(r'^[Тт]аблица\s+\d+', text):
        return "table_ref"
    if re.match(r'^[Рр]исунок\s+\d+', text):
        return "figure_ref"
    return "text"


def get_expected(chunk_text: str) -> bool:
    """Определяет ожидаемое нарушение для чанка."""
    for keyword, expected in EXPECTED_VIOLATIONS.items():
        if keyword.lower() in chunk_text.lower():
            return expected
    return False


def compute_metrics(results: list[dict], chunks: list[dict]) -> dict:
    tp = fp = fn = tn = 0
    for result, chunk in zip(results, chunks):
        predicted = result.get("has_violation", False)
        expected = get_expected(chunk["text"])
        if predicted and expected:
            tp += 1
        elif predicted and not expected:
            fp += 1
        elif not predicted and expected:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(precision, 3),
        "recall":    round(recall, 3),
        "f1":        round(f1, 3),
    }


def run_chunk_size_experiments():
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ OPENROUTER_API_KEY не задан")
        return

    index_path = "data/gost.index"
    meta_path  = "data/gost_rules_meta.pkl"

    if not Path(index_path).exists():
        print(f"❌ Индекс не найден: {index_path}")
        return

    print("Загрузка системы...")
    normkontroler = AINormkontroler.load_from_index(
        index_path=index_path,
        meta_path=meta_path,
        api_key=api_key,
    )

    all_results = []

    for chunk_size in CHUNK_SIZES:
        print(f"\n{'='*50}")
        print(f"Эксперимент: chunk_size={chunk_size}")
        print(f"{'='*50}")

        chunks = split_into_chunks(FULL_TEXT, chunk_size)
        print(f"  Получено чанков: {len(chunks)}")

        chunk_results = []
        latencies = []

        for chunk in chunks:
            print(f"  [{chunk['id']}] {chunk['text'][:40]}...")
            start = time.time()
            result = normkontroler.analyze_chunk(chunk)
            latency = round(time.time() - start, 2)
            latencies.append(latency)

            predicted = result.get("has_violation", False)
            expected  = get_expected(chunk["text"])
            correct   = "✅" if predicted == expected else "❌"
            print(f"    {correct} predicted={predicted}, expected={expected}, latency={latency}s")

            chunk_results.append(result)

        metrics = compute_metrics(chunk_results, chunks)
        avg_latency = round(sum(latencies) / len(latencies), 2) if latencies else 0

        print(f"\n  Метрики: precision={metrics['precision']} "
              f"recall={metrics['recall']} F1={metrics['f1']}")
        print(f"  Чанков: {len(chunks)}, среднее время: {avg_latency}s")

        all_results.append({
            "experiment": f"chunk_size={chunk_size}",
            "chunk_size": chunk_size,
            "num_chunks": len(chunks),
            "metrics": metrics,
            "avg_latency_sec": avg_latency,
        })

    # ── Итоговая таблица ─────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ — Эксперимент 2: chunk_size")
    print(f"{'='*50}")
    print(f"{'chunk_size':<15} {'Чанков':>7} {'P':>6} {'R':>6} {'F1':>6} {'Время':>8}")
    print("-" * 50)

    best = max(all_results, key=lambda x: x["metrics"]["f1"])

    for r in all_results:
        m = r["metrics"]
        marker = " ← лучший" if r["chunk_size"] == best["chunk_size"] else ""
        print(f"{r['experiment']:<15} {r['num_chunks']:>7} {m['precision']:>6} "
              f"{m['recall']:>6} {m['f1']:>6} {r['avg_latency_sec']:>7}s{marker}")

    # ── Сохранение ───────────────────────────────────────────────────────────
    output = {
        "experiment_name": "chunk_size_experiment",
        "timestamp": datetime.now().isoformat(),
        "best_chunk_size": best["chunk_size"],
        "best_metrics": best["metrics"],
        "all_experiments": all_results,
        "conclusion": (
            f"Оптимальный размер чанка: {best['chunk_size']} символов "
            f"(F1={best['metrics']['f1']})"
        ),
    }

    output_path = "data/rag_experiment2_chunk_size.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Результаты сохранены в {output_path}")
    print(f"🏆 Лучший chunk_size: {best['chunk_size']} (F1={best['metrics']['f1']})")

    return output


if __name__ == "__main__":
    run_chunk_size_experiments()
