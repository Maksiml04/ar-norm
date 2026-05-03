"""
Эксперименты с параметрами RAG-системы AI Нормконтролер.
 
Варьируем:
- top_k: количество извлекаемых правил (3, 5, 10)
- chunk_size: размер чанка (300, 500, 800)
- prompt_style: структура промпта (strict, detailed, minimal)
 
Запуск:
    PYTHONPATH=. OPENROUTER_API_KEY=... python rag_experiments.py
"""
 
from __future__ import annotations
 
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any
 
from src.main import AINormkontroler
from src.retriever import GOSTRetriever
 
# ── Тестовые чанки с заранее известными нарушениями ─────────────────────────
 
TEST_CHUNKS = [
    {
        "id": "test_header_1",
        "text": "1.1 введение.",          # нарушение: строчная буква + точка в конце
        "chunk_type": "section_header",
        "location": {"page": 1},
        "metadata": {"is_centered": False, "is_bold": False, "font_size": 14.0},
        "expected_violation": True,
        "expected_rule": "4.1.9",
        "description": "Заголовок со строчной буквы и точкой в конце"
    },
    {
        "id": "test_header_2",
        "text": "2. Основная часть",      # правильный заголовок
        "chunk_type": "section_header",
        "location": {"page": 2},
        "metadata": {"is_centered": False, "is_bold": True, "font_size": 14.0},
        "expected_violation": False,
        "expected_rule": None,
        "description": "Корректный заголовок"
    },
    {
        "id": "test_table_1",
        "text": "таблица 1 результаты",   # нарушение: строчная буква
        "chunk_type": "table_ref",
        "location": {"page": 3},
        "metadata": {},
        "expected_violation": True,
        "expected_rule": "4.2.1",
        "description": "Заголовок таблицы со строчной буквы"
    },
    {
        "id": "test_text_1",
        "text": "Результаты испытаний приведены в табл. 1 и рис. 2.",
        "chunk_type": "text",
        "location": {"page": 4},
        "metadata": {},
        "expected_violation": False,
        "expected_rule": None,
        "description": "Корректный текст со ссылками"
    },
    {
        "id": "test_figure_1",
        "text": "Рисунок 1 схема установки",  # нарушение: нет тире перед названием
        "chunk_type": "figure_ref",
        "location": {"page": 5},
        "metadata": {},
        "expected_violation": True,
        "expected_rule": "4.3.1",
        "description": "Подпись рисунка без тире"
    },
]
 
# ── Конфигурации экспериментов ───────────────────────────────────────────────
 
EXPERIMENTS = [
    # Варьируем top_k
    {"name": "top_k=3",  "top_k": 3,  "prompt_style": "strict"},
    {"name": "top_k=5",  "top_k": 5,  "prompt_style": "strict"},
    {"name": "top_k=10", "top_k": 10, "prompt_style": "strict"},
 
    # Варьируем структуру промпта
    {"name": "prompt_detailed", "top_k": 5, "prompt_style": "detailed"},
    {"name": "prompt_minimal",  "top_k": 5, "prompt_style": "minimal"},
]
 
PROMPT_STYLES = {
    "strict": (
        "Ты эксперт-нормоконтролёр. Проверяй ТОЛЬКО нарушения из списка правил. "
        "Не выдумывай правила. Отвечай строго JSON."
    ),
    "detailed": (
        "Ты опытный инженер-нормоконтролёр технической документации. "
        "Твоя задача — найти нарушения ГОСТ в тексте. "
        "Проверяй каждое правило внимательно. Отвечай в формате JSON."
    ),
    "minimal": (
        "Проверь текст на нарушения ГОСТ. JSON ответ."
    ),
}
 
 
# ── Метрики ──────────────────────────────────────────────────────────────────
 
def compute_metrics(results: list[dict], chunks: list[dict]) -> dict:
    """Считает precision, recall, F1 по результатам."""
    tp = fp = fn = 0
 
    for result, chunk in zip(results, chunks):
        predicted = result.get("has_violation", False)
        expected = chunk["expected_violation"]
 
        if predicted and expected:
            tp += 1
        elif predicted and not expected:
            fp += 1
        elif not predicted and expected:
            fn += 1
 
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
 
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(precision, 3),
        "recall":    round(recall, 3),
        "f1":        round(f1, 3),
    }
 
 
# ── Патч промпта ─────────────────────────────────────────────────────────────
 
def patch_prompt_style(normkontroler: AINormkontroler, style: str):
    """Подменяет system-промпт в LLMAnalyzer."""
    if normkontroler.analyzer:
        normkontroler.analyzer._system_prompt_override = PROMPT_STYLES[style]
 
 
# ── Основной цикл ────────────────────────────────────────────────────────────
 
def run_experiments():
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ OPENROUTER_API_KEY не задан — запуск без LLM (метрики будут нулевые)")
 
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
 
    for exp in EXPERIMENTS:
        print(f"\n{'='*50}")
        print(f"Эксперимент: {exp['name']}")
        print(f"  top_k={exp['top_k']}, prompt={exp['prompt_style']}")
        print(f"{'='*50}")
 
        # Патчим top_k в методе analyze_chunk
        original_search = normkontroler.search_rules
 
        def patched_search(chunk_text, chunk_type="text", context_query="", top_k=5):
            return original_search(
                chunk_text=chunk_text,
                chunk_type=chunk_type,
                context_query=context_query,
                top_k=exp["top_k"],
            )
 
        normkontroler.search_rules = patched_search
 
        # Прогоняем тестовые чанки
        chunk_results = []
        latencies = []
 
        for chunk in TEST_CHUNKS:
            print(f"  [{chunk['id']}] {chunk['description'][:40]}...")
            start = time.time()
 
            result = normkontroler.analyze_chunk(chunk)
            latency = round(time.time() - start, 2)
            latencies.append(latency)
 
            predicted = result.get("has_violation", False)
            expected  = chunk["expected_violation"]
            correct   = "✅" if predicted == expected else "❌"
 
            print(f"    {correct} predicted={predicted}, expected={expected}, "
                  f"latency={latency}s")
 
            chunk_results.append(result)
 
        # Метрики
        metrics = compute_metrics(chunk_results, TEST_CHUNKS)
        avg_latency = round(sum(latencies) / len(latencies), 2)
 
        print(f"\n  Метрики: precision={metrics['precision']} "
              f"recall={metrics['recall']} F1={metrics['f1']}")
        print(f"  Среднее время: {avg_latency}s")
 
        all_results.append({
            "experiment": exp["name"],
            "config": exp,
            "metrics": metrics,
            "avg_latency_sec": avg_latency,
            "chunk_results": chunk_results,
        })
 
        # Восстанавливаем оригинальный метод
        normkontroler.search_rules = original_search
 
    # ── Итоговый отчёт ───────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
    print(f"{'='*50}")
    print(f"{'Эксперимент':<20} {'P':>6} {'R':>6} {'F1':>6} {'Время':>8}")
    print("-" * 50)
 
    best = max(all_results, key=lambda x: x["metrics"]["f1"])
 
    for r in all_results:
        m = r["metrics"]
        marker = " ← лучший" if r["experiment"] == best["experiment"] else ""
        print(f"{r['experiment']:<20} {m['precision']:>6} {m['recall']:>6} "
              f"{m['f1']:>6} {r['avg_latency_sec']:>7}s{marker}")
 
    # Сохраняем в JSON
    output = {
        "timestamp": datetime.now().isoformat(),
        "best_config": best["config"],
        "best_metrics": best["metrics"],
        "all_experiments": [
            {k: v for k, v in r.items() if k != "chunk_results"}
            for r in all_results
        ],
    }
 
    output_path = "data/rag_experiment_results.json"
    Path("data").mkdir(exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
 
    print(f"\n✅ Результаты сохранены в {output_path}")
    print(f"🏆 Лучший конфиг: {best['config']['name']} "
          f"(F1={best['metrics']['f1']})")
 
    return output
 
 
if __name__ == "__main__":
    run_experiments()