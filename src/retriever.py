"""

GOSTRetriever v2 — улучшенный семантический поиск правил ГОСТ.



Ключевые улучшения по сравнению с v1:

1. Загружает модель SentenceTransformer из meta.pkl — больше никаких внешних embedding_func.

2. Строит обогащённый запрос: сырой текст чанка + context_query из PDF-парсера.

3. Мягкий скоринг по типу (вместо жёсткого фильтра).

4. Возвращает score для каждого правила.

5. Поддержка batch-поиска для нескольких чанков сразу.

"""

from __future__ import annotations



import pickle

from pathlib import Path

from typing import Any



import faiss

import numpy as np

from sentence_transformers import SentenceTransformer



from .logging_config import get_logger



logger = get_logger(__name__)





# Бонус к score для правил, чей applies_to совпадает с chunk_type чанка

TYPE_MATCH_BONUS = 0.1  # ← убран ведущий ноль



# context_query по умолчанию для каждого типа чанка

DEFAULT_CONTEXT_QUERIES: dict[str, str] = {

    "title_page":      "титульный лист оформление ГОСТ 7.32 отчёт",

    "toc":             "содержание оглавление нумерация страниц ГОСТ 7.32",

    "section_header":  "заголовок раздела подраздела оформление нумерация ГОСТ 2.105",

    "text":            "текст абзац отступ интервал шрифт ГОСТ 2.105",

    "figure_ref":      "рисунок подпись нумерация ГОСТ 2.105",

    "table_ref":       "таблица заголовок нумерация ГОСТ 2.105",

    "bibliography":    "список литературы источники оформление ГОСТ 7.32",

}





class GOSTRetriever:

    """

    Семантический поиск правил ГОСТ по тексту чанка документа.

    """



    def __init__(

        self,

        index: faiss.Index,

        rules: list[dict[str, Any]],

        model: SentenceTransformer,

    ) -> None:

        self.index = index

        self.rules = rules

        self.model = model

        logger.info(

            f"GOSTRetriever инициализирован. "

            f"Правил: {len(rules)}, векторов: {index.ntotal}"

        )



    # ── Фабричный метод ─────────────────────────────────────────────────────



    @classmethod

    def load(

        cls,

        index_path: str,

        meta_path: str,

        device: str = "cpu",

    ) -> "GOSTRetriever":

        """

        Загружает retriever из файлов, созданных GOSTIndexBuilder.save().

        """

        index = faiss.read_index(index_path)

        logger.info(f"FAISS-индекс загружен. Векторов: {index.ntotal}")



        with open(meta_path, "rb") as f:

            meta = pickle.load(f)



        if isinstance(meta, list):

            logger.warning(

                "Обнаружен старый формат meta (list). "

                "model_name будет взят по умолчанию."

            )

            rules = meta

            model_name = "cointegrated/LaBSE-en-ru"

        else:

            rules = meta.get("rules", [])

            model_name = meta.get("model_name", "cointegrated/LaBSE-en-ru")



        logger.info(f"Загружено {len(rules)} правил. Модель: {model_name}")

        model = SentenceTransformer(model_name, device=device)



        return cls(index=index, rules=rules, model=model)



    # ── Внутренние методы ───────────────────────────────────────────────────



    def _build_query(

        self,

        chunk_text: str,

        chunk_type: str,

        context_query: str = "",

    ) -> str:

        """

        Строит обогащённый запрос для эмбеддинга.

        Берёт context_query + первые 300 символов текста.

        """

        ctx = context_query or DEFAULT_CONTEXT_QUERIES.get(

            chunk_type, "ГОСТ требования"

        )

        snippet = chunk_text[:300].strip()

        query = f"{ctx} | {snippet}"

        logger.debug(f"[{chunk_type}] query: {query[:120]}...")

        return query



    def _embed(self, texts: list[str]) -> np.ndarray:

        """Кодирует тексты в нормализованные векторы."""

        vecs = self.model.encode(

            texts,

            convert_to_numpy=True,

            normalize_embeddings=True,

            show_progress_bar=False,

        )

        return vecs.astype("float32")



    @staticmethod

    def _soft_type_score(rule: dict[str, Any], chunk_type: str) -> float:

        """

        Возвращает бонус к similarity score если правило подходит по типу.

        """

        applies_to: list[str] | str | None = rule.get("applies_to")

        if not applies_to:

            return 0.0



        applies_list = (

            applies_to if isinstance(applies_to, list) else [applies_to]

        )

        return TYPE_MATCH_BONUS if chunk_type in applies_list else 0.0



    # ── Публичный API ───────────────────────────────────────────────────────



    def search(

        self,

        chunk_text: str,

        chunk_type: str = "text",

        context_query: str = "",

        top_k: int = 5,

        search_pool: int = 30,

    ) -> list[dict[str, Any]]:

        """

        Ищет top_k наиболее релевантных правил для чанка.

        """

        if not self.rules or self.index is None:

            logger.warning("Retriever не готов: индекс или правила отсутствуют")

            return []



        query = self._build_query(chunk_text, chunk_type, context_query)

        query_vec = self._embed([query])



        k = min(search_pool, self.index.ntotal)

        scores, indices = self.index.search(query_vec, k)

        scores, indices = scores[0], indices[0]



        candidates: list[dict[str, Any]] = []

        for raw_score, idx in zip(scores, indices):

            if idx < 0:

                continue

            rule = self.rules[idx]

            bonus = self._soft_type_score(rule, chunk_type)

            final_score = float(raw_score) + bonus



            enriched: dict[str, Any] = {

                **rule,

                "_score": round(final_score, 4),

                "_type_bonus": bonus,

            }

            candidates.append(enriched)



        candidates.sort(key=lambda r: r["_score"], reverse=True)

        result = candidates[:top_k]



        logger.info(

            f"search [{chunk_type}] → {len(result)} правил. "

            f"Scores: {[r['_score'] for r in result]}"

        )

        return result



    def search_batch(

        self,

        chunks: list[dict[str, Any]],

        top_k: int = 5,

        search_pool: int = 30,

    ) -> list[list[dict[str, Any]]]:

        """

        Батч-поиск для списка чанков. Эффективнее, чем search() в цикле.

        """

        if not chunks:

            return []



        queries = [

            self._build_query(

                c.get("text", ""),

                c.get("chunk_type", "text"),

                c.get("context_query", ""),

            )

            for c in chunks

        ]



        query_vecs = self._embed(queries)

        k = min(search_pool, self.index.ntotal)

        all_scores, all_indices = self.index.search(query_vecs, k)



        results: list[list[dict[str, Any]]] = []

        for i, chunk in enumerate(chunks):

            chunk_type = chunk.get("chunk_type", "text")

            scores, indices = all_scores[i], all_indices[i]



            candidates = []

            for raw_score, idx in zip(scores, indices):

                if idx < 0:

                    continue

                rule = self.rules[idx]

                bonus = self._soft_type_score(rule, chunk_type)

                enriched = {

                    **rule,

                    "_score": round(float(raw_score) + bonus, 4),

                    "_type_bonus": bonus,

                }

                candidates.append(enriched)



            candidates.sort(key=lambda r: r["_score"], reverse=True)

            results.append(candidates[:top_k])



        return results