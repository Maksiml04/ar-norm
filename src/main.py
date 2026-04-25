"""

Основной модуль системы AI Нормконтролер v2.

Теперь search_rules работает через GOSTRetriever, а не заглушку.

"""

from __future__ import annotations



import os

from typing import Any, Optional



try:

    import faiss

except ImportError:

    faiss = None



from src.logging_config import get_logger

from src.llm_analyzer import LLMAnalyzer

from src.retriever import GOSTRetriever



logger = get_logger(__name__)





class AINormkontroler:

    """Основной класс системы нормоконтроля."""



    def __init__(

        self,

        rules: list[dict[str, Any]],

        retriever: Optional[GOSTRetriever] = None,

        analyzer: Optional[LLMAnalyzer] = None,

    ) -> None:

        self.rules = rules

        self.retriever = retriever

        self.analyzer = analyzer



        logger.info(f"AINormkontroler инициализирован. Правил: {len(rules)}")

        logger.info(f"Retriever: {'✅' if retriever else '❌ не подключён'}")

        logger.info(f"LLM анализатор: {'✅' if analyzer else '❌ не подключён'}")



    # ── Фабричный метод ─────────────────────────────────────────────────────



    @classmethod

    def load_from_index(

        cls,

        index_path: str,

        meta_path: str,

        api_key: Optional[str] = None,

        device: str = "cpu",

    ) -> "AINormkontroler":

        """

        Загружает систему из файлов индекса.



        Args:

            index_path: путь к gost.index

            meta_path:  путь к gost_rules_meta.pkl

            api_key:    ключ для LLM (OpenRouter / OpenAI)

            device:     'cpu' или 'cuda'

        """

        if faiss is None:

            raise ImportError("Установите faiss-cpu: pip install faiss-cpu")



        # 1. GOSTRetriever загружает индекс, правила и модель в одном месте

        logger.info("Загрузка GOSTRetriever...")

        retriever = GOSTRetriever.load(index_path, meta_path, device=device)

        rules = retriever.rules



        # 2. LLM анализатор (опционально)

        analyzer: Optional[LLMAnalyzer] = None

        if api_key:

            try:

                analyzer = LLMAnalyzer(api_key=api_key)

                logger.info("✅ LLM анализатор инициализирован.")

            except Exception as e:

                logger.error(f"Не удалось создать LLMAnalyzer: {e}")

        else:

            logger.warning("API ключ не предоставлен. LLM анализатор отключён.")



        return cls(rules=rules, retriever=retriever, analyzer=analyzer)



    # ── Поиск правил ────────────────────────────────────────────────────────



    def search_rules(

        self,

        chunk_text: str,

        chunk_type: str = "text",

        context_query: str = "",

        top_k: int = 5,

    ) -> list[dict[str, Any]]:

        if self.retriever is None:

            logger.warning("Retriever не загружен — возвращаю первые правила как fallback")

            return self.rules[:top_k]



        return self.retriever.search(

            chunk_text=chunk_text,

            chunk_type=chunk_type,

            context_query=context_query,

            top_k=top_k,

        )



    # ── Анализ чанка ────────────────────────────────────────────────────────



    def analyze_chunk(self, chunk: dict[str, Any]) -> dict[str, Any]:

        """

        Анализирует чанк документа на нарушения ГОСТ.



        Args:

            chunk: словарь с ключами 'id', 'text', 'chunk_type', 'location',

                   опционально 'context_query' (из PDFChunker).

        Returns:

            Словарь с полями has_violation, violations, confidence, …

        """

        chunk_id = chunk.get("id", "unknown")

        chunk_text = chunk.get("text", "")

        chunk_type = chunk.get("chunk_type", "text")

        context_query = chunk.get("context_query", "")



        # ── 1. Семантический поиск релевантных правил ────────────────────────

        relevant_rules = self.search_rules(

            chunk_text=chunk_text,

            chunk_type=chunk_type,

            context_query=context_query,

            top_k=5,

        )

        logger.debug(

            f"[{chunk_id}] Найдено {len(relevant_rules)} правил: "

            f"{[r.get('gost_id') for r in relevant_rules]}"

        )



        # ── 2. LLM анализ ────────────────────────────────────────────────────

        if not self.analyzer:

            return {

                "chunk_id": chunk_id,
                "text": chunk_text,  # ← ДОБАВЛЯЕМ
                "has_violation": False,
                "violations": [],

                "is_correct": True,

                "confidence": 0.0,

                "applied_rules": [r.get("gost_id") for r in relevant_rules],

                "warning": "LLM анализатор не доступен.",

            }



        try:

            # ← ИСПРАВЛЕНО: передаём полный словарь chunk, а не строку chunk_text

            result = self.analyzer.analyze_chunk(

                chunk=chunk,

                rules=relevant_rules,

            )

            result["chunk_id"] = chunk_id
            result["location"] = chunk.get("location", {})
            result["text"] = chunk_text  # ← ДОБАВЛЯЕМ текст чанка в ответ
            result["applied_rules"] = [r.get("id") for r in relevant_rules]

            return result



        except Exception as e:

            logger.error(f"Ошибка анализа чанка {chunk_id}: {e}", exc_info=True)

            return {

                "chunk_id": chunk_id,

                "has_violation": False,

                "violations": [],

                "is_correct": True,

                "confidence": 0.0,

                "applied_rules": [],

                "error": str(e),

            }



    def analyze_chunks_batch(

        self, chunks: list[dict[str, Any]]

    ) -> list[dict[str, Any]]:

        """

        Анализирует список чанков, используя батч-поиск правил.

        """

        if not chunks:

            return []



        # Батч-поиск правил для всех чанков за один encode-вызов

        if self.retriever:

            all_rules = self.retriever.search_batch(chunks, top_k=5)

        else:

            all_rules = [self.rules[:5]] * len(chunks)



        results = []

        for chunk, relevant_rules in zip(chunks, all_rules):

            chunk_id = chunk.get("id", "unknown")



            if not self.analyzer:

                results.append({

                    "chunk_id": chunk_id,

                    "has_violation": False,

                    "violations": [],

                    "is_correct": True,

                    "confidence": 0.0,

                    "applied_rules": [r.get("gost_id") for r in relevant_rules],

                    "warning": "LLM анализатор не доступен.",

                })

                continue



            try:

                # ← ИСПРАВЛЕНО: передаём полный словарь chunk

                result = self.analyzer.analyze_chunk(

                    chunk=chunk,

                    rules=relevant_rules,

                )

                result["chunk_id"] = chunk_id

                result["location"] = chunk.get("location", {})

                result["applied_rules"] = [r.get("gost_id") for r in relevant_rules]

                results.append(result)

            except Exception as e:

                logger.error(f"Ошибка анализа {chunk_id}: {e}", exc_info=True)

                results.append({

                    "chunk_id": chunk_id,

                    "has_violation": False,

                    "violations": [],

                    "is_correct": True,

                    "confidence": 0.0,

                    "applied_rules": [],

                    "error": str(e),

                })



        return results