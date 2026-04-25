"""

Модуль построения векторного индекса для AI-нормконтролера.

v2: Сохраняет имя модели вместе с индексом для корректной загрузки в retriever.

"""

from __future__ import annotations



import json

import pickle

from pathlib import Path

from typing import Any



from sentence_transformers import SentenceTransformer

import faiss

import numpy as np



from .logging_config import get_logger



logger = get_logger(__name__)





class GOSTIndexBuilder:

    """

    Строит FAISS-индекс из правил ГОСТ.



    Изменения v2:

    - Сохраняет имя модели в meta.pkl, чтобы retriever мог загрузить ту же модель.

    - Строит текст для эмбеддинга из всех значимых полей правила, а не только text.

    """



    def __init__(

        self,

        model_name: str = "cointegrated/LaBSE-en-ru",

        device: str = "cpu"

    ) -> None:

        self.model_name = model_name

        self.device = device

        self.rules: list[dict[str, Any]] = []

        self.model: SentenceTransformer | None = None

        self.index: faiss.Index | None = None

        logger.info(f"GOSTIndexBuilder init (model={model_name}, device={device})")



    def load_rules(self, rules_path: str) -> None:

        logger.info(f"Загрузка правил из: {rules_path}")

        with open(rules_path, "r", encoding="utf-8") as f:

            self.rules = json.load(f)



        if not self.rules:

            raise ValueError("Список правил пуст")



        logger.info(f"Загружено {len(self.rules)} правил")

        logger.info(f"Пример правила: {self.rules[0]}")



    @staticmethod

    def _rule_to_text(rule: dict[str, Any]) -> str:

        """

        Формирует строку для эмбеддинга из всех полей правила.



        Чем богаче текст — тем точнее поиск. Используем:

          gost_id   — номер пункта ГОСТа (например '2.105-95 п.3.1')

          text      — формулировка требования

          keywords  — ключевые слова (если есть в JSON)

          applies_to — типы элементов (если есть)

        """

        parts: list[str] = []



        if rule_id := rule.get("id"):

            parts.append(str(rule_id))

        if text := rule.get("text"):

            parts.append(text)

        if keywords := rule.get("keywords"):

            kw_str = ", ".join(keywords) if isinstance(keywords, list) else str(keywords)

            parts.append(kw_str)

        if applies_to := rule.get("applies_to"):

            at_str = ", ".join(applies_to) if isinstance(applies_to, list) else str(applies_to)

            parts.append(at_str)

        if gost := rule.get("gost"):

            parts.append(str(gost))  # ← ИСПРАВЛЕНО: было isinstance(w, list)

        if chunk_type := rule.get("chunk_type"):

            parts.append(str(chunk_type))  # ← ИСПРАВЛЕНО: для строк не нужен join

        if category := rule.get("category"):

            parts.append(str(category))    # ← ИСПРАВЛЕНО: аналогично



        return " | ".join(parts)



    def build_index(self) -> None:

        if not self.rules:

            raise ValueError("Сначала загрузите правила через load_rules()")



        logger.info(f"Загрузка модели: {self.model_name}")

        self.model = SentenceTransformer(self.model_name, device=self.device)



        texts = [self._rule_to_text(r) for r in self.rules]

        logger.info(f"Генерация эмбеддингов для {len(texts)} правил...")



        embeddings: np.ndarray = self.model.encode(

            texts,

            batch_size=32,

            show_progress_bar=True,

            convert_to_numpy=True,

            normalize_embeddings=True,  # нормализация → IP = косинусное сходство

        )



        dim = embeddings.shape[1]

        logger.info(f"Размерность эмбеддингов: {dim}")



        # IndexFlatIP + нормализованные векторы = косинусное сходство

        self.index = faiss.IndexFlatIP(dim)

        self.index.add(embeddings.astype("float32"))

        logger.info(f"Индекс построен. Векторов: {self.index.ntotal}")



    def save(self, save_dir: str) -> None:

        if self.index is None:

            raise ValueError("Сначала постройте индекс через build_index()")



        save_path = Path(save_dir)

        save_path.mkdir(parents=True, exist_ok=True)



        index_path = save_path / "gost.index"

        meta_path = save_path / "gost_rules_meta.pkl"



        faiss.write_index(self.index, str(index_path))

        logger.info(f"Индекс сохранён: {index_path}")



        # ── КЛЮЧЕВОЕ ИЗМЕНЕНИЕ ──────────────────────────────────────────────

        # Сохраняем правила И имя модели в одном pkl.

        # Это позволяет GOSTRetriever загружать правильную модель автоматически.

        meta = {

            "rules": self.rules,

            "model_name": self.model_name,

        }

        with open(meta_path, "wb") as f:

            pickle.dump(meta, f)

        logger.info(f"Метаданные сохранены: {meta_path}")





def main():

    base_dir = Path(__file__).resolve().parent.parent

    data_dir = base_dir / "data"

    rules_path = data_dir / "gost_2_105_95_rules.json"



    if not rules_path.exists():

        logger.error(f"Файл правил не найден: {rules_path}")

        return



    builder = GOSTIndexBuilder(device="cpu")

    builder.load_rules(str(rules_path))

    builder.build_index()

    builder.save(str(data_dir))



    logger.info("✅ Индекс построен и сохранён")





if __name__ == "__main__":

    main()