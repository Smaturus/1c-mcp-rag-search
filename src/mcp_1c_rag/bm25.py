"""BM25: токенизация, классический Okapi и глобальный инвертированный индекс.

Модуль используется в двух ролях:

1. ``BM25Okapi`` — счёт по небольшому переданному корпусу (кандидаты cosine),
   как в ``hybrid.py``: без изменений в поведении, обратная совместимость.
2. ``Bm25Index`` — глобальный лексический индекс по всем чанкам коллекции:
   строится один раз, сохраняется на диск, затем отвечает на запросы
   ``bm25_search(query, top_k)`` для RRF-гибрида.

Токенизатор: нижний регистр, латиница + кириллица, цифры и подчёркивания,
дефисные коды (``НФ-001``, ``НДС-18``) сохраняются как один токен.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Латинские/кириллические слова, цифры, подчёркивания; дефисные коды — целиком.
_TOKEN = re.compile(
    r"[0-9A-Za-zА-Яа-яЁё_]+(?:-[0-9A-Za-zА-Яа-яЁё_]+)*",
    re.UNICODE,
)


def tokenize(text: str) -> list[str]:
    """Токены в нижнем регистре; аббревиатуры и коды сохраняются целиком."""
    return [t.lower() for t in _TOKEN.findall(text or "")]


class BM25Okapi:
    """Okapi BM25 по маленькому корпусу (например, кандидаты cosine)."""

    def __init__(
        self,
        corpus_tokens: list[list[str]],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.corpus = corpus_tokens
        self.n_docs = len(corpus_tokens)
        self.doc_len = [len(doc) for doc in corpus_tokens]
        self.avgdl = (
            sum(self.doc_len) / self.n_docs if self.n_docs else 0.0
        )
        df: Counter[str] = Counter()
        for doc in corpus_tokens:
            for term in set(doc):
                df[term] += 1
        self.df = df
        self.idf = {
            term: math.log(1.0 + (self.n_docs - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def score(self, query_tokens: list[str], doc_index: int) -> float:
        """BM25-скор запроса против документа по индексу ``doc_index``."""
        if self.n_docs == 0 or self.avgdl <= 0:
            return 0.0
        doc = self.corpus[doc_index]
        tf = Counter(doc)
        dl = self.doc_len[doc_index]
        score = 0.0
        for term in query_tokens:
            if term not in tf:
                continue
            idf = self.idf.get(term, 0.0)
            freq = tf[term]
            denom = freq + self.k1 * (1.0 - self.b + self.b * dl / self.avgdl)
            score += idf * (freq * (self.k1 + 1.0)) / denom
        return score

    def scores(self, query: str) -> list[float]:
        """Скоры для всех документов корпуса (линейно по корпусу)."""
        q = tokenize(query)
        return [self.score(q, i) for i in range(self.n_docs)]


# --------------------------------------------------------------------------
# Глобальный инвертированный индекс
# --------------------------------------------------------------------------

INDEX_FILE = "index.npz"
META_FILE = "meta.json"


@dataclass
class Bm25Hit:
    """Результат лексического поиска."""

    doc_id: str
    score: float


class Bm25Index:
    """Глобальный BM25: словарь -> постинги (doc_id, tf), сохранение на диск.

    Хранение компактное: ``np.int32`` для индексов документов и ``np.uint16``
    для частот, поэтому индекс по 124k чанкам занимает десятки мегабайт.
    """

    def __init__(
        self,
        *,
        vocab: dict[str, int],
        postings_doc: np.ndarray,
        postings_tf: np.ndarray,
        offsets: np.ndarray,
        doc_len: np.ndarray,
        doc_ids: list[str],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.vocab = vocab
        self.postings_doc = postings_doc
        self.postings_tf = postings_tf
        self.offsets = offsets
        self.doc_len = doc_len
        self.doc_ids = doc_ids
        self.k1 = float(k1)
        self.b = float(b)
        self.n_docs = len(doc_ids)
        self.avgdl = float(doc_len.mean()) if self.n_docs else 0.0
        df = np.diff(offsets)  # число постингов на термин == df
        self.idf = np.log(1.0 + (self.n_docs - df + 0.5) / (df + 0.5))

    # ---- построение -----------------------------------------------------

    @classmethod
    def build(
        cls,
        docs: Iterable[tuple[str, str]],
        *,
        k1: float = 1.5,
        b: float = 0.75,
        batch_log: int = 20000,
    ) -> Bm25Index:
        """Построить индекс по парам ``(doc_id, text)``.

        Один проход: токены каждого документа считаются один раз, постинги
        накапливаются в ``array``-структурах с минимальным оверхедом.
        """
        term_ids: dict[str, int] = {}
        post_docs: list[list[int]] = []
        post_tf: list[list[int]] = []
        doc_len: list[int] = []
        doc_ids: list[str] = []

        for n, (doc_id, text) in enumerate(docs, start=1):
            tokens = tokenize(text)
            doc_len.append(len(tokens))
            doc_ids.append(doc_id)
            idx = n - 1
            for term, tf in Counter(tokens).items():
                tid = term_ids.get(term)
                if tid is None:
                    tid = len(post_docs)
                    term_ids[term] = tid
                    post_docs.append([])
                    post_tf.append([])
                post_docs[tid].append(idx)
                post_tf[tid].append(tf)
            if batch_log and n % batch_log == 0:
                print(f"  построено {n} документов, {len(term_ids)} термов", flush=True)

        if not doc_ids:
            raise ValueError("пустой корпус: нечего индексировать")

        lengths = np.fromiter((len(x) for x in post_docs), dtype=np.int64, count=len(post_docs))
        offsets = np.zeros(len(post_docs) + 1, dtype=np.int64)
        np.cumsum(lengths, out=offsets[1:])
        total = int(offsets[-1])

        p_doc = np.empty(total, dtype=np.int32)
        p_tf = np.empty(total, dtype=np.uint16)
        for tid, (dlist, tlist) in enumerate(zip(post_docs, post_tf, strict=False)):
            start, end = int(offsets[tid]), int(offsets[tid + 1])
            p_doc[start:end] = dlist
            p_tf[start:end] = tlist

        return cls(
            vocab=term_ids,
            postings_doc=p_doc,
            postings_tf=p_tf,
            offsets=offsets,
            doc_len=np.asarray(doc_len, dtype=np.int32),
            doc_ids=doc_ids,
            k1=k1,
            b=b,
        )

    # ---- сохранение / загрузка -----------------------------------------

    def save(self, directory: Path) -> None:
        """Сохранить индекс в каталог (``index.npz`` + ``meta.json``)."""
        directory.mkdir(parents=True, exist_ok=True)
        np.savez(
            directory / INDEX_FILE,
            postings_doc=self.postings_doc,
            postings_tf=self.postings_tf,
            offsets=self.offsets,
            doc_len=self.doc_len,
        )
        (directory / META_FILE).write_text(
            json.dumps(
                {
                    "k1": self.k1,
                    "b": self.b,
                    "n_docs": self.n_docs,
                    "avgdl": self.avgdl,
                    "doc_ids": self.doc_ids,
                    "vocab": self.vocab,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: Path) -> Bm25Index:
        """Загрузить индекс, сохранённый :meth:`save`."""
        with np.load(directory / INDEX_FILE) as data:
            arrays = {k: data[k] for k in data.files}
        meta = json.loads((directory / META_FILE).read_text(encoding="utf-8"))
        return cls(
            vocab=meta["vocab"],
            postings_doc=arrays["postings_doc"],
            postings_tf=arrays["postings_tf"],
            offsets=arrays["offsets"],
            doc_len=arrays["doc_len"],
            doc_ids=meta["doc_ids"],
            k1=meta.get("k1", 1.5),
            b=meta.get("b", 0.75),
        )

    # ---- поиск ----------------------------------------------------------

    def scores(self, query: str) -> np.ndarray:
        """BM25-скоры всех документов (разреженно по постингам)."""
        out = np.zeros(self.n_docs, dtype=np.float32)
        if not self.n_docs or self.avgdl <= 0:
            return out
        for term in tokenize(query):
            tid = self.vocab.get(term)
            if tid is None:
                continue
            start, end = int(self.offsets[tid]), int(self.offsets[tid + 1])
            docs = self.postings_doc[start:end]
            tfs = self.postings_tf[start:end].astype(np.float32)
            dl = self.doc_len[docs].astype(np.float32)
            denom = tfs + self.k1 * (1.0 - self.b + self.b * dl / self.avgdl)
            out[docs] += float(self.idf[tid]) * (tfs * (self.k1 + 1.0)) / denom
        return out

    def search(self, query: str, top_k: int = 50) -> list[Bm25Hit]:
        """Top-k документов по BM25; отдаёт пары ``(doc_id, score)``."""
        scores = self.scores(query)
        if not scores.size:
            return []
        top_k = max(1, min(int(top_k), self.n_docs))
        order = np.argpartition(-scores, top_k - 1)[:top_k]
        order = order[np.argsort(-scores[order])]
        return [
            Bm25Hit(doc_id=self.doc_ids[int(i)], score=float(scores[int(i)]))
            for i in order
            if scores[int(i)] > 0
        ]

    def stats(self) -> dict[str, float | int]:
        """Краткая статистика индекса для логов и README."""
        return {
            "n_docs": self.n_docs,
            "n_terms": len(self.vocab),
            "n_postings": int(self.postings_doc.size),
            "avgdl": round(self.avgdl, 1),
        }


def docs_from_sqlite(sqlite_path: Path) -> Iterable[tuple[str, str]]:
    """Поток пар ``(chroma_id, document)`` прямо из sqlite Chroma.

    Быстрее открытия Chroma-клиента (тот на старте пересобирает HNSW) и не
    требует прав записи: используется режим ``mode=ro``.
    """
    import sqlite3

    db = sqlite3.connect(f"file:{sqlite_path.as_posix()}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    cur = db.execute(
        """
        SELECT e.embedding_id AS cid, md.string_value AS doc
        FROM embeddings e
        JOIN embedding_metadata md ON md.id = e.id AND md.key = 'chroma:document'
        ORDER BY e.id
        """
    )
    try:
        for row in cur:
            if row["doc"]:
                yield row["cid"], row["doc"]
    finally:
        db.close()


__all__ = [
    "BM25Okapi",
    "Bm25Hit",
    "Bm25Index",
    "docs_from_sqlite",
    "tokenize",
]
