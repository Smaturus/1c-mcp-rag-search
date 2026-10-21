"""Единый слой поиска: vector / hybrid (RRF) / hybrid_rerank / pearson-baseline.

Схема пайплайна::

    query ──► e5 (?query:?) ──► Chroma cosine top-N ─┐
          └─► BM25 (глобальный индекс) top-N ───────┴─► RRF ─► flashrank rerank ─► top-k

Режимы:
    ``vector``        — только cosine (baseline, как было до RRF);
    ``hybrid``        — cosine + BM25 через Reciprocal Rank Fusion;
    ``hybrid_rerank`` — RRF + cross-encoder вторым этапом (по умолчанию);
    ``pearson``       — старый взвешенный гибрид cos/BM25/Pearson (для сравнения).

Сырые скоры cosine и BM25 никогда не смешиваются: RRF работает с рангами.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

from mcp_1c_rag.config import (
    BM25_DIR,
    HYBRID_TOP_K,
    RERANK_CANDIDATES,
    RRF_K,
    SEARCH_DEFAULT_MODE,
)
from mcp_1c_rag.models import Candidate
from mcp_1c_rag.rrf import rrf_fuse, rrf_order

if TYPE_CHECKING:  # pragma: no cover - только для аннотаций
    from mcp_1c_rag.bm25 import Bm25Index

SearchMode = Literal["vector", "hybrid", "hybrid_rerank", "pearson"]

# Префикс chroma-id для каждой коллекции (задаётся индексатором).
_ID_PREFIX = {"code": "bsl:", "metadata": "meta:"}


_bm25: Bm25Index | None = None
_masks: dict[str, np.ndarray] = {}


def get_collection(name: str = "code"):
    """Вернуть Chroma-коллекцию по имени (``code`` или ``metadata``)."""
    from mcp_1c_rag.store import get_code_collection, get_metadata_collection

    key = (name or "code").strip().lower()
    if key in ("metadata", "meta"):
        return get_metadata_collection()
    return get_code_collection()


def get_bm25() -> Bm25Index:
    """Ленивая загрузка глобального BM25-индекса (собирается build_bm25.py)."""
    global _bm25
    if _bm25 is None:
        from mcp_1c_rag.bm25 import Bm25Index

        if not (BM25_DIR / "index.npz").exists():
            raise FileNotFoundError(
                f"BM25-индекс не найден: {BM25_DIR}. "
                "Соберите его: python scripts/build_bm25.py"
            )
        _bm25 = Bm25Index.load(BM25_DIR)
    return _bm25


def bm25_available() -> bool:
    """Есть ли собранный BM25-индекс (для честного fallback в MCP-туле)."""
    return (BM25_DIR / "index.npz").exists()


def _prefix_mask(index: Bm25Index, prefix: str) -> np.ndarray:
    mask = _masks.get(prefix)
    if mask is None:
        mask = np.fromiter(
            (doc_id.startswith(prefix) for doc_id in index.doc_ids),
            dtype=bool,
            count=index.n_docs,
        )
        _masks[prefix] = mask
    return mask


def bm25_search(
    query: str,
    *,
    collection: str = "code",
    top_k: int = HYBRID_TOP_K,
) -> list[tuple[str, float]]:
    """Глобальный лексический поиск, ограниченный одной коллекцией."""
    index = get_bm25()
    scores = index.scores(query)
    prefix = _ID_PREFIX.get(collection, "")
    if prefix:
        scores = np.where(_prefix_mask(index, prefix), scores, -np.inf)
    if not np.isfinite(scores).any():
        return []
    n = max(1, min(int(top_k), int(np.isfinite(scores).sum())))
    top = np.argpartition(-scores, n - 1)[:n]
    top = top[np.argsort(-scores[top])]
    return [
        (index.doc_ids[int(i)], float(scores[int(i)]))
        for i in top
        if np.isfinite(scores[int(i)]) and scores[int(i)] > 0
    ]


def _raw_query(
    collection_name: str,
    query: str,
    top_k: int,
    *,
    include_embeddings: bool = False,
) -> dict[str, list]:
    """Chroma-запрос с сохранением id (нужны для RRF) и, опционально, векторов."""
    from mcp_1c_rag.embeddings import embed_query

    coll = get_collection(collection_name)
    count = coll.count()
    if count == 0:
        return {"ids": [], "documents": [], "metadatas": [], "distances": [], "embeddings": []}
    include = ["documents", "metadatas", "distances"]
    if include_embeddings:
        include.append("embeddings")
    raw = coll.query(
        query_embeddings=[embed_query(query)],
        n_results=max(1, min(int(top_k), count)),
        include=include,
    )
    return {
        "ids": (raw.get("ids") or [[]])[0],
        "documents": (raw.get("documents") or [[]])[0],
        "metadatas": (raw.get("metadatas") or [[]])[0],
        "distances": (raw.get("distances") or [[]])[0],
        "embeddings": (raw.get("embeddings") or [[]])[0],
    }


def vector_search(
    query: str,
    *,
    collection: str = "code",
    top_k: int = 5,
) -> list[Candidate]:
    """Режим ``vector``: только cosine (baseline)."""
    raw = _raw_query(collection, query, top_k)
    return [
        Candidate(
            id=doc_id,
            text=text or "",
            metadata=meta or {},
            score=1.0 - float(dist),
            source_scores={"cosine": 1.0 - float(dist)},
        )
        for doc_id, text, meta, dist in zip(
            raw["ids"], raw["documents"], raw["metadatas"], raw["distances"], strict=False
        )
    ]


def hybrid_search(
    query: str,
    *,
    collection: str = "code",
    top_k: int = HYBRID_TOP_K,
    k_rrf: int = RRF_K,
) -> list[Candidate]:
    """Режим ``hybrid``: cosine + BM25, объединённые Reciprocal Rank Fusion."""
    vector = vector_search(query, collection=collection, top_k=top_k)
    lexical = bm25_search(query, collection=collection, top_k=top_k)

    fused = rrf_fuse(
        [[c.id for c in vector], [doc_id for doc_id, _ in lexical]],
        k=k_rrf,
    )
    if not fused:
        return []

    by_id: dict[str, Candidate] = {c.id: c for c in vector}
    missing = [doc_id for doc_id in fused if doc_id not in by_id]
    if missing:
        coll = get_collection(collection)
        got = coll.get(ids=missing, include=["documents", "metadatas"])
        for doc_id, text, meta in zip(
            got.get("ids") or [], got.get("documents") or [], got.get("metadatas") or [], strict=False
        ):
            by_id[doc_id] = Candidate(id=doc_id, text=text or "", metadata=meta or {})

    vector_rank = {c.id: rank for rank, c in enumerate(vector, start=1)}
    bm25_rank = {doc_id: rank for rank, (doc_id, _) in enumerate(lexical, start=1)}
    result: list[Candidate] = []
    for doc_id in rrf_order(fused):
        base = by_id.get(doc_id)
        if base is None:
            continue
        result.append(
            Candidate(
                id=base.id,
                text=base.text,
                metadata=dict(base.metadata),
                score=fused[doc_id],
                source_scores={
                    "rrf": fused[doc_id],
                    "vector_rank": float(vector_rank.get(doc_id, 0)),
                    "bm25_rank": float(bm25_rank.get(doc_id, 0)),
                },
            )
        )
    return result


def pearson_search(
    query: str,
    *,
    collection: str = "code",
    top_k: int = 5,
) -> list[Candidate]:
    """Режим ``pearson``: прежний взвешенный гибрид (baseline для сравнения).

    Эмбеддинги пассажей берутся прямо из Chroma, а не пересчитываются моделью:
    это честнее по латентности и не меняет формулу Пирсона.
    """
    from mcp_1c_rag.hybrid import hybrid_rerank

    raw = _raw_query(collection, query, max(top_k * 4, top_k), include_embeddings=True)
    hits = [
        {
            "id": doc_id,
            "text": text,
            "metadata": meta,
            "distance": dist,
        }
        for doc_id, text, meta, dist in zip(
            raw["ids"], raw["documents"], raw["metadatas"], raw["distances"], strict=False
        )
    ]
    vectors = [vec for vec in raw["embeddings"] if vec is not None]
    reranked = hybrid_rerank(
        query,
        hits,
        limit=top_k,
        passage_vectors=vectors if len(vectors) == len(hits) else None,
    )
    return [
        Candidate(
            id=str(hit.get("id", "")),
            text=hit.get("text") or "",
            metadata=hit.get("metadata") or {},
            score=float(hit.get("hybrid_score", 0.0)),
            source_scores={
                "cosine": float(hit.get("cosine", 0.0)),
                "bm25": float(hit.get("bm25", 0.0)),
                "pearson": float(hit.get("pearson", 0.0)),
            },
        )
        for hit in reranked
    ]


def search(
    query: str,
    *,
    top_k: int = 5,
    mode: str | None = None,
    collection: str = "code",
) -> list[Candidate]:
    """Единая точка входа: выбрать режим и вернуть top_k кандидатов."""
    chosen = (mode or SEARCH_DEFAULT_MODE).strip().lower()
    if chosen not in ("vector", "hybrid", "hybrid_rerank", "pearson"):
        raise ValueError(
            f"неизвестный режим {chosen!r}; ожидается vector|hybrid|hybrid_rerank|pearson"
        )
    top_k = max(1, int(top_k))

    if chosen == "vector":
        return vector_search(query, collection=collection, top_k=top_k)[:top_k]
    if chosen == "pearson":
        return pearson_search(query, collection=collection, top_k=top_k)
    if chosen == "hybrid":
        return hybrid_search(query, collection=collection, top_k=top_k)[:top_k]

    from mcp_1c_rag.rerank import rerank

    candidates = hybrid_search(
        query,
        collection=collection,
        top_k=max(RERANK_CANDIDATES, top_k),
    )
    return rerank(query, candidates, top_k=top_k)


def format_candidates(candidates: list[Candidate]) -> str:
    """Текстовый вывод кандидатов для MCP-тула."""
    if not candidates:
        return "Ничего не найдено. Проверьте индекс и BM25-индекс (scripts/build_bm25.py)."
    blocks: list[str] = []
    for i, c in enumerate(candidates, start=1):
        meta = c.metadata or {}
        header = (
            f"[{i}] source={meta.get('source', '?')} "
            f"chunk={meta.get('chunk_index', '?')} score={c.score:.4f}"
        )
        extra = {k: round(v, 4) for k, v in c.source_scores.items() if k != "score"}
        if extra:
            header += " " + " ".join(f"{k}={v}" for k, v in extra.items())
        blocks.append(header + "\n" + (c.text or ""))
    return "\n\n---\n\n".join(blocks)


__all__ = [
    "Candidate",
    "bm25_available",
    "bm25_search",
    "format_candidates",
    "get_bm25",
    "get_collection",
    "hybrid_search",
    "pearson_search",
    "search",
    "vector_search",
]
