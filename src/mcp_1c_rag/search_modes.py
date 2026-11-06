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

from mcp_1c_rag.collections import all_specs, canonical_name
from mcp_1c_rag.config import (
    COLLECTION_CODE,
    HYBRID_TOP_K,
    MAX_PER_SOURCE,
    RERANK_CANDIDATES,
    RRF_K,
    SEARCH_DEFAULT_MODE,
)
from mcp_1c_rag.dedup import DEFAULT_START_MULTIPLIER, cap_by_source, iterative_dedup
from mcp_1c_rag.formatting import format_candidates
from mcp_1c_rag.models import Candidate
from mcp_1c_rag.rrf import rrf_fuse, rrf_order
from mcp_1c_rag.search_context import SearchContext

if TYPE_CHECKING:  # pragma: no cover - только для аннотаций
    from mcp_1c_rag.bm25 import Bm25Index

SearchMode = Literal["vector", "hybrid", "hybrid_rerank", "pearson"]

# Стартовый пул для source-dedup: top_k * DEDUP_START_MULTIPLIER, затем ×2.
DEDUP_START_MULTIPLIER = DEFAULT_START_MULTIPLIER

# Префикс chroma-id для каждой коллекции (из registry; без ручных значений).
_ID_PREFIX = {spec.name: spec.id_prefix for spec in all_specs()}

# Процессный default-контекст (обратная совместимость без явного context=).
default_context = SearchContext()


def get_collection(name: str | None = None):
    """Вернуть Chroma-коллекцию по каноническому имени (aliases резолвятся)."""
    from mcp_1c_rag.store import get_collection as store_get_collection

    canonical = canonical_name(name or COLLECTION_CODE)
    return store_get_collection(canonical)


def _collection_count(name: str | None = None) -> int:
    """Count коллекции с кэшем (shim → default_context)."""
    return default_context.collection_count(name)


def get_bm25() -> Bm25Index:
    """Ленивая загрузка BM25-индекса (shim → default_context)."""
    return default_context.bm25()


def bm25_available() -> bool:
    """Есть ли собранный BM25-индекс (для честного fallback в MCP-туле)."""
    return default_context.bm25_available()


def _prefix_mask(index: Bm25Index, prefix: str) -> np.ndarray:
    """Чистая маска по doc_ids (без модульного кэша; кэш — в SearchContext)."""
    return np.fromiter(
        (doc_id.startswith(prefix) for doc_id in index.doc_ids),
        dtype=bool,
        count=index.n_docs,
    )


def _masked_bm25_scores(query: str, collection: str) -> np.ndarray:
    """BM25-скоры с prefix-mask коллекции (shim → default_context)."""
    return default_context.masked_bm25_scores(query, collection)


def bm25_search(
    query: str,
    *,
    collection: str = COLLECTION_CODE,
    top_k: int = HYBRID_TOP_K,
    scores: np.ndarray | None = None,
    context: SearchContext | None = None,
) -> list[tuple[str, float]]:
    """Глобальный лексический поиск, ограниченный одной коллекцией."""
    if context is None:
        index = get_bm25()
        if scores is None:
            scores = _masked_bm25_scores(query, collection)
    else:
        index = context.bm25()
        if scores is None:
            scores = context.masked_bm25_scores(query, collection)
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


def lexical_search(
    query: str,
    *,
    collection: str = COLLECTION_CODE,
    top_k: int = 5,
    context: SearchContext | None = None,
) -> list[Candidate]:
    """Режим ``lexical``: только глобальный BM25 (grep-подобный поиск)."""
    hits = bm25_search(
        query, collection=collection, top_k=top_k, context=context
    )
    if not hits:
        return []
    coll = get_collection(collection)
    requested = [doc_id for doc_id, _ in hits]
    raw = coll.get(ids=requested, include=["documents", "metadatas"])
    ids = raw.get("ids") or []
    documents = raw.get("documents") or []
    metadatas = raw.get("metadatas") or []
    by_id: dict[str, tuple[str, dict]] = {}
    for index, doc_id in enumerate(ids):
        text = documents[index] if index < len(documents) else ""
        meta = metadatas[index] if index < len(metadatas) else {}
        by_id[str(doc_id)] = (text or "", dict(meta or {}))
    candidates: list[Candidate] = []
    for doc_id, score in hits[:top_k]:
        text, meta = by_id.get(doc_id, ("", {}))
        candidates.append(
            Candidate(id=doc_id, text=text, metadata=meta, score=float(score),
                      source_scores={"bm25": float(score)})
        )
    return candidates


def _raw_query(
    collection_name: str,
    query: str,
    top_k: int,
    *,
    include_embeddings: bool = False,
    query_embedding: list[float] | None = None,
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
    emb = query_embedding if query_embedding is not None else embed_query(query)
    raw = coll.query(
        query_embeddings=[emb],
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
    collection: str = COLLECTION_CODE,
    top_k: int = 5,
    query_embedding: list[float] | None = None,
) -> list[Candidate]:
    """Режим ``vector``: только cosine (baseline)."""
    raw = _raw_query(collection, query, top_k, query_embedding=query_embedding)
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
    collection: str = COLLECTION_CODE,
    top_k: int = HYBRID_TOP_K,
    k_rrf: int = RRF_K,
    query_embedding: list[float] | None = None,
    bm25_scores: np.ndarray | None = None,
    context: SearchContext | None = None,
) -> list[Candidate]:
    """Режим ``hybrid``: cosine + BM25, объединённые Reciprocal Rank Fusion."""
    vector = vector_search(
        query, collection=collection, top_k=top_k, query_embedding=query_embedding
    )
    lexical = bm25_search(
        query,
        collection=collection,
        top_k=top_k,
        scores=bm25_scores,
        context=context,
    )

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
    collection: str = COLLECTION_CODE,
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


def _search_impl(
    query: str,
    top_k: int,
    mode: str | None,
    collection: str,
    *,
    context: SearchContext | None = None,
) -> list[Candidate]:
    """Прежнее тело search(): режим → top_k без source-dedup."""
    collection = collection or COLLECTION_CODE
    chosen = (mode or SEARCH_DEFAULT_MODE).strip().lower()
    if chosen not in ("lexical", "vector", "hybrid", "hybrid_rerank", "pearson"):
        raise ValueError(
            "неизвестный режим "
            f"{chosen!r}; ожидается lexical|vector|hybrid|hybrid_rerank|pearson"
        )
    top_k = max(1, int(top_k))

    if chosen == "lexical":
        return lexical_search(
            query, collection=collection, top_k=top_k, context=context
        )
    if chosen == "vector":
        return vector_search(query, collection=collection, top_k=top_k)[:top_k]
    if chosen == "pearson":
        return pearson_search(query, collection=collection, top_k=top_k)
    if chosen == "hybrid":
        return hybrid_search(
            query, collection=collection, top_k=top_k, context=context
        )[:top_k]

    from mcp_1c_rag.rerank import rerank

    candidates = hybrid_search(
        query,
        collection=collection,
        top_k=max(RERANK_CANDIDATES, top_k),
        context=context,
    )
    return rerank(query, candidates, top_k=top_k)


def _hybrid_search_with_dedup(
    query: str,
    *,
    collection: str,
    top_k: int,
    max_per_source: int,
    pool_multiplier: int,
    total: int,
    context: SearchContext | None = None,
) -> list[Candidate]:
    """Hybrid + source cap: один embed, один BM25.scores, Chroma при росте пула."""
    from mcp_1c_rag.embeddings import embed_query

    qvec = embed_query(query)
    if context is None:
        scores = _masked_bm25_scores(query, collection)
    else:
        scores = context.masked_bm25_scores(query, collection)
    pool = min(top_k * max(1, int(pool_multiplier)), total)
    if pool <= 0:
        return []

    while True:
        fused = hybrid_search(
            query,
            collection=collection,
            top_k=pool,
            query_embedding=qvec,
            bm25_scores=scores,
            context=context,
        )
        candidates = fused[:pool]
        capped = cap_by_source(candidates, max_per_source)
        if len(capped) >= top_k:
            return capped[:top_k]
        if len(candidates) < pool or pool >= total:
            return capped[:top_k]
        next_pool = min(pool * 2, total)
        if next_pool <= pool:
            return capped[:top_k]
        pool = next_pool


def search(
    query: str,
    *,
    top_k: int = 5,
    mode: str | None = None,
    collection: str = COLLECTION_CODE,
    max_per_source: int = MAX_PER_SOURCE,
    pool_multiplier: int = DEDUP_START_MULTIPLIER,
    context: SearchContext | None = None,
) -> list[Candidate]:
    """Единая точка входа: режим → top_k; опционально cap по source + backfill.

    При ``max_per_source <= 0`` поведение байт-в-байт как у прежнего ``search``.
    При dedup стартовый пул = ``top_k * pool_multiplier`` (default 3), затем ×2.
    """
    top_k = max(1, int(top_k))
    collection = collection or COLLECTION_CODE
    if max_per_source <= 0:
        return _search_impl(query, top_k, mode, collection, context=context)

    chosen = (mode or SEARCH_DEFAULT_MODE).strip().lower()
    if context is None:
        total = _collection_count(collection)
    else:
        total = context.collection_count(collection)
    if chosen == "hybrid":
        return _hybrid_search_with_dedup(
            query,
            collection=collection,
            top_k=top_k,
            max_per_source=max_per_source,
            pool_multiplier=pool_multiplier,
            total=total,
            context=context,
        )

    # Прочие режимы: iterative_dedup; qvec кэшируем через замыкание где возможно.
    from mcp_1c_rag.embeddings import embed_query

    qvec = embed_query(query)

    def fetch(pool: int) -> list[Candidate]:
        if chosen == "vector":
            return vector_search(
                query, collection=collection, top_k=pool, query_embedding=qvec
            )[:pool]
        if chosen == "lexical":
            return lexical_search(
                query, collection=collection, top_k=pool, context=context
            )
        if chosen == "pearson":
            return pearson_search(query, collection=collection, top_k=pool)
        # hybrid_rerank: полный _search_impl (rerank зависит от размера пула)
        return _search_impl(query, pool, mode, collection, context=context)

    return iterative_dedup(
        fetch,
        top_k=top_k,
        max_per_source=max_per_source,
        pool_multiplier=pool_multiplier,
        total=total,
    )


__all__ = [
    "Candidate",
    "DEDUP_START_MULTIPLIER",
    "SearchContext",
    "bm25_available",
    "bm25_search",
    "default_context",
    "format_candidates",
    "get_bm25",
    "get_collection",
    "hybrid_search",
    "pearson_search",
    "search",
    "vector_search",
]
