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
    COLLECTION_CODE,
    COLLECTION_METADATA,
    HYBRID_TOP_K,
    MAX_PER_SOURCE,
    RERANK_CANDIDATES,
    RRF_K,
    SEARCH_DEFAULT_MODE,
)
from mcp_1c_rag.dedup import DEFAULT_START_MULTIPLIER, cap_by_source, iterative_dedup
from mcp_1c_rag.models import Candidate
from mcp_1c_rag.rrf import rrf_fuse, rrf_order

if TYPE_CHECKING:  # pragma: no cover - только для аннотаций
    from mcp_1c_rag.bm25 import Bm25Index

SearchMode = Literal["vector", "hybrid", "hybrid_rerank", "pearson"]

# Стартовый пул для source-dedup: top_k * DEDUP_START_MULTIPLIER, затем ×2.
DEDUP_START_MULTIPLIER = DEFAULT_START_MULTIPLIER

# Префикс chroma-id для каждой коллекции (задаётся индексатором).
_ID_PREFIX = {
    "code": "bsl:",
    "metadata": "meta:",
    "metadata_xml": "metaobj:",
    "code_proc": "bsl_proc:",
    "config_xml": "xml:",
}
_ID_PREFIX.setdefault(COLLECTION_CODE, "bsl_proc:")
_ID_PREFIX.setdefault(COLLECTION_METADATA, "metaobj:")


_bm25: Bm25Index | None = None
_masks: dict[str, np.ndarray] = {}
_collection_counts: dict[str, int] = {}


def get_collection(name: str = "code"):
    """Вернуть Chroma-коллекцию по имени (``code``/``metadata`` и любые новые)."""
    from mcp_1c_rag.store import get_collection as store_get_collection

    key = (name or "code").strip()
    if key.lower() in ("meta", "metadata"):
        key = "metadata"
    return store_get_collection(key)


def _collection_count(name: str) -> int:
    """Count коллекции с кэшем на процесс (индекс неизменен в runtime)."""
    key = (name or "code").strip()
    if key not in _collection_counts:
        _collection_counts[key] = int(get_collection(key).count())
    return _collection_counts[key]


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


def _masked_bm25_scores(query: str, collection: str) -> np.ndarray:
    """BM25-скоры с prefix-mask коллекции (один раз на запрос)."""
    index = get_bm25()
    scores = index.scores(query)
    prefix = _ID_PREFIX.get(collection, "")
    if prefix:
        scores = np.where(_prefix_mask(index, prefix), scores, -np.inf)
    return scores


def bm25_search(
    query: str,
    *,
    collection: str = COLLECTION_CODE,
    top_k: int = HYBRID_TOP_K,
    scores: np.ndarray | None = None,
) -> list[tuple[str, float]]:
    """Глобальный лексический поиск, ограниченный одной коллекцией."""
    index = get_bm25()
    if scores is None:
        scores = _masked_bm25_scores(query, collection)
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
) -> list[Candidate]:
    """Режим ``lexical``: только глобальный BM25 (grep-подобный поиск)."""
    hits = bm25_search(query, collection=collection, top_k=top_k)
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
) -> list[Candidate]:
    """Режим ``hybrid``: cosine + BM25, объединённые Reciprocal Rank Fusion."""
    vector = vector_search(
        query, collection=collection, top_k=top_k, query_embedding=query_embedding
    )
    lexical = bm25_search(
        query, collection=collection, top_k=top_k, scores=bm25_scores
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
        return lexical_search(query, collection=collection, top_k=top_k)
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


def _hybrid_search_with_dedup(
    query: str,
    *,
    collection: str,
    top_k: int,
    max_per_source: int,
    pool_multiplier: int,
    total: int,
) -> list[Candidate]:
    """Hybrid + source cap: один embed, один BM25.scores, Chroma при росте пула."""
    from mcp_1c_rag.embeddings import embed_query

    qvec = embed_query(query)
    scores = _masked_bm25_scores(query, collection)
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
) -> list[Candidate]:
    """Единая точка входа: режим → top_k; опционально cap по source + backfill.

    При ``max_per_source <= 0`` поведение байт-в-байт как у прежнего ``search``.
    При dedup стартовый пул = ``top_k * pool_multiplier`` (default 3), затем ×2.
    """
    top_k = max(1, int(top_k))
    collection = collection or COLLECTION_CODE
    if max_per_source <= 0:
        return _search_impl(query, top_k, mode, collection)

    chosen = (mode or SEARCH_DEFAULT_MODE).strip().lower()
    total = _collection_count(collection)
    if chosen == "hybrid":
        return _hybrid_search_with_dedup(
            query,
            collection=collection,
            top_k=top_k,
            max_per_source=max_per_source,
            pool_multiplier=pool_multiplier,
            total=total,
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
            return lexical_search(query, collection=collection, top_k=pool)
        if chosen == "pearson":
            return pearson_search(query, collection=collection, top_k=pool)
        # hybrid_rerank: полный _search_impl (rerank зависит от размера пула)
        return _search_impl(query, pool, mode, collection)

    return iterative_dedup(
        fetch,
        top_k=top_k,
        max_per_source=max_per_source,
        pool_multiplier=pool_multiplier,
        total=total,
    )


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
    "DEDUP_START_MULTIPLIER",
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
