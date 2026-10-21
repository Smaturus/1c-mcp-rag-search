"""Semantic search helpers used by MCP tools."""

from __future__ import annotations

from typing import Any

from mcp_1c_homework.config import (
    HYBRID_CANDIDATE_MULT,
    HYBRID_ENABLED,
)
from mcp_1c_homework.embeddings import embed_query
from mcp_1c_homework.hybrid import hybrid_rerank
from mcp_1c_homework.index_guard import IndexBusy
from mcp_1c_homework.store import (
    get_code_collection,
    get_metadata_collection,
    query_collection,
)


def _format_hits(hits: list[dict[str, Any]]) -> str:
    if not hits:
        return "No results. Build the index first: python -m mcp_1c_homework.indexer --reset"
    blocks: list[str] = []
    for i, hit in enumerate(hits, start=1):
        meta = hit.get("metadata") or {}
        source = meta.get("source", "?")
        chunk_index = meta.get("chunk_index", "?")
        dist = hit.get("distance")
        header = f"[{i}] source={source} chunk={chunk_index} distance={dist:.4f}"
        if "hybrid_score" in hit:
            header += (
                f" hybrid={hit['hybrid_score']:.4f}"
                f" cosine={hit.get('cosine', float('nan')):.4f}"
                f" bm25={hit.get('bm25', float('nan')):.4f}"
                f" pearson={hit.get('pearson', float('nan')):.4f}"
            )
        blocks.append(header + "\n" + (hit.get("text") or ""))
    return "\n\n---\n\n".join(blocks)


def _search_collection(collection, query: str, limit: int) -> list[dict[str, Any]]:
    """Cosine via Chroma; optional BM25+cosine hybrid rerank."""
    if collection.count() == 0:
        return []
    if HYBRID_ENABLED:
        cand = max(limit * max(HYBRID_CANDIDATE_MULT, 1), limit)
        cand = min(cand, max(collection.count(), 1))
        raw = query_collection(collection, embed_query(query), cand)
        return hybrid_rerank(query, raw, limit=limit)
    return query_collection(collection, embed_query(query), limit)


def codesearch(query: str, limit: int = 5) -> str:
    """Search 1C BSL code (cosine; optional BM25+cosine hybrid)."""
    try:
        collection = get_code_collection()
    except IndexBusy as exc:
        return str(exc)
    if collection.count() == 0:
        return "Code index is empty. Run: python -m mcp_1c_homework.indexer --target code --reset"
    return _format_hits(_search_collection(collection, query, limit))


def metadatasearch(query: str, limit: int = 20) -> str:
    """Search configuration report / metadata (cosine; optional hybrid)."""
    try:
        collection = get_metadata_collection()
    except IndexBusy as exc:
        return str(exc)
    if collection.count() == 0:
        return (
            "Metadata index is empty. Run: "
            "python -m mcp_1c_homework.indexer --target metadata --reset"
        )
    return _format_hits(_search_collection(collection, query, limit))
