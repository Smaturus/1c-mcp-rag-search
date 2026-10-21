"""
Hybrid rerank over Chroma cosine candidates:

  s = w_cos * cosine_norm + w_bm25 * bm25_norm + w_pearson * pearson_norm

Cosine from Chroma distance; BM25 on candidate texts; Pearson on e5 vectors
(query vs passage) — educational third signal, not HNSW space change.
"""

from __future__ import annotations

from typing import Any

from mcp_1c_rag.bm25 import BM25Okapi, tokenize
from mcp_1c_rag.config import (
    HYBRID_ALPHA,
    HYBRID_BM25_WEIGHT,
    HYBRID_PEARSON_WEIGHT,
)
from mcp_1c_rag.embeddings import embed_passages, embed_query
from mcp_1c_rag.metrics import pearson_correlation


def _minmax_norm(values: list[float]) -> list[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return [1.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def cosine_from_distance(distance: float | None) -> float:
    if distance is None:
        return 0.0
    return 1.0 - float(distance)


def _normalized_weights() -> tuple[float, float, float]:
    """Return (w_cos, w_bm25, w_pearson) summing to 1."""
    # If user only tweaks ALPHA (old docs), keep backward-compatible split
    # when BM25/PEARSON env left at defaults... Actually always use three weights.
    w_cos = max(0.0, HYBRID_ALPHA)
    w_bm = max(0.0, HYBRID_BM25_WEIGHT)
    w_pr = max(0.0, HYBRID_PEARSON_WEIGHT)
    total = w_cos + w_bm + w_pr
    if total <= 1e-12:
        return 1.0, 0.0, 0.0
    return w_cos / total, w_bm / total, w_pr / total


def hybrid_rerank(
    query: str,
    hits: list[dict[str, Any]],
    *,
    limit: int,
    passage_vectors: list[Any] | None = None,
) -> list[dict[str, Any]]:
    """
    Reorder cosine hits by hybrid score; return top ``limit``.

    Enriched keys: cosine, bm25, pearson, hybrid_score, weights.

    Args:
        passage_vectors: Готовые эмбеддинги пассажей (из Chroma), чтобы не
            пересчитывать их моделью на каждый запрос. Если не переданы,
            векторы считаются через ``embed_passages``.
    """
    if not hits:
        return []

    w_cos, w_bm, w_pr = _normalized_weights()
    cosines = [cosine_from_distance(h.get("distance")) for h in hits]

    corpus_texts: list[str] = []
    for h in hits:
        meta = h.get("metadata") or {}
        source = str(meta.get("source", ""))
        body = str(h.get("text") or "")
        corpus_texts.append(f"{source}\n{body}")

    bm25_raw = [0.0] * len(hits)
    if w_bm > 0:
        bm25 = BM25Okapi([tokenize(t) for t in corpus_texts])
        bm25_raw = bm25.scores(query)

    pearsons = [0.0] * len(hits)
    if w_pr > 0:
        q_vec = embed_query(query)
        if passage_vectors is not None and len(passage_vectors) == len(hits):
            p_vecs = passage_vectors
        else:
            # passage embeddings for Pearson (same model as index)
            p_vecs = embed_passages(corpus_texts)
        pearsons = [
            pearson_correlation(q_vec, list(pv)) for pv in p_vecs if pv is not None
        ]
        if len(pearsons) != len(hits):
            pearsons = [0.0] * len(hits)

    cos_n = _minmax_norm(cosines)
    bm_n = _minmax_norm(bm25_raw) if w_bm > 0 else [0.0] * len(hits)
    pr_n = _minmax_norm(pearsons) if w_pr > 0 else [0.0] * len(hits)

    ranked: list[tuple[float, dict[str, Any]]] = []
    for h, c, cn, b, bn, p, pn in zip(
        hits, cosines, cos_n, bm25_raw, bm_n, pearsons, pr_n, strict=False
    ):
        hybrid = w_cos * cn + w_bm * bn + w_pr * pn
        enriched = dict(h)
        enriched["cosine"] = c
        enriched["bm25"] = b
        enriched["pearson"] = p
        enriched["hybrid_score"] = hybrid
        enriched["weights"] = {
            "cosine": w_cos,
            "bm25": w_bm,
            "pearson": w_pr,
        }
        ranked.append((hybrid, enriched))

    ranked.sort(key=lambda x: x[0], reverse=True)
    return [h for _, h in ranked[: max(1, limit)]]
