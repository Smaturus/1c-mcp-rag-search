"""
Interactive / MCP evaluation of one retrieval query (homework highlight).

Example MCP call: evaluate_retrieval(query=\"НДС\")
Computes Hit@k, Recall@k, MRR, top cosine, optional null-tail vs background.
"""

from __future__ import annotations

import math
import random

from mcp_1c_rag.embeddings import embed_passages, embed_query
from mcp_1c_rag.index_guard import IndexBusy
from mcp_1c_rag.metrics import pearson_correlation
from mcp_1c_rag.store import (
    get_code_collection,
    get_metadata_collection,
    query_collection,
)


def _cosine_from_distance(distance: float | None) -> float:
    if distance is None:
        return float("nan")
    return 1.0 - float(distance)


def _is_relevant(source: str, needles: list[str]) -> bool:
    src = source.replace("\\", "/")
    return any(n and n in src for n in needles)


def _get_collection(name: str):
    key = (name or "code").strip().lower()
    if key in ("code", "bsl", "codesearch"):
        return get_code_collection(), "code"
    if key in ("metadata", "meta", "metadatasearch"):
        return get_metadata_collection(), "metadata"
    raise ValueError("collection must be 'code' or 'metadata'")


def _parse_needles(query: str, relevant_sources: str) -> list[str]:
    if relevant_sources and relevant_sources.strip():
        return [p.strip() for p in relevant_sources.split(",") if p.strip()]
    # Default: use the query itself as path/text needle (e.g. "НДС")
    return [query.strip()] if query.strip() else []


def _null_tail_prob(
    collection,
    top_sim: float,
    *,
    n_pairs: int,
    seed: int,
) -> float | None:
    """Share of random (noise query, random doc) cosines >= top_sim."""
    if n_pairs <= 0 or collection.count() == 0:
        return None
    rng = random.Random(seed)
    pool = min(collection.count(), max(n_pairs * 2, 100), 800)
    try:
        raw = collection.get(limit=pool, include=["documents", "embeddings"])
    except Exception:
        return None
    docs = raw.get("documents") or []
    embs = raw.get("embeddings")
    if not docs:
        return None
    noise_bank = [
        "weather forecast tomorrow",
        "recipe for apple pie",
        "фото котиков",
        "bitcoin wallet cluster",
        "расписание на Марс",
    ]
    scores: list[float] = []
    for _ in range(n_pairs):
        q = rng.choice(noise_bank) + f" #{rng.randint(1, 99999)}"
        qv = embed_query(q)
        idx = rng.randrange(len(docs))
        if embs is not None and len(embs) == len(docs) and embs[idx] is not None:
            pv = embs[idx]
        else:
            pv = embed_passages([docs[idx] or " "])[0]
        scores.append(sum(a * b for a, b in zip(qv, pv, strict=False)))
    if not scores:
        return None
    return sum(1 for s in scores if s >= top_sim) / len(scores)


def evaluate_retrieval(
    query: str,
    *,
    k: int = 5,
    collection: str = "code",
    relevant_sources: str = "",
    null_pairs: int = 40,
) -> str:
    """
    Evaluate one query against the local index (MCP highlight tool).

    Args:
        query: Search query, e.g. «НДС».
        k: Top-k for Hit/Recall/MRR.
        collection: «code» or «metadata».
        relevant_sources: Comma-separated path needles; default = query itself.
        null_pairs: Background cosine samples for null-tail (0 to skip).
    """
    q = (query or "").strip()
    if not q:
        return "query is empty. Example: evaluate_retrieval(query=\"НДС\")"

    try:
        coll, coll_name = _get_collection(collection)
    except (ValueError, IndexBusy) as exc:
        return str(exc)

    if coll.count() == 0:
        return (
            f"Collection «{coll_name}» is empty. "
            "Finish indexing: python -m mcp_1c_rag.indexer --reset"
        )

    needles = _parse_needles(q, relevant_sources)
    k = max(1, int(k))
    try:
        hits = query_collection(coll, embed_query(q), k)
    except Exception as exc:  # noqa: BLE001
        return (
            f"Search failed (index busy/locked?): {exc}\n"
            "Wait until indexing finishes, then retry."
        )

    sources = [(h.get("metadata") or {}).get("source", "") for h in hits]
    sims = [_cosine_from_distance(h.get("distance")) for h in hits]
    flags = [_is_relevant(s, needles) for s in sources]

    hit = 1.0 if any(flags) else 0.0
    needles_hit = {
        n for n in needles if any(n in s.replace("\\", "/") for s in sources)
    }
    recall = len(needles_hit) / len(needles) if needles else float("nan")
    mrr = 0.0
    for rank, flag in enumerate(flags, start=1):
        if flag:
            mrr = 1.0 / rank
            break

    top_sim = sims[0] if sims else float("nan")
    null_tail = None
    if null_pairs > 0 and sims and not math.isnan(top_sim):
        try:
            null_tail = _null_tail_prob(
                coll, top_sim, n_pairs=null_pairs, seed=hash(q) % 10_000
            )
        except Exception:
            null_tail = None

    # Extra signal for the report: Pearson(query, top passages)
    pearsons: list[float] = []
    try:
        q_vec = embed_query(q)
        texts = []
        for h in hits:
            meta = h.get("metadata") or {}
            texts.append(f"{meta.get('source', '')}\n{h.get('text') or ''}")
        p_vecs = embed_passages(texts) if texts else []
        pearsons = [pearson_correlation(q_vec, pv) for pv in p_vecs]
    except Exception:
        pearsons = []

    lines = [
        f"evaluate_retrieval(query={q!r}, k={k}, collection={coll_name})",
        f"needles={needles}",
        f"Hit@{k}={hit:.0f}  Recall@{k}={recall:.3f}  MRR={mrr:.3f}",
        f"top1_cosine={top_sim:.4f}",
    ]
    if pearsons:
        lines.append(f"top1_pearson={pearsons[0]:.4f}")
    if null_tail is not None:
        lines.append(
            f"null_tail_prob(top1)={null_tail:.3f}  "
            f"( доля случайных пар с cosine ≥ top1; меньше = top1 отличимее от фона )"
        )
    lines.append("top hits:")
    for i, (src, sim, flag) in enumerate(zip(sources, sims, flags, strict=False), start=1):
        mark = "HIT" if flag else "—"
        pr = f" pearson={pearsons[i-1]:.4f}" if i <= len(pearsons) else ""
        lines.append(f"  [{i}] {mark} cos={sim:.4f}{pr}  {src}")

    lines.append(
        "\nTip: уточните эталон через relevant_sources="
        "\"CommonModules.УчетНДС,НДС\" "
        "или прогоните пакетный eval: scripts/eval_retrieval_fluctuations.py"
    )
    return "\n".join(lines)
