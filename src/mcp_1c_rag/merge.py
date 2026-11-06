"""Объединение результатов поиска по нескольким коллекциям.

Нормализация скоров между коллекциями (max / rrf / minmax). Не импортирует
``server``. Дефолт ``normalization=\"max\"`` — прежнее поведение ``_merged_search``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal

from mcp_1c_rag.config import COLLECTION_CODE, COLLECTION_METADATA, RRF_K
from mcp_1c_rag.models import Candidate
from mcp_1c_rag.rrf import rrf_fuse, rrf_order

Normalization = Literal["max", "rrf", "minmax"]
Searcher = Callable[..., list[Candidate]]

DEFAULT_COLLECTIONS: tuple[str, ...] = (COLLECTION_CODE, COLLECTION_METADATA)


def _default_searcher(
    query: str,
    *,
    top_k: int,
    mode: str,
    collection: str,
) -> list[Candidate]:
    from mcp_1c_rag import search_modes

    return search_modes.search(
        query=query, top_k=top_k, mode=mode, collection=collection
    )


def _fetch_per_collection(
    query: str,
    mode: str,
    limit: int,
    collections: Sequence[str],
    searcher: Searcher,
) -> list[tuple[str, list[Candidate]]]:
    """Собрать кандидатов по коллекциям; FileNotFoundError — пропуск."""
    out: list[tuple[str, list[Candidate]]] = []
    for collection in collections:
        try:
            candidates = searcher(
                query, top_k=limit, mode=mode, collection=collection
            )
        except FileNotFoundError:
            continue
        if not candidates:
            continue
        for candidate in candidates:
            candidate.metadata["collection"] = collection
        out.append((collection, candidates))
    return out


def _scored_max(
    per_collection: Sequence[tuple[str, list[Candidate]]],
) -> list[tuple[float, Candidate]]:
    """score / max_score внутри коллекции (baseline)."""
    found: list[tuple[float, Candidate]] = []
    for _name, candidates in per_collection:
        top = max(candidate.score for candidate in candidates) or 1.0
        for candidate in candidates:
            found.append((candidate.score / top, candidate))
    return found


def _scored_minmax(
    per_collection: Sequence[tuple[str, list[Candidate]]],
) -> list[tuple[float, Candidate]]:
    """Min-max нормализация скоров внутри коллекции."""
    found: list[tuple[float, Candidate]] = []
    for _name, candidates in per_collection:
        scores = [candidate.score for candidate in candidates]
        lo, hi = min(scores), max(scores)
        span = (hi - lo) if hi > lo else 0.0
        for candidate in candidates:
            norm = 1.0 if span == 0.0 else (candidate.score - lo) / span
            found.append((norm, candidate))
    return found


def _scored_rrf(
    per_collection: Sequence[tuple[str, list[Candidate]]],
    *,
    k_rrf: int,
) -> list[tuple[float, Candidate]]:
    """RRF по рангам внутри каждой коллекции."""
    rankings: list[list[str]] = []
    by_id: dict[str, Candidate] = {}
    for _name, candidates in per_collection:
        rankings.append([candidate.id for candidate in candidates])
        for candidate in candidates:
            by_id.setdefault(candidate.id, candidate)
    fused = rrf_fuse(rankings, k=k_rrf)
    return [(fused[doc_id], by_id[doc_id]) for doc_id in rrf_order(fused)]


def _dedup_limit(
    scored: Sequence[tuple[float, Candidate]],
    limit: int,
) -> list[Candidate]:
    """Сортировка по убыванию нормализованного скора, dedup id, обрезка."""
    ordered = sorted(scored, key=lambda pair: pair[0], reverse=True)
    seen: set[str] = set()
    unique: list[Candidate] = []
    for _norm, candidate in ordered:
        if candidate.id in seen:
            continue
        seen.add(candidate.id)
        unique.append(candidate)
        if len(unique) >= limit:
            break
    return unique


def merged_search(
    query: str,
    mode: str,
    limit: int,
    collections: Sequence[str] | None = None,
    *,
    searcher: Searcher | None = None,
    normalization: Normalization = "max",
    k_rrf: int = RRF_K,
) -> list[Candidate]:
    """Поиск по коллекциям с нормализацией скоров и объединением.

    Дефолт ``normalization=\"max\"`` повторяет прежнее поведение server._merged_search.
    """
    cols = tuple(collections) if collections is not None else DEFAULT_COLLECTIONS
    runner = searcher if searcher is not None else _default_searcher
    per_collection = _fetch_per_collection(query, mode, limit, cols, runner)
    if not per_collection:
        return []
    if normalization == "max":
        scored = _scored_max(per_collection)
    elif normalization == "minmax":
        scored = _scored_minmax(per_collection)
    elif normalization == "rrf":
        scored = _scored_rrf(per_collection, k_rrf=k_rrf)
    else:
        raise ValueError(
            f"неизвестная normalization {normalization!r}; ожидается max|rrf|minmax"
        )
    return _dedup_limit(scored, max(1, int(limit)))


__all__ = [
    "DEFAULT_COLLECTIONS",
    "Normalization",
    "Searcher",
    "merged_search",
]
