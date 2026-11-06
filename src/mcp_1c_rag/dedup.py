"""Source-level dedup: cap по файлу-источнику и iterative backfill.

Чистый модуль без Chroma/эмбеддингов — пригоден для unit-тестов и harness.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from mcp_1c_rag.models import Candidate

# Стартовый пул = top_k * DEFAULT_START_MULTIPLIER (search / harness alignment).
DEFAULT_START_MULTIPLIER = 3


def cap_by_source(
    candidates: Sequence[Candidate],
    max_per_source: int,
) -> list[Candidate]:
    """Оставить не более ``max_per_source`` кандидатов на каждый ``source``.

    Порядок сохраняется. ``max_per_source <= 0`` — вернуть список как есть.
    Используется ``Candidate.source`` (property), не словарь.
    """
    if max_per_source <= 0:
        return list(candidates)
    counts: dict[str, int] = {}
    out: list[Candidate] = []
    for cand in candidates:
        src = cand.source
        n = counts.get(src, 0)
        if n >= max_per_source:
            continue
        counts[src] = n + 1
        out.append(cand)
    return out


def iterative_dedup(
    fetch: Callable[[int], list[Candidate]],
    top_k: int,
    max_per_source: int,
    pool_multiplier: int = DEFAULT_START_MULTIPLIER,
    total: int | None = None,
) -> list[Candidate]:
    """Запросить пул, применить cap, при нехватке увеличить пул (×2).

    Args:
        fetch: ``fetch(pool) -> list[Candidate]`` в порядке релевантности,
            не длиннее запрошенного ``pool``.
        top_k: целевой размер результата.
        max_per_source: лимит чанков на один source.
        pool_multiplier: стартовый пул = ``top_k * pool_multiplier``
            (default ``DEFAULT_START_MULTIPLIER``; при нехватке удваивается).
        total: размер коллекции (потолок пула); ``None`` — без потолка.

    Returns:
        Не более ``top_k`` кандидатов; дубли сверх cap не добираются.
        Если уникальных source мало — меньше ``top_k``.
    """
    top_k = max(1, int(top_k))
    if max_per_source <= 0:
        return fetch(top_k)[:top_k]

    pool = top_k * max(1, int(pool_multiplier))
    if total is not None:
        total = max(0, int(total))
        if total == 0:
            return []
        pool = min(pool, total)

    while True:
        candidates = fetch(pool)
        capped = cap_by_source(candidates, max_per_source)
        if len(capped) >= top_k:
            return capped[:top_k]
        if len(candidates) < pool:
            return capped[:top_k]
        if total is not None and pool >= total:
            return capped[:top_k]
        next_pool = pool * 2
        if total is not None:
            next_pool = min(next_pool, total)
        if next_pool <= pool:
            return capped[:top_k]
        pool = next_pool


__all__ = ["DEFAULT_START_MULTIPLIER", "cap_by_source", "iterative_dedup"]
