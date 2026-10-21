"""Reciprocal Rank Fusion (RRF) для гибридного поиска.

Идея: сырые скоры cosine и BM25 несравнимы по масштабу, поэтому смешиваем
только позиции в ранжированиях:

    score(d) = Σ_i  w_i / (k_rrf + rank_i(d))

где ``rank_i(d)`` — позиция документа в i-м списке (нумерация с 1).
Классическое значение ``k_rrf = 60``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence


def rrf_fuse(
    rankings: Sequence[Sequence[str]],
    *,
    k: int = 60,
    weights: Sequence[float] | None = None,
) -> dict[str, float]:
    """Слить несколько ранжирований (списков id) в один словарь скоров.

    Args:
        rankings: Списки id, упорядоченные от лучшего к худшему.
        k: Сглаживающая константа RRF (обычно 60).
        weights: Необязательные веса источников; по умолчанию все равны 1.

    Returns:
        ``{doc_id: rrf_score}``; чем больше, тем релевантнее.
    """
    if not rankings:
        return {}
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("weights и rankings должны быть одинаковой длины")

    fused: dict[str, float] = {}
    for ranking, weight in zip(rankings, weights, strict=False):
        if weight <= 0:
            continue
        seen: set[str] = set()
        for rank, doc_id in enumerate(ranking, start=1):
            if doc_id in seen:  # дубликат внутри одного списка не учитываем
                continue
            seen.add(doc_id)
            fused[doc_id] = fused.get(doc_id, 0.0) + weight / (k + rank)
    return fused


def rrf_order(fused: dict[str, float]) -> list[str]:
    """Отсортировать id по убыванию RRF-скора (детерминированно)."""
    return sorted(fused, key=lambda doc_id: (-fused[doc_id], doc_id))


def rrf_ranks(ranking: Iterable[str]) -> dict[str, int]:
    """Позиции (с 1) элементов ранжирования — удобно для отладки и тестов."""
    return {doc_id: rank for rank, doc_id in enumerate(ranking, start=1)}


__all__ = ["rrf_fuse", "rrf_order", "rrf_ranks"]
