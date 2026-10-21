"""Тесты Reciprocal Rank Fusion."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mcp_1c_rag.rrf import rrf_fuse, rrf_order, rrf_ranks  # noqa: E402


def test_rrf_ranks_are_one_based() -> None:
    """Нумерация позиций начинается с 1."""
    assert rrf_ranks(["a", "b", "c"]) == {"a": 1, "b": 2, "c": 3}


def test_rrf_document_in_both_lists_wins() -> None:
    """Документ, найденный обоими источниками, получает больший скор."""
    cosine = ["a", "b", "c"]
    bm25 = ["c", "d"]
    fused = rrf_fuse([cosine, bm25], k=60)
    assert fused["c"] > fused["a"]
    assert fused["c"] > fused["b"]


def test_rrf_score_formula() -> None:
    """Скор равен сумме 1/(k+rank) по источникам."""
    fused = rrf_fuse([["a", "b"], ["b"]], k=10)
    assert fused["a"] == pytest.approx(1 / 11)
    assert fused["b"] == pytest.approx(1 / 12 + 1 / 11)


def test_rrf_weights_scale_sources() -> None:
    """Веса источников масштабируют вклад."""
    fused = rrf_fuse([["a"], ["b"]], k=10, weights=[1.0, 3.0])
    assert fused["b"] == pytest.approx(3 / 11)
    assert fused["a"] == pytest.approx(1 / 11)


def test_rrf_duplicates_inside_one_ranking_counted_once() -> None:
    """Дубликат id внутри одного списка учитывается один раз."""
    fused = rrf_fuse([["a", "a", "b"]], k=10)
    assert fused["a"] == pytest.approx(1 / 11)


def test_rrf_order_is_deterministic() -> None:
    """Порядок при равных скорах детерминированный (по id)."""
    fused = {"b": 0.1, "a": 0.1}
    assert rrf_order(fused) == ["a", "b"]


def test_rrf_empty_input() -> None:
    """Пустые списки дают пустой результат."""
    assert rrf_fuse([]) == {}
    assert rrf_fuse([[], []]) == {}


def test_rrf_length_mismatch_raises() -> None:
    """Несовпадение числа весов и ранжирований — ошибка."""
    with pytest.raises(ValueError):
        rrf_fuse([["a"]], weights=[1.0, 2.0])
