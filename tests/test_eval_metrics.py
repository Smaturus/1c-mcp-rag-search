"""Тесты метрик retrieval (file-level, чистые функции без индекса)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mcp_1c_homework.eval_metrics import (  # noqa: E402
    aggregate,
    evaluate_query,
    format_markdown_table,
    hit_at_1,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
)

RELEVANT = {"a.bsl", "b.bsl"}


def test_hit_at_1_positive() -> None:
    """Первый результат из эталона — Hit@1 = 1."""
    assert hit_at_1(["a.bsl", "x.bsl"], RELEVANT) == 1.0


def test_hit_at_1_negative() -> None:
    """Первый результат не из эталона — Hit@1 = 0."""
    assert hit_at_1(["x.bsl", "a.bsl"], RELEVANT) == 0.0
    assert hit_at_1([], RELEVANT) == 0.0


def test_mrr_uses_first_relevant_rank() -> None:
    """MRR = 1/rank первого релевантного результата."""
    assert mrr_at_k(["x.bsl", "y.bsl", "a.bsl"], RELEVANT, k=10) == pytest.approx(1 / 3)


def test_mrr_zero_when_nothing_found() -> None:
    """Релевантных в top-k нет — MRR = 0."""
    assert mrr_at_k(["x.bsl"], RELEVANT, k=10) == 0.0


def test_recall_counts_unique_sources() -> None:
    """Recall@k считает уникальные найденные файлы."""
    assert recall_at_k(["a.bsl", "a.bsl", "x.bsl"], RELEVANT, k=10) == pytest.approx(0.5)


def test_recall_capped_by_k() -> None:
    """Если релевантных файлов больше k, знаменатель — k."""
    relevant = {f"f{i}.bsl" for i in range(20)}
    assert recall_at_k(["f0.bsl", "f1.bsl"], relevant, k=10) == pytest.approx(0.2)


def test_ndcg_perfect_ranking_is_one() -> None:
    """Идеальный порядок даёт nDCG = 1."""
    assert ndcg_at_k(["a.bsl", "b.bsl"], RELEVANT, k=10) == pytest.approx(1.0)


def test_ndcg_penalizes_late_hits() -> None:
    """Позднее попадание даёт nDCG < 1."""
    score = ndcg_at_k(["x.bsl", "y.bsl", "a.bsl"], RELEVANT, k=10)
    assert 0.0 < score < 1.0


def test_evaluate_query_combines_metrics() -> None:
    """evaluate_query возвращает все метрики и источники."""
    metrics = evaluate_query(["a.bsl", "x.bsl"], RELEVANT, k=10, latency_ms=42.0)
    assert metrics.hit_at_1 == 1.0
    assert metrics.mrr_at_10 == 1.0
    assert metrics.recall_at_10 == pytest.approx(0.5)
    assert metrics.latency_ms == 42.0
    assert metrics.retrieved_sources == ("a.bsl", "x.bsl")


def test_aggregate_averages_metrics() -> None:
    """aggregate усредняет метрики по запросам."""
    items = [
        evaluate_query(["a.bsl"], RELEVANT, k=10, latency_ms=10.0),
        evaluate_query(["x.bsl"], RELEVANT, k=10, latency_ms=30.0),
    ]
    agg = aggregate(items)
    assert agg.n_queries == 2
    assert agg.values["hit@1"] == pytest.approx(0.5)
    assert agg.values["latency_ms"] == pytest.approx(20.0)


def test_aggregate_empty_is_safe() -> None:
    """Пустой список даёт пустые средние без деления на ноль."""
    agg = aggregate([])
    assert agg.n_queries == 0
    assert agg.values == {}


def test_markdown_table_contains_all_modes() -> None:
    """Markdown-таблица содержит строку на каждую конфигурацию."""
    items = [evaluate_query(["a.bsl"], RELEVANT, k=10)]
    table = format_markdown_table({"vector": aggregate(items)})
    assert "| `vector` |" in table
    assert "Hit@1" in table
