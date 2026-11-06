"""Unit-тесты scripts/eval_regression_harness.py (без модели и индексов)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from eval_regression_harness import (  # noqa: E402
    build_metric_report,
    filter_gold_by_scope,
    gate_non_inferiority,
    load_goldset,
    metric_value,
    paired_compare,
    resolve_delta_floor,
)
from mcp_1c_rag.dedup import cap_by_source  # noqa: E402
from mcp_1c_rag.models import Candidate  # noqa: E402
from mcp_1c_rag.rrf import rrf_fuse, rrf_order  # noqa: E402


def test_delta_floor_v2():
    assert resolve_delta_floor(Path("data/eval/gold_set_v2.jsonl"), None) == pytest.approx(
        0.028
    )


def test_delta_floor_v3():
    assert resolve_delta_floor(
        Path("data/eval/gold_set_v3_holdout.jsonl"), None
    ) == pytest.approx(0.020)


def test_delta_floor_v4():
    assert resolve_delta_floor(
        Path("data/eval/gold_set_v4_holdout.jsonl"), None
    ) == pytest.approx(0.020)


def test_delta_floor_explicit_overrides():
    assert resolve_delta_floor(Path("data/eval/gold_set_v2.jsonl"), 0.05) == pytest.approx(
        0.05
    )


def test_delta_floor_unknown_fails():
    with pytest.raises(SystemExit):
        resolve_delta_floor(Path("data/eval/unknown_gold.jsonl"), None)


def test_scope_filter(tmp_path: Path) -> None:
    """Фильтр по scope проверяется на синтетическом gold-set (без приватных данных)."""
    rows = [
        {"id": "1", "query": "q1", "scope": "code", "relevant_sources": ["a.bsl"]},
        {"id": "2", "query": "q2", "scope": "none", "relevant_sources": []},
        {"id": "3", "query": "q3", "scope": "code", "relevant_sources": ["b.bsl"]},
    ]
    golden = tmp_path / "gold.jsonl"
    golden.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    loaded = load_goldset(golden)
    assert len(filter_gold_by_scope(loaded, None)) == 3
    assert len(filter_gold_by_scope(loaded, "code")) == 2
    assert len(filter_gold_by_scope(loaded, "none")) == 1

    # v2-подобный набор без поля scope: фильтр игнорируется
    no_scope = tmp_path / "gold_no_scope.jsonl"
    no_scope.write_text(
        "\n".join(
            json.dumps({"id": str(i), "query": f"q{i}", "relevant_sources": []})
            for i in range(3)
        )
        + "\n",
        encoding="utf-8",
    )
    assert len(filter_gold_by_scope(load_goldset(no_scope), "code")) == 3


def test_metrics_not_mixed():
    """Две метрики дают две независимые записи (не один смешанный список)."""
    baseline_ndcg = [0.4, 0.5, 0.3]
    candidate_ndcg = [0.42, 0.49, 0.33]
    baseline_recall = [0.6, 0.7, 0.5]
    candidate_recall = [0.65, 0.68, 0.55]

    rep_ndcg = build_metric_report(
        baseline_ndcg, candidate_ndcg, metric="ndcg@10", delta_floor=0.028
    )
    rep_recall = build_metric_report(
        baseline_recall, candidate_recall, metric="recall@10", delta_floor=0.028
    )

    assert rep_ndcg["baseline"] == pytest.approx(sum(baseline_ndcg) / 3)
    assert rep_recall["baseline"] == pytest.approx(sum(baseline_recall) / 3)
    assert rep_ndcg["delta"] != pytest.approx(rep_recall["delta"])
    assert rep_ndcg["n"] == 3 and rep_recall["n"] == 3


def test_gate_non_inferiority():
    # CI_lower > -δ → pass
    assert gate_non_inferiority(-0.01, 0.028) is True
    # CI_lower == -δ → fail (строгое >)
    assert gate_non_inferiority(-0.028, 0.028) is False
    # CI_lower < -δ → fail
    assert gate_non_inferiority(-0.05, 0.028) is False

    # Через build_metric_report: небольшой плюс → gate pass
    b = [0.4, 0.5, 0.3, 0.45]
    c = [0.41, 0.51, 0.31, 0.46]
    rep = build_metric_report(b, c, metric="ndcg@10", delta_floor=0.028)
    assert rep["gate"] is True

    # Сильный минус → gate fail
    c_bad = [0.1, 0.1, 0.1, 0.1]
    rep_bad = build_metric_report(b, c_bad, metric="ndcg@10", delta_floor=0.028)
    assert rep_bad["ci_low"] < -0.028
    assert rep_bad["gate"] is False


def test_vector_sanity_zero():
    """Одинаковые vector-метрики → Δ=0, CI=[0,0], gate True."""
    vals = [0.3, 0.4, 0.5, 0.6]
    rep = build_metric_report(vals, vals, metric="ndcg@10", delta_floor=0.028)
    assert rep["delta"] == pytest.approx(0.0)
    assert rep["ci_low"] == pytest.approx(0.0)
    assert rep["ci_high"] == pytest.approx(0.0)
    assert rep["wins"] == 0 and rep["losses"] == 0 and rep["ties"] == 4
    # Для gated-метрики gate смотрит non-inferiority; 0 > -δ → True
    assert rep["gate"] is True
    # Явная проверка sanity-условия
    assert abs(rep["delta"]) < 1e-12
    assert abs(rep["ci_low"]) < 1e-12 and abs(rep["ci_high"]) < 1e-12


def test_cap_by_source_shared_helper():
    cands = [
        Candidate(id=f"a{i}", text="", metadata={"source": "file1"}, score=1.0 - i * 0.01)
        for i in range(5)
    ] + [
        Candidate(id=f"b{i}", text="", metadata={"source": "file2"}, score=0.5 - i * 0.01)
        for i in range(3)
    ]
    capped = cap_by_source(cands, max_per_source=2)
    assert len(capped) == 4
    assert sum(c.source == "file1" for c in capped) == 2
    assert sum(c.source == "file2" for c in capped) == 2
    assert cap_by_source(cands, 0) == cands


def test_rrf_fuse_order_unit():
    """RRF без модели: детерминированный порядок."""
    fused = rrf_fuse([["a", "b", "c"], ["b", "d", "a"]], k=60)
    order = rrf_order(fused)
    assert order[0] == "b"  # лучший в обоих списках
    assert set(order) == {"a", "b", "c", "d"}


def test_unique_sources_metric():
    sources = ["f1", "f1", "f2", "f3", "f2"]
    assert metric_value("unique_sources@10", sources, ["f1"], k=10) == 3.0


def test_paired_compare_wtl():
    b = [0.4, 0.5, 0.3]
    c = [0.42, 0.49, 0.33]
    r = paired_compare(b, c)
    assert r["wins"] == 2
    assert r["losses"] == 1
    assert r["ties"] == 0
    assert r["n"] == 3
