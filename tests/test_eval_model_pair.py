"""Unit-тесты scripts/eval_model_pair.py (синтетика, без реальных индексов).

Покрывают контракт из плана (фаза 1.1):
- одинаковые id -> корректный Δ;
- разные длины (несовпадающие наборы id) -> fail;
- отсутствующий id в одной модели -> fail;
- нулевая дисперсия -> Δ=0, все ничьи, без падения;
- перестановка порядка запросов -> тот же результат.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from eval_model_pair import (  # noqa: E402
    PairError,
    average_runs,
    compare,
    metric_by_id,
)

MODE = "hybrid"
METRIC = "ndcg@10"


def write_report(path: Path, rows: list[tuple[str, float]]) -> None:
    """Записать синтетический report_pipeline.json с одним режимом."""
    payload = {
        "per_query": {
            MODE: [
                {"id": qid, METRIC: value}
                for qid, value in rows
            ]
        }
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_same_ids_computes_delta() -> None:
    small = {"a": 0.30, "b": 0.40, "c": 0.50}
    large = {"a": 0.35, "b": 0.41, "c": 0.52}
    result = compare(small, large, n_boot=1000, seed=42)
    assert result["n_queries"] == 3
    assert result["mean_delta"] == pytest.approx(0.08 / 3, abs=1e-9)
    assert result["mean_small"] == pytest.approx(0.40, abs=1e-9)
    assert result["mean_large"] == pytest.approx(0.426666666666, abs=1e-6)
    # Δ = +0.02, +0.01, +0.02 -> все победы large
    assert (result["wins"], result["ties"], result["losses"]) == (3, 0, 0)


def test_different_lengths_fail() -> None:
    small = {"a": 0.3, "b": 0.4}
    large = {"a": 0.3}  # не хватает b
    with pytest.raises(PairError):
        compare(small, large, n_boot=100, seed=42)


def test_missing_id_fail() -> None:
    small = {"a": 0.3, "b": 0.4, "c": 0.5}
    large = {"a": 0.3, "b": 0.4, "d": 0.5}  # c <-> d
    with pytest.raises(PairError):
        compare(small, large, n_boot=100, seed=42)


def test_zero_variance() -> None:
    small = {"a": 0.3, "b": 0.4, "c": 0.5, "d": 0.2}
    large = {"a": 0.3, "b": 0.4, "c": 0.5, "d": 0.2}
    result = compare(small, large, n_boot=1000, seed=42)
    assert result["mean_delta"] == 0.0
    assert (result["wins"], result["ties"], result["losses"]) == (0, 4, 0)
    assert result["bootstrap"]["ci_low"] == 0.0
    assert result["bootstrap"]["ci_high"] == 0.0
    assert result["bootstrap"]["p_value"] == 1.0
    # t-test с нулевой дисперсией даёт nan — главное, что не падает
    assert math.isnan(result["ttest"]["p_value"])


def test_permuted_order_same_result(tmp_path: Path) -> None:
    rows_small = [("a", 0.30), ("b", 0.40), ("c", 0.50), ("d", 0.25)]
    rows_large = [("c", 0.52), ("a", 0.35), ("d", 0.25), ("b", 0.41)]
    f_small = tmp_path / "small.json"
    f_large = tmp_path / "large.json"
    write_report(f_small, rows_small)
    write_report(f_large, rows_large)
    small = average_runs([f_small], MODE, METRIC)
    large = average_runs([f_large], MODE, METRIC)
    assert list(small) == sorted(small)
    assert small == {"a": 0.30, "b": 0.40, "c": 0.50, "d": 0.25}
    assert large == {"a": 0.35, "b": 0.41, "c": 0.52, "d": 0.25}
    result = compare(small, large, n_boot=1000, seed=42)
    assert result["n_queries"] == 4
    # Δ: +0.05, +0.01, +0.02, 0.0 -> 3 победы, 1 ничья
    assert (result["wins"], result["ties"], result["losses"]) == (3, 1, 0)


def test_average_runs_requires_consistent_ids(tmp_path: Path) -> None:
    f1 = tmp_path / "r1.json"
    f2 = tmp_path / "r2.json"
    write_report(f1, [("a", 0.3), ("b", 0.4)])
    write_report(f2, [("a", 0.5), ("c", 0.6)])  # b <-> c
    with pytest.raises(PairError):
        average_runs([f1, f2], MODE, METRIC)


def test_metric_by_id_missing_metric(tmp_path: Path) -> None:
    with pytest.raises(PairError):
        metric_by_id([{"id": "a"}], METRIC, MODE)
