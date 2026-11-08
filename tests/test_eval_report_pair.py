"""Unit-тесты scripts/eval_report_pair.py (синтетика, без реальных индексов).

Покрывают контракт из плана (фаза 4.4):
- одинаковые id -> корректный Δ = new − old;
- разные наборы id -> fail;
- отсутствующий id / повтор id / нет метрики -> fail;
- нулевая дисперсия -> Δ=0, все ничьи, p=1, CI [0,0], без падения;
- перестановка порядка запросов -> тот же результат.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from eval_report_pair import (  # noqa: E402
    PairError,
    RunSet,
    analyse,
    bootstrap_ci,
    metric_by_id,
)

MODE = "hybrid"
METRIC = "ndcg@10"


def make_run_set(values: dict[str, float]) -> RunSet:
    return RunSet(description="synthetic", files=[], n_runs=1, values=values)


def test_same_ids_correct_delta():
    old = make_run_set({"a": 0.40, "b": 0.50, "c": 0.30})
    new = make_run_set({"a": 0.42, "b": 0.49, "c": 0.33})
    r = analyse(old, new, metric=METRIC, mode=MODE)
    assert r["n_queries"] == 3
    assert r["mean_old"] == pytest.approx((0.40 + 0.50 + 0.30) / 3)
    assert r["mean_new"] == pytest.approx((0.42 + 0.49 + 0.33) / 3)
    assert r["mean_delta"] == pytest.approx(r["mean_new"] - r["mean_old"])
    # Δ: a=+0.02 (win), b=−0.01 (loss), c=+0.03 (win) → 2 wins / 1 loss / 0 ties
    assert r["wins"] == 2
    assert r["losses"] == 1
    assert r["ties"] == 0
    assert "per_query" in r and len(r["per_query"]) == 3


def test_different_id_sets_fail():
    old = make_run_set({"a": 0.4, "b": 0.5})
    new = make_run_set({"a": 0.4, "c": 0.5})
    with pytest.raises(PairError):
        analyse(old, new, metric=METRIC, mode=MODE)


def test_metric_by_id_missing_metric_fail():
    items = [{"id": "x", "type": "exact"}]
    with pytest.raises(PairError):
        metric_by_id(items, METRIC, MODE)


def test_metric_by_id_duplicate_id_fail():
    items = [
        {"id": "x", METRIC: 0.5},
        {"id": "x", METRIC: 0.6},
    ]
    with pytest.raises(PairError):
        metric_by_id(items, METRIC, MODE)


def test_zero_variance():
    old = make_run_set({"a": 0.4, "b": 0.5, "c": 0.6, "d": 0.7})
    new = make_run_set({"a": 0.4, "b": 0.5, "c": 0.6, "d": 0.7})
    r = analyse(old, new, metric=METRIC, mode=MODE)
    assert r["mean_delta"] == pytest.approx(0.0)
    assert r["wins"] == 0
    assert r["losses"] == 0
    assert r["ties"] == 4
    assert r["bootstrap"]["ci_low"] == pytest.approx(0.0)
    assert r["bootstrap"]["ci_high"] == pytest.approx(0.0)
    assert r["bootstrap"]["p_value"] == pytest.approx(1.0)


def test_permutation_invariant():
    vals = {f"q{i}": 0.3 + 0.01 * i for i in range(20)}
    old = make_run_set(vals)
    new = make_run_set({k: v + 0.001 * (i % 3) for i, (k, v) in enumerate(vals.items())})
    r1 = analyse(old, new, metric=METRIC, mode=MODE)
    # Переставить порядок ключей в dict — результат не должен измениться.
    shuffled = make_run_set(dict(reversed(list(vals.items()))))
    shuffled_new = make_run_set(dict(reversed(list(new.values.items()))))
    r2 = analyse(shuffled, shuffled_new, metric=METRIC, mode=MODE)
    assert r1["mean_delta"] == pytest.approx(r2["mean_delta"])
    assert r1["wins"] == r2["wins"]
    assert r1["losses"] == r2["losses"]
    assert r1["ttest"]["statistic"] == pytest.approx(r2["ttest"]["statistic"])


def test_bootstrap_ci_contains_mean():
    rng = np.random.default_rng(0)
    deltas = rng.normal(0.02, 0.05, size=50)
    low, high, p = bootstrap_ci(deltas, n_boot=2000, seed=42)
    assert low <= float(deltas.mean()) <= high
    assert 0.0 <= p <= 1.0


def test_too_few_pairs_fail():
    old = make_run_set({"a": 0.4})
    new = make_run_set({"a": 0.5})
    with pytest.raises(PairError):
        analyse(old, new, metric=METRIC, mode=MODE)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
