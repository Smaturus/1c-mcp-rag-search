"""Unit-тесты SearchContext: кэши, маски, изоляция двух контекстов."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mcp_1c_rag.search_context import SearchContext  # noqa: E402


class FakeBm25:
    """Минимальный BM25-двойник: doc_ids + scores(query)."""

    def __init__(
        self,
        doc_ids: list[str],
        score_map: dict[str, float] | None = None,
    ) -> None:
        self.doc_ids = doc_ids
        self.n_docs = len(doc_ids)
        self._score_map = score_map or {}
        self.scores_calls = 0

    def scores(self, query: str) -> np.ndarray:
        self.scores_calls += 1
        out = np.zeros(self.n_docs, dtype=np.float32)
        for i, doc_id in enumerate(self.doc_ids):
            out[i] = float(self._score_map.get(doc_id, 0.0))
        return out


def _inject_fake(ctx: SearchContext, fake: FakeBm25) -> None:
    ctx._bm25 = fake  # type: ignore[assignment]


def test_bm25_returns_injected_fake() -> None:
    ctx = SearchContext()
    fake = FakeBm25(["bsl_proc:1", "metaobj:2"])
    _inject_fake(ctx, fake)
    assert ctx.bm25() is fake


def test_masked_bm25_scores_filters_by_prefix() -> None:
    ctx = SearchContext()
    fake = FakeBm25(
        ["bsl_proc:a", "metaobj:b", "bsl_proc:c"],
        {"bsl_proc:a": 3.0, "metaobj:b": 9.0, "bsl_proc:c": 2.0},
    )
    _inject_fake(ctx, fake)
    code_scores = ctx.masked_bm25_scores("q", "code_proc")
    assert code_scores[0] == pytest.approx(3.0)
    assert not np.isfinite(code_scores[1])
    assert code_scores[2] == pytest.approx(2.0)
    meta_scores = ctx.masked_bm25_scores("q", "metadata_xml")
    assert not np.isfinite(meta_scores[0])
    assert meta_scores[1] == pytest.approx(9.0)
    assert not np.isfinite(meta_scores[2])


def test_prefix_mask_cached() -> None:
    ctx = SearchContext()
    fake = FakeBm25(["bsl_proc:1", "metaobj:2"])
    _inject_fake(ctx, fake)
    m1 = ctx.prefix_mask("bsl_proc:")
    m2 = ctx.prefix_mask("bsl_proc:")
    assert m1 is m2
    assert list(m1) == [True, False]
    assert "bsl_proc:" in ctx._masks


def test_collection_count_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    class FakeColl:
        def count(self) -> int:
            return 42

    def fake_get(name: str, *, reset: bool = False) -> Any:
        calls.append(name)
        return FakeColl()

    monkeypatch.setattr("mcp_1c_rag.store.get_collection", fake_get)
    ctx = SearchContext()
    assert ctx.collection_count("code_proc") == 42
    assert ctx.collection_count("code_proc") == 42
    assert calls == ["code_proc"]


def test_two_contexts_isolated_in_one_process() -> None:
    """Главная цель P1a: два SearchContext → разные BM25-результаты."""
    ctx_a = SearchContext()
    ctx_b = SearchContext()
    fake_a = FakeBm25(
        ["bsl_proc:a1", "bsl_proc:a2"],
        {"bsl_proc:a1": 5.0, "bsl_proc:a2": 1.0},
    )
    fake_b = FakeBm25(
        ["bsl_proc:b1", "bsl_proc:b2"],
        {"bsl_proc:b1": 1.0, "bsl_proc:b2": 7.0},
    )
    _inject_fake(ctx_a, fake_a)
    _inject_fake(ctx_b, fake_b)

    scores_a = ctx_a.masked_bm25_scores("q", "code_proc")
    scores_b = ctx_b.masked_bm25_scores("q", "code_proc")
    assert ctx_a.bm25() is fake_a
    assert ctx_b.bm25() is fake_b
    assert ctx_a.bm25() is not ctx_b.bm25()
    assert float(scores_a.max()) == pytest.approx(5.0)
    assert float(scores_b.max()) == pytest.approx(7.0)
    assert list(ctx_a.bm25().doc_ids) == ["bsl_proc:a1", "bsl_proc:a2"]
    assert list(ctx_b.bm25().doc_ids) == ["bsl_proc:b1", "bsl_proc:b2"]


def test_bm25_missing_index_raises(tmp_path: Path) -> None:
    ctx = SearchContext(bm25_dir=tmp_path)
    with pytest.raises(FileNotFoundError) as exc_info:
        ctx.bm25()
    msg = str(exc_info.value)
    assert "BM25-индекс не найден" in msg
    assert str(tmp_path) in msg
    assert "build_bm25.py" in msg


def test_reset_clears_caches(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeColl:
        def count(self) -> int:
            return 7

    monkeypatch.setattr(
        "mcp_1c_rag.store.get_collection",
        lambda name, *, reset=False: FakeColl(),
    )
    ctx = SearchContext()
    fake = FakeBm25(["bsl_proc:1"])
    _inject_fake(ctx, fake)
    _ = ctx.prefix_mask("bsl_proc:")
    _ = ctx.collection_count("code_proc")
    assert ctx._bm25 is fake
    assert ctx._masks
    assert ctx._collection_counts
    ctx.reset()
    assert ctx._bm25 is None
    assert ctx._masks == {}
    assert ctx._collection_counts == {}
