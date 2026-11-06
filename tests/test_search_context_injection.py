"""Проброс context= через search / bm25_search; изоляция от default_context."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mcp_1c_rag import search_modes  # noqa: E402
from mcp_1c_rag.models import Candidate  # noqa: E402
from mcp_1c_rag.search_context import SearchContext  # noqa: E402


class FakeBm25:
    def __init__(self, doc_ids: list[str], values: list[float]) -> None:
        self.doc_ids = doc_ids
        self.n_docs = len(doc_ids)
        self._values = np.asarray(values, dtype=np.float32)

    def scores(self, query: str) -> np.ndarray:
        return self._values.copy()


@pytest.fixture(autouse=True)
def _reset_default_context() -> Any:
    search_modes.default_context.reset()
    yield
    search_modes.default_context.reset()


def test_bm25_search_uses_passed_context_not_default() -> None:
    default_fake = FakeBm25(["bsl_proc:default"], [99.0])
    ctx_fake = FakeBm25(["bsl_proc:ctx"], [4.0])
    search_modes.default_context._bm25 = default_fake  # type: ignore[assignment]
    ctx = SearchContext()
    ctx._bm25 = ctx_fake  # type: ignore[assignment]

    hits = search_modes.bm25_search("q", collection="code_proc", top_k=5, context=ctx)
    assert hits == [("bsl_proc:ctx", 4.0)]
    assert search_modes.default_context.bm25() is default_fake


def test_search_passes_context_to_search_impl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx = SearchContext()
    seen: dict[str, Any] = {}

    def fake_impl(
        query: str,
        top_k: int,
        mode: str | None,
        collection: str,
        *,
        context: SearchContext | None = None,
    ) -> list[Candidate]:
        seen["context"] = context
        seen["query"] = query
        return [Candidate(id="x", text="", metadata={}, score=1.0)]

    monkeypatch.setattr(search_modes, "_search_impl", fake_impl)
    out = search_modes.search("hello", top_k=3, max_per_source=0, context=ctx)
    assert len(out) == 1
    assert seen["context"] is ctx
    assert seen["query"] == "hello"


def test_default_and_explicit_context_are_isolated() -> None:
    default_fake = FakeBm25(
        ["bsl_proc:d1", "bsl_proc:d2"],
        [1.0, 2.0],
    )
    other_fake = FakeBm25(
        ["bsl_proc:o1", "bsl_proc:o2"],
        [8.0, 0.5],
    )
    search_modes.default_context._bm25 = default_fake  # type: ignore[assignment]
    other = SearchContext()
    other._bm25 = other_fake  # type: ignore[assignment]

    via_default = search_modes._masked_bm25_scores("q", "code_proc")
    via_other = other.masked_bm25_scores("q", "code_proc")
    assert float(via_default.max()) == pytest.approx(2.0)
    assert float(via_other.max()) == pytest.approx(8.0)
    assert search_modes.default_context.bm25() is not other.bm25()
