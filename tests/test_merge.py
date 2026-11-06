"""Тесты merge.merged_search: нормализация, dedup, limit, searcher."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mcp_1c_rag.merge import merged_search  # noqa: E402
from mcp_1c_rag.models import Candidate  # noqa: E402


def _cand(cid: str, score: float, source: str = "s") -> Candidate:
    return Candidate(
        id=cid,
        text=cid,
        metadata={"source": source},
        score=score,
    )


def _fake_searcher(mapping: dict[str, list[Candidate]]):
    def searcher(
        query: str,
        *,
        top_k: int,
        mode: str,
        collection: str,
    ) -> list[Candidate]:
        rows = mapping.get(collection, [])
        return [
            Candidate(
                id=c.id,
                text=c.text,
                metadata=dict(c.metadata),
                score=c.score,
                source_scores=dict(c.source_scores),
            )
            for c in rows[:top_k]
        ]

    return searcher


def test_max_normalization_order_and_collection_meta() -> None:
    """max: score/max_score; более «относительный» победитель выше."""
    searcher = _fake_searcher(
        {
            "code_proc": [_cand("bsl_proc:a", 10.0), _cand("bsl_proc:b", 5.0)],
            "metadata_xml": [_cand("metaobj:x", 2.0), _cand("metaobj:y", 1.0)],
        }
    )
    out = merged_search(
        "q",
        "hybrid",
        10,
        collections=("code_proc", "metadata_xml"),
        searcher=searcher,
        normalization="max",
    )
    # a: 10/10=1, x: 2/2=1, b: 5/10=0.5, y: 1/2=0.5
    assert [c.id for c in out[:2]] == ["bsl_proc:a", "metaobj:x"]
    assert out[0].metadata["collection"] == "code_proc"
    assert out[1].metadata["collection"] == "metadata_xml"


def test_dedup_by_id_keeps_first_after_sort() -> None:
    shared = "shared:1"
    searcher = _fake_searcher(
        {
            "code_proc": [_cand(shared, 9.0, "code")],
            "metadata_xml": [_cand(shared, 3.0, "meta")],
        }
    )
    out = merged_search(
        "q",
        "hybrid",
        5,
        collections=("code_proc", "metadata_xml"),
        searcher=searcher,
        normalization="max",
    )
    assert len(out) == 1
    assert out[0].id == shared
    # оба дают norm=1.0; побеждает тот, кто раньше в отсортированном списке
    # (стабильно по reverse sort — первый встреченный с max norm)
    assert out[0].metadata["collection"] in {"code_proc", "metadata_xml"}


def test_limit_respected() -> None:
    searcher = _fake_searcher(
        {
            "code_proc": [_cand(f"bsl_proc:{i}", float(10 - i)) for i in range(5)],
            "metadata_xml": [_cand(f"metaobj:{i}", float(5 - i)) for i in range(5)],
        }
    )
    out = merged_search(
        "q",
        "hybrid",
        3,
        collections=("code_proc", "metadata_xml"),
        searcher=searcher,
    )
    assert len(out) == 3


def test_rrf_prefers_top_of_both_lists() -> None:
    searcher = _fake_searcher(
        {
            "code_proc": [
                _cand("bsl_proc:top", 1.0),
                _cand("bsl_proc:low", 0.1),
            ],
            "metadata_xml": [
                _cand("bsl_proc:top", 0.5),
                _cand("metaobj:only", 0.4),
            ],
        }
    )
    out = merged_search(
        "q",
        "hybrid",
        10,
        collections=("code_proc", "metadata_xml"),
        searcher=searcher,
        normalization="rrf",
    )
    assert out[0].id == "bsl_proc:top"


def test_minmax_stretches_collection_range() -> None:
    searcher = _fake_searcher(
        {
            "code_proc": [_cand("bsl_proc:hi", 100.0), _cand("bsl_proc:lo", 0.0)],
            "metadata_xml": [_cand("metaobj:mid", 50.0)],
        }
    )
    out = merged_search(
        "q",
        "hybrid",
        10,
        collections=("code_proc", "metadata_xml"),
        searcher=searcher,
        normalization="minmax",
    )
    # hi -> 1.0, mid alone -> 1.0, lo -> 0.0
    assert {c.id for c in out[:2]} == {"bsl_proc:hi", "metaobj:mid"}
    assert out[-1].id == "bsl_proc:lo"


def test_default_normalization_is_max() -> None:
    searcher = _fake_searcher(
        {
            "code_proc": [_cand("bsl_proc:a", 8.0), _cand("bsl_proc:b", 4.0)],
            "metadata_xml": [_cand("metaobj:x", 3.0)],
        }
    )
    default = merged_search(
        "q", "hybrid", 10, collections=("code_proc", "metadata_xml"), searcher=searcher
    )
    explicit = merged_search(
        "q",
        "hybrid",
        10,
        collections=("code_proc", "metadata_xml"),
        searcher=searcher,
        normalization="max",
    )
    assert [c.id for c in default] == [c.id for c in explicit]


def test_unknown_normalization_raises() -> None:
    searcher = _fake_searcher({"code_proc": [_cand("a", 1.0)]})
    with pytest.raises(ValueError, match="normalization"):
        merged_search(
            "q",
            "hybrid",
            5,
            collections=("code_proc",),
            searcher=searcher,
            normalization="nope",  # type: ignore[arg-type]
        )
