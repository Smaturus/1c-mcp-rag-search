"""Тесты presentation-слоя format_candidates."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mcp_1c_rag import search_modes  # noqa: E402
from mcp_1c_rag.formatting import format_candidates  # noqa: E402
from mcp_1c_rag.models import Candidate  # noqa: E402


def test_format_empty_message() -> None:
    text = format_candidates([])
    assert "Ничего не найдено" in text
    assert "build_bm25.py" in text


def test_format_source_chunk_score_and_source_scores() -> None:
    cands = [
        Candidate(
            id="bsl_proc:1",
            text="тело процедуры",
            metadata={"source": "Doc/ObjectModule.bsl", "chunk_index": 2},
            score=0.8754321,
            source_scores={"rrf": 0.0164, "bm25_rank": 1.0, "score": 9.9},
        )
    ]
    text = format_candidates(cands)
    assert "[1] source=Doc/ObjectModule.bsl chunk=2 score=0.8754" in text
    assert "rrf=0.0164" in text
    assert "bm25_rank=1.0" in text
    assert "score=9.9" not in text  # ключ score в source_scores пропускается
    assert "тело процедуры" in text


def test_format_reexport_matches_formatting() -> None:
    cands = [
        Candidate(
            id="x",
            text="t",
            metadata={"source": "s", "chunk_index": 0},
            score=1.0,
            source_scores={"cosine": 0.5},
        )
    ]
    assert format_candidates(cands) == search_modes.format_candidates(cands)
