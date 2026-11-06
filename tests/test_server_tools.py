"""MCP tool surface: 4 tools; search(collection=all); legacy aliases."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mcp_1c_rag import server  # noqa: E402
from mcp_1c_rag.models import Candidate  # noqa: E402


def _tool_names() -> list[str]:
    tools = asyncio.run(server.mcp.list_tools())
    return sorted(t.name for t in tools)


def test_mcp_exposes_exactly_four_tools() -> None:
    assert _tool_names() == [
        "evaluate_retrieval",
        "index_stats",
        "route_query",
        "search",
    ]


def test_legacy_aliases_callable_but_not_tools() -> None:
    for name in (
        "codesearch",
        "metadatasearch",
        "search_1c_metadata",
        "search_knowledge_base",
        "search_hybrid",
    ):
        assert callable(getattr(server, name))
        assert name not in _tool_names()


def test_search_collection_all_uses_merged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """collection='all' идёт через _merged_search (обе коллекции)."""
    seen: list[tuple[str, str, int]] = []

    def fake_merged(
        query: str,
        mode: str,
        limit: int,
        collections: tuple[str, ...] | None = None,
    ) -> str:
        seen.append((query, mode, limit))
        assert collections is None  # default: code + metadata
        return "MERGED"

    monkeypatch.setattr(server, "_merged_search", fake_merged)
    out = server.search("НДС", top_k=3, mode="hybrid", collection="all")
    assert out == "MERGED"
    assert seen == [("НДС", "hybrid", 3)]


def test_search_collection_metadata_and_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """metadata_xml и legacy code передаются в search_modes.search."""
    calls: list[dict[str, Any]] = []

    def fake_search(**kwargs: Any) -> list[Candidate]:
        calls.append(kwargs)
        return [
            Candidate(
                id="x",
                text="t",
                metadata={"source": "s"},
                score=0.9,
            )
        ]

    monkeypatch.setattr(server.search_modes, "search", fake_search)
    monkeypatch.setattr(
        server.search_modes,
        "format_candidates",
        lambda cands: f"n={len(cands)}",
    )
    assert server.search("q", collection="metadata_xml") == "n=1"
    assert calls[-1]["collection"] == "metadata_xml"
    assert server.search("q", collection="code") == "n=1"
    assert calls[-1]["collection"] == "code"


def test_route_query_mcp_format() -> None:
    text = server.route_query("Какие проводки у документа РеализацияТоваровУслуг?")
    assert "tool=search" in text
    assert "mode=lexical" in text
    assert "collection=metadata_xml" in text
    assert "route=lexical" in text
