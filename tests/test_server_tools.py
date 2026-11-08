"""MCP tool surface: 4 tools; search(collection=all); legacy aliases."""

from __future__ import annotations

import asyncio
import inspect
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


def test_search_collection_all_uses_merge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """collection='all' идёт через merge.merged_search (обе коллекции)."""
    seen: list[tuple[str, str, int]] = []

    def fake_merged(
        query: str,
        mode: str,
        limit: int,
        collections: tuple[str, ...] | None = None,
        **kwargs: Any,
    ) -> list[Candidate]:
        seen.append((query, mode, limit))
        assert collections is None  # default: code + metadata
        return [Candidate(id="1", text="t", metadata={}, score=1.0)]

    monkeypatch.setattr(server.merge, "merged_search", fake_merged)
    monkeypatch.setattr(server, "format_candidates", lambda cands: "MERGED")
    out = server.search("НДС", top_k=3, mode="hybrid", collection="all")
    assert out == "MERGED"
    assert seen == [("НДС", "hybrid", 3)]


def test_search_collection_metadata_and_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """metadata_xml передаётся как есть; legacy code/metadata → served коллекции."""
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
        server,
        "format_candidates",
        lambda cands: f"n={len(cands)}",
    )
    assert server.search("q", collection="metadata_xml") == "n=1"
    assert calls[-1]["collection"] == "metadata_xml"
    assert server.search("q", collection="code") == "n=1"
    assert calls[-1]["collection"] == server.COLLECTION_CODE
    assert server.search("q", collection="metadata") == "n=1"
    assert calls[-1]["collection"] == server.COLLECTION_METADATA


def _tool_map() -> dict[str, Any]:
    return {t.name: t for t in asyncio.run(server.mcp.list_tools())}


def test_tools_are_annotated_read_only() -> None:
    for tool in _tool_map().values():
        hints = tool.annotations
        assert hints is not None, tool.name
        assert hints.read_only_hint is True
        assert hints.destructive_hint is False
        assert hints.idempotent_hint is True
        assert hints.open_world_hint is False


def test_search_schema_enumerates_mode_and_collection() -> None:
    props = _tool_map()["search"].input_schema["properties"]
    assert set(props["mode"]["enum"]) == {
        "",
        "lexical",
        "vector",
        "hybrid",
        "hybrid_rerank",
        "pearson",
    }
    collections = set(props["collection"]["enum"])
    assert {"code_proc", "metadata_xml", "all"} <= collections
    assert "config_xml" not in collections
    assert props["top_k"]["minimum"] == 1
    assert props["top_k"]["maximum"] == 50


def test_evaluate_retrieval_schema_accepts_canonical_collections() -> None:
    props = _tool_map()["evaluate_retrieval"].input_schema["properties"]
    assert {"code_proc", "metadata_xml"} <= set(props["collection"]["enum"])
    assert "config_xml" not in props["collection"]["enum"]


@pytest.mark.parametrize(
    "arguments",
    [
        {"query": "x", "mode": "bogus"},
        {"query": "x", "collection": "config_xml"},
        {"query": "x", "top_k": 0},
    ],
)
def test_search_rejects_out_of_schema_arguments(arguments: dict[str, Any]) -> None:
    from mcp.server.mcpserver.exceptions import ToolError

    with pytest.raises(ToolError, match="validation error"):
        asyncio.run(server.mcp.call_tool("search", arguments))


def test_eval_get_collection_accepts_canonical_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(server.eval_mod, "get_code_collection", lambda: "C")
    monkeypatch.setattr(server.eval_mod, "get_metadata_collection", lambda: "M")
    assert server.eval_mod._get_collection("code_proc") == ("C", "code")
    assert server.eval_mod._get_collection("metadata_xml") == ("M", "metadata")
    assert server.eval_mod._get_collection("") == ("C", "code")


def test_route_query_mcp_format() -> None:
    text = server.route_query("Какие проводки у документа РеализацияТоваровУслуг?")
    assert "tool=search" in text
    assert "mode=lexical" in text
    assert "collection=metadata_xml" in text
    assert "route=lexical" in text


def test_server_has_no_inline_score_normalization() -> None:
    """Бизнес-нормализация скоров живёт в merge, не в server."""
    source = inspect.getsource(server)
    assert "_merged_search" not in source
    assert "score / top" not in source
    assert "max_score" not in source
    assert "merged_search" in source
    assert "format_candidates" in source
