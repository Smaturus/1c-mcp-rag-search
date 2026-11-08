"""Интеграция observability в server tools (без live индекса)."""

from __future__ import annotations

import asyncio
import inspect
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mcp_1c_rag import observability as obs  # noqa: E402
from mcp_1c_rag import server  # noqa: E402
from mcp_1c_rag.models import Candidate  # noqa: E402
from mcp_1c_rag.startup import ValidationReport  # noqa: E402


@pytest.fixture(autouse=True)
def _reset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MCP_1C_LOG_LEVEL", "WARNING")
    obs.reset_metrics()
    obs.set_startup_report(ValidationReport(issues=[]))
    obs.setup_logging(level="WARNING", json_logs=False, log_queries=False)


def test_search_success_increments_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_search(**kwargs: Any) -> list[Candidate]:
        return [
            Candidate(id="a", text="t", metadata={"source": "s"}, score=0.9),
            Candidate(id="b", text="t2", metadata={"source": "s2"}, score=0.8),
        ]

    monkeypatch.setattr(server.search_modes, "search", fake_search)
    monkeypatch.setattr(server, "format_candidates", lambda c: f"n={len(c)}")
    out = server.search("НДС", top_k=2, mode="hybrid", collection="code_proc")
    assert out == "n=2"
    snap = obs.snapshot()
    assert snap["requests_total"] == 1
    assert snap["errors_total"] == 0
    assert snap["empty_results_total"] == 0
    assert snap["latency_ms_last"] > 0
    assert snap["by_tool"]["search"]["requests"] == 1


def test_search_error_increments_errors_keeps_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(**kwargs: Any) -> list[Candidate]:
        raise FileNotFoundError("BM25-индекс не найден: /tmp/x")

    monkeypatch.setattr(server.search_modes, "search", boom)
    out = server.search("q", top_k=3, mode="hybrid", collection="code_proc")
    assert "BM25-индекс не найден" in out
    assert "build_bm25.py" in out
    snap = obs.snapshot()
    assert snap["requests_total"] == 1
    assert snap["errors_total"] == 1
    assert snap["by_tool"]["search"]["errors"] == 1


def test_index_stats_contains_status_and_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeColl:
        def count(self) -> int:
            return 7

    class FakeClient:
        def get_collection(self, name: str) -> FakeColl:
            return FakeColl()

    monkeypatch.setattr("mcp_1c_rag.store.get_client", lambda: FakeClient())
    obs.record_request("search", latency_ms=11.0, ok=True, result_count=1)
    obs.record_request(
        "search", latency_ms=4.0, ok=False, error_type="ValueError"
    )
    text = server.index_stats()
    assert "status=ok" in text
    assert "startup_warnings=0" in text
    assert "requests_total=2" in text
    assert "errors_total=1" in text
    assert "empty_results_total=0" in text
    assert "latency_ms_avg=" in text
    assert "latency_ms_max=" in text
    assert "default_mode=" in text
    assert "query=" not in text


def test_tool_surface_still_four() -> None:
    tools = asyncio.run(server.mcp.list_tools())
    names = sorted(t.name for t in tools)
    assert names == [
        "evaluate_retrieval",
        "index_stats",
        "route_query",
        "search",
    ]
    source = inspect.getsource(server)
    assert source.count("@mcp.tool(") == 4
    assert "def health" not in source
