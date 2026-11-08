"""Фоновый прогрев: порядок шагов, сбои, ожидание, интеграция с server tools."""

from __future__ import annotations

import sys
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mcp_1c_rag import server, warmup  # noqa: E402
from mcp_1c_rag.models import Candidate  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_warmup() -> Iterator[None]:
    warmup.reset()
    yield
    warmup.reset()


def _join(thread: threading.Thread | None) -> None:
    assert thread is not None
    thread.join(timeout=5)
    assert not thread.is_alive()


def test_not_started_is_ready_immediately() -> None:
    assert warmup.wait_ready(timeout=0) is True
    assert warmup.is_running() is False
    assert warmup.status_lines() == ["warmup=off"]


def test_steps_run_in_order_and_mark_done() -> None:
    calls: list[str] = []
    thread = warmup.start(
        [("a", lambda: calls.append("a")), ("b", lambda: calls.append("b"))]
    )
    _join(thread)
    assert calls == ["a", "b"]
    assert warmup.wait_ready(timeout=0) is True
    lines = warmup.status_lines()
    assert "warmup=done" in lines
    assert any(line.startswith("warmup_duration_s=") for line in lines)


def test_failed_step_does_not_stop_others() -> None:
    calls: list[str] = []

    def boom() -> None:
        raise RuntimeError("index busy")

    thread = warmup.start([("bad", boom), ("good", lambda: calls.append("good"))])
    _join(thread)
    assert calls == ["good"]
    lines = warmup.status_lines()
    assert "warmup=partial" in lines
    assert "warmup_failed_steps=bad" in lines


def test_wait_ready_blocks_until_finished_and_no_double_start() -> None:
    release = threading.Event()
    thread = warmup.start([("slow", lambda: release.wait(5))])
    assert warmup.is_running() is True
    assert warmup.wait_ready(timeout=0.05) is False
    assert warmup.start([("other", lambda: None)]) is None
    release.set()
    _join(thread)
    assert warmup.wait_ready(timeout=0) is True


def test_delay_keeps_status_running_before_steps() -> None:
    calls: list[str] = []
    thread = warmup.start([("a", lambda: calls.append("a"))], delay_s=0.3)
    assert warmup.is_running() is True
    assert calls == []
    assert warmup.wait_ready(timeout=0.05) is False
    _join(thread)
    assert calls == ["a"]
    assert "warmup=done" in warmup.status_lines()


def test_search_tools_wait_for_warmup(monkeypatch: pytest.MonkeyPatch) -> None:
    waited: list[str] = []
    monkeypatch.setattr(server.warmup, "wait_ready", lambda: waited.append("w") or True)
    monkeypatch.setattr(
        server.search_modes,
        "search",
        lambda **_kw: [Candidate(id="x", text="t", metadata={}, score=1.0)],
    )
    monkeypatch.setattr(server, "format_candidates", lambda cands: "ok")
    monkeypatch.setattr(server.eval_mod, "evaluate_retrieval", lambda **_kw: "eval")
    assert server.search("q", collection="metadata_xml") == "ok"
    assert server.evaluate_retrieval("q", collection="metadata_xml") == "eval"
    assert waited == ["w", "w"]


def test_index_stats_counts_sqlite_while_warming(monkeypatch: pytest.MonkeyPatch) -> None:
    """Во время прогрева counts идут из sqlite: Chroma не открывается."""

    def no_client() -> Any:
        raise AssertionError("index_stats must not open Chroma during warm-up")

    monkeypatch.setattr("mcp_1c_rag.store.get_client", no_client)
    monkeypatch.setattr(
        "mcp_1c_rag.manifest.list_collection_counts",
        lambda _path: {"code_proc": 3, "metadata_xml": 1},
    )
    release = threading.Event()
    thread = warmup.start([("slow", lambda: release.wait(5))])
    try:
        text = server.index_stats()
    finally:
        release.set()
        _join(thread)
    assert "warmup=running" in text
    assert "collection[code_proc]=3" in text
    assert "collection[metadata_xml]=1" in text
    assert "chroma_error=" not in text
