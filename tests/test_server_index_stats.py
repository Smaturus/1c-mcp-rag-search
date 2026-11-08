"""index_stats берёт counts из sqlite, не открывая Chroma."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mcp_1c_rag import server  # noqa: E402


def test_index_stats_counts_come_from_sqlite(monkeypatch: pytest.MonkeyPatch) -> None:
    """collection[...] из list_collection_counts; Chroma-клиент не открывается."""

    def no_client() -> object:
        raise AssertionError("index_stats must not open Chroma")

    monkeypatch.setattr("mcp_1c_rag.store.get_client", no_client)
    monkeypatch.setattr(
        "mcp_1c_rag.manifest.list_collection_counts",
        lambda _path: {"code_proc": 10, "metadata_xml": 5},
    )
    monkeypatch.setattr("mcp_1c_rag.config.BM25_DIR", Path("/tmp/no-bm25-dir-for-test"))
    text = server.index_stats()
    assert "collection[code_proc]=10" in text
    assert "collection[metadata_xml]=5" in text
    assert "chroma_error=" not in text


def test_index_stats_survives_unreadable_sqlite(monkeypatch: pytest.MonkeyPatch) -> None:
    """Нечитаемый sqlite не роняет tool — строка chroma_error=..."""

    def boom(_path: Path) -> dict[str, int]:
        raise FileNotFoundError("нет chroma.sqlite3")

    monkeypatch.setattr("mcp_1c_rag.manifest.list_collection_counts", boom)
    text = server.index_stats()
    assert "chroma_error=FileNotFoundError" in text
