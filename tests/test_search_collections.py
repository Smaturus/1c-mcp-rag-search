"""Резолв коллекций в search_modes через registry."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mcp_1c_rag import search_modes  # noqa: E402
from mcp_1c_rag.collections import (  # noqa: E402
    UnknownCollectionError,
    all_specs,
    id_prefix,
)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("meta", "metadata"),
        ("metadata", "metadata"),
        ("code_proc", "code_proc"),
        (" CODE ", "code"),
        ("metadata_xml", "metadata_xml"),
    ],
)
def test_get_collection_passes_canonical_to_store(
    monkeypatch: pytest.MonkeyPatch, name: str, expected: str
) -> None:
    """get_collection резолвит alias и вызывает store с canonical."""
    seen: list[str] = []

    def fake_store_get(coll_name: str, *, reset: bool = False) -> Any:
        seen.append(coll_name)
        return object()

    monkeypatch.setattr("mcp_1c_rag.store.get_collection", fake_store_get)
    search_modes.get_collection(name)
    assert seen == [expected]


def test_id_prefix_table_matches_registry() -> None:
    """_ID_PREFIX сгенерирован из registry, без ручных расхождений."""
    expected = {spec.name: spec.id_prefix for spec in all_specs()}
    assert expected == search_modes._ID_PREFIX


def test_bm25_prefix_code_proc() -> None:
    """Префикс BM25 для code_proc — bsl_proc:."""
    assert id_prefix("code_proc") == "bsl_proc:"
    assert search_modes._ID_PREFIX["code_proc"] == "bsl_proc:"


def test_masked_bm25_unknown_collection_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Неизвестная коллекция в BM25-mask → UnknownCollectionError (до загрузки индекса)."""

    def boom() -> Any:
        raise AssertionError("get_bm25 не должен вызываться для неизвестной коллекции")

    monkeypatch.setattr(search_modes, "get_bm25", boom)
    with pytest.raises(UnknownCollectionError) as exc_info:
        search_modes._masked_bm25_scores("НДС", "no_such_collection_xyz")
    assert "no_such_collection_xyz" in str(exc_info.value)
