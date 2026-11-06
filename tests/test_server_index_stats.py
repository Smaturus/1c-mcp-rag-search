"""index_stats берёт имена коллекций из registry."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mcp_1c_rag import server  # noqa: E402
from mcp_1c_rag.collections import all_specs  # noqa: E402


class _FakeCollection:
    def __init__(self, name: str, count: int) -> None:
        self.name = name
        self._count = count

    def count(self) -> int:
        return self._count


class _FakeClient:
    def __init__(self, present: dict[str, int]) -> None:
        self._present = present

    def get_collection(self, name: str) -> Any:
        if name not in self._present:
            raise ValueError(f"missing {name}")
        return _FakeCollection(name, self._present[name])


def test_index_stats_uses_registry_names(monkeypatch: pytest.MonkeyPatch) -> None:
    """Вывод collection[...] для имён из all_specs; отсутствующие пропускаются."""
    present = {
        "code_proc": 10,
        "metadata_xml": 5,
        "code": 1,
        # metadata / config_xml отсутствуют — должны быть пропущены
    }
    monkeypatch.setattr("mcp_1c_rag.store.get_client", lambda: _FakeClient(present))
    monkeypatch.setattr(
        "mcp_1c_rag.config.BM25_DIR", Path("/tmp/no-bm25-dir-for-test")
    )
    text = server.index_stats()
    for name, count in present.items():
        assert f"collection[{name}]={count}" in text
    for spec in all_specs():
        if spec.name not in present:
            assert f"collection[{spec.name}]=" not in text
    assert "chroma_error=" not in text


def test_index_stats_survives_client_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ошибка клиента не роняет tool — строка chroma_error=..."""

    def boom() -> Any:
        raise RuntimeError("client down")

    monkeypatch.setattr("mcp_1c_rag.store.get_client", boom)
    text = server.index_stats()
    assert "chroma_error=RuntimeError" in text
