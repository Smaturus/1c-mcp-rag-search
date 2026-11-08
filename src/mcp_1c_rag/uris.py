"""Сборка URI для выдачи search. Только stdlib, без MCP SDK и config."""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Any
from urllib.parse import quote


def bsl_uri(source: str) -> str | None:
    """URI модуля для чанка кода; None, если источник — не относительный .bsl."""
    path = (source or "").replace("\\", "/")
    if path.startswith("/"):
        return None
    parts = PurePosixPath(path).parts
    if not path.lower().endswith(".bsl") or ".." in parts:
        return None
    return "bsl://" + quote(path, safe="/")


def metadata_uri(metadata: dict[str, Any]) -> str | None:
    """URI карточки объекта XML-метаданных; None без kind_tag/name."""
    kind = str(metadata.get("kind_tag") or "").strip()
    name = str(metadata.get("name") or "").strip()
    if not kind or not name:
        return None
    return f"metadata://{quote(kind, safe='')}/{quote(name, safe='')}"


__all__ = ["bsl_uri", "metadata_uri"]
