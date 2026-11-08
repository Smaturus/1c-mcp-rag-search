"""Ресурсы MCP: модуль BSL с диска и карточка XML-метаданных из Chroma.

``bsl://{path}`` читает модуль целиком внутри ``MCP_1C_STUDY_ROOT`` (``safe_join``
отклоняет выход за каталог, абсолютные пути и ссылки-симлинки). Выбор одной
процедуры (``?proc``) — отдельная задача, этот ресурс её не поддерживает.
``metadata://{kind}/{name}`` отдаёт карточку уровня «объект» из коллекции
``metadata_xml`` (XML-дамп), без разбора XML на диске.
"""

from __future__ import annotations

import os
from pathlib import Path

from mcp.server.mcpserver.exceptions import ResourceError
from mcp.shared.path_security import PathEscapeError, safe_join

from mcp_1c_rag.config import COLLECTION_METADATA, STUDY_ROOT
from mcp_1c_rag.uris import bsl_uri, metadata_uri

BSL_MAX_BYTES = int(os.environ.get("MCP_1C_BSL_MAX_BYTES", "262144"))


def read_bsl_module(
    path: str, *, root: Path | None = None, max_bytes: int = BSL_MAX_BYTES
) -> str:
    """Прочитать модуль .bsl из корня корпуса; ResourceError при отказе."""
    base = root if root is not None else STUDY_ROOT
    if not base.is_dir():
        raise ResourceError(f"каталог BSL не найден: {base} (MCP_1C_STUDY_ROOT)")
    if not path.lower().endswith(".bsl"):
        raise ResourceError("разрешены только файлы .bsl")
    try:
        full = safe_join(base, path)
    except PathEscapeError as exc:
        raise ResourceError("путь вне каталога BSL") from exc
    if not full.is_file():
        raise ResourceError(f"модуль не найден: {path}")
    size = full.stat().st_size
    if size > max_bytes:
        raise ResourceError(f"модуль {size} байт больше лимита {max_bytes}")
    return full.read_text(encoding="utf-8", errors="replace")


def _fetch_metadata_card(kind: str, name: str) -> list[str]:
    """Тексты карточки объекта (level=object) из коллекции метаданных."""
    from mcp_1c_rag.store import get_collection

    found = get_collection(COLLECTION_METADATA).get(
        where={
            "$and": [
                {"kind_tag": kind},
                {"name": name},
                {"level": "object"},
            ]
        },
        include=["documents"],
    )
    return [doc for doc in (found.get("documents") or []) if doc]


def read_metadata_card(kind: str, name: str) -> str:
    """Карточка объекта XML-метаданных; ResourceError, если объекта нет."""
    from mcp_1c_rag.index_guard import IndexBusy

    try:
        texts = _fetch_metadata_card(kind, name)
    except IndexBusy as exc:
        raise ResourceError(f"индекс занят индексацией: {exc}") from exc
    except FileNotFoundError as exc:
        raise ResourceError(f"коллекция метаданных недоступна: {exc}") from exc
    if not texts:
        raise ResourceError(f"объект метаданных не найден: {kind}/{name}")
    return "\n\n".join(texts)


__all__ = [
    "BSL_MAX_BYTES",
    "bsl_uri",
    "metadata_uri",
    "read_bsl_module",
    "read_metadata_card",
]
