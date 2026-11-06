"""Канонический реестр коллекций RAG (чистый, без Chroma/config).

P0.1a: только объявление спецификаций и резолв имён. Интеграция в
``config`` / ``search_modes`` / ``server`` — отдельный шаг (P0.1b).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Role = Literal["code", "metadata", "aux"]

DEFAULT_CODE_COLLECTION = "code_proc"
DEFAULT_METADATA_COLLECTION = "metadata_xml"


class UnknownCollectionError(ValueError):
    """Неизвестное имя коллекции (канон / chroma / alias)."""


@dataclass(frozen=True)
class CollectionSpec:
    """Описание одной коллекции: публичное имя, Chroma, префикс id."""

    name: str
    chroma_name: str
    id_prefix: str
    display_name: str
    description: str
    aliases: tuple[str, ...] = ()
    role: Role = "aux"


SPECS: tuple[CollectionSpec, ...] = (
    CollectionSpec(
        name="code_proc",
        chroma_name="code_proc",
        id_prefix="bsl_proc:",
        display_name="BSL procedures",
        description="Процедуры и функции BSL (чанк по процедуре).",
        aliases=(),
        role="code",
    ),
    CollectionSpec(
        name="metadata_xml",
        chroma_name="metadata_xml",
        id_prefix="metaobj:",
        display_name="Metadata objects (XML dump)",
        description="Карточки объектов метаданных из XML-дампа конфигурации.",
        aliases=(),
        role="metadata",
    ),
    CollectionSpec(
        name="config_xml",
        chroma_name="config_xml",
        id_prefix="xml:",
        display_name="Config XML files",
        description="Непустые *.xml из корпуса конфигурации (сырой XML).",
        aliases=(),
        role="aux",
    ),
    CollectionSpec(
        name="code",
        chroma_name="code",
        id_prefix="bsl:",
        display_name="BSL code (legacy)",
        description="Устаревшая коллекция BSL (чанк по символам).",
        aliases=(),
        role="aux",
    ),
    CollectionSpec(
        name="metadata",
        chroma_name="metadata",
        id_prefix="meta:",
        display_name="Metadata report (legacy)",
        description="Устаревшая коллекция из ОтчетПоКонфигурации.",
        aliases=("meta",),
        role="aux",
    ),
)


def _available_names() -> tuple[str, ...]:
    return tuple(spec.name for spec in SPECS)


def _build_lookup() -> dict[str, CollectionSpec]:
    table: dict[str, CollectionSpec] = {}
    for spec in SPECS:
        keys = (spec.name, spec.chroma_name, *spec.aliases)
        for key in keys:
            normalized = key.strip().lower()
            if not normalized:
                continue
            existing = table.get(normalized)
            if existing is not None and existing is not spec:
                raise ValueError(
                    f"дублирующий ключ реестра коллекций: {normalized!r}"
                )
            table[normalized] = spec
    return table


_LOOKUP: dict[str, CollectionSpec] = _build_lookup()


def all_specs() -> tuple[CollectionSpec, ...]:
    """Все спецификации в стабильном порядке объявления."""
    return SPECS


def get_spec(name: str) -> CollectionSpec:
    """Резолв по name / chroma_name / aliases (strip + lower)."""
    key = (name or "").strip().lower()
    spec = _LOOKUP.get(key)
    if spec is None:
        available = ", ".join(_available_names())
        raise UnknownCollectionError(
            f"неизвестная коллекция {name!r}; доступны: {available}"
        )
    return spec


def canonical_name(name: str) -> str:
    """Каноническое публичное имя коллекции."""
    return get_spec(name).name


def id_prefix(name: str) -> str:
    """Префикс embedding_id для коллекции (например ``bsl_proc:``)."""
    return get_spec(name).id_prefix


def validate_collection_name(name: str) -> str:
    """Вернуть каноническое имя или поднять ``UnknownCollectionError``."""
    return canonical_name(name)


__all__ = [
    "DEFAULT_CODE_COLLECTION",
    "DEFAULT_METADATA_COLLECTION",
    "CollectionSpec",
    "SPECS",
    "UnknownCollectionError",
    "all_specs",
    "canonical_name",
    "get_spec",
    "id_prefix",
    "validate_collection_name",
]
