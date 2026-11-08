"""Token-aware нарезка сырых XML-файлов конфигурации (без границ Процедура/Функция).

Бюджет совпадает с BSL-токенным чанкером: полный вход модели
``passage:`` + ``// file: <source>`` + тело ≤ ``max_tokens`` (500),
перекрытие 64, хвост короче ``min_tokens`` (32) приклеивается,
потолок ``max_chars`` 2000. Режется по строкам; невлезающая строка — по токенам.
Id сборки: ``xml:<sha1(source)[:16]>:<chunk_index>``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from mcp_1c_rag.chunking import read_text_file
from mcp_1c_rag.chunking_tokens import (
    DEFAULT_MAX_CHARS,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MIN_TOKENS,
    DEFAULT_OVERLAP_TOKENS,
    _split_body_tokens,
    get_tokenizer,
)

__all__ = [
    "XmlTokenChunk",
    "chunk_xml_file_tokens",
    "chunk_xml_text_tokens",
    "xml_file_prefix",
]


@dataclass(frozen=True)
class XmlTokenChunk:
    """Чанк XML-файла с токенным бюджетом."""

    text: str
    source_path: str
    chunk_index: int
    kind: str = "xml"
    part: int = 1
    parts: int = 1


def xml_file_prefix(source: str) -> str:
    """Префикс чанка XML (без proc — в XML нет процедур)."""
    return f"// file: {source}"


def chunk_xml_text_tokens(
    text: str,
    source: str,
    *,
    model_name: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
    min_tokens: int = DEFAULT_MIN_TOKENS,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> list[XmlTokenChunk]:
    """Нарезать XML-текст токенным бюджетом (без Процедура/Функция)."""
    if not model_name:
        raise ValueError("model_name is required")
    # Явная загрузка токенизатора — тот же get_tokenizer, что у BSL.
    get_tokenizer(model_name)
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if overlap_tokens < 0 or overlap_tokens >= max_tokens:
        raise ValueError("overlap_tokens must be >= 0 and < max_tokens")
    if min_tokens <= 0 or min_tokens >= max_tokens:
        raise ValueError("min_tokens must be > 0 and < max_tokens")
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")

    body = text.replace("\r\n", "\n").strip()
    if not body:
        return []

    prefix = xml_file_prefix(source)
    pieces = _split_body_tokens(
        prefix,
        body,
        max_tokens,
        overlap_tokens,
        min_tokens,
        max_chars,
        model_name,
    )
    total = len(pieces)
    chunks: list[XmlTokenChunk] = []
    for index, piece in enumerate(pieces):
        chunks.append(
            XmlTokenChunk(
                text=f"{prefix}\n\n{piece}",
                source_path=source,
                chunk_index=index,
                kind="xml",
                part=index + 1,
                parts=total,
            )
        )
    return chunks


def chunk_xml_file_tokens(
    path: Path,
    root: Path,
    *,
    model_name: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
    min_tokens: int = DEFAULT_MIN_TOKENS,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> list[XmlTokenChunk]:
    """Прочитать .xml и вернуть токенные чанки с относительным путём."""
    rel = str(path.relative_to(root)).replace("\\", "/")
    return chunk_xml_text_tokens(
        read_text_file(path),
        rel,
        model_name=model_name,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
        min_tokens=min_tokens,
        max_chars=max_chars,
    )
