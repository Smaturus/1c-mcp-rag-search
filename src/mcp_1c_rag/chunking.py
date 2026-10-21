"""Text chunking for BSL modules and configuration report files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TextChunk:
    """One searchable fragment with source path and ordinal."""

    text: str
    source_path: str
    chunk_index: int


def read_text_file(path: Path) -> str:
    """Read text trying common 1C / Windows encodings (incl. UTF-16 reports)."""
    raw = path.read_bytes()
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16")
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw.decode("utf-8-sig")
    # UTF-16-LE without BOM: many NULs on odd bytes in the sample
    sample = raw[: min(512, len(raw))]
    if len(sample) >= 4 and sample[1:2] == b"\x00" and sample[3:4] == b"\x00":
        try:
            return raw.decode("utf-16-le")
        except UnicodeDecodeError:
            pass
    for encoding in ("utf-8", "cp1251"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def chunk_text(
    text: str,
    source_path: str,
    chunk_chars: int,
    overlap: int,
) -> list[TextChunk]:
    """Split plain text into overlapping character windows."""
    cleaned = text.replace("\r\n", "\n").strip()
    if not cleaned:
        return []

    if chunk_chars <= 0:
        raise ValueError("chunk_chars must be positive")
    if overlap < 0 or overlap >= chunk_chars:
        raise ValueError("overlap must be >= 0 and < chunk_chars")

    chunks: list[TextChunk] = []
    start = 0
    index = 0
    length = len(cleaned)
    while start < length:
        end = min(start + chunk_chars, length)
        piece = cleaned[start:end].strip()
        if piece:
            chunks.append(
                TextChunk(text=piece, source_path=source_path, chunk_index=index)
            )
            index += 1
        if end >= length:
            break
        start = end - overlap
    return chunks


def chunk_file(
    path: Path,
    root: Path,
    chunk_chars: int,
    overlap: int,
) -> list[TextChunk]:
    """Read a file and return relative-path chunks."""
    rel = str(path.relative_to(root)).replace("\\", "/")
    return chunk_text(read_text_file(path), rel, chunk_chars, overlap)
