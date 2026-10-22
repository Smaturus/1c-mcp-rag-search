"""Chunking BSL-модулей по границам процедур и функций.

Зачем: символьное окно (1200 символов) режет логику посередине и не даёт
эмбеддеру целостной единицы. Процедура — естественная единица BSL: у неё есть
имя, назначение и границы. Дополнительно в текст чанка добавляется строка
``// file: <path> · proc: <name>`` — это усиливает точные имена (лексический
сигнал BM25 и контекст для эмбеддинга).

Ограничения:
- длинная процедура режется на части (``part k/m``) с перекрытием;
- код вне процедур (заголовок модуля, директивы, «шапка») становится отдельным
  чанком ``kind="module"``;
- если в файле нет процедур, весь файл режется символьным окном.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from mcp_1c_rag.chunking import chunk_text, read_text_file

__all__ = ["ProcedureChunk", "PROC_PREFIX", "chunk_bsl_text", "chunk_bsl_file"]

# «Процедура Имя(...) Экспорт», «Функция Имя() Экспорт», с ведущими пробелами.
START = re.compile(
    r"^\s*(?:Процедура|Функция)\s+([A-Za-zА-Яа-я_][\w]*)\s*\(",
    re.IGNORECASE,
)
END = re.compile(r"^\s*(?:КонецПроцедуры|КонецФункции)\b", re.IGNORECASE)

PROC_PREFIX = "// file: {source} · proc: {name}"
MODULE_PREFIX = "// file: {source} · part: module"


@dataclass(frozen=True)
class ProcedureChunk:
    """Чанк, соответствующий процедуре (или её части)."""

    text: str
    source_path: str
    chunk_index: int
    procedure_name: str
    kind: str = "procedure"
    part: int = 1
    parts: int = 1


def _split_long(text: str, max_chars: int, overlap: int) -> list[str]:
    """Разбить длинный фрагмент на части с перекрытием (по символам)."""
    if len(text) <= max_chars:
        return [text]
    parts: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        parts.append(text[start:end].strip())
        if end >= len(text):
            break
        start = end - overlap
    return parts


def _module_chunk(
    text: str,
    source: str,
    max_chars: int,
    overlap: int,
    start_index: int,
) -> list[ProcedureChunk]:
    """Обычный символьный чанкинг (модули без процедур)."""
    if not text.strip():
        return []
    return [
        ProcedureChunk(
            text=f"{PROC_PREFIX.format(source=source, name='module')}\n\n{piece.text}",
            source_path=source,
            chunk_index=start_index + piece.chunk_index,
            procedure_name="module",
            kind="module",
        )
        for piece in chunk_text(text, source, max_chars, overlap)
    ]


def chunk_bsl_text(
    text: str,
    source: str,
    *,
    max_chars: int = 1200,
    overlap: int = 200,
) -> list[ProcedureChunk]:
    """Нарезать BSL-текст по процедурам.

    Args:
        text: Исходный код модуля.
        source: Путь файла для метаданных и префикса чанка.
        max_chars: Максимальный размер части длинной процедуры.
        overlap: Перекрытие частей длинной процедуры.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if overlap < 0 or overlap >= max_chars:
        raise ValueError("overlap must be >= 0 and < max_chars")

    lines = text.replace("\r\n", "\n").split("\n")
    chunks: list[ProcedureChunk] = []
    index = 0

    header: list[str] = []
    current: list[str] = []
    current_name: str | None = None
    pending_directives: list[str] = []

    def flush_procedure() -> None:
        nonlocal index, current, current_name
        if current_name is None or not "".join(current).strip():
            current, current_name = [], None
            return
        body = "\n".join(current).strip()
        pieces = _split_long(body, max_chars, overlap)
        total = len(pieces)
        for part, piece in enumerate(pieces, start=1):
            chunks.append(
                ProcedureChunk(
                    text=f"{PROC_PREFIX.format(source=source, name=current_name)}\n\n{piece}",
                    source_path=source,
                    chunk_index=index,
                    procedure_name=current_name,
                    kind="procedure",
                    part=part,
                    parts=total,
                )
            )
            index += 1
        current, current_name = [], None

    for line in lines:
        start_match = START.match(line)
        if start_match and current_name is None:
            header_text = "\n".join(header).strip()
            if header_text:
                chunks.extend(_module_chunk(header_text, source, max_chars, overlap, index))
                index = len(chunks)
            header = []
            current_name = start_match.group(1)
            current = [*pending_directives, line]
            pending_directives = []
            continue
        if END.match(line) and current_name is not None:
            current.append(line)
            flush_procedure()
            continue
        if current_name is not None:
            current.append(line)
        elif line.lstrip().startswith("&"):
            # директива компиляции перед процедурой — приклеиваем к следующей
            pending_directives.append(line)
        else:
            pending_directives = []
            header.append(line)

    flush_procedure()
    header_text = "\n".join(header).strip()
    if header_text:
        chunks.extend(_module_chunk(header_text, source, max_chars, overlap, len(chunks)))

    if not chunks and text.strip():
        return _module_chunk(text, source, max_chars, overlap, 0)
    return chunks


def chunk_bsl_file(
    path: Path,
    root: Path,
    *,
    max_chars: int = 1200,
    overlap: int = 200,
) -> list[ProcedureChunk]:
    """Прочитать файл и вернуть чанки по процедурам с относительным путём."""
    rel = str(path.relative_to(root)).replace("\\", "/")
    return chunk_bsl_text(
        read_text_file(path), rel, max_chars=max_chars, overlap=overlap
    )
