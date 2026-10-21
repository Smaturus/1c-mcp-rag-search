"""Build local Chroma index from D:\\Study (.bsl) and D:\\Study_Report (.txt)."""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

from mcp_1c_homework.chunking import TextChunk, chunk_file
from mcp_1c_homework.config import (
    BSL_CHUNK_CHARS,
    BSL_CHUNK_OVERLAP,
    CHROMA_ADD_BATCH,
    DATA_DIR,
    EMBED_BATCH_SIZE,
    META_CHUNK_CHARS,
    META_CHUNK_OVERLAP,
    METADATA_XML_ROOT,
    STUDY_REPORT_ROOT,
    STUDY_ROOT,
)
from mcp_1c_homework.embeddings import embed_passages
from mcp_1c_homework.index_guard import (
    IndexBusy,
    IndexProgress,
    acquire_lock,
    assert_output_not_inside_sources,
    empty_progress,
    load_progress,
    progress_due,
    release_lock,
    save_progress,
)
from mcp_1c_homework.keep_awake import allow_sleep, prevent_sleep
from mcp_1c_homework.store import get_code_collection, get_metadata_collection


def _chunk_id(prefix: str, chunk: TextChunk) -> str:
    digest = hashlib.sha1(
        f"{chunk.source_path}::{chunk.chunk_index}::{chunk.text[:64]}".encode()
    ).hexdigest()[:16]
    return f"{prefix}:{digest}:{chunk.chunk_index}"


def _iter_files(root: Path, pattern: str, limit: int | None) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Source path not found: {root}")
    files = sorted(root.rglob(pattern))
    files = [p for p in files if p.is_file()]
    if limit is not None:
        files = files[:limit]
    return files


def _flush_batch(
    collection,
    ids: list[str],
    documents: list[str],
    metadatas: list[dict],
) -> None:
    if not ids:
        return
    embeddings = embed_passages(documents, batch_size=EMBED_BATCH_SIZE)
    # Upsert so a replay after a lost checkpoint does not fail on the same ids.
    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )


def _source_of(path: Path, root: Path) -> str:
    return str(path.relative_to(root)).replace("\\", "/")


def _prepare_progress(target: str, *, reset: bool, resume: bool) -> IndexProgress:
    """Start, continue, or clear the checkpoint for one target."""
    if reset and resume:
        raise ValueError("reset and resume cannot be used together")
    if reset:
        progress = empty_progress(target)
        save_progress(progress)
        return progress
    if resume:
        return load_progress(target) or empty_progress(target)
    return empty_progress(target)


def _remember_batch(
    progress: IndexProgress,
    metadatas: list[dict],
    current: str | None,
    current_finished: bool,
) -> None:
    progress.apply_batch(
        [str(item["source"]) for item in metadatas],
        current,
        current_finished,
    )
    save_progress(progress)


def _clock(progress: IndexProgress, started: float) -> float:
    """Seconds since the original run, kept across a continuation."""
    origin = progress.started_unix if progress.started_unix else started
    return time.time() - origin


def _remember_start(progress: IndexProgress, started: float) -> None:
    if progress.started_unix is None:
        progress.started_unix = started


def _note_previous(target: str, *, reset: bool, resume: bool) -> None:
    if reset or resume:
        return
    previous = load_progress(target)
    if previous and previous.stored_chunks():
        print(
            "Есть контрольная точка предыдущего прогона. "
            "Чтобы не встраивать векторы заново, добавьте --resume.",
            flush=True,
        )


def _flush_marked(
    collection,
    ids: list[str],
    documents: list[str],
    metadatas: list[dict],
    progress: IndexProgress,
    current: str | None,
    current_finished: bool,
) -> int:
    """Write one batch and store the checkpoint. Returns chunks written."""
    if not ids:
        return 0
    _flush_batch(collection, ids, documents, metadatas)
    _remember_batch(progress, metadatas, current, current_finished)
    return len(ids)


def index_bsl(
    *,
    reset: bool,
    limit: int | None,
    max_chunks: int | None,
    resume: bool = False,
) -> int:
    """Index .bsl files into the code collection. Returns chunk count."""
    _note_previous("code", reset=reset, resume=resume)
    progress = _prepare_progress("code", reset=reset, resume=resume)
    collection = get_code_collection(reset=reset)
    files = _iter_files(STUDY_ROOT, "*.bsl", limit)
    print(f"[code] files={len(files)} root={STUDY_ROOT}", flush=True)
    if resume and progress.stored_chunks():
        print(
            f"Продолжение кода: файлов готово {len(progress.completed)}, "
            f"фрагментов уже записано {progress.stored_chunks()}",
            flush=True,
        )

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict] = []
    total_chunks = progress.stored_chunks() if resume else 0
    started = time.time()
    _remember_start(progress, started)
    last_emit = started
    stop = False

    def _emit_code(i: int, *, force: bool) -> None:
        nonlocal last_emit
        now = time.time()
        if not force and not progress_due(last_emit, now):
            return
        print(
            f"[code] {i}/{len(files)} files, chunks_so_far={total_chunks + len(ids)}, "
            f"{_clock(progress, started):.1f}s",
            flush=True,
        )
        last_emit = now

    for i, path in enumerate(files, start=1):
        if stop:
            break
        source = _source_of(path, STUDY_ROOT)
        if resume and source in progress.completed:
            _emit_code(i, force=(i % 50 == 0 or i == len(files)))
            continue
        chunks = chunk_file(path, STUDY_ROOT, BSL_CHUNK_CHARS, BSL_CHUNK_OVERLAP)
        skip = progress.skip_chunks(source) if resume else 0
        file_complete = True
        for chunk in chunks:
            if skip > 0:
                skip -= 1
                continue
            if max_chunks is not None and total_chunks + len(ids) >= max_chunks:
                stop = True
                file_complete = False
                break
            ids.append(_chunk_id("bsl", chunk))
            documents.append(chunk.text)
            metadatas.append(
                {
                    "source": chunk.source_path,
                    "chunk_index": chunk.chunk_index,
                    "kind": "bsl",
                }
            )
            _emit_code(i, force=False)
            if len(ids) >= CHROMA_ADD_BATCH:
                total_chunks += _flush_marked(
                    collection,
                    ids,
                    documents,
                    metadatas,
                    progress,
                    source,
                    False,
                )
                ids, documents, metadatas = [], [], []

        if file_complete and not any(item["source"] == source for item in metadatas):
            progress.mark_source_fully_flushed(source)
            save_progress(progress)

        _emit_code(i, force=(i % 50 == 0 or i == len(files)))

    if ids:
        total_chunks += _flush_marked(
            collection,
            ids,
            documents,
            metadatas,
            progress,
            str(metadatas[-1]["source"]),
            not stop,
        )
    print(f"[code] done chunks={total_chunks} collection_count={collection.count()}", flush=True)
    return total_chunks


def _metadata_sources(
    limit: int | None,
) -> list[tuple[Path, Path, str, str]]:
    """Report texts, plus *.xml when MCP_1C_METADATA_XML_ROOT is set.

    Each row is (path, root, kind, id prefix). Files are only read.
    """
    rows: list[tuple[Path, Path, str, str]] = []
    for path in _iter_files(STUDY_REPORT_ROOT, "*.txt", limit):
        rows.append((path, STUDY_REPORT_ROOT, "metadata", "meta"))
    if METADATA_XML_ROOT is not None:
        for path in _iter_files(METADATA_XML_ROOT, "*.xml", limit):
            rows.append((path, METADATA_XML_ROOT, "xml", "xml"))
    return rows


def _source_roots() -> list[Path]:
    roots = [STUDY_ROOT, STUDY_REPORT_ROOT]
    if METADATA_XML_ROOT is not None:
        roots.append(METADATA_XML_ROOT)
    return roots


def index_metadata(
    *,
    reset: bool,
    limit: int | None,
    max_chunks: int | None,
    resume: bool = False,
) -> int:
    """Index the report and, if configured, dump XML. Returns chunk count."""
    _note_previous("metadata", reset=reset, resume=resume)
    progress = _prepare_progress("metadata", reset=reset, resume=resume)
    collection = get_metadata_collection(reset=reset)
    files = _metadata_sources(limit)
    report_n = sum(1 for row in files if row[2] == "metadata")
    xml_n = sum(1 for row in files if row[2] == "xml")
    print(
        f"[metadata] files={len(files)} report={report_n} xml={xml_n} "
        f"report_root={STUDY_REPORT_ROOT}",
        flush=True,
    )
    if resume and progress.stored_chunks():
        print(
            f"Продолжение метаданных: файлов готово {len(progress.completed)}, "
            f"фрагментов уже записано {progress.stored_chunks()}",
            flush=True,
        )

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict] = []
    total_chunks = progress.stored_chunks() if resume else 0
    started = time.time()
    _remember_start(progress, started)
    last_emit = started
    stop = False

    def _emit_meta(*, force: bool) -> None:
        nonlocal last_emit
        now = time.time()
        if not force and not progress_due(last_emit, now):
            return
        print(
            f"[metadata] flushed total={total_chunks + len(ids)}, "
            f"{_clock(progress, started):.1f}s",
            flush=True,
        )
        last_emit = now

    for i, (path, root, kind, prefix) in enumerate(files, start=1):
        if stop:
            break
        source = _source_of(path, root)
        if resume and source in progress.completed:
            continue
        chunks = chunk_file(path, root, META_CHUNK_CHARS, META_CHUNK_OVERLAP)
        if max_chunks is not None and total_chunks >= max_chunks:
            break
        if max_chunks is not None:
            chunks = chunks[: max(0, max_chunks - total_chunks)]
        if kind == "metadata" or i == 1 or i % 500 == 0:
            print(
                f"[metadata] passed {i}/{len(files)} pieces={len(chunks)}",
                flush=True,
            )
        skip = progress.skip_chunks(source) if resume else 0
        file_complete = True
        for chunk in chunks:
            if skip > 0:
                skip -= 1
                continue
            ids.append(_chunk_id(prefix, chunk))
            documents.append(chunk.text)
            metadatas.append(
                {
                    "source": chunk.source_path,
                    "chunk_index": chunk.chunk_index,
                    "kind": kind,
                }
            )
            _emit_meta(force=False)
            if len(ids) >= CHROMA_ADD_BATCH:
                total_chunks += _flush_marked(
                    collection,
                    ids,
                    documents,
                    metadatas,
                    progress,
                    source,
                    False,
                )
                ids, documents, metadatas = [], [], []
                _emit_meta(force=True)
                if max_chunks is not None and total_chunks >= max_chunks:
                    stop = True
                    file_complete = False
                    break
        if max_chunks is not None and total_chunks + len(ids) >= max_chunks:
            stop = True
            file_complete = False
        if file_complete and not any(item["source"] == source for item in metadatas):
            progress.mark_source_fully_flushed(source)
            save_progress(progress)

    if ids:
        total_chunks += _flush_marked(
            collection,
            ids,
            documents,
            metadatas,
            progress,
            str(metadatas[-1]["source"]),
            not stop,
        )
    print(
        f"[metadata] done chunks={total_chunks} collection_count={collection.count()}",
        flush=True,
    )
    return total_chunks


def build_index(
    *,
    target: str = "all",
    reset: bool = False,
    limit: int | None = None,
    max_chunks: int | None = None,
    resume: bool = False,
) -> None:
    """Build selected collections."""
    if reset and resume:
        raise ValueError("reset and resume cannot be used together")
    assert_output_not_inside_sources(DATA_DIR, _source_roots())
    # Drop stale checkpoints before any collection opens, so a later
    # continuation cannot skip files that were never written in this run.
    if reset:
        if target in ("all", "metadata"):
            save_progress(empty_progress("metadata"))
        if target in ("all", "code"):
            save_progress(empty_progress("code"))
    # One process: metadata, then code. Nothing stops the process between them.
    if target in ("all", "metadata"):
        # limit = max source files; max_chunks caps vectors (report is huge)
        index_metadata(reset=reset, limit=limit, max_chunks=max_chunks, resume=resume)
    if target in ("all", "code"):
        index_bsl(reset=reset, limit=limit, max_chunks=max_chunks, resume=resume)


def main(argv: list[str] | None = None) -> int:
    """CLI entry: python -m mcp_1c_homework.indexer ..."""
    parser = argparse.ArgumentParser(
        description="Build Chroma index for 1C BSL and metadata (own MCP homework)."
    )
    parser.add_argument(
        "--target",
        choices=("all", "code", "metadata"),
        default="all",
        help="Which collection(s) to (re)build",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop target collection(s) before indexing",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max source files per target (smoke test)",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="Stop after N chunks (useful for huge report / smoke test)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue from the last checkpoint; do not drop the collection",
    )
    args = parser.parse_args(argv)
    if args.reset and args.resume:
        print(
            "Нельзя одновременно очищать коллекцию и продолжать с контрольной точки.",
            file=sys.stderr,
        )
        return 2

    print(
        f"Indexing target={args.target} reset={args.reset} resume={args.resume} "
        f"limit={args.limit} max_chunks={args.max_chunks}",
        flush=True,
    )
    try:
        acquire_lock(args.target)
    except IndexBusy as exc:
        print(str(exc), file=sys.stderr)
        return 2
    # Current run already holds the lock in another process; this applies
    # on the next indexer start. The progress loop also holds wake while writing.
    prevent_sleep()
    try:
        build_index(
            target=args.target,
            reset=args.reset,
            limit=args.limit,
            max_chunks=args.max_chunks,
            resume=args.resume,
        )
    finally:
        allow_sleep()
        release_lock()
    return 0


if __name__ == "__main__":
    sys.exit(main())
