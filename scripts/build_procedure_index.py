"""Собрать отдельную Chroma-коллекцию с нарезкой BSL по процедурам.

Не трогает текущие коллекции ``code`` и ``metadata``: пишет в ``code_proc``
(имя задаётся ``--collection``). Это позволяет сравнивать два чанкинга на одних
и тех же данных, не разрушая работающий индекс.

Возобновляемость: чекпоинт в JSON, повторный запуск продолжает с последнего
обработанного файла. Lock — общий single-writer с индексатором.

Запуск:
    python scripts/build_procedure_index.py --files 300 --reset      # смоук
    python scripts/build_procedure_index.py                           # полная сборка
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mcp_1c_rag.chunking_procedures import chunk_bsl_file  # noqa: E402
from mcp_1c_rag.config import CHROMA_ADD_BATCH, DATA_DIR, STUDY_ROOT  # noqa: E402
from mcp_1c_rag.embeddings import embed_passages  # noqa: E402
from mcp_1c_rag.index_guard import acquire_lock, release_lock  # noqa: E402
from mcp_1c_rag.store import get_collection  # noqa: E402

DEFAULT_CHECKPOINT = DATA_DIR / "procedure_index_progress.json"


def _chunk_id(source: str, index: int) -> str:
    digest = hashlib.sha1(source.encode("utf-8")).hexdigest()[:16]
    return f"bsl_proc:{digest}:{index}"


def _write_checkpoint(path: Path, processed: int, chunks: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"processed": processed, "chunks": chunks}), encoding="utf-8")
    tmp.replace(path)


def _read_checkpoint(path: Path) -> tuple[int, int]:
    if not path.exists():
        return 0, 0
    data = json.loads(path.read_text(encoding="utf-8"))
    return int(data.get("processed", 0)), int(data.get("chunks", 0))


def main() -> int:
    ap = argparse.ArgumentParser(description="Build procedure-chunked collection")
    ap.add_argument("--root", type=Path, default=STUDY_ROOT)
    ap.add_argument("--collection", default="code_proc")
    ap.add_argument("--max-chars", type=int, default=1000)
    ap.add_argument("--overlap", type=int, default=150)
    ap.add_argument("--files", type=int, default=0, help="ограничить число файлов (0 = все)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--reset", action="store_true")
    ap.add_argument("--no-lock", action="store_true",
                    help="не брать общий lock индексатора (сборка в отдельный CHROMA_DIR)")
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    args = ap.parse_args()

    files = sorted(args.root.rglob("*.bsl"))
    if args.files:
        files = files[: args.files]
    if not files:
        raise SystemExit(f"нет .bsl в {args.root}")

    if args.reset:
        args.checkpoint.unlink(missing_ok=True)
        get_collection(args.collection, reset=True).count()

    processed, total_chunks = _read_checkpoint(args.checkpoint)
    if processed >= len(files):
        print(f"checkpoint: все {len(files)} файлов уже обработаны", flush=True)
        return 0

    lock_owned = not args.no_lock
    if lock_owned:
        acquire_lock(args.collection)
    started = time.time()
    collection = get_collection(args.collection)
    pending_ids: list[str] = []
    pending_docs: list[str] = []
    pending_meta: list[dict] = []

    def flush() -> None:
        nonlocal pending_ids, pending_docs, pending_meta
        if not pending_ids:
            return
        vectors = embed_passages(pending_docs, batch_size=args.batch_size)
        collection.upsert(
            ids=pending_ids,
            embeddings=vectors,
            documents=pending_docs,
            metadatas=pending_meta,
        )
        pending_ids, pending_docs, pending_meta = [], [], []

    try:
        for offset, path in enumerate(files[processed:], start=processed):
            try:
                source = str(path.relative_to(args.root)).replace("\\", "/")
                chunks = chunk_bsl_file(
                    path, args.root, max_chars=args.max_chars, overlap=args.overlap
                )
            except Exception as exc:  # noqa: BLE001 — один битый файл не должен ронять сборку
                print(f"[skip] {path}: {exc}", flush=True)
                continue
            for chunk in chunks:
                pending_ids.append(_chunk_id(source, chunk.chunk_index))
                pending_docs.append(chunk.text)
                pending_meta.append(
                    {
                        "source": source,
                        "chunk_index": chunk.chunk_index,
                        "procedure_name": chunk.procedure_name,
                        "kind": chunk.kind,
                        "part": chunk.part,
                        "parts": chunk.parts,
                    }
                )
                total_chunks += 1
                if len(pending_ids) >= CHROMA_ADD_BATCH:
                    flush()
            if offset % 25 == 0 or offset == len(files) - 1:
                elapsed = time.time() - started
                rate = (offset - processed + 1) / elapsed if elapsed else 0.0
                left = len(files) - offset - 1
                eta = left / rate / 60 if rate else 0.0
                print(
                    f"[{args.collection}] files={offset + 1}/{len(files)} "
                    f"chunks={total_chunks} rate={rate:.2f} files/s eta={eta:.0f} min",
                    flush=True,
                )
                _write_checkpoint(args.checkpoint, offset + 1, total_chunks)
        flush()
        _write_checkpoint(args.checkpoint, len(files), total_chunks)
        print(
            f"[{args.collection}] done files={len(files)} chunks={total_chunks} "
            f"count={collection.count()}",
            flush=True,
        )
    except KeyboardInterrupt:
        flush()
        print("[interrupted] checkpoint сохранён", flush=True)
    finally:
        if lock_owned:
            release_lock()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
