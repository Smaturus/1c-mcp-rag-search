"""Собрать Chroma-коллекцию на токенном чанкере.

Модель задаётся только флагом ``--embedding-model`` (пишется в окружение
до импорта ``config``). Пишет в отдельный CHROMA_DIR (через ``--chroma-dir``),
не трогает рабочие ``data/chroma``/``data/chroma_xml``. Возобновляемость —
JSON-чекпоинт; повторный запуск продолжает с последнего обработанного файла.
Lock — общий single-writer (можно отключить ``--no-lock`` для отдельного CHROMA_DIR).

Примеры:
    python scripts/build_token_index.py --embedding-model intfloat/multilingual-e5-base --files 300 --reset --chroma-dir data/chroma_large_pilot --checkpoint data/procedure_index_progress_large_pilot_b8.json
    python scripts/build_token_index.py --embedding-model intfloat/multilingual-e5-base --chroma-dir data/chroma_large --checkpoint data/procedure_index_progress_large.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CORPUS = ROOT / "1C-Corpus" / "1C_Corpus"
DEFAULT_CHROMA_DIR = ROOT / "data" / "chroma_token"
DEFAULT_CHECKPOINT = ROOT / "data" / "procedure_index_progress_token.json"


def peak_rss_bytes() -> int | None:
    """Пиковый WorkingSetSize процесса через Windows psapi."""
    if os.name != "nt":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
            wintypes.DWORD,
        ]
        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
        handle = k32.GetCurrentProcess()
        if psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            return int(counters.PeakWorkingSetSize)
    except Exception:  # noqa: BLE001 — метрика не критична
        return None
    return None


def _chunk_id(source: str, index: int) -> str:
    digest = hashlib.sha1(source.encode("utf-8")).hexdigest()[:16]
    return f"bsl_proc:{digest}:{index}"


def _write_checkpoint(path: Path, processed: int, chunks: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps({"processed": processed, "chunks": chunks}), encoding="utf-8"
    )
    tmp.replace(path)


def _read_checkpoint(path: Path) -> tuple[int, int]:
    if not path.exists():
        return 0, 0
    data = json.loads(path.read_text(encoding="utf-8"))
    return int(data.get("processed", 0)), int(data.get("chunks", 0))


def main() -> int:
    ap = argparse.ArgumentParser(description="Build token-aware chunked collection")
    ap.add_argument("--root", type=Path, default=DEFAULT_CORPUS)
    ap.add_argument("--chroma-dir", type=Path, default=DEFAULT_CHROMA_DIR)
    ap.add_argument("--collection", default="code_proc")
    ap.add_argument("--max-tokens", type=int, default=500)
    ap.add_argument("--overlap-tokens", type=int, default=64)
    ap.add_argument("--min-tokens", type=int, default=32)
    ap.add_argument("--max-chars", type=int, default=2000)
    ap.add_argument("--files", type=int, default=0, help="ограничить число файлов (0 = все)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--reset", action="store_true")
    ap.add_argument("--no-lock", action="store_true")
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    ap.add_argument("--checkpoint-interval", type=int, default=50)
    ap.add_argument(
        "--embedding-model",
        required=True,
        help="имя модели эмбеддинга/токенизатора; без флага модель в окружении не подменяется",
    )
    args = ap.parse_args()

    # Настроить окружение ДО импорта mcp-модулей (config читает env при импорте).
    os.environ["MCP_1C_CHROMA_DIR"] = str(args.chroma_dir)
    os.environ["MCP_1C_EMBEDDING_MODEL"] = args.embedding_model
    os.environ["MCP_1C_HNSW_SYNC_THRESHOLD"] = "1000000"
    sys.path.insert(0, str(ROOT / "src"))

    from mcp_1c_rag.chunking_tokens import chunk_bsl_file_tokens  # noqa: E402
    from mcp_1c_rag.config import CHROMA_ADD_BATCH  # noqa: E402
    from mcp_1c_rag.embeddings import embed_passages, get_model  # noqa: E402
    from mcp_1c_rag.index_guard import acquire_lock, release_lock  # noqa: E402
    from mcp_1c_rag.store import get_collection  # noqa: E402

    dim = int(get_model().get_sentence_embedding_dimension())
    print(f"embedding_model={args.embedding_model} dim={dim}", flush=True)

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
                chunks = chunk_bsl_file_tokens(
                    path,
                    args.root,
                    model_name=args.embedding_model,
                    max_tokens=args.max_tokens,
                    overlap_tokens=args.overlap_tokens,
                    min_tokens=args.min_tokens,
                    max_chars=args.max_chars,
                )
            except Exception as exc:  # noqa: BLE001
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
            if offset % args.checkpoint_interval == 0 or offset == len(files) - 1:
                now = time.time()
                elapsed = now - started
                done = offset - processed + 1
                rate = done / elapsed if elapsed else 0.0
                left = len(files) - offset - 1
                eta_min = left / rate / 60 if rate else 0.0
                rss_mb = (peak_rss_bytes() or 0) / 1048576
                print(
                    f"[{args.collection}] files={offset + 1}/{len(files)} "
                    f"chunks={total_chunks} rate={rate:.2f} files/s "
                    f"eta={eta_min:.0f} min rss={rss_mb:.0f} MB",
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
