"""Build the ``meta_upp11`` collection from parsed XML metadata chunks.

Input: data/metadata_xml/chunks.jsonl (see scripts/parse_config_xml.py).
Writes into a separate Chroma dir (default data/chroma_xml), so the legacy collection
index in data/chroma stays untouched.

Usage:
    $env:MCP_1C_CHROMA_DIR = "data\\chroma_xml"
    python scripts/build_metadata_index.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mcp_1c_rag.config import CHROMA_ADD_BATCH  # noqa: E402
from mcp_1c_rag.embeddings import embed_passages  # noqa: E402
from mcp_1c_rag.store import get_collection  # noqa: E402

DEFAULT_CHUNKS = ROOT / "data" / "metadata_xml" / "chunks.jsonl"

# Шумовые виды (картинки, стили, интерфейсы) не индексируем: они не отвечают
# на вопросы аналитика и размывают выдачу.
EXCLUDE_KINDS = {
    "CommonPicture",
    "Style",
    "StyleItem",
    "Interface",
    "Language",
    "WSReference",
    "XDTOPackage",
    "Role",
}


def main() -> int:
    ap = argparse.ArgumentParser(description="Build meta_upp11 collection")
    ap.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    ap.add_argument("--collection", default="meta_upp11")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--reset", action="store_true")
    args = ap.parse_args()

    if not args.chunks.exists():
        raise SystemExit(f"нет чанков: {args.chunks} (сначала parse_config_xml.py)")

    items: list[dict] = []
    scanned = 0
    for line in args.chunks.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        scanned += 1
        chunk = json.loads(line)
        if chunk["metadata"].get("kind_tag") in EXCLUDE_KINDS:
            continue
        items.append(chunk)
        if args.limit and len(items) >= args.limit:
            break

    kinds = Counter(item["metadata"]["kind"] for item in items)
    print(f"scanned {scanned}, indexed {len(items)}, kinds: {dict(kinds.most_common(8))}")

    collection = get_collection(args.collection, reset=args.reset)
    started = time.time()
    seen_ids: set[str] = set()
    ids: list[str] = []
    docs: list[str] = []
    metas: list[dict] = []

    def flush() -> None:
        nonlocal ids, docs, metas
        if not ids:
            return
        vectors = embed_passages(docs, batch_size=args.batch_size)
        collection.upsert(ids=ids, embeddings=vectors, documents=docs, metadatas=metas)
        print(
            f"  upserted {len(ids)} chunks, collection={collection.count()}, "
            f"{time.time() - started:.0f}s",
            flush=True,
        )
        ids, docs, metas = [], [], []

    try:
        for index, item in enumerate(items, start=1):
            chunk_id = item["id"]
            if chunk_id in seen_ids:
                chunk_id = f"{chunk_id}#{len(seen_ids)}"
            seen_ids.add(chunk_id)
            ids.append(chunk_id)
            docs.append(item["text"])
            metas.append(item["metadata"])
            if len(ids) >= CHROMA_ADD_BATCH:
                flush()
            elif index % 500 == 0:
                print(f"  prepared {index}/{len(items)}", flush=True)
        flush()
    except KeyboardInterrupt:
        flush()
        print("[interrupted]")

    print(
        f"done: collection {args.collection} count={collection.count()} "
        f"in {time.time() - started:.0f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
