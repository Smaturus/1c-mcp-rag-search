"""Lexical anchor regression check for the v3 corpus.

Anchors are short exact-name phrases that must return a specific file through
the ``lexical`` (global BM25) route. The check is file-level: duplicate chunks
of the same source are collapsed, like in eval_metrics (ADR 0004).

Usage:
    python scripts/eval_anchors.py
    python scripts/eval_anchors.py --top-k 3
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mcp_1c_rag.bm25 import Bm25Index  # noqa: E402

DEFAULT_ANCHORS = ROOT / "data" / "eval" / "anchors_v3.jsonl"
DEFAULT_BM25 = ROOT / "data" / "bm25_all"
DEFAULT_SQLITE = ROOT / "data" / "chroma_xml" / "chroma.sqlite3"
DEFAULT_PREFIX = "bsl_proc:"


def load_anchors(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def source_map(sqlite_path: Path, prefix: str) -> dict[str, str]:
    conn = sqlite3.connect(f"file:{sqlite_path.as_posix()}?mode=ro", uri=True)
    try:
        return {
            str(doc_id): source
            for doc_id, source in conn.execute(
                """
                select e.embedding_id, m.string_value
                from embeddings e
                join embedding_metadata m on m.id = e.id and m.key = 'source'
                where e.embedding_id like ?
                """,
                (prefix + "%",),
            )
        }
    finally:
        conn.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Lexical anchor regression check")
    parser.add_argument("--anchors", type=Path, default=DEFAULT_ANCHORS)
    parser.add_argument("--bm25-dir", type=Path, default=DEFAULT_BM25)
    parser.add_argument("--sqlite", type=Path, default=DEFAULT_SQLITE)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--top-k", type=int, default=None, help="override per-anchor top_k")
    parser.add_argument("--limit", type=int, default=300, help="BM25 candidates to inspect")
    args = parser.parse_args()

    started = time.perf_counter()
    anchors = load_anchors(args.anchors)
    index = Bm25Index.load(args.bm25_dir)
    mapping = source_map(args.sqlite, args.prefix)
    print(
        f"anchors={len(anchors)} bm25_docs={index.n_docs} "
        f"mapped_sources={len(mapping)} load={time.perf_counter() - started:.1f}s"
    )

    failed = 0
    for anchor in anchors:
        top_k = args.top_k or int(anchor["top_k"])
        hits = index.search(anchor["phrase"], top_k=args.limit)
        seen: list[str] = []
        for hit in hits:
            if not hit.doc_id.startswith(args.prefix):
                continue
            source = mapping.get(hit.doc_id)
            if source and source not in seen:
                seen.append(source)
            if len(seen) >= top_k:
                break
        position = seen.index(anchor["target_source"]) + 1 if anchor["target_source"] in seen else None
        ok = position is not None and position <= top_k
        failed += 0 if ok else 1
        mark = "PASS" if ok else "FAIL"
        print(f"[{mark}] {anchor['id']} top_k={top_k} pos={position} target={anchor['target_source']}")
        if not ok:
            for rank, source in enumerate(seen, start=1):
                print(f"        {rank}. {source}")

    elapsed = time.perf_counter() - started
    print(f"\n{len(anchors) - failed}/{len(anchors)} anchors passed in {elapsed:.1f}s")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
