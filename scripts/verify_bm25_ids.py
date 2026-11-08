"""Проверка §3.4: множество id нового Chroma == doc_ids нового BM25.

Читает BM25 (meta.json) и Chroma sqlite (read-only). Печатает JSON.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mcp_1c_rag.bm25 import Bm25Index  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chroma-dir", type=Path, default=ROOT / "data" / "chroma_token")
    ap.add_argument("--bm25-dir", type=Path, default=ROOT / "data" / "bm25_token")
    args = ap.parse_args()

    bm25 = Bm25Index.load(args.bm25_dir)
    bm25_ids = set(bm25.doc_ids)

    db = sqlite3.connect(
        f"file:{(args.chroma_dir / 'chroma.sqlite3').as_posix()}?mode=ro", uri=True
    )
    chroma_ids = {
        r[0]
        for r in db.execute("SELECT embedding_id FROM embeddings WHERE embedding_id LIKE 'bsl_proc:%'")
    }
    db.close()

    only_bm25 = sorted(bm25_ids - chroma_ids)
    only_chroma = sorted(chroma_ids - bm25_ids)
    result = {
        "bm25_n_docs": len(bm25_ids),
        "chroma_bsl_proc_n": len(chroma_ids),
        "ids_equal": bm25_ids == chroma_ids,
        "only_in_bm25": only_bm25[:10],
        "only_in_chroma": only_chroma[:10],
        "gate": bm25_ids == chroma_ids,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0 if result["gate"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
