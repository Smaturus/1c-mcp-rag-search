"""Собрать индекс-манифест по существующим Chroma + BM25 (только чтение).

Пример::

    python scripts/build_manifest.py \\
      --chroma-dir data/chroma_xml --bm25-dir data/bm25_all \\
      --out data/manifests/chroma_xml.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mcp_1c_rag.config import (  # noqa: E402
    BM25_DIR,
    CHROMA_DIR,
    EMBEDDING_MODEL,
    MANIFEST_PATH,
)
from mcp_1c_rag.manifest import (  # noqa: E402
    IndexManifest,
    collect_manifest,
    write_manifest,
)


def main() -> int:
    ap = argparse.ArgumentParser(description="Build index manifest (read-only)")
    ap.add_argument("--chroma-dir", type=Path, default=CHROMA_DIR)
    ap.add_argument("--bm25-dir", type=Path, default=BM25_DIR)
    ap.add_argument("--out", type=Path, default=MANIFEST_PATH)
    ap.add_argument("--embedding-model", default=EMBEDDING_MODEL)
    ap.add_argument(
        "--collections",
        default="",
        help="comma-separated collection names; empty = all registry names present",
    )
    args = ap.parse_args()
    names = [x.strip() for x in args.collections.split(",") if x.strip()] or None
    data = collect_manifest(
        args.chroma_dir,
        args.bm25_dir,
        collections=names,
        embedding_model=args.embedding_model,
    )
    write_manifest(args.out, data)
    man = IndexManifest.from_dict(data)
    print(f"wrote {args.out}", flush=True)
    print(f"embedding_model={man.embedding_model}", flush=True)
    for name, coll in man.collections.items():
        print(
            f"collection[{name}] count={coll.count} fp={coll.fingerprint[:12]}…",
            flush=True,
        )
    print(
        f"bm25 n_docs={man.bm25.n_docs} k1={man.bm25.k1} b={man.bm25.b} "
        f"fp={man.bm25.fingerprint[:12]}…",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
