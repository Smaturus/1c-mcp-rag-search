"""Smoke-запрос поиска: hybrid (и vector), проверка префикса id первого hybrid.

Модель эмбеддинга из окружения не подменяет: задайте MCP_1C_EMBEDDING_MODEL
до запуска, если индекс не на дефолтном e5-small.

Пример:
    python scripts/smoke_search.py --chroma-dir data/chroma_large --bm25-dir data/bm25_large \\
        --collection code_proc --query \"как начислить НДС с авансов полученных\"
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_EXPECTED_PREFIX = {
    "code_proc": "bsl_proc:",
    "metadata_xml": "metaobj:",
    "config_xml": "xml:",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chroma-dir", type=Path, required=True)
    ap.add_argument("--bm25-dir", type=Path, required=True)
    ap.add_argument("--collection", default="code_proc")
    ap.add_argument("--query", default="НДС с аванса")
    ap.add_argument("--top-k", type=int, default=10)
    args = ap.parse_args()

    os.environ["MCP_1C_CHROMA_DIR"] = str(args.chroma_dir)
    os.environ["MCP_1C_BM25_DIR"] = str(args.bm25_dir)
    os.environ["MCP_1C_CODE_COLLECTION"] = args.collection
    # Модель эмбеддинга намеренно не трогаем.
    sys.path.insert(0, str(ROOT / "src"))

    from mcp_1c_rag.search_modes import search  # noqa: E402

    expected = _EXPECTED_PREFIX.get(args.collection)
    ok = True
    hybrid_id: str | None = None

    for mode in ("vector", "hybrid"):
        res = search(args.query, top_k=args.top_k, mode=mode, collection=args.collection)
        n = len(res)
        print(f"[{mode}] top_k={args.top_k} results={n}", flush=True)
        if n < 1:
            ok = False
            continue
        first = res[0]
        print(
            f"  first: id={first.id} source={first.metadata.get('source', '?')} "
            f"score={first.score:.4f}",
            flush=True,
        )
        if mode == "hybrid":
            hybrid_id = first.id

    if hybrid_id is None:
        print("SMOKE_FAIL: нет попаданий hybrid", flush=True)
        return 1
    if expected is not None and not hybrid_id.startswith(expected):
        print(
            f"SMOKE_FAIL: id {hybrid_id!r} не начинается с {expected!r} "
            f"(collection={args.collection})",
            flush=True,
        )
        return 1
    if not ok:
        print("SMOKE_FAIL", flush=True)
        return 1
    print(f"SMOKE_PASS hybrid_id={hybrid_id}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
