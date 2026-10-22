"""Собрать глобальный BM25-индекс по всем чанкам существующей Chroma-базы.

Читает документы прямо из sqlite (без открытия Chroma-клиента) и сохраняет
индекс в ``data/bm25/``. Повторный запуск перезаписывает индекс.

Пример:
    python scripts/build_bm25.py
    python scripts/build_bm25.py --chroma-dir data/chroma --out data/bm25
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from collections.abc import Iterable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mcp_1c_rag.bm25 import Bm25Index, docs_from_sqlite  # noqa: E402
from mcp_1c_rag.config import BM25_B, BM25_DIR, BM25_K1, CHROMA_DIR  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Построить глобальный BM25-индекс")
    ap.add_argument("--chroma-dir", type=Path, default=CHROMA_DIR)
    ap.add_argument("--out", type=Path, default=BM25_DIR)
    ap.add_argument("--k1", type=float, default=BM25_K1)
    ap.add_argument("--b", type=float, default=BM25_B)
    ap.add_argument("--limit", type=int, default=0, help="ограничить число чанков (смоук)")
    ap.add_argument(
        "--id-prefix",
        default="",
        help="включить только документы с префиксом id (например bsl_proc:)",
    )
    args = ap.parse_args()

    sqlite_path = args.chroma_dir / "chroma.sqlite3"
    if not sqlite_path.exists():
        raise SystemExit(f"не найдена база: {sqlite_path}")

    def stream() -> Iterable[tuple[str, str]]:
        """Документы из sqlite (опционально ограниченные для смоука)."""
        items = docs_from_sqlite(sqlite_path)
        if args.id_prefix:
            items = ((doc_id, text) for doc_id, text in items if doc_id.startswith(args.id_prefix))
        return itertools.islice(items, args.limit) if args.limit else items

    t0 = time.time()
    print(f"читаю документы из {sqlite_path} ...", flush=True)
    index = Bm25Index.build(stream(), k1=args.k1, b=args.b)
    print(f"индекс построен за {time.time() - t0:.0f} c: {index.stats()}", flush=True)
    index.save(args.out)
    size_mb = sum(f.stat().st_size for f in args.out.glob("*")) / 1048576
    print(f"сохранено в {args.out} ({size_mb:.1f} МБ) за {time.time() - t0:.0f} c")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
