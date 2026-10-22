"""Проверить, укладываются ли чанки индекса в лимит e5-small (512 токенов).

Модель intfloat/multilingual-e5-small обучена с max_seq_length=512; всё, что
длиннее, молча обрезается при эмбеддинге — хвост текста не влияет на вектор.
Скрипт считает распределение длин чанков и долю превышающих лимит.

Запуск:
    python scripts/check_embedding_tokens.py
    python scripts/check_embedding_tokens.py --limit 5000   # быстрый смоук
"""

from __future__ import annotations

import argparse
import sqlite3
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mcp_1c_rag.config import (  # noqa: E402
    CHROMA_DIR,
    DATA_DIR,
    EMBEDDING_MODEL,  # noqa: E402
)

MAX_TOKENS = 512
REPORT = DATA_DIR / "eval" / "embedding_tokens.md"


def _docs(sqlite_path: Path, limit: int):
    db = sqlite3.connect(f"file:{sqlite_path.as_posix()}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    query = """
        SELECT e.embedding_id AS cid, d.string_value AS doc
        FROM embeddings e
        JOIN embedding_metadata d ON d.id = e.id AND d.key = 'chroma:document'
        ORDER BY e.id
    """
    rows = db.execute(query)
    try:
        for index, row in enumerate(rows, start=1):
            if limit and index > limit:
                break
            if row["doc"]:
                yield row["cid"], row["doc"]
    finally:
        db.close()


def main() -> int:
    ap = argparse.ArgumentParser(description="e5 token limit check")
    ap.add_argument("--chroma-dir", type=Path, default=CHROMA_DIR)
    ap.add_argument("--limit", type=int, default=0, help="первые N чанков (0 = все)")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    sqlite_path = args.chroma_dir / "chroma.sqlite3"
    print(f"tokenizer: {EMBEDDING_MODEL}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)

    started = time.time()
    lengths: list[int] = []
    by_kind: dict[str, int] = {"code": 0, "metadata": 0}
    over_by_kind: dict[str, int] = {"code": 0, "metadata": 0}
    over_examples: list[tuple[int, str]] = []

    for cid, doc in _docs(sqlite_path, args.limit):
        tokens = len(tokenizer.encode(doc, add_special_tokens=False))
        lengths.append(tokens)
        kind = "code" if cid.startswith("bsl:") else "metadata"
        by_kind[kind] += 1
        if tokens > MAX_TOKENS:
            over_by_kind[kind] += 1
            if len(over_examples) < 3:
                over_examples.append((tokens, cid))

    if not lengths:
        raise SystemExit("нет документов для проверки")

    lengths_sorted = sorted(lengths)
    over = sum(1 for value in lengths_sorted if value > MAX_TOKENS)
    def pct(share: float) -> int:
        index = min(len(lengths_sorted) - 1, int(share * len(lengths_sorted)))
        return lengths_sorted[index]

    report = [
        "# Проверка лимита токенов embedder'а",
        "",
        f"- модель: `{EMBEDDING_MODEL}`",
        f"- лимит: **{MAX_TOKENS}** токенов (max_seq_length)",
        f"- чанков проверено: **{len(lengths)}**",
        f"- превышают лимит: **{over}** ({over / len(lengths) * 100:.2f} %)",
        "",
        "| Перцентиль | Токенов |",
        "|---|---|",
        f"| p50 | {pct(0.50)} |",
        f"| p90 | {pct(0.90)} |",
        f"| p95 | {pct(0.95)} |",
        f"| p99 | {pct(0.99)} |",
        f"| max | {lengths_sorted[-1]} |",
        "",
        f"- среднее: {statistics.fmean(lengths):.1f}",
        f"- `code` чанков: {by_kind['code']}, из них > {MAX_TOKENS}: {over_by_kind['code']}",
        f"- `metadata` чанков: {by_kind['metadata']}, из них > {MAX_TOKENS}: {over_by_kind['metadata']}",
        f"- время проверки: {time.time() - started:.0f} c",
    ]
    if over_examples:
        report += ["", "Примеры превышений:"] + [f"- {n} токенов, `{cid}`" for n, cid in over_examples]

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
