"""Распределение токенов large-токенизатора на чанках code_proc (bsl_proc:).

Читает тексты чанков только из ``data/chroma_xml`` (read-only) и считает длины
в токенах токенизатора ``intfloat/multilingual-e5-large`` (max_seq_length=512).

Полный перебор всех 154 560 чанков (перепись, а не выборка) — точная доля
>512 токенов для гейта 0.7. Результат пишется в локальные заметки, а НЕ в
``data/eval/embedding_tokens.md`` (опубликованный отчёт — только чтение).

Exit code: 0 = PASS (доля <= 1 %), 2 = STOP (доля > 1 %).
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

MAX_TOKENS = 512
MODEL = "intfloat/multilingual-e5-large"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chroma-db", type=Path, required=True)
    ap.add_argument("--out-md", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, required=True)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    print(f"tokenizer: {MODEL}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)

    db = sqlite3.connect(f"file:{args.chroma_db.as_posix()}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    started = time.time()
    lengths: list[int] = []
    over_examples: list[tuple[int, str]] = []

    cur = db.execute(
        """
        SELECT e.embedding_id AS cid, md.string_value AS doc
        FROM embeddings e
        JOIN embedding_metadata md ON md.id = e.id AND md.key = 'chroma:document'
        WHERE e.embedding_id LIKE 'bsl_proc:%'
        ORDER BY e.id
        """
    )
    n = 0
    for row in cur:
        doc = row["doc"] or ""
        if not doc.strip():
            continue
        n += 1
        tok = len(tokenizer.encode(doc, add_special_tokens=False))
        lengths.append(tok)
        if tok > MAX_TOKENS and len(over_examples) < 3:
            over_examples.append((tok, row["cid"]))
        if n % 50000 == 0:
            print(f"  ... {n} чанков, {time.time() - started:.0f} c", flush=True)
    db.close()

    if not lengths:
        raise SystemExit("нет чанков для проверки")

    lengths_sorted = sorted(lengths)
    over = sum(1 for v in lengths_sorted if v > MAX_TOKENS)
    frac = over / len(lengths)

    def pct(share: float) -> int:
        i = min(len(lengths_sorted) - 1, max(0, int(share * len(lengths_sorted))))
        return lengths_sorted[i]

    mean = statistics.fmean(lengths)
    result = {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "n_chunks": len(lengths),
        "n_over": over,
        "frac_over": round(frac, 6),
        "p50": pct(0.50),
        "p95": pct(0.95),
        "p99": pct(0.99),
        "max": lengths_sorted[-1],
        "mean": round(mean, 2),
        "elapsed_sec": round(time.time() - started, 1),
    }

    md_lines = [
        "# Токенизация large на чанках code_proc (bsl_proc:)",
        "",
        f"- модель: `{MODEL}`",
        f"- лимит: **{MAX_TOKENS}** токенов",
        f"- чанков проверено: **{len(lengths)}** (полный перебор, не выборка)",
        f"- превышают лимит: **{over}** ({frac * 100:.2f} %)",
        "",
        "| Перцентиль | Токенов |",
        "|---|---|",
        f"| p50 | {pct(0.50)} |",
        f"| p95 | {pct(0.95)} |",
        f"| p99 | {pct(0.99)} |",
        f"| max | {lengths_sorted[-1]} |",
        "",
        f"- среднее: {mean:.1f}",
        f"- время: {time.time() - started:.0f} c",
    ]
    if over_examples:
        md_lines += ["", "Примеры превышений:"]
        md_lines += [f"- {n_tok} токенов, `{cid}`" for n_tok, cid in over_examples]

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    args.out_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n".join(md_lines), flush=True)
    verdict = "PASS" if frac <= 0.01 else "STOP"
    print(f"\nRESULT: {verdict} (frac_over={frac * 100:.2f} %, порог 1 %)", flush=True)
    return 0 if frac <= 0.01 else 2


if __name__ == "__main__":
    raise SystemExit(main())
