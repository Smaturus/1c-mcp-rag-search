"""Post-build gates для data/chroma_token (фаза 3.2, без BM25/поиска).

Проверяет по sqlite (read-only):
- коллекция code_proc, dim = 384;
- все чанки <=500 (политика) и <=512 (sanity) токенов (полный перебор);
- нет пустых чанков;
- все source — .bsl (0 XML-путей), под 1C_Corpus;
- все relevant_sources v2/v3 code присутствуют.

Результат — JSON в stdout.
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

from mcp_1c_rag.chunking_tokens import get_tokenizer  # noqa: E402


def load_relevant_sources() -> list[str]:
    sources: set[str] = set()
    v2 = ROOT / "data" / "eval" / "gold_set_v2.jsonl"
    v3 = ROOT / "data" / "eval" / "gold_set_v3_holdout.jsonl"
    for p in (v2, v3):
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if p.name == "gold_set_v3_holdout.jsonl" and row.get("scope") != "code":
                continue
            for src in row.get("relevant_sources", []):
                sources.add(src.replace("\\", "/"))
    return sorted(sources)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, default=ROOT / "data" / "chroma_token" / "chroma.sqlite3")
    ap.add_argument(
        "--embedding-model",
        required=True,
        help="имя токенизатора для пересчёта длины чанков",
    )
    args = ap.parse_args()

    started = time.time()
    tok = get_tokenizer(args.embedding_model)

    db = sqlite3.connect(f"file:{args.db.as_posix()}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row

    dims = {r["name"]: r["dimension"] for r in db.execute("SELECT name, dimension FROM collections")}

    rows = db.execute(
        """
        SELECT e.embedding_id AS cid, d.string_value AS doc, s.string_value AS source
        FROM embeddings e
        JOIN embedding_metadata d ON d.id = e.id AND d.key = 'chroma:document'
        JOIN embedding_metadata s ON s.id = e.id AND s.key = 'source'
        WHERE e.embedding_id LIKE 'bsl_proc:%'
        """
    ).fetchall()
    db.close()

    n = len(rows)
    n_over_500 = 0
    n_over_512 = 0
    n_empty = 0
    n_non_bsl = 0
    n_xml = 0
    token_lengths: list[int] = []
    sources: set[str] = set()

    for r in rows:
        doc = r["doc"] or ""
        src = r["source"] or ""
        sources.add(src)
        if not doc.strip():
            n_empty += 1
        if not src.endswith(".bsl"):
            n_non_bsl += 1
        if src.lower().endswith(".xml") or "/Ext/" not in src and ".xml" in src.lower():
            n_xml += 1
        t = len(tok.encode(f"passage: {doc}", add_special_tokens=True, truncation=False))
        token_lengths.append(t)
        if t > 500:
            n_over_500 += 1
        if t > 512:
            n_over_512 += 1

    s = sorted(token_lengths)
    def pct(q):
        i = min(len(s) - 1, max(0, int(q * len(s))))
        return s[i] if s else 0

    relevant = load_relevant_sources()
    missing_relevant = [r for r in relevant if r not in sources]

    result = {
        "db": str(args.db),
        "collections_dims": dims,
        "n_chunks": n,
        "n_over_500": n_over_500,
        "n_over_512": n_over_512,
        "n_empty": n_empty,
        "n_non_bsl": n_non_bsl,
        "n_xml": n_xml,
        "token_stats": {
            "p50": pct(0.50), "p95": pct(0.95), "p99": pct(0.99),
            "max": s[-1] if s else None,
            "mean": round(statistics.fmean(token_lengths), 2) if token_lengths else None,
        },
        "n_relevant_required": len(relevant),
        "n_relevant_missing": len(missing_relevant),
        "missing_relevant": missing_relevant[:20],
        "elapsed_sec": round(time.time() - started, 1),
        "gates": {
            "dim_384": dims.get("code_proc") == 384,
            "zero_over_500": n_over_500 == 0,
            "zero_over_512": n_over_512 == 0,
            "zero_empty": n_empty == 0,
            "zero_xml": n_xml == 0,
            "all_relevant_present": len(missing_relevant) == 0,
        },
    }
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
