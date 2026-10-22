"""Сравнить чанкинг: коллекция `code` (символьные окна) vs `code_proc` (процедуры).

Запуск после полной сборки `code_proc`:

    python scripts/eval_chunking.py

Что делает:
1. Считает vector-поиск по gold-set v2 в обеих базах и сравнивает file-level
   метрики (одна и та же модель e5-small, разница только в чанкинге).
2. Если для `code_proc` собран отдельный BM25 (`--bm25-proc`), дополнительно
   сравнивает hybrid.
3. Пишет отчёт `data/eval/chunking_comparison.md` и JSON.

Базы:
- `code`: data/chroma (текущая), BM25: data/bm25;
- `code_proc`: data/chroma_proc, BM25: data/bm25_code_proc.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mcp_1c_rag.config import DATA_DIR  # noqa: E402
from mcp_1c_rag.eval_metrics import ndcg_at_k, recall_at_k  # noqa: E402

GOLDEN = DATA_DIR / "eval" / "gold_set_v2.jsonl"
REPORT_MD = DATA_DIR / "eval" / "chunking_comparison.md"
REPORT_JSON = DATA_DIR / "eval" / "chunking_comparison.json"
K = 10


def _rows() -> list[dict]:
    return [
        json.loads(line)
        for line in GOLDEN.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _evaluate(rows: list[dict], *, chroma_dir: Path, bm25_dir: Path | None,
              collection: str, modes: tuple[str, ...]) -> dict[str, dict[str, float]]:
    """Прогнать запросы в заданной базе; окружение задаётся до импорта движка."""
    os.environ["MCP_1C_CHROMA_DIR"] = str(chroma_dir)
    if bm25_dir is not None:
        os.environ["MCP_1C_BM25_DIR"] = str(bm25_dir)

    # импорт после подмены env: config читает переменные при импорте
    from mcp_1c_rag.search_modes import search

    result: dict[str, dict[str, float]] = {}
    for mode in modes:
        ndcgs: list[float] = []
        recalls: list[float] = []
        for row in rows:
            candidates = search(row["query"], top_k=K, mode=mode, collection=collection)
            sources = [c.source for c in candidates]
            ndcgs.append(ndcg_at_k(sources, row["relevant_sources"], k=K))
            recalls.append(recall_at_k(sources, row["relevant_sources"], k=K))
        result[mode] = {
            "ndcg@10": sum(ndcgs) / len(ndcgs),
            "recall@10": sum(recalls) / len(recalls),
        }
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare char vs procedure chunking")
    ap.add_argument("--chroma", type=Path, default=DATA_DIR / "chroma")
    ap.add_argument("--bm25", type=Path, default=DATA_DIR / "bm25")
    ap.add_argument("--chroma-proc", type=Path, default=DATA_DIR / "chroma_proc")
    ap.add_argument("--bm25-proc", type=Path, default=DATA_DIR / "bm25_code_proc")
    args = ap.parse_args()

    rows = _rows()
    modes: tuple[str, ...] = ("vector", "hybrid")

    old = _evaluate(rows, chroma_dir=args.chroma, bm25_dir=args.bm25,
                    collection="code", modes=modes)
    if not args.chroma_proc.exists():
        raise SystemExit(f"нет базы процедур: {args.chroma_proc}")
    proc_bm25 = args.bm25_proc if args.bm25_proc.exists() else None
    proc_modes = modes if proc_bm25 else ("vector",)
    new = _evaluate(rows, chroma_dir=args.chroma_proc, bm25_dir=proc_bm25,
                    collection="code_proc", modes=proc_modes)

    lines = [
        "# Сравнение чанкинга: символьные окна vs процедуры",
        "",
        f"- gold-set: `{GOLDEN.name}`, запросов: {len(rows)}, k={K}",
        f"- `code`: `{args.chroma}` (окно 1200/200)",
        f"- `code_proc`: `{args.chroma_proc}` (процедуры, max_chars=1000/150)",
        "",
        "| Режим | Метрика | code | code_proc | Δ |",
        "|---|---|---|---|---|",
    ]
    for mode in modes:
        for metric in ("ndcg@10", "recall@10"):
            if mode in new:
                before = old[mode][metric]
                after = new[mode][metric]
                lines.append(
                    f"| {mode} | {metric} | {before:.3f} | {after:.3f} | {after - before:+.3f} |"
                )
    if proc_bm25 is None:
        lines += ["", "hybrid для `code_proc` не считался: нет BM25 "
                      "(`python scripts/build_bm25.py --chroma-dir data/chroma_proc "
                      "--id-prefix bsl_proc: --out data/bm25_code_proc`)."]
    lines += [
        "",
        "Вывод фиксируется в ADR 0007; дефолт меняется только если Δ устойчив "
        "и не ухудшает recall.",
    ]

    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    REPORT_JSON.write_text(
        json.dumps({"code": old, "code_proc": new, "bm25_proc": bool(proc_bm25)},
                   ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
