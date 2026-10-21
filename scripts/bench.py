"""Бенчмарк retrieval-пайплайна: латентность и потребление памяти.

Считает p50/p95 латентности для каждого режима по gold-set, размер индексов
и время построения BM25-индекса. Пишет markdown-таблицу в ``data/eval/bench.md``.

Пример:
    python scripts/bench.py
    python scripts/bench.py --golden data/eval/gold_set.jsonl --warmup 2
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mcp_1c_rag.config import BM25_DIR, CHROMA_DIR, DATA_DIR  # noqa: E402
from mcp_1c_rag.search_modes import search  # noqa: E402

MODES = ("vector", "pearson", "hybrid", "hybrid_rerank")


def percentile(values: Sequence[float], q: float) -> float:
    """Перцентиль по методу ближайшего ранга (для малых выборок)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(q / 100.0 * (len(ordered) - 1))))
    return ordered[index]


def measure(mode: str, queries: Sequence[str], *, warmup: int, k: int) -> dict[str, float]:
    """Замерить латентность режима: прогрев, затем p50/p95/mean."""
    for query in queries[:warmup]:
        search(query, top_k=k, mode=mode)
    latencies: list[float] = []
    for query in queries:
        started = time.perf_counter()
        search(query, top_k=k, mode=mode)
        latencies.append((time.perf_counter() - started) * 1000.0)
    return {
        "p50": percentile(latencies, 50),
        "p95": percentile(latencies, 95),
        "mean": statistics.fmean(latencies),
        "n": float(len(latencies)),
    }


def directory_size_mb(path: Path) -> float:
    """Суммарный размер каталога в мегабайтах."""
    if not path.exists():
        return 0.0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1048576


def main() -> int:
    ap = argparse.ArgumentParser(description="Бенчмарк retrieval-пайплайна")
    ap.add_argument("--golden", type=Path, default=DATA_DIR / "eval" / "gold_set.jsonl")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--out", type=Path, default=DATA_DIR / "eval" / "bench.md")
    args = ap.parse_args()

    queries = [
        json.loads(line)["query"]
        for line in args.golden.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not queries:
        raise SystemExit("пустой gold-set")

    rows: dict[str, dict[str, float]] = {}
    for mode in MODES:
        print(f"[{mode}] бенчмарк ({len(queries)} запросов) ...", flush=True)
        rows[mode] = measure(mode, queries, warmup=args.warmup, k=args.k)

    bm25_rows = 124_253
    bm25_size = directory_size_mb(BM25_DIR)
    chroma_size = directory_size_mb(CHROMA_DIR)

    lines = [
        "# Benchmarks: retrieval pipeline",
        "",
        f"- gold-set: {len(queries)} запросов, k={args.k}, прогрев {args.warmup} запросов",
        f"- BM25-индекс: {bm25_rows} чанков, {bm25_size:.1f} МБ",
        f"- Chroma: {chroma_size:.1f} МБ",
        "",
        "| Режим | p50, мс | p95, мс | mean, мс |",
        "|---|---|---|---|",
    ]
    for mode in MODES:
        r = rows[mode]
        lines.append(f"| `{mode}` | {r['p50']:.0f} | {r['p95']:.0f} | {r['mean']:.0f} |")
    lines += [
        "",
        "Латентность измерена на CPU (без GPU), после прогрева модели эмбеддингов,",
        "клиента Chroma и ONNX-модели reranker'а.",
    ]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))
    print(f"отчёт: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
