"""Eval retrieval-пайплайна: 4 конфигурации × gold-set, file-level метрики.

Конфигурации:
    ``vector``        — только cosine (baseline);
    ``pearson``       — прежний взвешенный cos+BM25+Pearson (бывший MVP);
    ``hybrid``        — cosine + глобальный BM25 через RRF;
    ``hybrid_rerank`` — RRF + cross-encoder (flashrank).

Метрики считаются по файлам-источникам (``src/mcp_1c_homework/eval_metrics.py``):
Hit@1, MRR@10, nDCG@10, Recall@10 + средняя латентность.

Пример:
    python scripts/eval_pipeline.py
    python scripts/eval_pipeline.py --golden data/eval/gold_set.jsonl --k 10
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sqlite3
import sys
import time
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mcp_1c_homework.config import CHROMA_DIR, DATA_DIR  # noqa: E402
from mcp_1c_homework.eval_metrics import (  # noqa: E402
    AggregateMetrics,
    QueryMetrics,
    aggregate,
    evaluate_query,
    format_markdown_table,
)
from mcp_1c_homework.search_modes import search  # noqa: E402

CONFIGS: list[tuple[str, str]] = [
    ("vector", "только cosine (baseline)"),
    ("pearson", "взвешенный cos+BM25+Pearson (бывший MVP)"),
    ("hybrid", "cosine + BM25 через RRF"),
    ("hybrid_rerank", "RRF + cross-encoder reranker"),
]

METRIC_KEYS = ("hit@1", "mrr@10", "ndcg@10", "recall@10", "latency_ms")


def load_goldset(path: Path) -> list[dict[str, Any]]:
    """Прочитать JSONL gold-set и проверить обязательные поля."""
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        missing = {"id", "query", "relevant_sources"} - set(row)
        if missing:
            raise SystemExit(f"{path}:{line_no}: нет полей {sorted(missing)}")
        rows.append(row)
    if not rows:
        raise SystemExit(f"{path}: пустой gold-set")
    return rows


def load_id_to_source(sqlite_path: Path) -> dict[str, str]:
    """Построить отображение chroma-id -> файл-источник (одним запросом)."""
    db = sqlite3.connect(f"file:{sqlite_path.as_posix()}?mode=ro", uri=True)
    try:
        return {
            row[0]: row[1]
            for row in db.execute(
                """
                SELECT e.embedding_id, s.string_value
                FROM embeddings e
                JOIN embedding_metadata s ON s.id = e.id AND s.key = 'source'
                """
            )
        }
    finally:
        db.close()


def evaluate_config(
    rows: Sequence[dict[str, Any]],
    mode: str,
    *,
    k: int,
    id_to_source: dict[str, str],
) -> tuple[AggregateMetrics, list[dict[str, Any]]]:
    """Прогнать один режим по всем запросам gold-set (с прогревом)."""
    details: list[dict[str, Any]] = []
    per_query: list[QueryMetrics] = []
    if rows:
        # Прогрев: загрузка модели/клиента Chroma/reranker не должна попадать
        # в латентность первого запроса.
        with contextlib.suppress(Exception):
            search(rows[0]["query"], top_k=k, mode=mode)
    for row in rows:
        started = time.perf_counter()
        candidates = search(row["query"], top_k=k, mode=mode)
        latency_ms = (time.perf_counter() - started) * 1000.0
        sources = [id_to_source.get(c.id, "") for c in candidates]
        metrics = evaluate_query(
            sources, row["relevant_sources"], k=k, latency_ms=latency_ms
        )
        per_query.append(metrics)
        details.append(
            {
                "id": row["id"],
                "type": row.get("type", "?"),
                "query": row["query"],
                "relevant_sources": row["relevant_sources"],
                "retrieved_sources": list(metrics.retrieved_sources),
                "hit@1": metrics.hit_at_1,
                "mrr@10": metrics.mrr_at_10,
                "ndcg@10": metrics.ndcg_at_10,
                "recall@10": metrics.recall_at_10,
                "latency_ms": round(latency_ms, 1),
            }
        )
    return aggregate(per_query), details


def group_by_type(details: Sequence[dict[str, Any]]) -> dict[str, AggregateMetrics]:
    """Средние метрики по типам запросов (exact / semantic / trap)."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in details:
        grouped[item["type"]].append(item)
    result: dict[str, AggregateMetrics] = {}
    for qtype, items in grouped.items():
        metrics = [
            QueryMetrics(
                hit_at_1=i["hit@1"],
                mrr_at_10=i["mrr@10"],
                ndcg_at_10=i["ndcg@10"],
                recall_at_10=i["recall@10"],
                latency_ms=i["latency_ms"],
            )
            for i in items
        ]
        result[qtype] = aggregate(metrics)
    return result


def build_report(
    overall: dict[str, AggregateMetrics],
    per_type: dict[str, dict[str, AggregateMetrics]],
    *,
    n_queries: int,
    k: int,
) -> str:
    """Markdown-отчёт: сводная таблица + разбивка по типам."""
    lines = [
        f"# Evaluation: retrieval pipeline ({n_queries} запросов, k={k})",
        "",
        "Релевантность — на уровне файла-источника (см. `eval_metrics.py`).",
        "",
        format_markdown_table(
            overall, descriptions=dict(CONFIGS), k=k
        ),
        "",
        "## По типам запросов",
        "",
        f"| Тип | Конфигурация | Hit@1 | MRR@{k} | nDCG@{k} | Recall@{k} |",
        "|---|---|---|---|---|---|",
    ]
    for qtype in ("exact", "semantic", "trap"):
        for mode, _desc in CONFIGS:
            metrics = per_type.get(qtype, {}).get(mode)
            if metrics is None:
                continue
            v = metrics.values
            lines.append(
                f"| {qtype} | `{mode}` | {v.get('hit@1', 0):.3f} | "
                f"{v.get('mrr@10', 0):.3f} | {v.get('ndcg@10', 0):.3f} | "
                f"{v.get('recall@10', 0):.3f} |"
            )
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="Eval retrieval: 4 конфигурации на gold-set")
    ap.add_argument("--golden", type=Path, default=DATA_DIR / "eval" / "gold_set.jsonl")
    ap.add_argument("--chroma-dir", type=Path, default=CHROMA_DIR)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--out-dir", type=Path, default=DATA_DIR / "eval")
    ap.add_argument("--modes", default=",".join(mode for mode, _ in CONFIGS))
    args = ap.parse_args()

    rows = load_goldset(args.golden)
    id_to_source = load_id_to_source(args.chroma_dir / "chroma.sqlite3")
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    print(f"gold-set: {len(rows)} запросов, k={args.k}, режимы: {modes}", flush=True)

    overall: dict[str, AggregateMetrics] = {}
    details: dict[str, list[dict[str, Any]]] = {}
    per_type: dict[str, dict[str, AggregateMetrics]] = defaultdict(dict)

    for mode in modes:
        started = time.time()
        print(f"[{mode}] ...", flush=True)
        metrics, items = evaluate_config(
            rows, mode, k=args.k, id_to_source=id_to_source
        )
        overall[mode] = metrics
        details[mode] = items
        for qtype, agg in group_by_type(items).items():
            per_type[qtype][mode] = agg
        v = metrics.values
        print(
            f"  Hit@1={v['hit@1']:.3f} MRR@10={v['mrr@10']:.3f} "
            f"nDCG@10={v['ndcg@10']:.3f} Recall@10={v['recall@10']:.3f} "
            f"({v['latency_ms']:.0f} мс/запрос, {time.time() - started:.0f} c)",
            flush=True,
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "k": args.k,
        "n_queries": len(rows),
        "configs": [{"mode": mode, "description": desc} for mode, desc in CONFIGS],
        "results": {mode: m.values for mode, m in overall.items()},
        "per_type": {
            qtype: {mode: m.values for mode, m in modes_map.items()}
            for qtype, modes_map in per_type.items()
        },
        "per_query": details,
        "golden": str(args.golden),
    }
    (args.out_dir / "report_pipeline.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    markdown = build_report(overall, per_type, n_queries=len(rows), k=args.k)
    (args.out_dir / "report_pipeline.md").write_text(markdown, encoding="utf-8")
    print("\n" + markdown)
    print(f"отчёт: {args.out_dir / 'report_pipeline.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
