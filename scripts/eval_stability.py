"""Агрегатор повторных прогонов eval: среднее и разброс по конфигурациям.

Зачем: HNSW в Chroma не сериализуется (см. ADR/README — `hnsw:sync_threshold`),
поэтому при каждом старте процесса граф соседей строится заново, и top-k может
слегка меняться. Чтобы метрики в README были честными, eval прогоняется
несколько раз в отдельных процессах, а отчёт показывает mean ± std.

Вход: ``data/eval/run_*.json`` — копии ``report_pipeline.json`` от отдельных
прогонов ``scripts/eval_pipeline.py``.

Пример:
    python scripts/eval_stability.py
    python scripts/eval_stability.py --out-dir data/eval
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mcp_1c_homework.config import DATA_DIR  # noqa: E402

METRICS = ("hit@1", "mrr@10", "ndcg@10", "recall@10", "latency_ms")
MODES_ORDER = ("vector", "pearson", "hybrid", "hybrid_rerank")


def load_runs(out_dir: Path) -> list[dict]:
    """Прочитать все run_*.json из каталога отчётов."""
    runs: list[dict] = []
    for path in sorted(out_dir.glob("run_*.json")):
        runs.append(json.loads(path.read_text(encoding="utf-8")))
    if not runs:
        raise SystemExit(f"нет файлов run_*.json в {out_dir}")
    return runs


def combine(runs: list[dict]) -> dict[str, dict[str, dict[str, float]]]:
    """Собрать mean/std по (режим, метрика) из нескольких прогонов."""
    values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for run in runs:
        for mode, metrics in (run.get("results") or {}).items():
            for key in METRICS:
                if key in metrics:
                    values[mode][key].append(float(metrics[key]))
    combined: dict[str, dict[str, dict[str, float]]] = {}
    for mode, metrics_map in values.items():
        combined[mode] = {
            key: {
                "mean": statistics.fmean(vals),
                "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
                "n": float(len(vals)),
            }
            for key, vals in metrics_map.items()
        }
    return combined


def to_markdown(combined: dict[str, dict[str, dict[str, float]]], n_runs: int) -> str:
    """Markdown-таблица mean ± std."""
    lines = [
        f"# Evaluation: retrieval pipeline ({n_runs} независимых прогонов)",
        "",
        "Релевантность — на уровне файла-источника. Значения — **mean ± std**",
        "по отдельным запускам процесса (HNSW пересобирается при старте, поэтому",
        "top-k может слегка меняться).",
        "",
        "| Конфигурация | Hit@1 | MRR@10 | nDCG@10 | Recall@10 | мс/запрос |",
        "|---|---|---|---|---|---|",
    ]
    for mode in MODES_ORDER:
        metrics = combined.get(mode)
        if not metrics:
            continue
        cells = []
        for key in METRICS:
            item = metrics.get(key)
            if not item:
                cells.append("—")
                continue
            if key == "latency_ms":
                cells.append(f"{item['mean']:.0f}")
            elif item["n"] > 1:
                cells.append(f"{item['mean']:.3f} ± {item['std']:.3f}")
            else:
                cells.append(f"{item['mean']:.3f}")
        lines.append(f"| `{mode}` | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="Среднее и разброс eval по прогонам")
    ap.add_argument("--out-dir", type=Path, default=DATA_DIR / "eval")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    runs = load_runs(args.out_dir)
    combined = combine(runs)
    markdown = to_markdown(combined, len(runs))
    out_path = args.out or (args.out_dir / "report_pipeline.md")
    out_path.write_text(markdown, encoding="utf-8")
    (args.out_dir / "report_stability.json").write_text(
        json.dumps({"n_runs": len(runs), "combined": combined}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(markdown)
    print(f"отчёт: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
