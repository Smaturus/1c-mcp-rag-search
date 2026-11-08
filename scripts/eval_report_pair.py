"""Парное сравнение старого и нового индекса по per-query метрикам.

Два набора прогонов (old и new), каждый — копии ``report_pipeline.json`` от
``scripts/eval_pipeline.py`` (содержат ``per_query[<mode>]``). Значения метрики
сначала усредняются ПО ЗАПРОСАМ внутри каждого набора, затем наборы соединяются
по id запроса и сравниваются парно: Δ = new − old.

Считается: mean old/new, Δ, парный bootstrap 95% CI (seed 42, 10 000),
paired t-test, p, победы/ничьи/поражения, n.

Примеры:
    python scripts/eval_report_pair.py \
        --old "data/eval/_runs_token/old_v2_run*.json" \
        --new "data/eval/_runs_token/new_v2_run*.json" \
        --mode hybrid --metric ndcg@10
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import ttest_rel

ROOT = Path(__file__).resolve().parents[1]

METRICS = ("hit@1", "mrr@10", "ndcg@10", "recall@10")
TIE_EPS = 1e-12
CI_LEVEL = 95.0
N_BOOT = 10000
SEED = 42


class PairError(RuntimeError):
    """Ожидаемая ошибка входных данных — без traceback."""


@dataclass
class RunSet:
    """Один набор прогонов: средние per-query по прогонам."""

    description: str
    files: list[Path]
    n_runs: int
    values: dict[str, float]  # id -> mean metric


def load_per_query(report_path: Path) -> dict[str, list[dict[str, Any]]]:
    """Прочитать ``per_query`` из отчёта eval_pipeline.json."""
    if not report_path.exists():
        raise PairError(f"Отчёт не найден: {report_path}")
    try:
        raw = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PairError(f"{report_path}: невалидный JSON ({exc})") from exc
    per_query = raw.get("per_query")
    if not isinstance(per_query, dict):
        raise PairError(
            f"{report_path}: нет объекта per_query (ожидается формат scripts/eval_pipeline.py)"
        )
    return per_query


def metric_by_id(items: list[dict[str, Any]], metric: str, mode: str) -> dict[str, float]:
    """Собрать ``id -> значение метрики`` для одного режима."""
    values: dict[str, float] = {}
    for item in items:
        query_id = str(item.get("id", ""))
        if not query_id:
            raise PairError(f"Режим {mode!r}: запись без id")
        if query_id in values:
            raise PairError(f"Режим {mode!r}: повторяющийся id {query_id!r}")
        if metric not in item:
            raise PairError(f"Режим {mode!r}, запрос {query_id!r}: нет метрики {metric!r}")
        values[query_id] = float(item[metric])
    return values


def resolve_glob(pattern: str) -> list[Path]:
    raw = Path(pattern)
    files = sorted(raw.parent.glob(raw.name)) if raw.is_absolute() else sorted(ROOT.glob(pattern))
    if not files:
        raise PairError(f"Нет файлов по шаблону {pattern!r} (корень {ROOT})")
    return files


def load_run_set(pattern: str, metric: str, mode: str) -> RunSet:
    """Средние per-query по прогонам одного набора."""
    files = resolve_glob(pattern)
    collected: dict[str, list[float]] = {}
    for path in files:
        per_query = load_per_query(path)
        if mode not in per_query:
            raise PairError(f"{path}: нет режима {mode!r}")
        by_id = metric_by_id(per_query[mode], metric, mode)
        for query_id, value in by_id.items():
            collected.setdefault(query_id, []).append(value)
    averaged: dict[str, float] = {}
    for query_id, values in collected.items():
        if len(values) != len(files):
            raise PairError(
                f"Запрос {query_id!r}: значений {len(values)}, а прогонов {len(files)}"
            )
        averaged[query_id] = float(np.mean(values))
    return RunSet(
        description=f"{len(files)} прогонов по шаблону `{pattern}`",
        files=files,
        n_runs=len(files),
        values=averaged,
    )


def bootstrap_ci(deltas: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED) -> tuple[float, float, float]:
    """Percentile 95% CI среднего Δ и двусторонний bootstrap p-value."""
    n = len(deltas)
    if n == 0 or n_boot <= 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = deltas[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    low = float(np.percentile(means, (100.0 - CI_LEVEL) / 2.0))
    high = float(np.percentile(means, 100.0 - (100.0 - CI_LEVEL) / 2.0))
    below = float((means <= 0.0).mean())
    above = float((means >= 0.0).mean())
    p_value = min(1.0, 2.0 * min(below, above))
    return low, high, p_value


def analyse(old: RunSet, new: RunSet, *, metric: str, mode: str) -> dict[str, Any]:
    """Сравнить два набора парно по id."""
    old_ids = set(old.values)
    new_ids = set(new.values)
    if old_ids != new_ids:
        only_old = sorted(old_ids - new_ids)
        only_new = sorted(new_ids - old_ids)
        raise PairError(
            "Наборы id не совпадают: "
            f"только в old {only_old[:5]}; только в new {only_new[:5]}"
        )
    ids = sorted(old_ids)
    n = len(ids)
    if n < 3:
        raise PairError(f"Слишком мало парных запросов: n={n}")

    arr_old = np.array([old.values[i] for i in ids], dtype=float)
    arr_new = np.array([new.values[i] for i in ids], dtype=float)
    deltas = arr_new - arr_old  # Δ = new − old
    ci_low, ci_high, boot_p = bootstrap_ci(deltas)
    t_test = ttest_rel(arr_new, arr_old)
    wins = int((deltas > TIE_EPS).sum())
    losses = int((deltas < -TIE_EPS).sum())
    ties = n - wins - losses
    return {
        "metric": metric,
        "mode": mode,
        "n_queries": n,
        "n_runs_old": old.n_runs,
        "n_runs_new": new.n_runs,
        "old": {"description": old.description, "files": [str(p) for p in old.files]},
        "new": {"description": new.description, "files": [str(p) for p in new.files]},
        "mean_old": float(arr_old.mean()),
        "mean_new": float(arr_new.mean()),
        "mean_delta": float(deltas.mean()),
        "median_delta": float(np.median(deltas)),
        "sd_delta": float(deltas.std(ddof=1)) if n > 1 else 0.0,
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "bootstrap": {
            "n_resamples": N_BOOT,
            "seed": SEED,
            "ci_level": CI_LEVEL,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "p_value": boot_p,
        },
        "ttest": {
            "statistic": float(t_test.statistic),
            "p_value": float(t_test.pvalue),
            "df": int(t_test.df),
        },
        "per_query": [
            {"id": qid, "old": float(o), "new": float(nv), "delta": float(d)}
            for qid, o, nv, d in zip(ids, arr_old, arr_new, deltas, strict=True)
        ],
    }


def build_markdown(result: dict[str, Any]) -> str:
    ci_low = result["bootstrap"]["ci_low"]
    ci_high = result["bootstrap"]["ci_high"]
    lines = [
        f"# Парное сравнение: новый vs старый индекс (`{result['mode']}`)",
        "",
        f"**Метрика:** {result['metric']} · **Запросов:** {result['n_queries']} · "
        f"**Δ = new − old**",
        "",
        "| Показатель | Значение |",
        "|---|---|",
        f"| Среднее old | {result['mean_old']:.4f} |",
        f"| Среднее new | {result['mean_new']:.4f} |",
        f"| **Δ = new − old, среднее** | **{result['mean_delta']:+.4f}** |",
        f"| 95% bootstrap CI (percentile) | [{ci_low:+.4f}; {ci_high:+.4f}] |",
        f"| **p (paired t-test)** | **{result['ttest']['p_value']:.4f}** |",
        f"| **p (bootstrap, two-sided)** | **{result['bootstrap']['p_value']:.4f}** |",
        f"| Побед / ничьих / поражений | {result['wins']} / {result['ties']} / {result['losses']} |",
        f"| Пар (n) | {result['n_queries']} |",
        f"| Прогонов old/new | {result['n_runs_old']} / {result['n_runs_new']} |",
        "",
        "Δ > 0 означает, что новый индекс лучше старого; Δ < 0 — наоборот.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="Парное сравнение old vs new индекса")
    ap.add_argument("--old", required=True, help="glob run_*.json старого контроля")
    ap.add_argument("--new", required=True, help="glob run_*.json нового индекса")
    ap.add_argument("--mode", default="hybrid", help="режим поиска (hybrid/vector)")
    ap.add_argument("--metric", default="ndcg@10", choices=METRICS)
    ap.add_argument("--out-md", type=Path, default=None)
    ap.add_argument("--out-json", type=Path, default=None)
    args = ap.parse_args()

    try:
        old = load_run_set(args.old, args.metric, args.mode)
        new = load_run_set(args.new, args.metric, args.mode)
        result = analyse(old, new, metric=args.metric, mode=args.mode)
    except PairError as exc:
        print(f"Пропуск: {exc}", file=sys.stderr)
        return 2

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown = build_markdown(result)
    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(markdown, encoding="utf-8")

    print(markdown)
    if args.out_json:
        print(f"JSON: {args.out_json}")
    if args.out_md:
        print(f"Markdown: {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
