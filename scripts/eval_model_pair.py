"""Парное сравнение двух embedding-моделей по per-query nDCG@10.

Читает отчёты ``report_pipeline.json`` (формат ``scripts/eval_pipeline.py``)
одного режима для двух моделей (e5-small и e5-large), усредняет per-query
значение метрики по прогонам, соединяет по ``id`` и считает
Δ = large − small.

Считается:
- средние по моделям, среднее Δ и парный bootstrap 95% CI (percentile);
- paired t-test (scipy ``ttest_rel``) и bootstrap p-value;
- побед / ничьих / поражений (Δ > TIE_EPS / |Δ| ≤ TIE_EPS / Δ < −TIE_EPS);
- n = число парных запросов.

Соединение по id строгое: если наборы id не равны — fail (exit 2).

Пример:
    python scripts/eval_model_pair.py \
        --small data/eval/_pilot_1c/small/report_pipeline.json \
        --large data/eval/_pilot_1c/large/report_pipeline.json \
        --mode hybrid
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import ttest_rel

METRICS = ("hit@1", "mrr@10", "ndcg@10", "recall@10")
TIE_EPS = 1e-12
CI_LEVEL = 95.0


class PairError(RuntimeError):
    """Ожидаемая ошибка входных данных — без traceback."""


def load_per_query(report_path: Path) -> dict[str, list[dict[str, Any]]]:
    if not report_path.exists():
        raise PairError(f"Отчёт не найден: {report_path}")
    try:
        raw = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PairError(f"{report_path}: невалидный JSON ({exc})") from exc
    per_query = raw.get("per_query")
    if not isinstance(per_query, dict):
        raise PairError(
            f"{report_path}: нет объекта per_query (ожидается формат scripts/eval_pipeline.py)"
        )
    return per_query


def metric_by_id(items: list[dict[str, Any]], metric: str, mode: str) -> dict[str, float]:
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


def average_runs(files: list[Path], mode: str, metric: str) -> dict[str, float]:
    """Усреднить per-query метрику по прогонам одной модели: id -> mean."""
    collected: dict[str, list[float]] = {}
    for path in files:
        per_query = load_per_query(path)
        if mode not in per_query:
            raise PairError(f"{path}: режим {mode!r} отсутствует; "
                            f"доступны: {', '.join(sorted(per_query))}")
        by_id = metric_by_id(per_query[mode], metric, mode)
        for qid, value in by_id.items():
            collected.setdefault(qid, []).append(value)
    if not collected:
        raise PairError("нет данных для усреднения")
    n_files = len(files)
    for qid, values in collected.items():
        if len(values) != n_files:
            raise PairError(
                f"запрос {qid!r}: значений {len(values)}, а прогонов {n_files}"
            )
    return {qid: float(np.mean(vals)) for qid, vals in collected.items()}


def bootstrap_stats(deltas: np.ndarray, *, n_boot: int, seed: int) -> tuple[float, float, float]:
    """Percentile 95% CI среднего Δ и двусторонний bootstrap p-value."""
    n = len(deltas)
    if n_boot <= 0 or n == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = deltas[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    low, high = np.percentile(
        means, [(100.0 - CI_LEVEL) / 2.0, 100.0 - (100.0 - CI_LEVEL) / 2.0]
    )
    below = float((means <= 0.0).mean())
    above = float((means >= 0.0).mean())
    p_value = min(1.0, 2.0 * min(below, above))
    return float(low), float(high), p_value


def compare(
    small: dict[str, float],
    large: dict[str, float],
    *,
    n_boot: int = 10000,
    seed: int = 42,
    small_label: str = "e5-small",
    large_label: str = "e5-large",
    metric: str = "ndcg@10",
    mode: str = "hybrid",
) -> dict[str, Any]:
    """Δ = large − small, парно по id. Строгая проверка равенства наборов id."""
    ids = sorted(large)
    if set(small) != set(large):
        only_small = sorted(set(small) - set(large))
        only_large = sorted(set(large) - set(small))
        raise PairError(
            "Наборы id не равны: "
            f"только в {small_label!r}: {only_small[:5]}; "
            f"только в {large_label!r}: {only_large[:5]}"
        )
    n = len(ids)
    if n < 3:
        raise PairError(f"Слишком мало парных запросов: n={n}")

    arr_small = np.array([small[i] for i in ids], dtype=float)
    arr_large = np.array([large[i] for i in ids], dtype=float)
    deltas = arr_large - arr_small
    mean_delta = float(deltas.mean())
    sd_delta = float(deltas.std(ddof=1))
    wins = int((deltas > TIE_EPS).sum())
    losses = int((deltas < -TIE_EPS).sum())
    ties = n - wins - losses
    ci_low, ci_high, boot_p = bootstrap_stats(deltas, n_boot=n_boot, seed=seed)
    t_test = ttest_rel(arr_large, arr_small)
    return {
        "mode": mode,
        "metric": metric,
        "n_queries": n,
        "mean_small": float(arr_small.mean()),
        "mean_large": float(arr_large.mean()),
        "mean_delta": mean_delta,
        "median_delta": float(np.median(deltas)),
        "sd_delta": sd_delta,
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "bootstrap": {
            "n_resamples": int(n_boot),
            "seed": int(seed),
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
            {"id": qid, "small": float(x), "large": float(y), "delta": float(d)}
            for qid, x, y, d in zip(ids, arr_small, arr_large, deltas, strict=True)
        ],
    }


def build_markdown(result: dict[str, Any], small_label: str, large_label: str) -> str:
    boot = result["bootstrap"]
    lines = [
        f"# Парное сравнение моделей: `{large_label}` vs `{small_label}`",
        "",
        f"**Метрика:** {result['metric']} · **Режим:** `{result['mode']}` · "
        f"**Запросов:** {result['n_queries']}",
        "",
        "| Показатель | Значение |",
        "|---|---|",
        f"| Среднее `{small_label}` | {result['mean_small']:.4f} |",
        f"| Среднее `{large_label}` | {result['mean_large']:.4f} |",
        f"| **Δ = `{large_label}` − `{small_label}`** | **{result['mean_delta']:+.4f}** |",
        f"| Медиана Δ | {result['median_delta']:+.4f} |",
        f"| 95% bootstrap CI (percentile) | [{boot['ci_low']:+.4f}; {boot['ci_high']:+.4f}] |",
        f"| **p (paired t-test)** | **{result['ttest']['p_value']:.4f}** |",
        f"| **p (bootstrap, two-sided)** | **{boot['p_value']:.4f}** |",
        f"| Побед / ничьих / поражений | {result['wins']} / {result['ties']} / {result['losses']} |",
        f"| Пар (n) | {result['n_queries']} |",
        "",
        f"Δ > 0 означает, что `{large_label}` лучше `{small_label}`.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(
        description="Парное сравнение embedding-моделей по per-query метрикам"
    )
    ap.add_argument("--small", nargs="+", required=True, type=Path,
                    help="отчёты report_pipeline.json для e5-small (один или несколько прогонов)")
    ap.add_argument("--large", nargs="+", required=True, type=Path,
                    help="отчёты report_pipeline.json для e5-large")
    ap.add_argument("--mode", default="hybrid", help="режим (vector|hybrid)")
    ap.add_argument("--metric", default="ndcg@10", choices=METRICS)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-md", type=Path, default=None)
    ap.add_argument("--out-json", type=Path, default=None)
    ap.add_argument("--small-label", default="e5-small")
    ap.add_argument("--large-label", default="e5-large")
    args = ap.parse_args()

    try:
        small = average_runs(list(args.small), args.mode, args.metric)
        large = average_runs(list(args.large), args.mode, args.metric)
        result = compare(
            small, large,
            n_boot=args.n_boot, seed=args.seed,
            small_label=args.small_label, large_label=args.large_label,
            metric=args.metric, mode=args.mode,
        )
    except PairError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2

    md = build_markdown(result, args.small_label, args.large_label)
    print(md)
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"JSON: {args.out_json}")
    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(md, encoding="utf-8")
        print(f"Markdown: {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
