"""Парное сравнение двух режимов поиска по per-query метрикам eval-отчётов.

Основной режим — усреднение нескольких независимых прогонов: ``run_*.json``
(формат ``scripts/eval_pipeline.py``) содержат ``per_query[<mode>]`` для одних и
тех же 70 запросов. Значения метрики сначала усредняются ПО ЗАПРОСАМ между
прогонами, затем режимы сравниваются парно. Это убирает шум HNSW между
прогонами и даёт те же числа, что и «снимок» метрик в отчётах.

Считается:
* Δ = a − b по каждому запросу, среднее Δ и парный bootstrap 95% CI;
* paired t-test (scipy ``ttest_rel``) и bootstrap p-value — ОСНОВНЫЕ тесты;
* Cohen's dz, число побед / ничьих / поражений;
* Wilcoxon signed-rank — только справочно: при большом числе нулевых Δ
  ``zero_method='wilcox'`` отбрасывает ничьи и занижает p, поэтому он не
  используется для вывода.

Примеры:
    python scripts/eval_paired.py                       # 4 прогона run_hybrid_*.json
    python scripts/eval_paired.py --a vector --b hybrid --metric ndcg@10
    python scripts/eval_paired.py --runs-glob "data/eval/code_proc/run_hybrid_*.json"
    python scripts/eval_paired.py --report data/eval/code_proc/report_pipeline.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import ttest_rel, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS_GLOB = "data/eval/code_proc/run_hybrid_*.json"
DEFAULT_OUT_DIR = ROOT / "data" / "eval" / "code_proc"

METRICS = ("hit@1", "mrr@10", "ndcg@10", "recall@10")
# Дельты меньше этого порога считаем ничьей (значения метрик — доли).
TIE_EPS = 1e-12
CI_LEVEL = 95.0


class PairError(RuntimeError):
    """Ожидаемая ошибка входных данных (нет режима/метрики/файлов) — без traceback."""


@dataclass
class MetricSource:
    """Источник парных значений: один отчёт или усреднение нескольких прогонов."""

    kind: str
    description: str
    files: list[Path]
    n_runs: int
    values: dict[str, dict[str, float]]


def load_per_query(report_path: Path) -> dict[str, list[dict[str, Any]]]:
    """Прочитать ``per_query`` из report_pipeline.json и проверить формат."""
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
    """Собрать отображение ``id -> значение метрики`` для одного режима."""
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
    """Развернуть glob-шаблон (относительно корня репозитория) в список файлов."""
    raw = Path(pattern)
    files = sorted(raw.parent.glob(raw.name)) if raw.is_absolute() else sorted(ROOT.glob(pattern))
    if not files:
        raise PairError(f"Нет файлов по шаблону {pattern!r} (корень {ROOT})")
    return files


def load_report_values(report_path: Path, metric: str) -> dict[str, dict[str, float]]:
    """Значения метрики из одного отчёта: ``mode -> id -> значение``."""
    per_query = load_per_query(report_path)
    return {mode: metric_by_id(items, metric, mode) for mode, items in per_query.items()}


def load_runs_values(files: list[Path], metric: str) -> dict[str, dict[str, float]]:
    """Усреднить метрику по запросам между прогонами: ``mode -> id -> mean``."""
    collected: dict[str, dict[str, list[float]]] = {}
    for path in files:
        per_query = load_per_query(path)
        for mode, items in per_query.items():
            by_id = metric_by_id(items, metric, mode)
            bucket = collected.setdefault(mode, {})
            if bucket and set(bucket) != set(by_id):
                raise PairError(
                    f"{path}: набор запросов режима {mode!r} не совпадает с другими прогонами"
                )
            for query_id, value in by_id.items():
                bucket.setdefault(query_id, []).append(value)
    averaged: dict[str, dict[str, float]] = {}
    for mode, per_query_values in collected.items():
        for query_id, values in per_query_values.items():
            if len(values) != len(files):
                raise PairError(
                    f"Режим {mode!r}, запрос {query_id!r}: значений {len(values)}, "
                    f"а прогонов {len(files)}"
                )
        averaged[mode] = {qid: float(np.mean(vals)) for qid, vals in per_query_values.items()}
    return averaged


def build_source(args: argparse.Namespace, metric: str) -> MetricSource:
    """Выбрать источник: один отчёт, явный glob или glob по умолчанию."""
    if args.runs_glob and args.report:
        raise PairError("Укажите что-то одно: --report или --runs-glob")
    if args.runs_glob:
        files = resolve_glob(args.runs_glob)
        return MetricSource(
            kind="runs_glob",
            description=f"{len(files)} прогонов по шаблону `{args.runs_glob}`",
            files=files,
            n_runs=len(files),
            values=load_runs_values(files, metric),
        )
    if args.report:
        return MetricSource(
            kind="report",
            description=f"один отчёт `{args.report}`",
            files=[args.report],
            n_runs=1,
            values=load_report_values(args.report, metric),
        )
    files = resolve_glob(DEFAULT_RUNS_GLOB)
    return MetricSource(
        kind="runs_glob",
        description=f"{len(files)} прогонов по шаблону `{DEFAULT_RUNS_GLOB}`",
        files=files,
        n_runs=len(files),
        values=load_runs_values(files, metric),
    )


def bootstrap_stats(deltas: np.ndarray, *, n_boot: int, seed: int) -> tuple[float, float, float]:
    """Percentile 95% CI среднего Δ и двусторонний bootstrap p-value."""
    n = len(deltas)
    if n_boot <= 0 or n == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = deltas[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    low, high = np.percentile(means, [(100.0 - CI_LEVEL) / 2.0, 100.0 - (100.0 - CI_LEVEL) / 2.0])
    below = float((means <= 0.0).mean())
    above = float((means >= 0.0).mean())
    p_value = min(1.0, 2.0 * min(below, above))
    return float(low), float(high), p_value


def wilcoxon_supplementary(deltas: np.ndarray) -> dict[str, Any]:
    """Wilcoxon signed-rank как справка (не основной тест при большом числе ничьих)."""
    nonzero = deltas[np.abs(deltas) > TIE_EPS]
    n_zero = int(len(deltas) - len(nonzero))
    abs_nonzero = np.abs(nonzero)
    has_ties = len(np.unique(abs_nonzero)) < len(nonzero)
    result = wilcoxon(deltas, alternative="two-sided", zero_method="wilcox", method="auto")
    return {
        "p_value": float(result.pvalue),
        "statistic": float(result.statistic),
        "n_nonzero": int(len(nonzero)),
        "n_zero": n_zero,
        "has_ties": bool(has_ties),
        "note": (
            "Справочно: zero_method='wilcox' отбрасывает нулевые Δ и при большом числе "
            "ничьих занижает p; основной вывод — paired t-test + bootstrap"
        ),
    }


def analyse(
    source: MetricSource, a: str, b: str, metric: str, *, n_boot: int, seed: int
) -> dict[str, Any]:
    """Посчитать все числа парного сравнения и вернуть JSON-совместимый dict."""
    available = sorted(source.values)
    missing = [mode for mode in (a, b) if mode not in source.values]
    if missing:
        raise PairError(f"Режимы {missing} отсутствуют; доступны: {', '.join(available)}")
    if a == b:
        raise PairError("--a и --b совпадают: сравнивать режим с собой нечего")

    values_a = source.values[a]
    values_b = source.values[b]
    ids = sorted(values_a)
    if set(values_a) != set(values_b):
        only_a = sorted(set(values_a) - set(values_b))
        only_b = sorted(set(values_b) - set(values_a))
        raise PairError(
            "Наборы запросов не совпадают: "
            f"только в {a!r}: {only_a[:5]}; только в {b!r}: {only_b[:5]}"
        )
    n = len(ids)
    if n < 3:
        raise PairError(f"Слишком мало парных запросов: n={n}")

    arr_a = np.array([values_a[i] for i in ids], dtype=float)
    arr_b = np.array([values_b[i] for i in ids], dtype=float)
    deltas = arr_a - arr_b
    mean_delta = float(deltas.mean())
    sd_delta = float(deltas.std(ddof=1))
    wins = int((deltas > TIE_EPS).sum())
    losses = int((deltas < -TIE_EPS).sum())
    ties = n - wins - losses
    ci_low, ci_high, boot_p = bootstrap_stats(deltas, n_boot=n_boot, seed=seed)
    t_test = ttest_rel(arr_a, arr_b)
    significant = (ci_low > 0.0 or ci_high < 0.0) and float(t_test.pvalue) < 0.05
    return {
        "source": {
            "kind": source.kind,
            "description": source.description,
            "files": [str(path) for path in source.files],
            "n_runs": source.n_runs,
        },
        "a": a,
        "b": b,
        "metric": metric,
        "n_queries": n,
        "n_runs": source.n_runs,
        "mean_a": float(arr_a.mean()),
        "mean_b": float(arr_b.mean()),
        "mean_delta": mean_delta,
        "median_delta": float(np.median(deltas)),
        "sd_delta": sd_delta,
        "cohens_dz": mean_delta / sd_delta if sd_delta > 0 else float("nan"),
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
        "wilcoxon": wilcoxon_supplementary(deltas),
        "significant_at_5pct": significant,
        "per_query": [
            {"id": query_id, "a": float(x), "b": float(y), "delta": float(d)}
            for query_id, x, y, d in zip(ids, arr_a, arr_b, deltas, strict=True)
        ],
    }


def build_markdown(result: dict[str, Any]) -> str:
    """Компактный markdown-отчёт на русском."""
    a, b = result["a"], result["b"]
    wil = result["wilcoxon"]
    boot = result["bootstrap"]
    ci_low = boot["ci_low"]
    ci_high = boot["ci_high"]
    verdict = (
        "различие статистически значимо на уровне 5%"
        if result["significant_at_5pct"]
        else "различие НЕ значимо на уровне 5% (CI включает 0 или p ≥ 0.05)"
    )
    lines = [
        f"# Парное сравнение режимов: `{a}` vs `{b}`",
        "",
        f"**Метрика:** {result['metric']} · **Запросов:** {result['n_queries']} · "
        f"**Источник:** {result['source']['description']}",
        "",
        "| Показатель | Значение |",
        "|---|---|",
        f"| Среднее `{a}` | {result['mean_a']:.4f} |",
        f"| Среднее `{b}` | {result['mean_b']:.4f} |",
        f"| **Δ = `{a}` − `{b}`, среднее** | **{result['mean_delta']:+.4f}** |",
        f"| Медиана Δ | {result['median_delta']:+.4f} |",
        f"| 95% bootstrap CI (percentile) | [{ci_low:+.4f}; {ci_high:+.4f}] |",
        f"| **p (paired t-test)** | **{result['ttest']['p_value']:.4f}** |",
        f"| **p (bootstrap, two-sided)** | **{boot['p_value']:.4f}** |",
        f"| Cohen's dz | {result['cohens_dz']:.3f} |",
        f"| Побед / ничьих / поражений | {result['wins']} / {result['ties']} / {result['losses']} |",
        f"| Пар (n) | {result['n_queries']} |",
        f"| Прогонов (усреднение per-query) | {result['n_runs']} |",
        "",
        f"**Вывод:** Δ = {result['mean_delta']:+.4f}, {verdict}.",
        "",
        "## Метод",
        "",
        f"- Источник: {result['source']['description']} "
        f"({', '.join(Path(p).name for p in result['source']['files'])}).",
        "- Значения метрики сначала усредняются по запросам между прогонами, затем режимы "
        "сравниваются парно — так убирается шум HNSW между прогонами.",
        f"- Bootstrap: {boot['n_resamples']} ресэмплингов пар запросов, seed={boot['seed']}, "
        "percentile-интервал; p-value = 2·min(P(Δ*≤0), P(Δ*≥0)).",
        "- Основные тесты: paired t-test (scipy.stats.ttest_rel) и bootstrap CI/p.",
        "- Wilcoxon (справочно, НЕ основной тест): "
        f"p = {wil['p_value']:.4f}, нулевых Δ = {wil['n_zero']}/{result['n_queries']}, "
        f"ненулевых = {wil['n_nonzero']}. При таком числе ничьих `zero_method='wilcox'` "
        "отбрасывает их и занижает p — как вывод не используется.",
        f"- Cohen's dz = mean(Δ) / sd(Δ), sd по выборке (ddof=1): {result['sd_delta']:.4f}.",
        "",
        f"Δ > 0 означает, что `{a}` лучше `{b}`; Δ < 0 — наоборот.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    # Консоль Windows часто в cp1251, а отчёт печатает Δ/±/—; файлы всегда UTF-8.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(
        description="Парное сравнение режимов retrieval по per-query метрикам (t-test + bootstrap)"
    )
    ap.add_argument(
        "--report",
        type=Path,
        default=None,
        help="один report_pipeline.json (per_query[<mode>]); несовместим с --runs-glob",
    )
    ap.add_argument(
        "--runs-glob",
        default=None,
        help=f"glob run_*.json для усреднения per-query (default: {DEFAULT_RUNS_GLOB})",
    )
    ap.add_argument("--a", required=True, help="режим A (например, vector)")
    ap.add_argument("--b", required=True, help="режим B (например, hybrid)")
    ap.add_argument("--metric", default="ndcg@10", choices=METRICS)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-md", type=Path, default=None)
    ap.add_argument("--out-json", type=Path, default=None)
    args = ap.parse_args()

    try:
        source = build_source(args, args.metric)
        result = analyse(source, args.a, args.b, args.metric, n_boot=args.n_boot, seed=args.seed)
    except PairError as exc:
        print(f"Пропуск: {exc}", file=sys.stderr)
        return 2
    out_md = args.out_md or (DEFAULT_OUT_DIR / "report_paired.md")
    out_json = args.out_json or (DEFAULT_OUT_DIR / "report_paired.json")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown = build_markdown(result)
    out_md.write_text(markdown, encoding="utf-8")

    print(markdown)
    print(f"JSON: {out_json}")
    print(f"Markdown: {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
