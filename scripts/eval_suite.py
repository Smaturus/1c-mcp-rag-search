"""Локальный/nightly eval-контур против expected nDCG@10 (приватные gold).

Не для публичного CI. Сверяет манифест (counts/model/bm25), затем прогоняет
hybrid на v2/v3/v4 и сравнивает с data/manifests/eval_expected.json.

Пример::

    python scripts/eval_suite.py --manifest data/manifests/chroma_xml.json \\
      --out data/local_notes/eval_nightly --sets v2 --record
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mcp_1c_rag.config import (  # noqa: E402
    BM25_DIR,
    CHROMA_DIR,
    EMBEDDING_MODEL,
    MANIFEST_PATH,
)
from mcp_1c_rag.eval_metrics import aggregate, evaluate_query  # noqa: E402
from mcp_1c_rag.manifest import (  # noqa: E402
    load_manifest,
    validate_manifest,
)
from mcp_1c_rag.models import Candidate  # noqa: E402

GOLD_FILES = {
    "v2": ROOT / "data" / "eval" / "gold_set_v2.jsonl",
    "v3": ROOT / "data" / "eval" / "gold_set_v3_holdout.jsonl",
    "v4": ROOT / "data" / "eval" / "gold_set_v4_holdout.jsonl",
}
DEFAULT_DELTA = {"v2": 0.028, "v3": 0.020, "v4": 0.020}
DEFAULT_SCOPE = {"v2": None, "v3": "code", "v4": "code"}
Searcher = Callable[..., list[Candidate]]


def load_gold(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def filter_scope(
    rows: Sequence[dict[str, Any]], scope: str | None
) -> list[dict[str, Any]]:
    if scope is None:
        return list(rows)
    if not any("scope" in row for row in rows):
        return list(rows)
    return [row for row in rows if row.get("scope") == scope]


def default_searcher(
    query: str, *, top_k: int, mode: str, collection: str
) -> list[Candidate]:
    from mcp_1c_rag.search_modes import search

    return search(query=query, top_k=top_k, mode=mode, collection=collection)


def run_set(
    rows: Sequence[dict[str, Any]],
    *,
    searcher: Searcher,
    k: int = 10,
    mode: str = "hybrid",
    collection: str = "code_proc",
) -> dict[str, Any]:
    metrics = []
    for row in rows:
        started = time.perf_counter()
        candidates = searcher(
            row["query"], top_k=k, mode=mode, collection=collection
        )
        latency = (time.perf_counter() - started) * 1000.0
        sources = [c.source for c in candidates]
        metrics.append(
            evaluate_query(
                sources, row["relevant_sources"], k=k, latency_ms=latency
            )
        )
    agg = aggregate(metrics)
    return {"n_queries": len(rows), "metrics": agg.values}


def load_expected(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"schema_version": 1, "sets": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def compare_ndcg(
    actual: float, expected: float, delta: float
) -> tuple[bool, float]:
    diff = float(actual) - float(expected)
    return diff >= -float(delta), diff


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Local eval suite vs expected nDCG")
    ap.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    ap.add_argument(
        "--expected",
        type=Path,
        default=ROOT / "data" / "manifests" / "eval_expected.json",
    )
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "local_notes" / "eval_nightly")
    ap.add_argument("--sets", default="v2,v3,v4")
    ap.add_argument("--n-runs", type=int, default=1)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--mode", default="hybrid")
    ap.add_argument("--collection", default="code_proc")
    ap.add_argument("--chroma-dir", type=Path, default=CHROMA_DIR)
    ap.add_argument("--bm25-dir", type=Path, default=BM25_DIR)
    ap.add_argument("--record", action="store_true", help="write expected.json")
    ap.add_argument(
        "--check-fingerprints",
        action="store_true",
        help="full fingerprint check (slow)",
    )
    args = ap.parse_args(argv)

    if not args.manifest.is_file():
        print(f"ERROR: manifest missing: {args.manifest}", file=sys.stderr)
        return 2
    manifest = load_manifest(args.manifest)
    issues = validate_manifest(
        manifest,
        chroma_dir=args.chroma_dir,
        bm25_dir=args.bm25_dir,
        embedding_model=EMBEDDING_MODEL,
        check_fingerprints=args.check_fingerprints,
    )
    errors = [i for i in issues if i.level == "error"]
    if errors:
        for issue in errors:
            print(f"ERROR [{issue.code}] {issue.message}: {issue.actual}", file=sys.stderr)
        return 2

    set_names = [s.strip() for s in args.sets.split(",") if s.strip()]
    expected = load_expected(args.expected)
    expected_sets: dict[str, Any] = dict(expected.get("sets") or {})
    report: dict[str, Any] = {
        "manifest": str(args.manifest),
        "mode": args.mode,
        "collection": args.collection,
        "n_runs": args.n_runs,
        "sets": {},
        "status": "ok",
    }
    partial = False
    failed = False

    for name in set_names:
        gold_path = GOLD_FILES.get(name)
        if gold_path is None or not gold_path.is_file():
            print(f"WARNING: gold missing for {name}, skip", flush=True)
            report["sets"][name] = {"status": "skipped", "reason": "gold_missing"}
            partial = True
            continue
        rows = filter_scope(load_gold(gold_path), DEFAULT_SCOPE.get(name))
        run_metrics: list[float] = []
        last_detail: dict[str, Any] = {}
        for _run in range(max(1, args.n_runs)):
            last_detail = run_set(
                rows,
                searcher=default_searcher,
                k=args.k,
                mode=args.mode,
                collection=args.collection,
            )
            run_metrics.append(float(last_detail["metrics"]["ndcg@10"]))
        mean_ndcg = sum(run_metrics) / len(run_metrics)
        exp = expected_sets.get(name) or {}
        delta = float(exp.get("delta", DEFAULT_DELTA.get(name, 0.03)))
        entry: dict[str, Any] = {
            "status": "ok",
            "n_queries": last_detail.get("n_queries", 0),
            "ndcg@10": mean_ndcg,
            "metrics": last_detail.get("metrics", {}),
            "delta": delta,
        }
        if args.record or name not in expected_sets:
            expected_sets[name] = {
                "ndcg@10": mean_ndcg,
                "delta": delta,
                "scope": DEFAULT_SCOPE.get(name),
            }
            entry["recorded"] = True
        else:
            ok, diff = compare_ndcg(mean_ndcg, float(exp["ndcg@10"]), delta)
            entry["expected_ndcg@10"] = float(exp["ndcg@10"])
            entry["diff"] = diff
            if not ok:
                entry["status"] = "regression"
                failed = True
        report["sets"][name] = entry
        print(
            f"[{name}] nDCG@10={mean_ndcg:.4f} status={entry['status']}",
            flush=True,
        )

    if args.record or any(
        report["sets"].get(n, {}).get("recorded") for n in set_names
    ):
        args.expected.parent.mkdir(parents=True, exist_ok=True)
        payload = {"schema_version": 1, "sets": expected_sets}
        args.expected.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"wrote expected {args.expected}", flush=True)

    if failed:
        report["status"] = "fail"
    elif partial:
        report["status"] = "partial"
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# eval_suite",
        "",
        f"status={report['status']}",
        f"mode={args.mode} collection={args.collection}",
        "",
        "| set | n | nDCG@10 | status |",
        "|---|---|---|---|",
    ]
    for name, entry in report["sets"].items():
        lines.append(
            f"| {name} | {entry.get('n_queries', 0)} | "
            f"{entry.get('ndcg@10', 0):.4f} | {entry.get('status')} |"
        )
    (args.out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.out / 'report.md'}", flush=True)
    if failed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
