"""Router accuracy on the independent v3 hold-out.

Compares ``route_query`` decisions with the ``expected_route`` labels of
``data/eval/gold_set_v3_holdout.jsonl`` and prints overall/per-category accuracy
plus a confusion matrix and the list of mismatches.

Usage:
    python scripts/eval_router_v3.py
    python scripts/eval_router_v3.py --answerable-only
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mcp_1c_rag.router import route_query  # noqa: E402

DEFAULT_GOLDEN = ROOT / "data" / "eval" / "gold_set_v3_holdout.jsonl"
ROUTES = ("lexical", "vector", "hybrid")


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Router accuracy on the v3 hold-out")
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    parser.add_argument("--answerable-only", action="store_true")
    parser.add_argument("--show-errors", type=int, default=20)
    args = parser.parse_args()

    rows = load_rows(args.golden)
    if args.answerable_only:
        rows = [row for row in rows if row.get("answerable", True)]

    total = correct = 0
    by_category: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    by_expected: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    confusion: Counter[tuple[str, str]] = Counter()
    errors: list[tuple[int, str, str, str, float]] = []

    for row in rows:
        expected = row["expected_route"]
        decision = route_query(row["query"])
        got = str(decision.route)
        total += 1
        by_category[row["category"]][1] += 1
        by_expected[expected][1] += 1
        confusion[(expected, got)] += 1
        if got == expected:
            correct += 1
            by_category[row["category"]][0] += 1
            by_expected[expected][0] += 1
        else:
            errors.append((row["id"], expected, got, decision.reason, decision.confidence))

    print(f"router accuracy: {correct}/{total} = {correct / total:.3f}" if total else "no rows")
    print("\nby expected route:")
    for route in ROUTES:
        ok, n = by_expected[route]
        if n:
            print(f"  {route:8s} {ok}/{n} = {ok / n:.3f}")
    print("\nby category:")
    for category, (ok, n) in sorted(by_category.items()):
        print(f"  {ok:2d}/{n:2d}  {category}")
    print("\nconfusion (expected -> got):")
    for (expected, got), count in sorted(confusion.items()):
        if expected != got:
            print(f"  {expected} -> {got}: {count}")
    if errors:
        print(f"\nmismatches (first {args.show_errors}):")
        for qid, expected, got, reason, confidence in errors[: args.show_errors]:
            print(f"  #{qid:3d} expected={expected:8s} got={got:8s} conf={confidence:.2f} reason={reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
