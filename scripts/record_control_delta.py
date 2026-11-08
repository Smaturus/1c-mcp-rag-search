"""Записать δ = max(floor, std) vector nDCG@10 контроля в state.json.

Прогоны не запускает — только читает готовые run_*.json по глобу ``--runs``.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description="Record control δ into state.json")
    ap.add_argument(
        "--runs",
        required=True,
        help="glob готовых прогонов, напр. data/eval/_runs_large/old_v2_run*.json",
    )
    ap.add_argument("--mode", default="vector")
    ap.add_argument("--metric", default="ndcg@10")
    ap.add_argument("--floor", type=float, required=True)
    ap.add_argument("--key", required=True, help="ключ в state.json, напр. delta_v2")
    ap.add_argument("--state", type=Path, required=True)
    args = ap.parse_args()

    paths = sorted(Path().glob(args.runs))
    if not paths:
        print(f"нет файлов по глобу: {args.runs}", file=sys.stderr)
        return 1

    values: list[float] = []
    for p in paths:
        data = json.loads(p.read_text(encoding="utf-8"))
        res = (data.get("results") or {}).get(args.mode) or {}
        if args.metric not in res:
            print(f"нет {args.mode}.{args.metric} в {p}", file=sys.stderr)
            return 1
        values.append(float(res[args.metric]))

    std = statistics.pstdev(values) if len(values) >= 1 else 0.0
    delta = max(float(args.floor), float(std))

    state: dict = {}
    if args.state.exists():
        state = json.loads(args.state.read_text(encoding="utf-8"))
        if not isinstance(state, dict):
            print(f"state не объект JSON: {args.state}", file=sys.stderr)
            return 1

    state[args.key] = delta
    state[f"{args.key}_meta"] = {
        "mode": args.mode,
        "metric": args.metric,
        "floor": args.floor,
        "std": std,
        "n_runs": len(values),
        "values": values,
        "runs": paths,
        "delta": delta,
    }
    args.state.parent.mkdir(parents=True, exist_ok=True)
    args.state.write_text(
        json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"{args.key}={delta:.6f} (floor={args.floor}, std={std:.6f}, n={len(values)}) "
        f"-> {args.state}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
