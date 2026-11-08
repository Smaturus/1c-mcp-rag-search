"""Выбрать и скопировать pilot-корпус (300–500 файлов) для кандидата.

Состав:
- ВСЕ code-relevant пути из повторяемых ``--gold`` (объединение §5);
- остальные — стратифицированная выборка по top-level типу × размеру
  (<1 КБ, 1–10, 10–100, >100 КБ) с фиксированным seed.

Копирует файлы с сохранением относительных путей в ``--out``.
Запуск без трёх ``--gold`` — стоп (молчаливый только-v2 запрещён).
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = ROOT / "1C-Corpus" / "1C_Corpus"
DEFAULT_OUT = ROOT / "data" / "large_pilot_corpus"

SIZE_BUCKETS = ("<1kb", "1-10kb", "10-100kb", ">100kb")


def size_bucket(size: int) -> str:
    if size < 1024:
        return "<1kb"
    if size < 10 * 1024:
        return "1-10kb"
    if size < 100 * 1024:
        return "10-100kb"
    return ">100kb"


def load_code_relevant(gold_paths: list[Path]) -> set[str]:
    """Объединение code-relevant_sources из gold-файлов (после замены слэшей)."""
    relevant: set[str] = set()
    for path in gold_paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            scope = row.get("scope")
            if scope is not None and scope != "code":
                continue
            for src in row.get("relevant_sources", []):
                norm = src.replace("\\", "/")
                if norm.lower().endswith(".bsl"):
                    relevant.add(norm)
    return relevant


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DEFAULT_CORPUS)
    ap.add_argument(
        "--gold",
        action="append",
        type=Path,
        required=True,
        help="gold JSONL; нужны ровно три файла (v2/v3/v4)",
    )
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--target", type=int, default=400)
    args = ap.parse_args()

    if len(args.gold) != 3:
        print(
            f"нужны ровно три --gold, получено {len(args.gold)}",
            file=sys.stderr,
        )
        return 1
    for g in args.gold:
        if not g.exists():
            print(f"нет gold-файла: {g}", file=sys.stderr)
            return 1

    relevant = load_code_relevant(args.gold)
    print(f"gold unique code relevant files: {len(relevant)}", flush=True)

    files = sorted(args.root.rglob("*.bsl"))
    by_rel: dict[str, Path] = {}
    for p in files:
        rel = str(p.relative_to(args.root)).replace("\\", "/")
        by_rel[rel] = p

    selected: list[str] = []
    for rel in sorted(relevant):
        if rel in by_rel:
            selected.append(rel)
    missing_relevant = sorted(relevant - set(by_rel))
    print(
        f"relevant selected: {len(selected)}, missing in corpus: {len(missing_relevant)}",
        flush=True,
    )
    if missing_relevant:
        print("missing relevant:", missing_relevant[:10], flush=True)

    rest = [rel for rel in by_rel if rel not in set(selected)]
    strata: dict[tuple[str, str], list[str]] = defaultdict(list)
    for rel in rest:
        size = by_rel[rel].stat().st_size
        top = rel.split("/")[0] if "/" in rel else "(root)"
        strata[(top, size_bucket(size))].append(rel)

    rng = random.Random(args.seed)
    remaining = args.target - len(selected)
    total_rest = len(rest)
    per_stratum: dict[tuple[str, str], int] = {}
    if remaining > 0 and total_rest > 0:
        for key, items in strata.items():
            per_stratum[key] = max(0, int(round(remaining * len(items) / total_rest)))
        deficit = remaining - sum(per_stratum.values())
        keys = sorted(strata, key=lambda k: -len(strata[k]))
        i = 0
        while deficit > 0 and keys:
            per_stratum[keys[i % len(keys)]] += 1
            deficit -= 1
            i += 1

        for key, items in strata.items():
            n = min(per_stratum.get(key, 0), len(items))
            selected.extend(rng.sample(items, n))

    selected = sorted(set(selected))
    print(f"total selected: {len(selected)}", flush=True)

    if args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True, exist_ok=True)
    for rel in selected:
        src = by_rel[rel]
        dst = args.out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    sel_buckets = Counter(size_bucket(by_rel[r].stat().st_size) for r in selected)
    sel_tops = Counter(r.split("/")[0] if "/" in r else "(root)" for r in selected)
    report = {
        "root": str(args.root),
        "out": str(args.out),
        "gold": [str(g) for g in args.gold],
        "seed": args.seed,
        "target": args.target,
        "n_selected": len(selected),
        "n_relevant": len([r for r in selected if r in relevant]),
        "missing_relevant": missing_relevant,
        "size_buckets": dict(sel_buckets),
        "top_level": dict(sel_tops.most_common(40)),
        "files": selected,
    }
    out_json = args.out.parent / "large_pilot_corpus_selection.json"
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps({k: v for k, v in report.items() if k != "files"}, ensure_ascii=False, indent=2),
        flush=True,
    )
    print(f"selection report: {out_json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
