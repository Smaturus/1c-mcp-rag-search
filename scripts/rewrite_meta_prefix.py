"""Переписать префикс id в JSONL метаданных (тексты без изменений).

Исходный файл не перезаписывается. Пример:
    python scripts/rewrite_meta_prefix.py \\
        --in data/metadata_xml/chunks.jsonl \\
        --out data/metadata_xml/chunks_metaobj.jsonl \\
        --from-prefix meta_upp: --to-prefix metaobj:
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description="Rewrite metadata chunk id prefixes")
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--from-prefix", required=True)
    ap.add_argument("--to-prefix", required=True)
    args = ap.parse_args()

    if not args.inp.exists():
        print(f"нет входного файла: {args.inp}", file=sys.stderr)
        return 1
    if args.inp.resolve() == args.out.resolve():
        print(" --out не должен совпадать с --in", file=sys.stderr)
        return 1

    n = 0
    n_changed = 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.inp.open(encoding="utf-8") as fin, args.out.open(
        "w", encoding="utf-8", newline="\n"
    ) as fout:
        for line in fin:
            if not line.strip():
                continue
            row = json.loads(line)
            cid = row.get("id", "")
            if isinstance(cid, str) and cid.startswith(args.from_prefix):
                row["id"] = args.to_prefix + cid[len(args.from_prefix) :]
                n_changed += 1
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1

    print(
        f"rewrite_meta_prefix: lines={n} changed={n_changed} "
        f"from={args.from_prefix!r} to={args.to_prefix!r} out={args.out}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
