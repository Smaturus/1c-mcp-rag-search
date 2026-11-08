"""Запустить N независимых eval-прогонов и собрать run_*.json.

Каждый прогон — отдельный процесс (HNSW пересобирается при старте Chroma, поэтому
нужны отдельные процессы для честного mean±std). Использует scripts/eval_pipeline.py.

Пример:
    python scripts/run_eval_runs.py --golden data/eval/gold_set_v2.jsonl \
        --chroma-dir data/chroma_xml_eval --bm25-dir data/bm25_all \
        --collection code_proc --n-runs 4 --prefix old_v2 --out-dir data/eval/_runs_token
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL_PY = ROOT / "scripts" / "eval_pipeline.py"
PYTHON = sys.executable


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", type=Path, required=True)
    ap.add_argument("--chroma-dir", type=Path, required=True)
    ap.add_argument("--bm25-dir", type=Path, required=True)
    ap.add_argument("--collection", default="code_proc")
    ap.add_argument("--modes", default="hybrid,vector")
    ap.add_argument("--n-runs", type=int, default=4)
    ap.add_argument("--prefix", default="run")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "data" / "eval" / "_runs_token")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--scope", default=None, help="v3: code")
    ap.add_argument("--answerable-only", action="store_true")
    ap.add_argument(
        "--embedding-model",
        default=None,
        help="если задан — MCP_1C_EMBEDDING_MODEL в окружении дочернего процесса",
    )
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tmp = args.out_dir / f"_tmp_{args.prefix}"
    tmp.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["MCP_1C_CHROMA_DIR"] = str(args.chroma_dir)
    env["MCP_1C_BM25_DIR"] = str(args.bm25_dir)
    env["MCP_1C_CODE_COLLECTION"] = args.collection
    if args.embedding_model:
        env["MCP_1C_EMBEDDING_MODEL"] = args.embedding_model

    for i in range(1, args.n_runs + 1):
        out_name = f"{args.prefix}_run{i}.json"
        dst = args.out_dir / out_name
        if dst.exists():
            print(f"[skip] {dst} существует", flush=True)
            continue
        # Чистим tmp от предыдущего прогона
        for f in tmp.glob("report_pipeline.*"):
            f.unlink()
        cmd = [
            str(PYTHON), str(EVAL_PY),
            "--golden", str(args.golden),
            "--chroma-dir", str(args.chroma_dir),
            "--collection", args.collection,
            "--modes", args.modes,
            "--k", str(args.k),
            "--out-dir", str(tmp),
        ]
        if args.scope:
            cmd += ["--scope", args.scope]
        if args.answerable_only:
            cmd += ["--answerable-only"]
        t0 = time.time()
        print(f"[{out_name}] run {i}/{args.n_runs} ...", flush=True)
        proc = subprocess.run(cmd, env=env, cwd=str(ROOT), capture_output=True, text=True)
        if proc.returncode != 0:
            print(f"[{out_name}] FAILED rc={proc.returncode}", file=sys.stderr)
            print(proc.stdout[-3000:], file=sys.stderr)
            print(proc.stderr[-3000:], file=sys.stderr)
            return 1
        src = tmp / "report_pipeline.json"
        if not src.exists():
            print(f"[{out_name}] нет report_pipeline.json", file=sys.stderr)
            return 1
        shutil.copy2(src, dst)
        print(f"[{out_name}] done {time.time() - t0:.0f}s", flush=True)

    # Итоговая сводка mean по прогонам (hit/mrr/ndcg/recall)
    runs = []
    for i in range(1, args.n_runs + 1):
        p = args.out_dir / f"{args.prefix}_run{i}.json"
        if p.exists():
            runs.append(json.loads(p.read_text(encoding="utf-8")))
    if runs:
        for mode in ("hybrid", "vector"):
            vals = {m: [] for m in ("hit@1", "mrr@10", "ndcg@10", "recall@10")}
            for r in runs:
                res = r.get("results", {}).get(mode, {})
                for m in vals:
                    if m in res:
                        vals[m].append(res[m])
            if vals["ndcg@10"]:
                import statistics
                print(f"[{args.prefix}] {mode} ndcg@10 = "
                      f"{statistics.fmean(vals['ndcg@10']):.4f} ± {statistics.pstdev(vals['ndcg@10']):.4f} "
                      f"(n={len(vals['ndcg@10'])})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
