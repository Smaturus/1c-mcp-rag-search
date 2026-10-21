"""
Оценка качества семантического поиска + «флуктуации» скоров/метрик.

Идеи из ДФ (стационарный фон, малые возмущения, МНА):
  - null-распределение cosine на случайных парах (фон)
  - Hit@k / Recall@k / MRR по golden-набору
  - CI среднего через нормальную аппроксимацию
  - флуктуации при paraphrases: Var(Hit@k), средний Jaccard top-k

Пример:
  python scripts/eval_retrieval_fluctuations.py ^
    --golden data/eval/golden_queries.example.json ^
    --null-pairs 500 ^
    --out data/eval/report.json

Пока идёт полная индексация Chroma может быть залочена — подождите или
укажите --chroma-dir на копию data/chroma.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mcp_1c_rag.embeddings import embed_passages, embed_query  # noqa: E402
from mcp_1c_rag.store import (  # noqa: E402
    get_code_collection,
    get_metadata_collection,
    query_collection,
)


@dataclass
class QueryResult:
    id: str
    query: str
    hit_at_k: float
    recall_at_k: float
    mrr: float
    top_sources: list[str]
    top_sims: list[float]
    null_tail_prob_top1: float | None


def _cosine_from_distance(distance: float | None) -> float:
    """Chroma cosine space: distance ≈ 1 - cos for normalized vectors."""
    if distance is None:
        return float("nan")
    return 1.0 - float(distance)


def _is_relevant(source: str, needles: list[str]) -> bool:
    src = source.replace("\\", "/")
    return any(n and n in src for n in needles)


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def _std_sample(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def _mna_ci(xs: list[float], z: float) -> tuple[float, float, float]:
    """Normal-approximation CI for the mean: (mean, lo, hi)."""
    m = _mean(xs)
    se = _std_sample(xs) / math.sqrt(len(xs)) if xs else float("nan")
    return m, m - z * se, m + z * se


def _jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _get_collection(name: str):
    if name == "code":
        return get_code_collection()
    if name == "metadata":
        return get_metadata_collection()
    raise SystemExit(f"Unknown collection: {name} (use code|metadata)")


def _search(collection, query: str, k: int) -> list[dict[str, Any]]:
    return query_collection(collection, embed_query(query), k)


def _eval_one(
    collection,
    *,
    qid: str,
    query: str,
    relevant_sources: list[str],
    k: int,
    null_scores: list[float] | None,
) -> QueryResult:
    hits = _search(collection, query, k)
    sources = [(h.get("metadata") or {}).get("source", "") for h in hits]
    sims = [_cosine_from_distance(h.get("distance")) for h in hits]

    rel_flags = [_is_relevant(s, relevant_sources) for s in sources]
    hit = 1.0 if any(rel_flags) else 0.0
    # binary recall@k = hit; fractional if several needles — share of distinct needles hit
    needles_hit = {
        n for n in relevant_sources if any(n in s.replace("\\", "/") for s in sources)
    }
    recall = (
        len(needles_hit) / len(relevant_sources) if relevant_sources else float("nan")
    )

    mrr = 0.0
    for rank, flag in enumerate(rel_flags, start=1):
        if flag:
            mrr = 1.0 / rank
            break

    null_tail = None
    if null_scores and sims:
        s0 = sims[0]
        null_tail = sum(1 for s in null_scores if s >= s0) / len(null_scores)

    return QueryResult(
        id=qid,
        query=query,
        hit_at_k=hit,
        recall_at_k=recall,
        mrr=mrr,
        top_sources=sources,
        top_sims=sims,
        null_tail_prob_top1=null_tail,
    )


def build_null_scores(
    collection,
    *,
    n_pairs: int,
    rng: random.Random,
) -> list[float]:
    """
    Stationary 'background' fluctuations: cosine(query_noise, random_passage).

    Sampling stored embeddings when available; otherwise re-embed random docs.
    """
    count = collection.count()
    if count == 0:
        raise SystemExit("Collection is empty — finish indexing first.")

    # Pull a pool of documents (capped for memory)
    pool = min(count, max(n_pairs * 2, 200), 2000)
    raw = collection.get(limit=pool, include=["documents", "embeddings"])
    docs = raw.get("documents") or []
    embs = raw.get("embeddings")
    if not docs:
        raise SystemExit("Could not read documents from collection.")

    # Noise queries: unrelated short strings (not from 1C domain)
    noise_bank = [
        "weather forecast tomorrow",
        "recipe for apple pie",
        "quantum chromodynamics lattice",
        "фото котиков в космосе",
        "расписание электричек на Марс",
        "bitcoin whale wallet cluster",
        "交响曲 allegro ma non troppo",
    ]

    scores: list[float] = []
    for _ in range(n_pairs):
        q = rng.choice(noise_bank) + f" #{rng.randint(1, 10_000)}"
        qv = embed_query(q)
        idx = rng.randrange(len(docs))
        if embs is not None and len(embs) == len(docs) and embs[idx] is not None:
            pv = embs[idx]
        else:
            pv = embed_passages([docs[idx] or " "])[0]
        # both normalized → cos = dot
        scores.append(sum(a * b for a, b in zip(qv, pv, strict=False)))
    return scores


def eval_paraphrase_fluctuations(
    collection,
    *,
    base_query: str,
    paraphrases: list[str],
    relevant_sources: list[str],
    k: int,
) -> dict[str, Any]:
    variants = [base_query, *paraphrases]
    results = [
        _eval_one(
            collection,
            qid=f"p{i}",
            query=q,
            relevant_sources=relevant_sources,
            k=k,
            null_scores=None,
        )
        for i, q in enumerate(variants)
    ]
    hits = [r.hit_at_k for r in results]
    jaccords = []
    base_src = results[0].top_sources
    for r in results[1:]:
        jaccords.append(_jaccard(base_src, r.top_sources))
    return {
        "n_variants": len(variants),
        "hit_mean": _mean(hits),
        "hit_std": _std_sample(hits),
        "hit_var": _std_sample(hits) ** 2,
        "jaccard_to_base_mean": _mean(jaccords) if jaccords else float("nan"),
        "jaccard_to_base_std": _std_sample(jaccords) if jaccords else float("nan"),
        "variants": [
            {"query": r.query, "hit_at_k": r.hit_at_k, "top_sources": r.top_sources[:k]}
            for r in results
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Retrieval quality + fluctuation-style diagnostics for MCP 1C search"
    )
    parser.add_argument(
        "--golden",
        type=Path,
        default=ROOT / "data" / "eval" / "golden_queries.example.json",
    )
    parser.add_argument("--null-pairs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data" / "eval" / "report_retrieval_fluctuations.json",
    )
    parser.add_argument(
        "--skip-null",
        action="store_true",
        help="Skip null-score background (faster)",
    )
    args = parser.parse_args()

    golden = json.loads(args.golden.read_text(encoding="utf-8"))
    collection_name = golden.get("collection", "code")
    k = int(golden.get("k", 5))
    z = float(golden.get("z_alpha", 1.96))
    queries = golden.get("queries") or []
    if not queries:
        raise SystemExit("No queries in golden file.")

    print(f"collection={collection_name} k={k} queries={len(queries)}", flush=True)
    try:
        collection = _get_collection(collection_name)
        print(f"collection.count={collection.count()}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(
            f"ERROR opening Chroma (index lock?): {exc}\n"
            "Wait for indexer to finish or copy data/chroma and point CHROMA via env.",
            file=sys.stderr,
        )
        return 2

    rng = random.Random(args.seed)
    null_scores: list[float] = []
    null_stats: dict[str, Any] = {}
    if not args.skip_null and args.null_pairs > 0:
        print(f"Building null cosine scores: n_pairs={args.null_pairs} ...", flush=True)
        t0 = time.time()
        null_scores = build_null_scores(
            collection, n_pairs=args.null_pairs, rng=rng
        )
        null_stats = {
            "n_pairs": len(null_scores),
            "mean": _mean(null_scores),
            "std": _std_sample(null_scores),
            "p95": sorted(null_scores)[int(0.95 * (len(null_scores) - 1))],
            "p99": sorted(null_scores)[int(0.99 * (len(null_scores) - 1))],
            "elapsed_sec": round(time.time() - t0, 1),
        }
        print(
            f"null: mean={null_stats['mean']:.4f} std={null_stats['std']:.4f} "
            f"p95={null_stats['p95']:.4f} p99={null_stats['p99']:.4f}",
            flush=True,
        )

    per_query: list[dict[str, Any]] = []
    hits: list[float] = []
    recalls: list[float] = []
    mrrs: list[float] = []
    paraphrase_reports: list[dict[str, Any]] = []

    for item in queries:
        qid = str(item.get("id") or item["query"][:40])
        query = item["query"]
        relevant = list(item.get("relevant_sources") or [])
        print(f"  eval {qid}: {query!r}", flush=True)
        r = _eval_one(
            collection,
            qid=qid,
            query=query,
            relevant_sources=relevant,
            k=k,
            null_scores=null_scores or None,
        )
        hits.append(r.hit_at_k)
        recalls.append(r.recall_at_k)
        mrrs.append(r.mrr)
        row = asdict(r)
        per_query.append(row)

        paras = list(item.get("paraphrases") or [])
        if paras:
            pf = eval_paraphrase_fluctuations(
                collection,
                base_query=query,
                paraphrases=paras,
                relevant_sources=relevant,
                k=k,
            )
            pf["id"] = qid
            paraphrase_reports.append(pf)

    hit_m, hit_lo, hit_hi = _mna_ci(hits, z)
    rec_m, rec_lo, rec_hi = _mna_ci(recalls, z)
    mrr_m, mrr_lo, mrr_hi = _mna_ci(mrrs, z)

    report = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "golden": str(args.golden),
        "collection": collection_name,
        "k": k,
        "z_alpha": z,
        "n_queries": len(queries),
        "metrics_mna": {
            "hit_at_k": {
                "mean": hit_m,
                "ci": [hit_lo, hit_hi],
                "std": _std_sample(hits),
                "formula": "mean ± z * s / sqrt(n)",
            },
            "recall_at_k": {
                "mean": rec_m,
                "ci": [rec_lo, rec_hi],
                "std": _std_sample(recalls),
            },
            "mrr": {
                "mean": mrr_m,
                "ci": [mrr_lo, mrr_hi],
                "std": _std_sample(mrrs),
            },
        },
        "null_background": null_stats,
        "per_query": per_query,
        "paraphrase_fluctuations": paraphrase_reports,
        "notes": [
            "null_background ≈ stationary score fluctuations under irrelevant queries",
            "paraphrase hit_var / jaccard ≈ sensitivity to small query perturbations",
            "MNA CI is normal approximation for the mean over queries",
        ],
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    md_path = args.out.with_suffix(".md")
    md_lines = [
        f"# Retrieval fluctuations report ({report['created_at']})",
        "",
        f"- collection: `{collection_name}`, k={k}, n={len(queries)}",
        f"- Hit@{k}: **{hit_m:.3f}** CI95≈[{hit_lo:.3f}, {hit_hi:.3f}]",
        f"- Recall@{k}: **{rec_m:.3f}** CI95≈[{rec_lo:.3f}, {rec_hi:.3f}]",
        f"- MRR: **{mrr_m:.3f}** CI95≈[{mrr_lo:.3f}, {mrr_hi:.3f}]",
    ]
    if null_stats:
        md_lines.append(
            f"- Null cosine: mean={null_stats['mean']:.4f}, "
            f"p95={null_stats['p95']:.4f}, p99={null_stats['p99']:.4f}"
        )
    if paraphrase_reports:
        md_lines.append("- Paraphrase fluctuations:")
        for pf in paraphrase_reports:
            md_lines.append(
                f"  - {pf['id']}: hit_var={pf['hit_var']:.3f}, "
                f"jaccard_mean={pf['jaccard_to_base_mean']:.3f}"
            )
    md_lines.append(f"\nFull JSON: `{args.out}`")
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print("\n=== summary ===", flush=True)
    print(f"Hit@{k}={hit_m:.3f} CI=[{hit_lo:.3f},{hit_hi:.3f}]", flush=True)
    print(f"Recall@{k}={rec_m:.3f} CI=[{rec_lo:.3f},{rec_hi:.3f}]", flush=True)
    print(f"MRR={mrr_m:.3f} CI=[{mrr_lo:.3f},{mrr_hi:.3f}]", flush=True)
    print(f"Wrote {args.out}", flush=True)
    print(f"Wrote {md_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
