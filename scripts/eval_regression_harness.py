"""Same-process regression harness: baseline vs candidate на общих vector-кандидатах.

Сравнивает два BM25-индекса (и опционально max_per_source) в одном процессе:
embedding-модель и Chroma-запрос считаются один раз на запрос; RRF строится
отдельно для baseline и candidate. Paired bootstrap CI + non-inferiority-гейт.

Не вызывает search_modes.search / bm25_search (глобальный BM25_DIR).

Пример:
    python scripts/eval_regression_harness.py \\
      --golden data/eval/gold_set_v2.jsonl \\
      --chroma-dir data/chroma_xml --collection code_proc \\
      --baseline-bm25-dir data/bm25_all --candidate-bm25-dir data/bm25_all \\
      --metrics ndcg@10,recall@10,unique_sources@10 \\
      --out-json data/local_notes/algo_glm5/harness_sanity_v2.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import ttest_rel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Лёгкие модули без чтения EMBEDDING_MODEL / CHROMA_DIR при импорте embeddings.
from mcp_1c_rag.bm25 import Bm25Index  # noqa: E402
from mcp_1c_rag.dedup import DEFAULT_START_MULTIPLIER, iterative_dedup  # noqa: E402
from mcp_1c_rag.eval_metrics import (  # noqa: E402
    hit_at_1,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
)
from mcp_1c_rag.models import Candidate  # noqa: E402
from mcp_1c_rag.rrf import rrf_fuse, rrf_order  # noqa: E402

DEFAULT_EMBEDDING = "intfloat/multilingual-e5-small"
DEFAULT_METRICS = "ndcg@10,recall@10,unique_sources@10"
RRF_K = 60
N_BOOT = 10000
BOOT_SEED = 42
CI_LEVEL = 95.0
TIE_EPS = 1e-12

# Префиксы chroma-id (как в search_modes._ID_PREFIX) — без импорта search_modes.
_ID_PREFIX = {
    "code": "bsl:",
    "metadata": "meta:",
    "metadata_xml": "metaobj:",
    "code_proc": "bsl_proc:",
    "config_xml": "xml:",
}

METRIC_GATED = frozenset({"ndcg@10", "recall@10"})


def resolve_delta_floor(golden: Path, explicit: float | None) -> float:
    """δ для non-inferiority: явный аргумент или по имени gold-файла."""
    if explicit is not None:
        return float(explicit)
    name = golden.name.lower()
    if re.search(r"v2", name):
        return 0.028
    if re.search(r"v3", name) or re.search(r"v4", name):
        return 0.020
    raise SystemExit(
        f"неизвестный gold-файл для --delta-floor: {golden.name}; "
        "задайте --delta-floor явно"
    )


def load_goldset(path: Path) -> list[dict[str, Any]]:
    """Прочитать JSONL gold-set."""
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        missing = {"id", "query", "relevant_sources"} - set(row)
        if missing:
            raise SystemExit(f"{path}:{line_no}: нет полей {sorted(missing)}")
        rows.append(row)
    if not rows:
        raise SystemExit(f"{path}: пустой gold-set")
    return rows


def filter_gold_by_scope(
    rows: Sequence[dict[str, Any]],
    scope: str | None,
) -> list[dict[str, Any]]:
    """Фильтр по scope. Для v2 (поля scope нет) — всегда все строки."""
    if scope is None or scope == "all":
        return list(rows)
    if not any("scope" in row for row in rows):
        return list(rows)
    return [row for row in rows if row.get("scope") == scope]


def bootstrap_ci(
    deltas: np.ndarray,
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
) -> tuple[float, float]:
    """Percentile 95% CI среднего Δ (paired bootstrap)."""
    n = len(deltas)
    if n == 0 or n_boot <= 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = deltas[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    low = float(np.percentile(means, (100.0 - CI_LEVEL) / 2.0))
    high = float(np.percentile(means, 100.0 - (100.0 - CI_LEVEL) / 2.0))
    return low, high


def paired_compare(
    baseline: Sequence[float],
    candidate: Sequence[float],
) -> dict[str, Any]:
    """Paired bootstrap CI, t-test, W/T/L по per-query значениям."""
    arr_b = np.asarray(baseline, dtype=float)
    arr_c = np.asarray(candidate, dtype=float)
    if arr_b.shape != arr_c.shape:
        raise ValueError("baseline и candidate массивы разной длины")
    n = int(arr_b.size)
    deltas = arr_c - arr_b
    ci_low, ci_high = bootstrap_ci(deltas)
    if n >= 2 and float(np.std(deltas)) > 0:
        t_stat = ttest_rel(arr_c, arr_b)
        p_value = float(t_stat.pvalue)
        t_statistic = float(t_stat.statistic)
    else:
        p_value = 1.0
        t_statistic = 0.0
    wins = int((deltas > TIE_EPS).sum())
    losses = int((deltas < -TIE_EPS).sum())
    ties = n - wins - losses
    return {
        "baseline": float(arr_b.mean()) if n else float("nan"),
        "candidate": float(arr_c.mean()) if n else float("nan"),
        "delta": float(deltas.mean()) if n else float("nan"),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value": p_value,
        "t_statistic": t_statistic,
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "n": n,
    }


def gate_non_inferiority(ci_low: float, delta_floor: float) -> bool:
    """Non-inferiority: нижняя граница CI строго выше −δ."""
    return float(ci_low) > -float(delta_floor)


def metric_value(
    name: str,
    sources: Sequence[str],
    relevant: Sequence[str],
    *,
    k: int,
    latency_ms: float = 0.0,
) -> float:
    """Одна метрика для одного запроса."""
    key = name.strip().lower()
    if key == "ndcg@10":
        return float(ndcg_at_k(sources, relevant, k=k))
    if key == "recall@10":
        return float(recall_at_k(sources, relevant, k=k))
    if key == "hit@1":
        return float(hit_at_1(sources, relevant))
    if key == "mrr@10":
        return float(mrr_at_k(sources, relevant, k=k))
    if key == "unique_sources@10":
        return float(len(set(sources[:k])))
    if key == "latency_ms":
        return float(latency_ms)
    raise ValueError(f"неизвестная метрика: {name!r}")


def bm25_top_ids(
    index: Bm25Index,
    query: str,
    *,
    collection: str,
    top_k: int,
) -> list[str]:
    """Top-k id по BM25 с prefix-mask коллекции (логика как bm25_search)."""
    scores = index.scores(query)
    prefix = _ID_PREFIX.get(collection, "")
    if prefix:
        mask = np.fromiter(
            (doc_id.startswith(prefix) for doc_id in index.doc_ids),
            dtype=bool,
            count=index.n_docs,
        )
        scores = np.where(mask, scores, -np.inf)
    if not np.isfinite(scores).any():
        return []
    n = max(1, min(int(top_k), int(np.isfinite(scores).sum())))
    top = np.argpartition(-scores, n - 1)[:n]
    top = top[np.argsort(-scores[top])]
    return [
        index.doc_ids[int(i)]
        for i in top
        if np.isfinite(scores[int(i)]) and scores[int(i)] > 0
    ]


def _vector_candidates_from_raw(raw: dict[str, list]) -> list[Candidate]:
    """Chroma query → list[Candidate] со score=cosine."""
    return [
        Candidate(
            id=str(doc_id),
            text=text or "",
            metadata=dict(meta or {}),
            score=1.0 - float(dist),
            source_scores={"cosine": 1.0 - float(dist)},
        )
        for doc_id, text, meta, dist in zip(
            raw.get("ids") or [],
            raw.get("documents") or [],
            raw.get("metadatas") or [],
            raw.get("distances") or [],
            strict=False,
        )
    ]


def _fetch_missing(
    collection: Any,
    doc_ids: Sequence[str],
    by_id: dict[str, Candidate],
) -> None:
    """Добрать documents/metadatas для id вне vector-пула."""
    missing = [doc_id for doc_id in doc_ids if doc_id not in by_id]
    if not missing:
        return
    got = collection.get(ids=list(missing), include=["documents", "metadatas"])
    for doc_id, text, meta in zip(
        got.get("ids") or [],
        got.get("documents") or [],
        got.get("metadatas") or [],
        strict=False,
    ):
        by_id[str(doc_id)] = Candidate(
            id=str(doc_id),
            text=text or "",
            metadata=dict(meta or {}),
        )


def fuse_hybrid(
    vector_cands: Sequence[Candidate],
    bm25_ids: Sequence[str],
    *,
    collection: Any,
    k_rrf: int = RRF_K,
    top_k: int,
) -> list[Candidate]:
    """RRF(vector, bm25) → Candidate со score=rrf; обрезать до top_k (без cap)."""
    vector_ids = [c.id for c in vector_cands]
    fused = rrf_fuse([vector_ids, list(bm25_ids)], k=k_rrf)
    if not fused:
        return []
    by_id: dict[str, Candidate] = {c.id: c for c in vector_cands}
    _fetch_missing(collection, list(fused.keys()), by_id)
    result: list[Candidate] = []
    for doc_id in rrf_order(fused):
        base = by_id.get(doc_id)
        if base is None:
            continue
        result.append(
            Candidate(
                id=base.id,
                text=base.text,
                metadata=dict(base.metadata),
                score=float(fused[doc_id]),
                source_scores={"rrf": float(fused[doc_id])},
            )
        )
    return result[:top_k]


def make_side_fetch(
    *,
    query: str,
    qvec: list[float],
    collection: Any,
    collection_name: str,
    bm25_index: Bm25Index,
    vector_cache: dict[int, list[Candidate]],
) -> Any:
    """fetch(pool): общий qvec + Chroma n_results=pool + BM25 side + RRF."""

    def fetch(req_pool: int) -> list[Candidate]:
        req_pool = max(1, int(req_pool))
        if req_pool not in vector_cache:
            raw = collection.query(
                query_embeddings=[qvec],
                n_results=req_pool,
                include=["documents", "metadatas", "distances"],
            )
            vector_raw = {
                "ids": (raw.get("ids") or [[]])[0],
                "documents": (raw.get("documents") or [[]])[0],
                "metadatas": (raw.get("metadatas") or [[]])[0],
                "distances": (raw.get("distances") or [[]])[0],
            }
            vector_cache[req_pool] = _vector_candidates_from_raw(vector_raw)
        vector_cands = vector_cache[req_pool]
        bm25_ids = bm25_top_ids(
            bm25_index, query, collection=collection_name, top_k=req_pool
        )
        return fuse_hybrid(
            vector_cands, bm25_ids, collection=collection, top_k=req_pool
        )

    return fetch


def side_hits(
    fetch: Any,
    *,
    top_k: int,
    max_per_source: int,
    pool: int,
    pool_multiplier: int,
    total: int,
) -> list[Candidate]:
    """max_per_source<=0: фиксированный pool; иначе iterative_dedup как в search()."""
    if max_per_source <= 0:
        return fetch(pool)[:top_k]
    return iterative_dedup(
        fetch,
        top_k=top_k,
        max_per_source=max_per_source,
        pool_multiplier=pool_multiplier,
        total=total,
    )


def build_metric_report(
    baseline_vals: Sequence[float],
    candidate_vals: Sequence[float],
    *,
    metric: str,
    delta_floor: float,
) -> dict[str, Any]:
    """Сводка + gate для одной метрики (отдельные массивы)."""
    stats = paired_compare(baseline_vals, candidate_vals)
    if metric in METRIC_GATED:
        stats["gate"] = gate_non_inferiority(stats["ci_low"], delta_floor)
    else:
        stats["gate"] = None
    return stats


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """CLI-контракт Task 1."""
    ap = argparse.ArgumentParser(
        description="Same-process regression harness: baseline vs candidate BM25"
    )
    ap.add_argument("--golden", type=Path, required=True)
    ap.add_argument("--chroma-dir", type=Path, required=True)
    ap.add_argument("--collection", required=True, help="например code_proc")
    ap.add_argument(
        "--scope",
        default=None,
        choices=["code", "metadata", "all"],
        help="фильтр gold по scope; для v2 игнорируется",
    )
    ap.add_argument("--embedding-model", default=DEFAULT_EMBEDDING)
    ap.add_argument("--baseline-bm25-dir", type=Path, required=True)
    ap.add_argument("--candidate-bm25-dir", type=Path, required=True)
    ap.add_argument("--baseline-max-per-source", type=int, default=0)
    ap.add_argument("--candidate-max-per-source", type=int, default=0)
    ap.add_argument("--metrics", default=DEFAULT_METRICS)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--pool", type=int, default=100)
    ap.add_argument("--delta-floor", type=float, default=None)
    ap.add_argument("--out-json", type=Path, required=True)
    return ap.parse_args(argv)


def run_harness(args: argparse.Namespace) -> dict[str, Any]:
    """Основной прогон (env уже выставлен до импорта embeddings)."""
    # Импорт после MCP_1C_CHROMA_DIR / MCP_1C_EMBEDDING_MODEL.
    import chromadb

    from mcp_1c_rag.embeddings import embed_query

    metrics = [m.strip() for m in str(args.metrics).split(",") if m.strip()]
    if not metrics:
        raise SystemExit("пустой --metrics")
    metrics_for_gate = (
        ["ndcg@10", *metrics] if "ndcg@10" not in metrics else metrics
    )

    delta_floor = resolve_delta_floor(args.golden, args.delta_floor)
    rows = filter_gold_by_scope(load_goldset(args.golden), args.scope)
    if not rows:
        raise SystemExit("после фильтрации не осталось запросов")

    chroma_dir = Path(args.chroma_dir)
    client = chromadb.PersistentClient(path=str(chroma_dir))
    collection = client.get_collection(args.collection)
    coll_count = int(collection.count())
    if coll_count == 0:
        raise SystemExit(f"коллекция {args.collection!r} пуста")

    baseline_bm25 = Bm25Index.load(Path(args.baseline_bm25_dir))
    candidate_bm25 = Bm25Index.load(Path(args.candidate_bm25_dir))

    pool = max(1, min(int(args.pool), coll_count))
    k = max(1, int(args.k))
    # Для max>0 — тот же стартовый multiplier, что у search() (default 3).
    pool_multiplier = DEFAULT_START_MULTIPLIER

    # Per-metric arrays — отдельные списки, не смешанные.
    baseline_by_metric: dict[str, list[float]] = {m: [] for m in metrics_for_gate}
    candidate_by_metric: dict[str, list[float]] = {m: [] for m in metrics_for_gate}
    # Vector-only sanity (общие vector-кандидаты → должны совпасть).
    vector_b_by_metric: dict[str, list[float]] = {
        m: [] for m in metrics_for_gate if m != "latency_ms"
    }
    vector_c_by_metric: dict[str, list[float]] = {
        m: [] for m in metrics_for_gate if m != "latency_ms"
    }

    # Прогрев: модель + первый Chroma/BM25 (не в latency).
    warm_q = rows[0]["query"]
    _ = embed_query(warm_q)
    _ = collection.query(
        query_embeddings=[embed_query(warm_q)],
        n_results=min(pool, coll_count),
        include=["documents", "metadatas", "distances"],
    )
    _ = baseline_bm25.scores(warm_q)
    _ = candidate_bm25.scores(warm_q)

    baseline_max = int(args.baseline_max_per_source)
    candidate_max = int(args.candidate_max_per_source)

    for row in rows:
        query = row["query"]
        relevant = list(row["relevant_sources"])

        t0 = time.perf_counter()
        qvec = embed_query(query)
        # Общий стартовый vector-пул (qvec один раз; без BM25 — только Chroma).
        raw0 = collection.query(
            query_embeddings=[qvec],
            n_results=pool,
            include=["documents", "metadatas", "distances"],
        )
        vector_cache: dict[int, list[Candidate]] = {
            pool: _vector_candidates_from_raw(
                {
                    "ids": (raw0.get("ids") or [[]])[0],
                    "documents": (raw0.get("documents") or [[]])[0],
                    "metadatas": (raw0.get("metadatas") or [[]])[0],
                    "distances": (raw0.get("distances") or [[]])[0],
                }
            )
        }
        t_vector = time.perf_counter() - t0

        fetch_b = make_side_fetch(
            query=query,
            qvec=qvec,
            collection=collection,
            collection_name=args.collection,
            bm25_index=baseline_bm25,
            vector_cache=vector_cache,
        )
        fetch_c = make_side_fetch(
            query=query,
            qvec=qvec,
            collection=collection,
            collection_name=args.collection,
            bm25_index=candidate_bm25,
            vector_cache=vector_cache,
        )

        vector_cands = vector_cache[pool]
        vector_top = vector_cands[:k]
        vector_sources = [c.source for c in vector_top]
        for m in vector_b_by_metric:
            val = metric_value(m, vector_sources, relevant, k=k)
            vector_b_by_metric[m].append(val)
            vector_c_by_metric[m].append(val)

        # Baseline arm (тот же vector_cache / qvec)
        t_b0 = time.perf_counter()
        b_hits = side_hits(
            fetch_b,
            top_k=k,
            max_per_source=baseline_max,
            pool=pool,
            pool_multiplier=pool_multiplier,
            total=coll_count,
        )
        t_baseline = (t_vector + (time.perf_counter() - t_b0)) * 1000.0
        b_sources = [c.source for c in b_hits]

        # Candidate arm
        t_c0 = time.perf_counter()
        c_hits = side_hits(
            fetch_c,
            top_k=k,
            max_per_source=candidate_max,
            pool=pool,
            pool_multiplier=pool_multiplier,
            total=coll_count,
        )
        t_candidate = (t_vector + (time.perf_counter() - t_c0)) * 1000.0
        c_sources = [c.source for c in c_hits]

        for m in metrics_for_gate:
            if m == "latency_ms":
                baseline_by_metric[m].append(t_baseline)
                candidate_by_metric[m].append(t_candidate)
            else:
                baseline_by_metric[m].append(
                    metric_value(m, b_sources, relevant, k=k, latency_ms=t_baseline)
                )
                candidate_by_metric[m].append(
                    metric_value(m, c_sources, relevant, k=k, latency_ms=t_candidate)
                )

    metric_reports: dict[str, Any] = {}
    for m in metrics_for_gate:
        metric_reports[m] = build_metric_report(
            baseline_by_metric[m],
            candidate_by_metric[m],
            metric=m,
            delta_floor=delta_floor,
        )

    vector_sanity: dict[str, Any] = {}
    vector_ok = True
    for m, b_vals in vector_b_by_metric.items():
        rep = build_metric_report(
            b_vals, vector_c_by_metric[m], metric=m, delta_floor=delta_floor
        )
        # Для sanity гейт: Δ=0 и CI=[0,0]
        delta_zero = abs(rep["delta"]) < TIE_EPS
        ci_zero = abs(rep["ci_low"]) < TIE_EPS and abs(rep["ci_high"]) < TIE_EPS
        rep["gate"] = bool(delta_zero and ci_zero)
        if not rep["gate"]:
            vector_ok = False
        vector_sanity[m] = rep
    vector_sanity["pass"] = vector_ok

    ndcg_gate = bool(metric_reports.get("ndcg@10", {}).get("gate"))
    status = "pass" if (ndcg_gate and vector_ok) else "fail"

    return {
        "metadata": {
            "golden": str(args.golden),
            "n": len(rows),
            "delta_floor": delta_floor,
            "chroma_dir": str(args.chroma_dir),
            "collection": args.collection,
            "scope": args.scope,
            "embedding_model": args.embedding_model,
            "baseline_bm25_dir": str(args.baseline_bm25_dir),
            "candidate_bm25_dir": str(args.candidate_bm25_dir),
            "baseline_max_per_source": int(args.baseline_max_per_source),
            "candidate_max_per_source": int(args.candidate_max_per_source),
            "k": k,
            "pool": pool,
            "metrics": metrics_for_gate,
            "rrf_k": RRF_K,
            "bootstrap": {"n_resamples": N_BOOT, "seed": BOOT_SEED, "ci_level": CI_LEVEL},
        },
        "metrics": metric_reports,
        "vector_sanity": vector_sanity,
        "status": status,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry: env → harness → JSON."""
    args = parse_args(argv)
    # До импорта embeddings/config внутри run_harness.
    os.environ["MCP_1C_CHROMA_DIR"] = str(Path(args.chroma_dir).resolve())
    os.environ["MCP_1C_EMBEDDING_MODEL"] = str(args.embedding_model)

    report = run_harness(args)
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"status={report['status']} n={report['metadata']['n']} "
        f"ndcg@10 Δ={report['metrics']['ndcg@10']['delta']:.6f} "
        f"CI=[{report['metrics']['ndcg@10']['ci_low']:.6f}; "
        f"{report['metrics']['ndcg@10']['ci_high']:.6f}] "
        f"→ {out}",
        flush=True,
    )
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
