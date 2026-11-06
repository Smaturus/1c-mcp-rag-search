"""Unit-тесты mcp_1c_rag.dedup и search(..., max_per_source) без модели/Chroma."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mcp_1c_rag import search_modes  # noqa: E402
from mcp_1c_rag.dedup import cap_by_source, iterative_dedup  # noqa: E402
from mcp_1c_rag.models import Candidate  # noqa: E402


def make(cid: str, source: str, score: float) -> Candidate:
    return Candidate(id=cid, text="", metadata={"source": source}, score=score)


def test_cap_by_source_7_plus_3():
    cands = [make(f"c{i}", "file1", 1.0 - i * 0.01) for i in range(7)] + [
        make(f"d{i}", "file2", 0.9 - i * 0.01) for i in range(3)
    ]
    capped = cap_by_source(cands, max_per_source=2)
    assert len(capped) == 4
    assert sum(c.source == "file1" for c in capped) == 2
    assert sum(c.source == "file2" for c in capped) == 2
    # порядок: первые два file1, затем первые два file2
    assert [c.id for c in capped] == ["c0", "c1", "d0", "d1"]


def test_cap_by_source_max_zero_unchanged():
    cands = [make(f"a{i}", "s", 1.0 - i * 0.01) for i in range(5)]
    assert cap_by_source(cands, 0) == cands


def test_iterative_dedup_fills_top_k():
    """100 кандидатов, 10 source → top_k=10, max=2 → ровно 10, cap соблюдён."""
    universe: list[Candidate] = []
    for s in range(10):
        for j in range(10):
            universe.append(make(f"s{s}_{j}", f"src{s}", 1.0 - s * 0.01 - j * 0.001))

    calls: list[int] = []

    def fetch(pool: int) -> list[Candidate]:
        calls.append(pool)
        return universe[:pool]

    out = iterative_dedup(fetch, top_k=10, max_per_source=2, pool_multiplier=10, total=100)
    assert len(out) == 10
    assert all(sum(c.source == src for c in out) <= 2 for src in {c.source for c in out})
    assert calls[0] == 100  # top_k * multiplier


def test_iterative_dedup_few_sources_no_padding():
    """2 source × много чанков, max=2, top_k=10 → 4, без добора дублей."""
    universe = [make(f"a{i}", "s1", 1.0 - i * 0.001) for i in range(50)] + [
        make(f"b{i}", "s2", 0.5 - i * 0.001) for i in range(50)
    ]
    calls: list[int] = []

    def fetch(pool: int) -> list[Candidate]:
        calls.append(pool)
        return universe[:pool]

    out = iterative_dedup(fetch, top_k=10, max_per_source=2, pool_multiplier=10, total=100)
    assert len(out) == 4
    assert sum(c.source == "s1" for c in out) == 2
    assert sum(c.source == "s2" for c in out) == 2
    # пул растёт, пока не упрётся в total
    assert calls[0] == 100
    assert calls[-1] == 100


def test_iterative_dedup_growing_pool():
    """fetch вызывается с растущим pool, пока не наберётся top_k."""
    # 5 source, по 1 кандидату на «ранних» позициях; остальные — дубли первого.
    # При pool=10 (top_k=5 * mult нужно меньше) — настроим mult=1 → pool=5,
    # capped даст мало; затем pool=10 наберёт 5 source.
    universe: list[Candidate] = []
    # Сначала 20 чанков одного source, потом уникальные
    for i in range(20):
        universe.append(make(f"dup{i}", "only", 1.0 - i * 0.001))
    for s in range(10):
        universe.append(make(f"u{s}", f"src{s}", 0.5 - s * 0.01))

    calls: list[int] = []

    def fetch(pool: int) -> list[Candidate]:
        calls.append(pool)
        return universe[:pool]

    out = iterative_dedup(
        fetch, top_k=5, max_per_source=1, pool_multiplier=1, total=len(universe)
    )
    assert len(out) == 5
    assert len({c.source for c in out}) == 5
    assert calls[0] == 5
    assert calls[1] == 10
    assert calls == sorted(calls)
    assert calls[-1] > calls[0]


def test_search_max_zero_delegates(monkeypatch: pytest.MonkeyPatch):
    synthetic = [make(f"x{i}", f"s{i}", 1.0 - i * 0.01) for i in range(5)]

    def fake_impl(query, top_k, mode, collection):
        return synthetic[:top_k]

    monkeypatch.setattr(search_modes, "_search_impl", fake_impl)
    out = search_modes.search("q", top_k=5, max_per_source=0)
    assert out == synthetic[:5]


def _patch_hybrid_dedup(
    monkeypatch: pytest.MonkeyPatch,
    universe: list[Candidate],
    *,
    total: int,
) -> None:
    """Подменить hybrid_search / embed / BM25 для unit-тестов без модели."""

    def fake_hybrid(
        query,
        *,
        collection="code",
        top_k=50,
        k_rrf=60,
        query_embedding=None,
        bm25_scores=None,
    ):
        return universe[:top_k]

    monkeypatch.setattr(search_modes, "hybrid_search", fake_hybrid)
    monkeypatch.setattr(
        search_modes, "_masked_bm25_scores", lambda q, c: __import__("numpy").zeros(1)
    )
    monkeypatch.setattr(search_modes, "_collection_count", lambda name="code": total)
    import mcp_1c_rag.embeddings as emb

    monkeypatch.setattr(emb, "embed_query", lambda t: [0.0])


def test_search_max_two_enough_sources(monkeypatch: pytest.MonkeyPatch):
    """≥10 source → search(top_k=10, max=2) == 10, cap соблюдён."""
    universe: list[Candidate] = []
    for s in range(20):
        for j in range(5):
            universe.append(make(f"s{s}_{j}", f"src{s}", 1.0 - s * 0.01 - j * 0.001))

    _patch_hybrid_dedup(monkeypatch, universe, total=1000)
    out = search_modes.search(
        "q", top_k=10, mode="hybrid", max_per_source=2, pool_multiplier=3
    )
    assert len(out) == 10
    assert all(sum(c.source == src for c in out) <= 2 for src in {c.source for c in out})


def test_search_max_two_few_sources(monkeypatch: pytest.MonkeyPatch):
    """2 source → ≤10 без добора дублей (ровно 4 при max=2)."""
    universe = [make(f"a{i}", "s1", 1.0 - i * 0.001) for i in range(50)] + [
        make(f"b{i}", "s2", 0.5 - i * 0.001) for i in range(50)
    ]
    _patch_hybrid_dedup(monkeypatch, universe, total=100)
    out = search_modes.search(
        "q", top_k=10, mode="hybrid", max_per_source=2, pool_multiplier=10
    )
    assert len(out) == 4
    assert sum(c.source == "s1" for c in out) == 2
    assert sum(c.source == "s2" for c in out) == 2


def test_iterative_dedup_start_multiplier_fewer_calls():
    """start_multiplier=3 → первый fetch меньше, чем при 10, при достаточном пуле."""
    universe: list[Candidate] = []
    for s in range(20):
        for j in range(5):
            universe.append(make(f"s{s}_{j}", f"src{s}", 1.0 - s * 0.01 - j * 0.001))

    calls3: list[int] = []
    calls10: list[int] = []

    def fetch3(pool: int) -> list[Candidate]:
        calls3.append(pool)
        return universe[:pool]

    def fetch10(pool: int) -> list[Candidate]:
        calls10.append(pool)
        return universe[:pool]

    out3 = iterative_dedup(
        fetch3, top_k=10, max_per_source=2, pool_multiplier=3, total=200
    )
    out10 = iterative_dedup(
        fetch10, top_k=10, max_per_source=2, pool_multiplier=10, total=200
    )
    assert len(out3) == 10 and len(out10) == 10
    assert calls3[0] == 30
    assert calls10[0] == 100
    assert calls3[0] < calls10[0]
    assert len(calls3) == 1  # хватило стартового пула


def test_search_max_zero_single_impl_call(monkeypatch: pytest.MonkeyPatch):
    """max=0: ровно один вызов _search_impl, без iterative_dedup."""
    synthetic = [make(f"x{i}", f"s{i}", 1.0 - i * 0.01) for i in range(5)]
    calls = {"n": 0}

    def fake_impl(query, top_k, mode, collection):
        calls["n"] += 1
        return synthetic[:top_k]

    monkeypatch.setattr(search_modes, "_search_impl", fake_impl)
    out = search_modes.search("q", top_k=5, max_per_source=0)
    assert out == synthetic[:5]
    assert calls["n"] == 1


def test_hybrid_dedup_reuses_qvec_and_bm25(monkeypatch: pytest.MonkeyPatch):
    """При backfill hybrid-dedup: embed_query и BM25.scores ровно по 1 разу."""
    embed_calls = {"n": 0}
    bm25_calls = {"n": 0}
    hybrid_pools: list[int] = []

    def fake_hybrid(
        query,
        *,
        collection="code",
        top_k=50,
        k_rrf=60,
        query_embedding=None,
        bm25_scores=None,
    ):
        assert query_embedding is not None
        assert bm25_scores is not None
        hybrid_pools.append(top_k)
        out = []
        for i in range(top_k):
            src = "only" if i < 20 else f"src{i}"
            out.append(make(f"id{i}", src, 1.0 - i * 0.001))
        return out

    def fake_embed(text: str):
        embed_calls["n"] += 1
        return [0.0, 1.0]

    def fake_masked(query: str, collection: str):
        bm25_calls["n"] += 1
        return __import__("numpy").zeros(10)

    monkeypatch.setattr(search_modes, "hybrid_search", fake_hybrid)
    monkeypatch.setattr(search_modes, "_masked_bm25_scores", fake_masked)
    monkeypatch.setattr(search_modes, "_collection_count", lambda name="code": 100)

    import mcp_1c_rag.embeddings as emb

    monkeypatch.setattr(emb, "embed_query", fake_embed)

    out = search_modes.search(
        "q",
        top_k=5,
        mode="hybrid",
        max_per_source=1,
        pool_multiplier=1,  # start pool=5 → все "only" → backfill
    )
    assert len(out) == 5
    assert len({c.source for c in out}) == 5
    assert embed_calls["n"] == 1
    assert bm25_calls["n"] == 1
    assert hybrid_pools[0] == 5
    assert hybrid_pools[-1] > hybrid_pools[0]
