"""Тесты обёртки над flashrank reranker (без загрузки реальной ONNX-модели)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mcp_1c_homework.rerank import rerank  # noqa: E402
from mcp_1c_homework.search_modes import Candidate  # noqa: E402


class FakeRanker:
    """Заглушка flashrank Ranker: ранжирует по длине текста (для детерминизма)."""

    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores
        self.requests: list = []

    def rerank(self, request):
        self.requests.append(request)
        items = [
            {"id": p["id"], "text": p["text"], "score": self.scores.get(p["id"], 0.0)}
            for p in request.passages
        ]
        return sorted(items, key=lambda x: x["score"], reverse=True)


def _candidates(*ids: str) -> list[Candidate]:
    return [
        Candidate(id=i, text=f"текст {i}", metadata={"source": f"{i}.bsl"}, score=0.01)
        for i in ids
    ]


def test_rerank_reorders_by_cross_encoder_score() -> None:
    """Reranker переставляет кандидатов по своим скорам."""
    fake = FakeRanker({"a": 0.1, "b": 0.9, "c": 0.5})
    result = rerank("запрос", _candidates("a", "b", "c"), top_k=3, ranker=fake)
    assert [c.id for c in result] == ["b", "c", "a"]


def test_rerank_keeps_top_k_only() -> None:
    """Возвращается ровно top_k кандидатов."""
    fake = FakeRanker({"a": 0.1, "b": 0.9, "c": 0.5})
    result = rerank("запрос", _candidates("a", "b", "c"), top_k=2, ranker=fake)
    assert len(result) == 2
    assert [c.id for c in result] == ["b", "c"]


def test_rerank_preserves_metadata_and_source_scores() -> None:
    """Метаданные сохраняются, RRF-скор кладётся в source_scores."""
    fake = FakeRanker({"a": 1.0})
    result = rerank("запрос", _candidates("a"), top_k=1, ranker=fake)
    assert result[0].metadata["source"] == "a.bsl"
    assert result[0].score == 1.0
    assert "rrf" in result[0].source_scores


def test_rerank_passes_query_in_request() -> None:
    """Оригинальный запрос передаётся reranker'у без изменений."""
    fake = FakeRanker({"a": 1.0})
    rerank("НДС с аванса", _candidates("a"), top_k=1, ranker=fake)
    assert fake.requests[0].query == "НДС с аванса"


def test_rerank_empty_candidates() -> None:
    """Пустой список кандидатов — пустой результат."""
    assert rerank("запрос", [], top_k=5, ranker=FakeRanker({})) == []
