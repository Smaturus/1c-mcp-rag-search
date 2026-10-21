"""Второй этап поиска: cross-encoder reranker на flashrank (CPU, ONNX).

Reranker работает поверх гибридных кандидатов (20–50 штук) и не заменяет
e5-small: он только переупорядочивает то, что уже отобрано vector+BM25.

Модель по умолчанию — ``ms-marco-MultiBERT-L-12`` (мультиязычная, 98 МБ ONNX),
загружается лениво при первом вызове и кэшируется на диск flashrank.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from mcp_1c_homework.config import RERANK_MAX_LENGTH, RERANK_MODEL
from mcp_1c_homework.models import Candidate


class _RankerLike(Protocol):
    def rerank(self, request: Any) -> list[dict[str, Any]]: ...


_ranker: _RankerLike | None = None


def _passage_text(candidate: Candidate, *, max_chars: int = 2000) -> str:
    """Текст пассажа для cross-encoder'а: путь к модулю + фрагмент кода.

    Имя файла — сильный сигнал для точных запросов («ПередЗаписью»,
    «СчетФактураВыданный»), поэтому оно идёт первым и не обрезается.
    """
    source = str((candidate.metadata or {}).get("source", "")).strip()
    body = (candidate.text or "").strip()
    text = f"{source}\n{body}" if source else body
    return (text[:max_chars] or " ").strip()


def get_ranker() -> _RankerLike:
    """Ленивая инициализация flashrank Ranker (скачивание модели при первом старте)."""
    global _ranker
    if _ranker is None:
        from flashrank import Ranker  # импорт здесь: torch не нужен, только onnxruntime

        _ranker = Ranker(model_name=RERANK_MODEL, max_length=RERANK_MAX_LENGTH)
    return _ranker


def rerank(
    query: str,
    candidates: Sequence[Candidate],
    *,
    top_k: int = 5,
    ranker: _RankerLike | None = None,
) -> list[Candidate]:
    """Пересортировать кандидатов cross-encoder'ом и вернуть top_k.

    Args:
        query: Исходный запрос (без префиксов — они не нужны reranker'у).
        candidates: Кандидаты от ``hybrid_search`` (id + text + метаданные).
        top_k: Сколько оставить после пересортировки.
        ranker: Позволяет подменить Ranker в тестах.

    Returns:
        Новый список кандидатов: score заменён на скор cross-encoder'а,
        исходный гибридный скор сохранён в ``source_scores['rrf']``.
    """
    from flashrank import RerankRequest

    if not candidates:
        return []
    top_k = max(1, min(int(top_k), len(candidates)))
    passages = [
        {"id": c.id, "text": _passage_text(c)}
        for c in candidates
    ]
    request = RerankRequest(query=query, passages=passages)
    ranked = (ranker or get_ranker()).rerank(request)

    by_id = {c.id: c for c in candidates}
    result: list[Candidate] = []
    for item in ranked[:top_k]:
        doc_id = str(item.get("id"))
        base = by_id.get(doc_id)
        if base is None:
            continue
        sources = dict(base.source_scores)
        sources["rrf"] = sources.get("score", base.score)
        result.append(
            Candidate(
                id=base.id,
                text=base.text,
                metadata=dict(base.metadata),
                score=float(item.get("score", 0.0)),
                source_scores=sources,
            )
        )
    return result


__all__ = ["get_ranker", "rerank"]
