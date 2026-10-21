"""Метрики качества retrieval: Hit@1, MRR@10, nDCG@10, Recall@10.

Релевантность оценивается на уровне **файла-источника** (source), а не чанка:
для корпуса из сотен чанков на файл это единственная устойчивая единица,
не зависящая от того, какой именно кусок модуля вернул поиск.

Модуль чистый (без Chroma и эмбеддингов), поэтому покрыт unit-тестами
и переиспользуется в ``scripts/eval_pipeline.py``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field


@dataclass(frozen=True)
class QueryMetrics:
    """Метрики одного запроса."""

    hit_at_1: float
    mrr_at_10: float
    ndcg_at_10: float
    recall_at_10: float
    latency_ms: float = 0.0
    retrieved_sources: tuple[str, ...] = ()


@dataclass
class AggregateMetrics:
    """Средние метрики по набору запросов."""

    n_queries: int = 0
    values: dict[str, float] = field(default_factory=dict)

    def as_row(self) -> dict[str, float]:
        """Плоская строка для отчёта."""
        return {"n_queries": float(self.n_queries), **self.values}


def hit_at_1(sources: Sequence[str], relevant: Iterable[str]) -> float:
    """1.0, если первый результат принадлежит релевантному файлу."""
    relevant_set = set(relevant)
    return 1.0 if sources and sources[0] in relevant_set else 0.0


def mrr_at_k(sources: Sequence[str], relevant: Iterable[str], *, k: int = 10) -> float:
    """Обратная позиция первого релевантного результата в top-k."""
    relevant_set = set(relevant)
    for rank, source in enumerate(sources[:k], start=1):
        if source in relevant_set:
            return 1.0 / rank
    return 0.0


def unique_sources(sources: Sequence[str]) -> list[str]:
    """Свернуть ранжирование до уникальных файлов, сохранив порядок первого вхождения.

    Один файл-источник может вернуться несколькими чанками. Если считать каждый
    чанк отдельным релевантным документом, метрики завышаются (а nDCG может даже
    превысить 1.0 при одном релевантном файле). Поэтому метрики считаются по
    уникальным файлам: повторные чанки того же файла не добавляют вклада.
    """
    seen: set[str] = set()
    ordered: list[str] = []
    for source in sources:
        if source in seen:
            continue
        seen.add(source)
        ordered.append(source)
    return ordered


def recall_at_k(sources: Sequence[str], relevant: Iterable[str], *, k: int = 10) -> float:
    """Доля релевантных файлов, найденных в top-k (по уникальным файлам).

    Знаменатель — ``min(|relevant|, k)``: если релевантных файлов больше k,
    максимум достижимого recall@k равен единице.
    """
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    found = {source for source in unique_sources(sources)[:k] if source in relevant_set}
    return len(found) / min(len(relevant_set), k)


def ndcg_at_k(sources: Sequence[str], relevant: Iterable[str], *, k: int = 10) -> float:
    """nDCG@k по уникальным файлам при бинарной релевантности (gain = 1).

    Повторные чанки одного файла схлопываются: иначе одна и та же релевантная
    единица учитывалась бы несколько раз и метрика была бы завышена.
    """
    relevant_set = set(relevant)
    ranked = unique_sources(sources)
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, source in enumerate(ranked[:k], start=1)
        if source in relevant_set
    )
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(relevant_set), k) + 1))
    return dcg / ideal if ideal > 0 else 0.0


def evaluate_query(
    sources: Sequence[str],
    relevant: Iterable[str],
    *,
    k: int = 10,
    latency_ms: float = 0.0,
) -> QueryMetrics:
    """Посчитать все метрики для одного запроса."""
    return QueryMetrics(
        hit_at_1=hit_at_1(sources, relevant),
        mrr_at_10=mrr_at_k(sources, relevant, k=k),
        ndcg_at_10=ndcg_at_k(sources, relevant, k=k),
        recall_at_10=recall_at_k(sources, relevant, k=k),
        latency_ms=latency_ms,
        retrieved_sources=tuple(sources[:k]),
    )


def aggregate(items: Sequence[QueryMetrics]) -> AggregateMetrics:
    """Усреднить метрики по запросам."""
    if not items:
        return AggregateMetrics()
    n = len(items)
    return AggregateMetrics(
        n_queries=n,
        values={
            "hit@1": sum(i.hit_at_1 for i in items) / n,
            "mrr@10": sum(i.mrr_at_10 for i in items) / n,
            "ndcg@10": sum(i.ndcg_at_10 for i in items) / n,
            "recall@10": sum(i.recall_at_10 for i in items) / n,
            "latency_ms": sum(i.latency_ms for i in items) / n,
        },
    )


def format_markdown_table(
    rows: dict[str, AggregateMetrics],
    *,
    descriptions: dict[str, str] | None = None,
    k: int = 10,
) -> str:
    """Markdown-таблица «конфигурация × метрики» для отчёта и README."""
    descriptions = descriptions or {}
    lines = [
        f"| Конфигурация | Hit@1 | MRR@{k} | nDCG@{k} | Recall@{k} | мс/запрос |",
        "|---|---|---|---|---|---|",
    ]
    for mode, metrics in rows.items():
        v = metrics.values
        lines.append(
            f"| `{mode}` | {v.get('hit@1', 0):.3f} | {v.get('mrr@10', 0):.3f} | "
            f"{v.get('ndcg@10', 0):.3f} | {v.get('recall@10', 0):.3f} | "
            f"{v.get('latency_ms', 0):.0f} |"
        )
    return "\n".join(lines)


__all__ = [
    "AggregateMetrics",
    "QueryMetrics",
    "aggregate",
    "evaluate_query",
    "format_markdown_table",
    "hit_at_1",
    "mrr_at_k",
    "ndcg_at_k",
    "recall_at_k",
    "unique_sources",
]
