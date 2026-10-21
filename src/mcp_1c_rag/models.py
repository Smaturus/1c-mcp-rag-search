"""Лёгкие модели данных retrieval-слоя (без тяжёлых зависимостей).

Модуль намеренно не импортирует Chroma, sentence-transformers и flashrank:
его можно использовать в unit-тестах и CI без установки ML-стека.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Candidate:
    """Кандидат поиска с единым интерфейсом для RRF и reranker'а.

    Attributes:
        id: Стабильный идентификатор чанка в Chroma (``bsl:...`` / ``meta:...``).
        text: Текст чанка.
        metadata: Метаданные чанка (``source``, ``chunk_index``, ``kind``).
        score: Итоговый скор в текущем режиме поиска.
        source_scores: Разложение скора по источникам (cosine, bm25, rrf, ...).
    """

    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
    source_scores: dict[str, float] = field(default_factory=dict)

    @property
    def source(self) -> str:
        """Путь к файлу-источнику (пустая строка, если метаданных нет)."""
        return str(self.metadata.get("source", ""))


__all__ = ["Candidate"]
