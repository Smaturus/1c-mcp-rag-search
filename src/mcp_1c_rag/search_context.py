"""Контекст поиска: BM25, prefix-маски и счётчики коллекций без модульных глобалов.

P1a: изолирует кэши в экземпляре ``SearchContext``, чтобы в одном процессе
можно было держать два независимых BM25-индекса (тесты, DI). Не меняет
scoring / выдачу / tool surface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from mcp_1c_rag.collections import canonical_name, id_prefix
from mcp_1c_rag.config import BM25_DIR, COLLECTION_CODE

if TYPE_CHECKING:  # pragma: no cover - только для аннотаций
    from mcp_1c_rag.bm25 import Bm25Index


@dataclass
class SearchContext:
    """Ленивые кэши BM25 / масок / count для одного поискового контура."""

    bm25_dir: Path | None = None
    _bm25: Bm25Index | None = field(default=None, init=False, repr=False)
    _masks: dict[str, np.ndarray] = field(default_factory=dict, init=False, repr=False)
    _collection_counts: dict[str, int] = field(
        default_factory=dict, init=False, repr=False
    )

    def _resolved_bm25_dir(self) -> Path:
        return self.bm25_dir if self.bm25_dir is not None else BM25_DIR

    def bm25(self) -> Bm25Index:
        """Ленивая загрузка BM25-индекса для этого контекста."""
        if self._bm25 is None:
            from mcp_1c_rag.bm25 import Bm25Index

            directory = self._resolved_bm25_dir()
            if not (directory / "index.npz").exists():
                raise FileNotFoundError(
                    f"BM25-индекс не найден: {directory}. "
                    "Соберите его: python scripts/build_bm25.py"
                )
            self._bm25 = Bm25Index.load(directory)
        return self._bm25

    def bm25_available(self) -> bool:
        """Есть ли собранный BM25-индекс в каталоге этого контекста."""
        return (self._resolved_bm25_dir() / "index.npz").exists()

    def prefix_mask(self, prefix: str) -> np.ndarray:
        """Булева маска doc_ids по префиксу; кэш на экземпляр."""
        mask = self._masks.get(prefix)
        if mask is None:
            index = self.bm25()
            mask = np.fromiter(
                (doc_id.startswith(prefix) for doc_id in index.doc_ids),
                dtype=bool,
                count=index.n_docs,
            )
            self._masks[prefix] = mask
        return mask

    def masked_bm25_scores(self, query: str, collection: str) -> np.ndarray:
        """BM25-скоры с prefix-mask коллекции (один раз на запрос)."""
        canonical = canonical_name(collection or COLLECTION_CODE)
        prefix = id_prefix(canonical)
        scores = self.bm25().scores(query)
        if prefix:
            scores = np.where(self.prefix_mask(prefix), scores, -np.inf)
        return scores

    def collection_count(self, name: str | None = None) -> int:
        """Count коллекции с кэшем на экземпляр (ключ — canonical)."""
        key = canonical_name(name or COLLECTION_CODE)
        if key not in self._collection_counts:
            from mcp_1c_rag.store import get_collection

            self._collection_counts[key] = int(get_collection(key).count())
        return self._collection_counts[key]

    def reset(self) -> None:
        """Очистить кэши (для тестов / смены индекса)."""
        self._bm25 = None
        self._masks.clear()
        self._collection_counts.clear()


__all__ = ["SearchContext"]
