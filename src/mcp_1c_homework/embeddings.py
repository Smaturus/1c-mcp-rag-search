"""Local multilingual-e5 embeddings with required query/passage prefixes."""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache

from sentence_transformers import SentenceTransformer

from mcp_1c_homework.config import EMBEDDING_MODEL


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    """Load embedding model once per process."""
    return SentenceTransformer(EMBEDDING_MODEL)


def embed_passages(texts: Sequence[str], batch_size: int = 8) -> list[list[float]]:
    """Embed documents/passages (index-time)."""
    model = get_model()
    prefixed = [f"passage: {t}" for t in texts]
    vectors = model.encode(
        prefixed,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    return [v.tolist() for v in vectors]


def embed_query(text: str) -> list[float]:
    """Embed a search query (query-time)."""
    model = get_model()
    vector = model.encode(
        [f"query: {text}"],
        batch_size=1,
        show_progress_bar=False,
        normalize_embeddings=True,
    )[0]
    return vector.tolist()
