"""Persistent Chroma collections for code and metadata."""

from __future__ import annotations

import contextlib
from typing import Any, cast

import chromadb
from chromadb.api.models.Collection import Collection

from mcp_1c_rag.config import (
    CHROMA_DIR,
    COLLECTION_CODE,
    COLLECTION_METADATA,
    HNSW_SYNC_THRESHOLD,
)
from mcp_1c_rag.index_guard import refuse_if_foreign_writer


def get_client() -> Any:
    """Open persistent Chroma client under project data/chroma.

    Returns ``Any`` because ``chromadb.PersistentClient`` is a factory function,
    not a type; the concrete client type is an implementation detail of Chroma.

    Refuses while another process holds the indexer lock, so search cannot
    open the base next to the writer.
    """
    refuse_if_foreign_writer()
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def get_collection(name: str, *, reset: bool = False) -> Collection:
    """Get or create a named collection; optionally drop first."""
    client = get_client()
    if reset:
        with contextlib.suppress(Exception):
            client.delete_collection(name)
    metadata: dict[str, Any] = {"hnsw:space": "cosine"}
    if HNSW_SYNC_THRESHOLD > 0:
        metadata["hnsw:sync_threshold"] = HNSW_SYNC_THRESHOLD
    return client.get_or_create_collection(
        name=name,
        metadata=metadata,
    )


def get_code_collection(*, reset: bool = False) -> Collection:
    return get_collection(COLLECTION_CODE, reset=reset)


def get_metadata_collection(*, reset: bool = False) -> Collection:
    return get_collection(COLLECTION_METADATA, reset=reset)


def query_collection(
    collection: Collection,
    query_embedding: list[float],
    limit: int,
) -> list[dict[str, Any]]:
    """Run nearest-neighbour search and flatten Chroma response."""
    if limit < 1:
        limit = 1
    raw = collection.query(
        # Chroma's stubs type query_embeddings as ndarray; a list of float lists
        # is the documented public form and works at runtime. ``cast`` instead of
        # ``# type: ignore`` keeps mypy --warn-unused-ignores happy in CI, where
        # chromadb may be absent and the argument degrades to Any.
        query_embeddings=cast("Any", [query_embedding]),
        n_results=min(limit, max(collection.count(), 1)),
        include=["documents", "metadatas", "distances"],
    )
    docs = (raw.get("documents") or [[]])[0]
    metas = (raw.get("metadatas") or [[]])[0]
    dists = (raw.get("distances") or [[]])[0]
    results: list[dict[str, Any]] = []
    for doc, meta, dist in zip(docs, metas, dists, strict=False):
        results.append(
            {
                "text": doc or "",
                "metadata": meta or {},
                "distance": dist,
            }
        )
    return results
