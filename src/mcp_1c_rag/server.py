"""MCP stdio server: search, evaluate_retrieval, route_query, index_stats."""

from __future__ import annotations

from mcp.server.mcpserver import MCPServer

from mcp_1c_rag import eval_retrieval as eval_mod
from mcp_1c_rag import merge, search_modes
from mcp_1c_rag import search as legacy_search
from mcp_1c_rag.config import COLLECTION_METADATA, SEARCH_DEFAULT_MODE
from mcp_1c_rag.formatting import format_candidates
from mcp_1c_rag.index_guard import IndexBusy
from mcp_1c_rag.router import route_query as route_decision

mcp = MCPServer(
    "1c-mcp-rag-search",
    instructions=(
        "1C retrieval MCP: four tools — search, evaluate_retrieval, route_query, "
        "index_stats. "
        "search(query, top_k, mode, collection) covers BSL and metadata: "
        "collection empty/code_proc/code → code; metadata_xml/metadata → metadata; "
        "all → both collections merged. "
        "Modes: lexical | vector | hybrid (default) | hybrid_rerank | pearson. "
        "route_query returns tool=search with mode/collection to call. "
        "Legacy Python aliases (codesearch, metadatasearch, search_1c_metadata, "
        "search_knowledge_base, search_hybrid) are not MCP tools."
    ),
)


@mcp.tool()
def search(
    query: str,
    top_k: int = 5,
    mode: str = "",
    collection: str = "",
) -> str:
    """Unified retrieval: vector | hybrid (RRF) | hybrid_rerank | pearson (baseline).

    Pipeline: e5 cosine top-N + global BM25 top-N -> RRF -> optional cross-encoder
    rerank (flashrank, ms-marco-MultiBERT-L-12). Default mode is ``hybrid``.

    Args:
        query: Natural-language or exact-name query (e.g. "НДС с аванса").
        top_k: How many results to return (default 5).
        mode: lexical | vector | hybrid | hybrid_rerank | pearson;
            empty = server default.
        collection: code_proc/code (BSL), metadata_xml/metadata, or all
            (both collections merged); empty = server default (code_proc).
    """
    key = (collection or "").strip().lower()
    chosen_mode = (mode or SEARCH_DEFAULT_MODE).strip().lower() or SEARCH_DEFAULT_MODE
    try:
        if key == "all":
            candidates = merge.merged_search(
                query, chosen_mode, max(1, int(top_k))
            )
        else:
            candidates = search_modes.search(
                query=query,
                top_k=top_k,
                mode=(mode or None),
                collection=collection,
            )
    except FileNotFoundError as exc:
        return (
            f"{exc}\n"
            "Подсказка: соберите BM25-индекс `python scripts/build_bm25.py` "
            "или используйте mode='vector'."
        )
    except IndexBusy as exc:
        return str(exc)
    except ValueError as exc:
        return f"{exc}"
    return format_candidates(candidates)


@mcp.tool()
def evaluate_retrieval(
    query: str,
    k: int = 5,
    collection: str = "",
    relevant_sources: str = "",
    null_pairs: int = 40,
) -> str:
    """Evaluate retrieval quality for one query (quality metrics).

    Example: evaluate_retrieval(query=\"НДС\")
    Returns Hit@k, Recall@k, MRR, top cosine, optional null-tail vs random pairs.

    Args:
        query: Query to evaluate, e.g. НДС or ПередЗаписью.
        k: Top-k cutoff (default 5).
        collection: code_proc (BSL) or metadata_xml (XML metadata);
            legacy aliases code/metadata still work.
        relevant_sources: Comma-separated path needles; empty = use query as needle.
        null_pairs: Random background pairs for null-tail (0 skips; default 40).
    """
    return eval_mod.evaluate_retrieval(
        query=query,
        k=k,
        collection=collection,
        relevant_sources=relevant_sources,
        null_pairs=null_pairs,
    )


@mcp.tool()
def route_query(query: str) -> str:
    """Explain which search() call to use for a query, and why.

    Applies the routing contract: lexical for exact names/relations, vector for
    how-to questions, hybrid for mixed or requirement queries (and for anything
    unclear). Returns route, tool=search, mode, collection, reason, confidence.

    Args:
        query: The user question to classify.
    """
    decision = route_decision(query)
    return (
        f"route={decision.route.value}\n"
        f"tool={decision.tool}\n"
        f"mode={decision.mode}\n"
        f"collection={decision.collection}\n"
        f"confidence={decision.confidence}\n"
        f"reason={decision.reason}"
    )


@mcp.tool()
def index_stats() -> str:
    """Report which index the server is actually serving (paths, counts, model).

    Useful before a demo or when results look unexpected: confirms the Chroma
    directory, collection sizes, BM25 index and the default search mode.
    """
    from pathlib import Path

    from mcp_1c_rag.config import (
        BM25_DIR,
        CHROMA_DIR,
        EMBEDDING_MODEL,
        SEARCH_DEFAULT_MODE,
    )

    lines = [
        f"chroma_dir={CHROMA_DIR}",
        f"bm25_dir={BM25_DIR}",
        f"embedding_model={EMBEDDING_MODEL}",
        f"default_mode={SEARCH_DEFAULT_MODE}",
    ]
    try:
        from mcp_1c_rag.collections import all_specs
        from mcp_1c_rag.store import get_client

        client = get_client()
        for spec in all_specs():
            try:
                collection = client.get_collection(spec.name)
                lines.append(f"collection[{spec.name}]={collection.count()}")
            except Exception:
                continue
    except Exception as exc:  # noqa: BLE001
        lines.append(f"chroma_error={type(exc).__name__}: {exc}")
    bm25_file = Path(BM25_DIR) / "index.npz"
    lines.append(f"bm25_index_present={bm25_file.exists()}")
    return "\n".join(lines)


def codesearch(query: str, limit: int = 5) -> str:
    """legacy Python alias, not an MCP tool — cosine search по BSL."""
    return legacy_search.codesearch(query=query, limit=limit)


def metadatasearch(query: str, limit: int = 20) -> str:
    """legacy Python alias, not an MCP tool — cosine search по метаданным."""
    return legacy_search.metadatasearch(query=query, limit=limit)


def search_1c_metadata(query: str, limit: int = 10) -> str:
    """legacy Python alias, not an MCP tool — lexical по metadata_xml."""
    return format_candidates(
        merge.merged_search(
            query,
            "lexical",
            max(1, int(limit)),
            collections=(COLLECTION_METADATA,),
        )
    )


def search_knowledge_base(query: str, limit: int = 10) -> str:
    """legacy Python alias, not an MCP tool — vector по обеим коллекциям."""
    return format_candidates(
        merge.merged_search(query, "vector", max(1, int(limit)))
    )


def search_hybrid(query: str, limit: int = 10) -> str:
    """legacy Python alias, not an MCP tool — hybrid по обеим коллекциям."""
    return format_candidates(
        merge.merged_search(query, "hybrid", max(1, int(limit)))
    )


def main() -> None:
    """Entry point for stdio transport (Cursor mcp.json)."""
    import sys

    from mcp_1c_rag.startup import (
        RuntimeValidationError,
        format_validation_error,
        format_validation_warnings,
        validate_runtime,
    )

    report = validate_runtime()
    if report.skipped:
        mcp.run(transport="stdio")
        return
    if report.errors:
        print(
            format_validation_error(
                RuntimeValidationError(report.errors, report.warnings)
            ),
            file=sys.stderr,
        )
        raise SystemExit(2)
    if report.warnings:
        print(format_validation_warnings(report), file=sys.stderr)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
