"""MCP stdio server: search, evaluate_retrieval, route_query, index_stats."""

from __future__ import annotations

import logging
import time
from typing import Annotated, Literal

from mcp.server.mcpserver import MCPServer
from mcp_types import ToolAnnotations
from pydantic import Field

from mcp_1c_rag import eval_retrieval as eval_mod
from mcp_1c_rag import merge, search_modes
from mcp_1c_rag import observability as obs
from mcp_1c_rag import search as legacy_search
from mcp_1c_rag.config import (
    COLLECTION_CODE,
    COLLECTION_METADATA,
    SEARCH_DEFAULT_MODE,
)
from mcp_1c_rag.formatting import format_candidates
from mcp_1c_rag.index_guard import IndexBusy
from mcp_1c_rag.router import route_query as route_decision

# Empty string means "server default"; config_xml stays out until it is in the
# served Chroma directory and the global BM25 index.
SearchMode = Literal["", "lexical", "vector", "hybrid", "hybrid_rerank", "pearson"]
SearchCollection = Literal["", "code_proc", "metadata_xml", "all", "code", "metadata"]
EvalCollection = Literal["", "code_proc", "metadata_xml", "code", "metadata"]

# Legacy collection names in search() point to the served collections.
_LEGACY_SEARCH_COLLECTIONS = {
    "code": COLLECTION_CODE,
    "metadata": COLLECTION_METADATA,
}

_READ_ONLY_TOOL = ToolAnnotations(
    read_only_hint=True,
    destructive_hint=False,
    idempotent_hint=True,
    open_world_hint=False,
)

mcp = MCPServer(
    "1c-mcp-rag-search",
    instructions=(
        "1C retrieval MCP: four read-only tools — search, evaluate_retrieval, "
        "route_query, index_stats. "
        "search(query, top_k, mode, collection) covers BSL and metadata: "
        "collection empty/code_proc/code → BSL procedures (code_proc); "
        "metadata_xml/metadata → XML metadata objects (metadata_xml); "
        "all → both collections merged. "
        "Modes: lexical | vector | hybrid (default) | hybrid_rerank | pearson. "
        "route_query returns tool=search with mode/collection to call. "
        "Legacy Python aliases (codesearch, metadatasearch, search_1c_metadata, "
        "search_knowledge_base, search_hybrid) are not MCP tools."
    ),
)


@mcp.tool(annotations=_READ_ONLY_TOOL)
def search(
    query: Annotated[
        str, Field(description='Natural-language or exact-name query, e.g. "НДС с аванса".')
    ],
    top_k: Annotated[
        int, Field(ge=1, le=50, description="How many results to return.")
    ] = 5,
    mode: Annotated[
        SearchMode, Field(description="Retrieval mode; empty = server default (MCP_1C_SEARCH_MODE).")
    ] = "",
    collection: Annotated[
        SearchCollection,
        Field(
            description=(
                "code_proc/code = BSL procedures, metadata_xml/metadata = XML "
                "metadata objects, all = both merged; empty = code_proc."
            )
        ),
    ] = "",
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
    coll_label = key or "default"
    started = time.perf_counter()
    request_id = obs.new_request_id()
    privacy = obs.query_privacy_fields(query)
    ok = True
    error_type: str | None = None
    result_count = 0
    empty = False
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
                collection=_LEGACY_SEARCH_COLLECTIONS.get(key, collection),
            )
        result_count = len(candidates)
        empty = result_count == 0
        return format_candidates(candidates)
    except FileNotFoundError as exc:
        ok = False
        error_type = "FileNotFoundError"
        return (
            f"{exc}\n"
            "Подсказка: соберите BM25-индекс `python scripts/build_bm25.py` "
            "или используйте mode='vector'."
        )
    except IndexBusy as exc:
        ok = False
        error_type = "IndexBusy"
        return str(exc)
    except ValueError as exc:
        ok = False
        error_type = "ValueError"
        return f"{exc}"
    finally:
        obs.record_request(
            "search",
            latency_ms=obs.elapsed_ms(started),
            ok=ok,
            empty=empty,
            error_type=error_type,
            request_id=request_id,
            mode=chosen_mode,
            collection=coll_label,
            top_k=max(1, int(top_k)),
            result_count=result_count,
            **privacy,
        )


@mcp.tool(annotations=_READ_ONLY_TOOL)
def evaluate_retrieval(
    query: Annotated[str, Field(description="Query to evaluate, e.g. НДС or ПередЗаписью.")],
    k: Annotated[int, Field(ge=1, le=50, description="Top-k cutoff.")] = 5,
    collection: Annotated[
        EvalCollection,
        Field(
            description=(
                "code_proc/code = BSL procedures, metadata_xml/metadata = XML "
                "metadata objects; empty = code_proc."
            )
        ),
    ] = "",
    relevant_sources: Annotated[
        str,
        Field(description="Comma-separated path needles; empty = use query as needle."),
    ] = "",
    null_pairs: Annotated[
        int, Field(ge=0, description="Random background pairs for null-tail; 0 skips.")
    ] = 40,
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
    started = time.perf_counter()
    request_id = obs.new_request_id()
    privacy = obs.query_privacy_fields(query)
    ok = True
    error_type: str | None = None
    try:
        return eval_mod.evaluate_retrieval(
            query=query,
            k=k,
            collection=collection,
            relevant_sources=relevant_sources,
            null_pairs=null_pairs,
        )
    except Exception as exc:
        ok = False
        error_type = type(exc).__name__
        raise
    finally:
        obs.record_request(
            "evaluate_retrieval",
            latency_ms=obs.elapsed_ms(started),
            ok=ok,
            error_type=error_type,
            request_id=request_id,
            collection=(collection or "").strip().lower() or "default",
            top_k=max(1, int(k)),
            **privacy,
        )


@mcp.tool(annotations=_READ_ONLY_TOOL)
def route_query(
    query: Annotated[str, Field(description="The user question to classify.")],
) -> str:
    """Explain which search() call to use for a query, and why.

    Applies the routing contract: lexical for exact names/relations, vector for
    how-to questions, hybrid for mixed or requirement queries (and for anything
    unclear). Returns route, tool=search, mode, collection, reason, confidence.

    Args:
        query: The user question to classify.
    """
    started = time.perf_counter()
    request_id = obs.new_request_id()
    privacy = obs.query_privacy_fields(query)
    ok = True
    error_type: str | None = None
    route_name = ""
    mode_name = ""
    coll_name = ""
    try:
        decision = route_decision(query)
        route_name = decision.route.value
        mode_name = decision.mode
        coll_name = decision.collection
        return (
            f"route={decision.route.value}\n"
            f"tool={decision.tool}\n"
            f"mode={decision.mode}\n"
            f"collection={decision.collection}\n"
            f"confidence={decision.confidence}\n"
            f"reason={decision.reason}"
        )
    except Exception as exc:
        ok = False
        error_type = type(exc).__name__
        raise
    finally:
        obs.record_request(
            "route_query",
            latency_ms=obs.elapsed_ms(started),
            ok=ok,
            error_type=error_type,
            request_id=request_id,
            route=route_name or None,
            mode=mode_name or None,
            collection=coll_name or None,
            **privacy,
        )


@mcp.tool(annotations=_READ_ONLY_TOOL)
def index_stats() -> str:
    """Report which index the server is actually serving (paths, counts, model).

    Useful before a demo or when results look unexpected: confirms the Chroma
    directory, collection sizes, BM25 index and the default search mode.
    Also includes readiness status and in-memory request metrics (no query text).
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
    lines.extend(obs.metrics_lines())
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
    import os
    import sys

    # Must be set before huggingface_hub is imported (it reads the flag at import).
    # The model is downloaded by the indexing scripts; HF_HUB_OFFLINE=0 re-enables it here.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    from mcp_1c_rag.config import (
        COLLECTION_CODE,
        COLLECTION_METADATA,
        SEARCH_DEFAULT_MODE,
    )
    from mcp_1c_rag.startup import (
        RuntimeValidationError,
        format_validation_error,
        format_validation_warnings,
        validate_runtime,
    )

    obs.setup_logging()
    report = validate_runtime()
    obs.set_startup_report(report)
    obs.log_event(
        "startup",
        level=logging.INFO,
        mode=SEARCH_DEFAULT_MODE,
        collection_code=COLLECTION_CODE,
        collection_metadata=COLLECTION_METADATA,
        status=obs.startup_status(),
        startup_skipped=report.skipped,
        startup_errors=len(report.errors),
        startup_warnings=len(report.warnings),
    )
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
