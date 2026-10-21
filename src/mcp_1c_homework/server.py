"""MCP stdio server: codesearch, metadatasearch, evaluate_retrieval, search."""

from __future__ import annotations

from mcp.server.mcpserver import MCPServer

from mcp_1c_homework import eval_retrieval as eval_mod
from mcp_1c_homework import search as legacy_search
from mcp_1c_homework import search_modes
from mcp_1c_homework.index_guard import IndexBusy

mcp = MCPServer(
    "mcp-1c-homework",
    instructions=(
        "Own homework MCP for 1C: semantic + lexical search over BSL and metadata. "
        "Tools: codesearch, metadatasearch (cosine baseline), "
        "search(query, top_k, mode, collection) with modes vector | hybrid (RRF) "
        "| hybrid_rerank | pearson, and evaluate_retrieval for quality metrics "
        "(Hit@k / MRR / nDCG / null-tail). Measured best mode: hybrid."
    ),
)


@mcp.tool()
def codesearch(query: str, limit: int = 5) -> str:
    """Search 1C BSL code in object/form/common modules.

    Args:
        query: Code fragment, procedure name, comment, or description to find.
        limit: Max results (default 5). Raise to see more; lower to save context.
    """
    return legacy_search.codesearch(query=query, limit=limit)


@mcp.tool()
def metadatasearch(query: str, limit: int = 20) -> str:
    """Search 1C metadata objects, fields and types in the configuration report.

    Args:
        query: Object name/path (e.g. Справочники.Номенклатура) or free-text description.
        limit: Max results (default 20). Raise for wide objects; lower to save context.
    """
    return legacy_search.metadatasearch(query=query, limit=limit)


@mcp.tool()
def evaluate_retrieval(
    query: str,
    k: int = 5,
    collection: str = "code",
    relevant_sources: str = "",
    null_pairs: int = 40,
) -> str:
    """Evaluate retrieval quality for one query (homework highlight).

    Example: evaluate_retrieval(query=\"НДС\")
    Returns Hit@k, Recall@k, MRR, top cosine, optional null-tail vs random pairs.

    Args:
        query: Query to evaluate, e.g. НДС or ПередЗаписью.
        k: Top-k cutoff (default 5).
        collection: code (BSL) or metadata (configuration report).
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
def search(
    query: str,
    top_k: int = 5,
    mode: str = "",
    collection: str = "code",
) -> str:
    """Unified retrieval: vector | hybrid (RRF) | hybrid_rerank | pearson (baseline).

    Pipeline: e5 cosine top-N + global BM25 top-N -> RRF -> optional cross-encoder
    rerank (flashrank, ms-marco-MultiBERT-L-12). Default mode is ``hybrid``:
    it wins on nDCG@10/MRR@10 in eval and is ~35x faster than rerank.

    Args:
        query: Natural-language or exact-name query (e.g. "НДС с аванса").
        top_k: How many results to return (default 5).
        mode: vector | hybrid | hybrid_rerank | pearson; empty = server default.
        collection: code (BSL) or metadata (configuration report).
    """
    try:
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
    return search_modes.format_candidates(candidates)


def main() -> None:
    """Entry point for stdio transport (Cursor mcp.json)."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
