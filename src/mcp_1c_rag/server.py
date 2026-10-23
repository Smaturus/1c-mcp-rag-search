"""MCP stdio server: codesearch, metadatasearch, evaluate_retrieval, search."""

from __future__ import annotations

from mcp.server.mcpserver import MCPServer

from mcp_1c_rag import eval_retrieval as eval_mod
from mcp_1c_rag import search as legacy_search
from mcp_1c_rag import search_modes
from mcp_1c_rag.index_guard import IndexBusy
from mcp_1c_rag.models import Candidate
from mcp_1c_rag.router import Route
from mcp_1c_rag.router import route_query as route_decision

mcp = MCPServer(
    "1c-mcp-rag-search",
    instructions=(
        "1C retrieval MCP: semantic + lexical search over BSL code and "
        "configuration metadata. "
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
    """Evaluate retrieval quality for one query (quality metrics).

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
    it wins on nDCG@10/MRR@10 in eval and is ~26x faster than rerank.

    Args:
        query: Natural-language or exact-name query (e.g. "НДС с аванса").
        top_k: How many results to return (default 5).
        mode: lexical | vector | hybrid | hybrid_rerank | pearson;
            empty = server default.
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


_MODE_BY_ROUTE = {
    Route.LEXICAL: "lexical",
    Route.VECTOR: "vector",
    Route.HYBRID: "hybrid",
}


def _merged_search(query: str, mode: str, limit: int) -> str:
    """Поиск по обеим коллекциям с нормализацией скоров и объединением."""
    found: list[tuple[float, Candidate]] = []
    for collection in ("code", "metadata"):
        try:
            candidates = search_modes.search(
                query=query, top_k=limit, mode=mode, collection=collection
            )
        except FileNotFoundError:
            continue
        if not candidates:
            continue
        top = max(candidate.score for candidate in candidates) or 1.0
        for candidate in candidates:
            candidate.metadata["collection"] = collection
            found.append((candidate.score / top, candidate))
    found.sort(key=lambda pair: pair[0], reverse=True)
    seen: set[str] = set()
    unique = []
    for _, candidate in found:
        if candidate.id in seen:
            continue
        seen.add(candidate.id)
        unique.append(candidate)
    return search_modes.format_candidates(unique[:limit])


@mcp.tool()
def search_1c_metadata(query: str, limit: int = 10) -> str:
    """Grep-like lexical search for exact 1C names, handlers and relations.

    Use for exact object/procedure names (РеализацияТоваровУслуг, ПередЗаписью),
    questions about "движения", "проводки", "кто пишет в регистр", "на основании",
    "связанные документы". Backed by the global BM25 index over both collections,
    so it ranks matches instead of dumping raw grep lines.

    Args:
        query: Exact name or a question with exact terms.
        limit: Max results (default 10).
    """
    return _merged_search(query, "lexical", max(1, int(limit)))


@mcp.tool()
def search_knowledge_base(query: str, limit: int = 10) -> str:
    """Semantic (vector) search for "how to" and troubleshooting questions.

    Use for how-to, instructions, "почему не работает", best practices, experience.
    Backed by e5-small cosine over both collections.

    Args:
        query: Natural-language question.
        limit: Max results (default 10).
    """
    return _merged_search(query, "vector", max(1, int(limit)))


@mcp.tool()
def search_hybrid(query: str, limit: int = 10) -> str:
    """Hybrid search (cosine + BM25 via RRF) for mixed or requirement queries.

    Use when the question mixes an exact term with meaning, or asks for current
    requirements (Confluence / ТЗ / регламент). This is also the safe default.

    Args:
        query: Mixed natural-language and exact-term question.
        limit: Max results (default 10).
    """
    return _merged_search(query, "hybrid", max(1, int(limit)))


@mcp.tool()
def route_query(query: str) -> str:
    """Explain which search tool to use for a query, and why.

    Applies the routing contract: lexical for exact names/relations, vector for
    how-to questions, hybrid for mixed or requirement queries (and for anything
    unclear). Returns route, tool, reason and confidence.

    Args:
        query: The user question to classify.
    """
    decision = route_decision(query)
    return (
        f"route={decision.route.value}\n"
        f"tool={decision.tool}\n"
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
        from mcp_1c_rag.store import get_client

        client = get_client()
        for name in ("code_proc", "metadata_xml", "code", "metadata"):
            try:
                collection = client.get_collection(name)
                lines.append(f"collection[{name}]={collection.count()}")
            except Exception:
                continue
    except Exception as exc:  # noqa: BLE001
        lines.append(f"chroma_error={type(exc).__name__}: {exc}")
    bm25_file = Path(BM25_DIR) / "index.npz"
    lines.append(f"bm25_index_present={bm25_file.exists()}")
    return "\n".join(lines)


def main() -> None:
    """Entry point for stdio transport (Cursor mcp.json)."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
