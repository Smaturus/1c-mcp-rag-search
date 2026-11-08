"""Текстовое представление кандидатов поиска (presentation-слой).

Только ``Candidate`` и сборка URI (``uris``, stdlib). Без Chroma, BM25 и MCP SDK.
"""

from __future__ import annotations

from mcp_1c_rag import uris
from mcp_1c_rag.models import Candidate


def format_candidates(candidates: list[Candidate]) -> str:
    """Текстовый вывод кандидатов для MCP-тула."""
    if not candidates:
        return "Ничего не найдено. Проверьте индекс и BM25-индекс (scripts/build_bm25.py)."
    blocks: list[str] = []
    for i, c in enumerate(candidates, start=1):
        meta = c.metadata or {}
        header = (
            f"[{i}] source={meta.get('source', '?')} "
            f"chunk={meta.get('chunk_index', '?')} score={c.score:.4f}"
        )
        extra = {k: round(v, 4) for k, v in c.source_scores.items() if k != "score"}
        if extra:
            header += " " + " ".join(f"{k}={v}" for k, v in extra.items())
        uri = (
            uris.metadata_uri(meta)
            if c.id.startswith("metaobj:")
            else uris.bsl_uri(str(meta.get("source", "")))
        )
        if uri:
            header += f"\nuri={uri}"
        blocks.append(header + "\n" + (c.text or ""))
    return "\n\n---\n\n".join(blocks)


__all__ = ["format_candidates"]
