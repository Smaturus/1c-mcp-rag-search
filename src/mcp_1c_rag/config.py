"""Paths and runtime settings for indexing and MCP search."""

from __future__ import annotations

import os
from pathlib import Path

# Project root: D:\MCP_1С_homework
PROJECT_ROOT = Path(__file__).resolve().parents[2]

STUDY_ROOT = Path(os.environ.get("MCP_1C_STUDY_ROOT", r"D:\Study"))
STUDY_REPORT_ROOT = Path(
    os.environ.get("MCP_1C_STUDY_REPORT_ROOT", r"D:\Study_Report")
)
# Optional extra metadata: *.xml from a configuration dump. Empty means off.
# Set only in the process environment. Do not store a personal path in the repo.
_xml_root = os.environ.get("MCP_1C_METADATA_XML_ROOT", "").strip()
METADATA_XML_ROOT = Path(_xml_root) if _xml_root else None

DATA_DIR = Path(os.environ.get("MCP_1C_DATA_DIR", PROJECT_ROOT / "data"))
CHROMA_DIR = Path(os.environ.get("MCP_1C_CHROMA_DIR", DATA_DIR / "chroma"))

EMBEDDING_MODEL = os.environ.get(
    "MCP_1C_EMBEDDING_MODEL", "intfloat/multilingual-e5-small"
)

COLLECTION_CODE = "code"
COLLECTION_METADATA = "metadata"

# Chunking defaults
BSL_CHUNK_CHARS = int(os.environ.get("MCP_1C_BSL_CHUNK_CHARS", "1200"))
BSL_CHUNK_OVERLAP = int(os.environ.get("MCP_1C_BSL_CHUNK_OVERLAP", "200"))
META_CHUNK_CHARS = int(os.environ.get("MCP_1C_META_CHUNK_CHARS", "1500"))
META_CHUNK_OVERLAP = int(os.environ.get("MCP_1C_META_CHUNK_OVERLAP", "200"))

# HNSW persistence workaround (chroma#7090): a very large sync threshold keeps
# the index out of the on-disk pickle, so a restart rebuilds it from the WAL
# instead of failing on a corrupted segment. 0 = library default.
HNSW_SYNC_THRESHOLD = int(os.environ.get("MCP_1C_HNSW_SYNC_THRESHOLD", "0"))

# Indexing throughput (keep modest on 16 GB RAM + CPU-only)
EMBED_BATCH_SIZE = int(os.environ.get("MCP_1C_EMBED_BATCH_SIZE", "8"))
CHROMA_ADD_BATCH = int(os.environ.get("MCP_1C_CHROMA_ADD_BATCH", "64"))

# Hybrid rerank (off by default): MCP_1C_HYBRID=1
# s = w_cos * cosine_n + w_bm25 * bm25_n + w_pearson * pearson_n  (weights normalized)
HYBRID_ENABLED = os.environ.get("MCP_1C_HYBRID", "0").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
# Legacy: if only ALPHA set, cosine=ALPHA and bm25=1-ALPHA (pearson=0 unless set)
HYBRID_ALPHA = float(os.environ.get("MCP_1C_HYBRID_ALPHA", "0.5"))
HYBRID_BM25_WEIGHT = float(os.environ.get("MCP_1C_HYBRID_BM25_WEIGHT", "0.3"))
HYBRID_PEARSON_WEIGHT = float(os.environ.get("MCP_1C_HYBRID_PEARSON_WEIGHT", "0.2"))
# Fetch this many cosine candidates before rerank (multiple of limit)
HYBRID_CANDIDATE_MULT = int(os.environ.get("MCP_1C_HYBRID_CANDIDATE_MULT", "4"))

# --- Lexical / hybrid retrieval (BM25 + RRF) -----------------------------
# Global BM25 index over all chunks (built by scripts/build_bm25.py).
BM25_DIR = Path(os.environ.get("MCP_1C_BM25_DIR", DATA_DIR / "bm25"))
BM25_K1 = float(os.environ.get("MCP_1C_BM25_K1", "1.5"))
BM25_B = float(os.environ.get("MCP_1C_BM25_B", "0.75"))
# RRF smoothing constant and how many candidates each source contributes.
RRF_K = int(os.environ.get("MCP_1C_RRF_K", "60"))
HYBRID_TOP_K = int(os.environ.get("MCP_1C_HYBRID_TOP_K", "50"))
# Default mode of the unified search tool: vector | hybrid | hybrid_rerank | pearson.
# Chosen by measurement: hybrid RRF wins on nDCG@10/MRR@10 and is ~35x faster
# than the cross-encoder reranker (see data/eval/report_pipeline.md).
SEARCH_DEFAULT_MODE = os.environ.get("MCP_1C_SEARCH_MODE", "hybrid").strip().lower()

# --- Cross-encoder reranker (second stage, flashrank / ONNX) --------------
RERANK_MODEL = os.environ.get("MCP_1C_RERANK_MODEL", "ms-marco-MultiBERT-L-12")
# 128 tokens keeps CPU latency ~0.5 s per query for 20 candidates.
RERANK_MAX_LENGTH = int(os.environ.get("MCP_1C_RERANK_MAX_LENGTH", "128"))
# How many RRF candidates to feed into the reranker (plan: keep <= 50).
RERANK_CANDIDATES = int(os.environ.get("MCP_1C_RERANK_CANDIDATES", "20"))
