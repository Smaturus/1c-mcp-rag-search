"""Индекс-манифест: counts/fingerprint коллекций и BM25 без импорта server.

Startup сравнивает быстрый путь (counts + model + bm25 meta).
Полный fingerprint — collect/build_manifest и eval_suite (ADR 0012).
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mcp_1c_rag.startup import ValidationIssue

SCHEMA_VERSION = 1
_ENV_REVISION = "MCP_1C_EMBEDDING_REVISION"


@dataclass
class CollectionManifest:
    """Снимок одной Chroma-коллекции."""

    count: int
    id_prefix: str
    fingerprint: str


@dataclass
class Bm25Manifest:
    """Снимок глобального BM25."""

    n_docs: int
    n_terms: int
    k1: float
    b: float
    fingerprint: str


@dataclass
class IndexManifest:
    """Машиночитаемое описание обслуживаемого индекса."""

    schema_version: int
    created_at: str
    git_commit: str | None
    chroma_dir: str
    bm25_dir: str
    embedding_model: str
    embedding_model_revision: str | None
    collections: dict[str, CollectionManifest]
    bm25: Bm25Manifest
    chunker: dict[str, int]
    hnsw: dict[str, Any]
    python: str
    chromadb: str | None = None
    sentence_transformers: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON-совместимый dict."""
        data = asdict(self)
        extra = data.pop("extra", {}) or {}
        data.update(extra)
        return data

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> IndexManifest:
        """Собрать dataclass из JSON."""
        collections = {
            name: CollectionManifest(**vals)
            for name, vals in (raw.get("collections") or {}).items()
        }
        bm25_raw = raw.get("bm25") or {}
        known = {
            "schema_version",
            "created_at",
            "git_commit",
            "chroma_dir",
            "bm25_dir",
            "embedding_model",
            "embedding_model_revision",
            "collections",
            "bm25",
            "chunker",
            "hnsw",
            "python",
            "chromadb",
            "sentence_transformers",
        }
        return cls(
            schema_version=int(raw.get("schema_version", SCHEMA_VERSION)),
            created_at=str(raw.get("created_at", "")),
            git_commit=raw.get("git_commit"),
            chroma_dir=str(raw.get("chroma_dir", "")),
            bm25_dir=str(raw.get("bm25_dir", "")),
            embedding_model=str(raw.get("embedding_model", "")),
            embedding_model_revision=raw.get("embedding_model_revision"),
            collections=collections,
            bm25=Bm25Manifest(
                n_docs=int(bm25_raw.get("n_docs", 0)),
                n_terms=int(bm25_raw.get("n_terms", 0)),
                k1=float(bm25_raw.get("k1", 1.5)),
                b=float(bm25_raw.get("b", 0.75)),
                fingerprint=str(bm25_raw.get("fingerprint", "")),
            ),
            chunker=dict(raw.get("chunker") or {}),
            hnsw=dict(raw.get("hnsw") or {}),
            python=str(raw.get("python", "")),
            chromadb=raw.get("chromadb"),
            sentence_transformers=raw.get("sentence_transformers"),
            extra={k: v for k, v in raw.items() if k not in known},
        )


def _sha256_sorted_ids(ids: list[str]) -> str:
    payload = "\n".join(sorted(ids)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _open_chroma_ro(chroma_dir: Path) -> sqlite3.Connection:
    db_path = Path(chroma_dir) / "chroma.sqlite3"
    if not db_path.is_file():
        raise FileNotFoundError(f"нет chroma.sqlite3: {db_path}")
    return sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)


def list_collection_counts(chroma_dir: Path) -> dict[str, int]:
    """Быстрый COUNT(*) по коллекциям (sqlite RO)."""
    conn = _open_chroma_ro(chroma_dir)
    try:
        rows = conn.execute(
            """
            SELECT c.name, COUNT(e.id)
            FROM collections c
            LEFT JOIN segments s
              ON s.collection = c.id
             AND s.type LIKE '%metadata/sqlite%'
            LEFT JOIN embeddings e ON e.segment_id = s.id
            GROUP BY c.name
            """
        ).fetchall()
    except sqlite3.Error as exc:
        raise FileNotFoundError(f"chroma sqlite unreadable: {exc}") from exc
    finally:
        conn.close()
    return {str(name): int(count) for name, count in rows}


def collection_fingerprint(chroma_dir: Path, collection: str) -> tuple[int, str]:
    """(count, sha256) по отсортированным embedding_id коллекции."""
    conn = _open_chroma_ro(chroma_dir)
    try:
        ids = [
            str(row[0])
            for row in conn.execute(
                """
                SELECT e.embedding_id
                FROM embeddings e
                JOIN segments s ON s.id = e.segment_id
                JOIN collections c ON c.id = s.collection
                WHERE c.name = ?
                  AND s.type LIKE '%metadata/sqlite%'
                ORDER BY e.embedding_id
                """,
                (collection,),
            )
        ]
    finally:
        conn.close()
    if not ids:
        return 0, _sha256_sorted_ids([])
    return len(ids), _sha256_sorted_ids(ids)


def bm25_fingerprint(bm25_dir: Path) -> tuple[dict[str, Any], str]:
    """Прочитать meta.json и fingerprint doc_ids."""
    meta_path = Path(bm25_dir) / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    doc_ids = [str(x) for x in meta.get("doc_ids") or []]
    vocab = meta.get("vocab") or {}
    info = {
        "n_docs": int(meta.get("n_docs") or len(doc_ids)),
        "n_terms": int(len(vocab)),
        "k1": float(meta.get("k1", 1.5)),
        "b": float(meta.get("b", 0.75)),
    }
    return info, _sha256_sorted_ids(doc_ids)


def _git_commit() -> str | None:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return (proc.stdout or "").strip() or None


def _pkg_version(name: str) -> str | None:
    try:
        mod = __import__(name)
    except ImportError:
        return None
    return getattr(mod, "__version__", None)


def collect_manifest(
    chroma_dir: Path,
    bm25_dir: Path,
    collections: list[str] | None = None,
    embedding_model: str | None = None,
) -> dict[str, Any]:
    """Собрать полный манифест (включая fingerprints) с диска."""
    from mcp_1c_rag.collections import all_specs, id_prefix
    from mcp_1c_rag.config import (
        BSL_CHUNK_CHARS,
        BSL_CHUNK_OVERLAP,
        EMBEDDING_MODEL,
        HNSW_SYNC_THRESHOLD,
        META_CHUNK_CHARS,
        META_CHUNK_OVERLAP,
    )

    chroma_path = Path(chroma_dir)
    bm25_path = Path(bm25_dir)
    model = embedding_model or EMBEDDING_MODEL
    names = collections or [spec.name for spec in all_specs()]
    present = set(list_collection_counts(chroma_path))
    coll_map: dict[str, CollectionManifest] = {}
    for name in names:
        if name not in present:
            continue
        count, fp = collection_fingerprint(chroma_path, name)
        coll_map[name] = CollectionManifest(
            count=count,
            id_prefix=id_prefix(name),
            fingerprint=fp,
        )
    bm25_info, bm25_fp = bm25_fingerprint(bm25_path)
    manifest = IndexManifest(
        schema_version=SCHEMA_VERSION,
        created_at=datetime.now(UTC).isoformat(),
        git_commit=_git_commit(),
        chroma_dir=str(chroma_path),
        bm25_dir=str(bm25_path),
        embedding_model=model,
        embedding_model_revision=os.environ.get(_ENV_REVISION) or None,
        collections=coll_map,
        bm25=Bm25Manifest(
            n_docs=int(bm25_info["n_docs"]),
            n_terms=int(bm25_info["n_terms"]),
            k1=float(bm25_info["k1"]),
            b=float(bm25_info["b"]),
            fingerprint=bm25_fp,
        ),
        chunker={
            "bsl_chunk_chars": BSL_CHUNK_CHARS,
            "bsl_chunk_overlap": BSL_CHUNK_OVERLAP,
            "meta_chunk_chars": META_CHUNK_CHARS,
            "meta_chunk_overlap": META_CHUNK_OVERLAP,
        },
        hnsw={"space": "cosine", "sync_threshold": HNSW_SYNC_THRESHOLD},
        python=sys.version.split()[0],
        chromadb=_pkg_version("chromadb"),
        sentence_transformers=_pkg_version("sentence_transformers"),
    )
    return manifest.to_dict()


def write_manifest(path: Path, manifest: IndexManifest | dict[str, Any]) -> None:
    """Записать манифест в JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = manifest.to_dict() if isinstance(manifest, IndexManifest) else dict(manifest)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_manifest(path: Path) -> IndexManifest:
    """Прочитать манифест с диска."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"манифест должен быть JSON-объектом: {path}")
    return IndexManifest.from_dict(raw)


def _err(
    code: str,
    message: str,
    *,
    path: Path | str | None = None,
    actual: Any = None,
    hint: str | None = None,
) -> ValidationIssue:
    return ValidationIssue(
        level="error",
        code=code,
        message=message,
        path=str(path) if path is not None else None,
        actual=None if actual is None else str(actual),
        hint=hint,
    )


def validate_manifest(
    manifest: IndexManifest,
    *,
    chroma_dir: Path,
    bm25_dir: Path,
    embedding_model: str,
    check_fingerprints: bool = False,
) -> list[ValidationIssue]:
    """Сверить манифест с диском.

    Быстрый путь (startup): counts + embedding_model + BM25 n_docs/k1/b.
    Полный путь: + fingerprints коллекций и BM25.
    """
    issues: list[ValidationIssue] = []
    chroma_path = Path(chroma_dir)
    bm25_path = Path(bm25_dir)

    if manifest.embedding_model != embedding_model:
        issues.append(
            _err(
                "manifest_embedding_mismatch",
                "embedding_model в манифесте не совпадает с runtime",
                path=chroma_path,
                actual=f"manifest={manifest.embedding_model!r} runtime={embedding_model!r}",
                hint="пересоберите манифест: python scripts/build_manifest.py",
            )
        )

    try:
        live_counts = list_collection_counts(chroma_path)
    except FileNotFoundError as exc:
        issues.append(
            _err(
                "manifest_chroma_unreadable",
                "не удалось открыть Chroma для сверки манифеста",
                path=chroma_path,
                actual=str(exc),
                hint="проверьте MCP_1C_CHROMA_DIR",
            )
        )
        return issues

    for name, expected in manifest.collections.items():
        actual = live_counts.get(name)
        if actual is None:
            issues.append(
                _err(
                    "manifest_collection_missing",
                    f"коллекция {name!r} из манифеста отсутствует на диске",
                    path=chroma_path,
                    actual="missing",
                    hint="пересоберите индекс или манифест",
                )
            )
            continue
        if int(actual) != int(expected.count):
            issues.append(
                _err(
                    "manifest_count_mismatch",
                    f"count коллекции {name!r} не совпадает с манифестом",
                    path=chroma_path,
                    actual=f"manifest={expected.count} live={actual}",
                    hint="python scripts/build_manifest.py после пересборки индекса",
                )
            )
        elif check_fingerprints:
            _count, live_fp = collection_fingerprint(chroma_path, name)
            if live_fp != expected.fingerprint:
                issues.append(
                    _err(
                        "manifest_fingerprint_mismatch",
                        f"fingerprint коллекции {name!r} не совпадает",
                        path=chroma_path,
                        actual=(
                            f"manifest={expected.fingerprint[:12]}… "
                            f"live={live_fp[:12]}…"
                        ),
                        hint="пересоберите манифест",
                    )
                )

    try:
        bm25_info, live_bm25_fp = bm25_fingerprint(bm25_path)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        issues.append(
            _err(
                "manifest_bm25_unreadable",
                "не удалось прочитать BM25 meta для сверки манифеста",
                path=bm25_path,
                actual=str(exc),
                hint="проверьте MCP_1C_BM25_DIR",
            )
        )
        return issues

    if int(bm25_info["n_docs"]) != int(manifest.bm25.n_docs):
        issues.append(
            _err(
                "manifest_bm25_n_docs_mismatch",
                "BM25 n_docs не совпадает с манифестом",
                path=bm25_path / "meta.json",
                actual=f"manifest={manifest.bm25.n_docs} live={bm25_info['n_docs']}",
                hint="пересоберите BM25 и манифест",
            )
        )
    if abs(float(bm25_info["k1"]) - float(manifest.bm25.k1)) > 1e-9:
        issues.append(
            _err(
                "manifest_bm25_k1_mismatch",
                "BM25 k1 не совпадает с манифестом",
                path=bm25_path / "meta.json",
                actual=f"manifest={manifest.bm25.k1} live={bm25_info['k1']}",
                hint="проверьте MCP_1C_BM25_K1 / пересоберите манифест",
            )
        )
    if abs(float(bm25_info["b"]) - float(manifest.bm25.b)) > 1e-9:
        issues.append(
            _err(
                "manifest_bm25_b_mismatch",
                "BM25 b не совпадает с манифестом",
                path=bm25_path / "meta.json",
                actual=f"manifest={manifest.bm25.b} live={bm25_info['b']}",
                hint="проверьте MCP_1C_BM25_B / пересоберите манифест",
            )
        )
    if check_fingerprints and live_bm25_fp != manifest.bm25.fingerprint:
        issues.append(
            _err(
                "manifest_bm25_fingerprint_mismatch",
                "BM25 fingerprint не совпадает с манифестом",
                path=bm25_path / "meta.json",
                actual=(
                    f"manifest={manifest.bm25.fingerprint[:12]}… "
                    f"live={live_bm25_fp[:12]}…"
                ),
                hint="пересоберите манифест",
            )
        )
    return issues


__all__ = [
    "Bm25Manifest",
    "CollectionManifest",
    "IndexManifest",
    "SCHEMA_VERSION",
    "bm25_fingerprint",
    "collect_manifest",
    "collection_fingerprint",
    "list_collection_counts",
    "load_manifest",
    "validate_manifest",
    "write_manifest",
]
