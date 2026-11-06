"""Тесты индекс-манифеста без ML и без server."""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mcp_1c_rag.manifest import (  # noqa: E402
    IndexManifest,
    bm25_fingerprint,
    collect_manifest,
    collection_fingerprint,
    list_collection_counts,
    load_manifest,
    validate_manifest,
    write_manifest,
)


def _write_bm25(bm25_dir: Path, doc_ids: list[str], *, k1: float = 1.5, b: float = 0.75) -> None:
    bm25_dir.mkdir(parents=True, exist_ok=True)
    (bm25_dir / "index.npz").write_bytes(b"NPZ")
    (bm25_dir / "meta.json").write_text(
        json.dumps(
            {
                "k1": k1,
                "b": b,
                "n_docs": len(doc_ids),
                "doc_ids": doc_ids,
                "vocab": {"ндс": 0, "аванс": 1},
            }
        ),
        encoding="utf-8",
    )


def _write_chroma(chroma_dir: Path, collections: dict[str, list[str]]) -> None:
    chroma_dir.mkdir(parents=True, exist_ok=True)
    db_path = chroma_dir / "chroma.sqlite3"
    if db_path.exists():
        db_path.unlink()
    db = sqlite3.connect(db_path)
    db.executescript(
        """
        CREATE TABLE collections (id TEXT PRIMARY KEY, name TEXT, dimension INTEGER,
            database TEXT, config_json_str TEXT);
        CREATE TABLE segments (id TEXT PRIMARY KEY, type TEXT NOT NULL, scope TEXT,
            collection TEXT NOT NULL);
        CREATE TABLE embeddings (
            id INTEGER PRIMARY KEY,
            segment_id TEXT NOT NULL,
            embedding_id TEXT NOT NULL,
            seq_id BLOB NOT NULL DEFAULT X'00',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
    )
    eid = 1
    for name, ids in collections.items():
        cid = f"c-{name}"
        sid = f"s-{name}"
        db.execute(
            "INSERT INTO collections (id, name) VALUES (?, ?)",
            (cid, name),
        )
        db.execute(
            "INSERT INTO segments (id, type, scope, collection) VALUES (?, ?, ?, ?)",
            (sid, "urn:chroma:segment/metadata/sqlite", "VECTOR", cid),
        )
        for emb_id in ids:
            db.execute(
                "INSERT INTO embeddings (id, segment_id, embedding_id) VALUES (?, ?, ?)",
                (eid, sid, emb_id),
            )
            eid += 1
    db.commit()
    db.close()


def test_collect_write_load_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    chroma = tmp_path / "chroma"
    bm25 = tmp_path / "bm25"
    _write_chroma(
        chroma,
        {
            "code_proc": ["bsl_proc:1", "bsl_proc:2"],
            "metadata_xml": ["metaobj:a"],
        },
    )
    _write_bm25(bm25, ["bsl_proc:1", "bsl_proc:2", "metaobj:a"])
    monkeypatch.setenv("MCP_1C_EMBEDDING_MODEL", "intfloat/multilingual-e5-small")
    data = collect_manifest(
        chroma,
        bm25,
        collections=["code_proc", "metadata_xml"],
        embedding_model="intfloat/multilingual-e5-small",
    )
    out = tmp_path / "manifest.json"
    write_manifest(out, data)
    loaded = load_manifest(out)
    assert loaded.collections["code_proc"].count == 2
    assert loaded.collections["metadata_xml"].count == 1
    assert loaded.bm25.n_docs == 3
    assert loaded.embedding_model == "intfloat/multilingual-e5-small"
    count, fp = collection_fingerprint(chroma, "code_proc")
    assert count == 2
    assert fp == loaded.collections["code_proc"].fingerprint


def test_validate_ok_and_mismatch_model_count(tmp_path: Path) -> None:
    chroma = tmp_path / "chroma"
    bm25 = tmp_path / "bm25"
    _write_chroma(chroma, {"code_proc": ["bsl_proc:1"]})
    _write_bm25(bm25, ["bsl_proc:1"])
    data = collect_manifest(
        chroma, bm25, collections=["code_proc"], embedding_model="model-a"
    )
    man = IndexManifest.from_dict(data)
    ok = validate_manifest(
        man, chroma_dir=chroma, bm25_dir=bm25, embedding_model="model-a"
    )
    assert ok == []

    bad_model = validate_manifest(
        man, chroma_dir=chroma, bm25_dir=bm25, embedding_model="other"
    )
    assert any(i.code == "manifest_embedding_mismatch" for i in bad_model)

    man.collections["code_proc"].count = 99
    bad_count = validate_manifest(
        man, chroma_dir=chroma, bm25_dir=bm25, embedding_model="model-a"
    )
    assert any(i.code == "manifest_count_mismatch" for i in bad_count)


def test_fingerprint_check_detects_id_change(tmp_path: Path) -> None:
    chroma = tmp_path / "chroma"
    bm25 = tmp_path / "bm25"
    _write_chroma(chroma, {"code_proc": ["bsl_proc:1", "bsl_proc:2"]})
    _write_bm25(bm25, ["bsl_proc:1", "bsl_proc:2"])
    man = IndexManifest.from_dict(
        collect_manifest(
            chroma, bm25, collections=["code_proc"], embedding_model="m"
        )
    )
    # mutate ids keeping count
    _write_chroma(chroma, {"code_proc": ["bsl_proc:1", "bsl_proc:9"]})
    issues = validate_manifest(
        man,
        chroma_dir=chroma,
        bm25_dir=bm25,
        embedding_model="m",
        check_fingerprints=True,
    )
    assert any(i.code == "manifest_fingerprint_mismatch" for i in issues)


def test_empty_collection_fingerprint(tmp_path: Path) -> None:
    chroma = tmp_path / "chroma"
    _write_chroma(chroma, {"code_proc": []})
    count, fp = collection_fingerprint(chroma, "code_proc")
    assert count == 0
    assert len(fp) == 64
    assert list_collection_counts(chroma)["code_proc"] == 0


def test_bm25_fingerprint_stable(tmp_path: Path) -> None:
    bm25 = tmp_path / "bm25"
    _write_bm25(bm25, ["b", "a"])
    info, fp1 = bm25_fingerprint(bm25)
    _, fp2 = bm25_fingerprint(bm25)
    assert info["n_docs"] == 2
    assert fp1 == fp2


def test_manifest_does_not_import_server() -> None:
    import mcp_1c_rag.manifest as mod

    src = Path(mod.__file__).read_text(encoding="utf-8")
    assert "mcp_1c_rag.server" not in src
    assert "from mcp_1c_rag import server" not in src
