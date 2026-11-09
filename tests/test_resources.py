"""Ресурсы bsl:// и metadata:// и URI в выдаче search."""

from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mcp.server.mcpserver.exceptions import ResourceError, ResourceNotFoundError  # noqa: E402
from mcp.shared.path_security import PathEscapeError  # noqa: E402

from mcp_1c_rag import resources, server  # noqa: E402
from mcp_1c_rag.formatting import format_candidates  # noqa: E402
from mcp_1c_rag.models import Candidate  # noqa: E402


def test_search_output_carries_bsl_and_metadata_uris() -> None:
    text = format_candidates(
        [
            Candidate(
                id="bsl_proc:1",
                text="код",
                metadata={"source": "Documents/A/Ext/ObjectModule.bsl", "chunk_index": 0},
                score=0.5,
            ),
            Candidate(
                id="metaobj:1",
                text="карточка",
                metadata={"kind_tag": "Document", "name": "Реализация", "source": "x"},
                score=0.4,
            ),
        ]
    )
    assert "uri=bsl://Documents/A/Ext/ObjectModule.bsl" in text
    assert "uri=metadata://Document/%D0%A0%D0%B5%D0%B0%D0%BB%D0%B8%D0%B7%D0%B0%D1%86%D0%B8%D1%8F" in text


def test_no_uri_for_unknown_chunk() -> None:
    text = format_candidates(
        [Candidate(id="x", text="t", metadata={"source": "s"}, score=1.0)]
    )
    assert "uri=" not in text


def test_bsl_uri_rejects_traversal() -> None:
    assert resources.bsl_uri("../secret.bsl") is None
    assert resources.bsl_uri("notes.txt") is None


def test_read_bsl_module(tmp_path: Path) -> None:
    module = tmp_path / "Documents" / "A" / "Ext"
    module.mkdir(parents=True)
    (module / "ObjectModule.bsl").write_text("Процедура П()\nКонецПроцедуры", encoding="utf-8")
    assert "Процедура П" in resources.read_bsl_module(
        "Documents/A/Ext/ObjectModule.bsl", root=tmp_path
    )


def test_read_bsl_rejects_symlink_escape(tmp_path: Path) -> None:
    """Ссылка внутри корня ведёт наружу: ResourceError, файл не прочитан.

    На Windows без права на symlink тот же обход проверяется junction:
    ``Path.resolve`` внутри ``safe_join`` следует и за тем, и за другим.
    """
    outside = tmp_path.parent / "secret_via_symlink.bsl"
    outside.write_text("SECRET", encoding="utf-8")
    link = tmp_path / "leak.bsl"
    rel = "leak.bsl"
    try:
        link.symlink_to(outside)
    except OSError:
        outside.unlink()
        outside_dir = tmp_path.parent / "secret_via_junction"
        outside_dir.mkdir()
        outside = outside_dir / "leak.bsl"
        outside.write_text("SECRET", encoding="utf-8")
        link = tmp_path / "alias"
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(outside_dir)],
            check=True,
            capture_output=True,
        )
        rel = "alias/leak.bsl"
    with pytest.raises(ResourceError, match="вне каталога"):
        resources.read_bsl_module(rel, root=tmp_path)
    assert outside.read_text(encoding="utf-8") == "SECRET"
    outside.unlink()


def test_read_bsl_rejects_escape_and_limits(tmp_path: Path) -> None:
    (tmp_path / "a.bsl").write_text("x" * 50, encoding="utf-8")
    outside = tmp_path.parent / "outside.bsl"
    with pytest.raises(ResourceError, match="вне каталога"):
        resources.read_bsl_module("../outside.bsl", root=tmp_path)
    with pytest.raises(ResourceError, match="только файлы .bsl"):
        resources.read_bsl_module("readme.md", root=tmp_path)
    with pytest.raises(ResourceError, match="не найден"):
        resources.read_bsl_module("missing.bsl", root=tmp_path)
    with pytest.raises(ResourceError, match="больше лимита"):
        resources.read_bsl_module("a.bsl", root=tmp_path, max_bytes=10)
    assert not outside.exists()


def test_safe_join_blocks_encoded_traversal(tmp_path: Path) -> None:
    with pytest.raises(PathEscapeError):
        from mcp.shared.path_security import safe_join

        safe_join(tmp_path, "..", "secret.bsl")


def test_read_metadata_card(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        resources,
        "_fetch_metadata_card",
        lambda kind, name: [f"{kind}/{name}"] if name == "Есть" else [],
    )
    assert resources.read_metadata_card("Document", "Есть") == "Document/Есть"
    with pytest.raises(ResourceError, match="не найден"):
        resources.read_metadata_card("Document", "Нет")


def test_resource_templates_and_static_resources() -> None:
    templates = asyncio.run(server.mcp.list_resource_templates())
    assert sorted(t.uri_template for t in templates) == [
        "bsl://{+path}",
        "metadata://{kind}/{name}",
    ]
    static = asyncio.run(server.mcp.list_resources())
    assert sorted(str(item.uri) for item in static) == [
        "index://manifest",
        "index://stats",
    ]


def test_traversal_uris_do_not_read_outside(tmp_path: Path) -> None:
    """%2e%2e, .. и абсолютный путь — ResourceNotFoundError, файл не прочитан."""
    secret = tmp_path.parent / "secret.bsl"
    secret.write_text("SECRET", encoding="utf-8")
    for uri in (
        "bsl://%2e%2e/secret.bsl",
        "bsl://../secret.bsl",
        "bsl:///C:/Windows/notepad.exe",
    ):
        with pytest.raises(ResourceNotFoundError, match="Unknown resource"):
            asyncio.run(server.mcp.read_resource(uri))
    assert secret.read_text(encoding="utf-8") == "SECRET"
    secret.unlink()


def test_metadata_card_maps_index_busy_and_missing_db(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mcp_1c_rag.index_guard import IndexBusy

    monkeypatch.setattr(
        resources, "_fetch_metadata_card", lambda _k, _n: (_ for _ in ()).throw(IndexBusy("lock"))
    )
    with pytest.raises(ResourceError, match="индекс занят"):
        resources.read_metadata_card("Document", "X")
    monkeypatch.setattr(
        resources,
        "_fetch_metadata_card",
        lambda _k, _n: (_ for _ in ()).throw(FileNotFoundError("нет базы")),
    )
    with pytest.raises(ResourceError, match="недоступна"):
        resources.read_metadata_card("Document", "X")


def test_manifest_resource_present_and_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing = tmp_path / "no.json"
    monkeypatch.setattr("mcp_1c_rag.config.MANIFEST_PATH", missing)
    with pytest.raises(ResourceError, match="не найден"):
        asyncio.run(server.mcp.read_resource("index://manifest"))

    present = tmp_path / "manifest.json"
    present.write_text("{}", encoding="utf-8")
    monkeypatch.setattr("mcp_1c_rag.config.MANIFEST_PATH", present)

    class _Manifest:
        def to_dict(self) -> dict[str, int]:
            return {"schema_version": 1}

    monkeypatch.setattr("mcp_1c_rag.manifest.load_manifest", lambda _path: _Manifest())
    found = asyncio.run(server.mcp.read_resource("index://manifest"))
    assert "schema_version" in found[0].content


def test_read_resource_through_server(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "m.bsl").write_text("модуль", encoding="utf-8")
    monkeypatch.setattr(resources, "STUDY_ROOT", tmp_path)
    monkeypatch.setattr(resources, "_fetch_metadata_card", lambda k, n: ["карточка XML"])
    bsl = asyncio.run(server.mcp.read_resource("bsl://m.bsl"))
    meta = asyncio.run(server.mcp.read_resource("metadata://Document/Реализация"))
    assert bsl[0].content == "модуль"
    assert meta[0].content == "карточка XML"
