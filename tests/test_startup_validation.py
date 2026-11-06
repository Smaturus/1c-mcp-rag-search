"""Тесты startup-валидации (без ML; Chroma — через fake client)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mcp_1c_rag.startup import (  # noqa: E402
    RuntimeValidationError,
    ValidationReport,
    format_validation_error,
    validate_runtime,
)


class _FakeColl:
    def __init__(self, count: int) -> None:
        self._count = count

    def count(self) -> int:
        return self._count


class _FakeClient:
    def __init__(self, counts: dict[str, int], *, missing: set[str] | None = None) -> None:
        self._counts = counts
        self._missing = missing or set()

    def get_collection(self, name: str) -> Any:
        if name in self._missing or name not in self._counts:
            raise ValueError(f"Collection {name} does not exist.")
        return _FakeColl(self._counts[name])


def _write_bm25(bm25_dir: Path, *, n_docs: int = 10) -> None:
    bm25_dir.mkdir(parents=True, exist_ok=True)
    (bm25_dir / "index.npz").write_bytes(b"NPZ")
    (bm25_dir / "meta.json").write_text(
        json.dumps({"k1": 1.5, "b": 0.75, "n_docs": n_docs}),
        encoding="utf-8",
    )


def _chroma_ok(chroma_dir: Path) -> None:
    chroma_dir.mkdir(parents=True, exist_ok=True)
    (chroma_dir / "chroma.sqlite3").write_bytes(b"SQLite")
    (chroma_dir / "extra.bin").write_bytes(b"x")


def test_tmp_path_without_chroma_sqlite(tmp_path: Path) -> None:
    """Каталог без chroma.sqlite3 → error."""
    chroma = tmp_path / "chroma"
    chroma.mkdir()
    (chroma / "orphan.txt").write_text("x", encoding="utf-8")
    bm25 = tmp_path / "bm25"
    _write_bm25(bm25)
    report = validate_runtime(
        chroma_dir=chroma,
        bm25_dir=bm25,
        search_default_mode="hybrid",
        client_factory=lambda _p: (_ for _ in ()).throw(AssertionError("no client")),
    )
    assert report.errors
    assert any(i.code == "chroma_sqlite_missing" for i in report.errors)


def test_empty_chroma_dir(tmp_path: Path) -> None:
    """Пустой CHROMA_DIR → error."""
    chroma = tmp_path / "chroma"
    chroma.mkdir()
    bm25 = tmp_path / "bm25"
    _write_bm25(bm25)
    report = validate_runtime(
        chroma_dir=chroma,
        bm25_dir=bm25,
        search_default_mode="hybrid",
        client_factory=lambda _p: (_ for _ in ()).throw(AssertionError("no client")),
    )
    codes = {i.code for i in report.errors}
    assert "chroma_empty" in codes
    assert "chroma_sqlite_missing" in codes


def test_missing_bm25(tmp_path: Path) -> None:
    """Нет index.npz/meta.json → error."""
    chroma = tmp_path / "chroma"
    _chroma_ok(chroma)
    bm25 = tmp_path / "bm25"
    bm25.mkdir()
    client = _FakeClient({"code_proc": 10, "metadata_xml": 5})
    report = validate_runtime(
        chroma_dir=chroma,
        bm25_dir=bm25,
        code_collection="code_proc",
        metadata_collection="metadata_xml",
        search_default_mode="hybrid",
        client_factory=lambda _p: client,
    )
    codes = {i.code for i in report.errors}
    assert "bm25_index_missing" in codes
    assert "bm25_meta_missing" in codes


def test_bm25_n_docs_zero(tmp_path: Path) -> None:
    """meta n_docs=0 → error."""
    chroma = tmp_path / "chroma"
    _chroma_ok(chroma)
    bm25 = tmp_path / "bm25"
    _write_bm25(bm25, n_docs=0)
    client = _FakeClient({"code_proc": 10, "metadata_xml": 5})
    report = validate_runtime(
        chroma_dir=chroma,
        bm25_dir=bm25,
        code_collection="code_proc",
        metadata_collection="metadata_xml",
        search_default_mode="hybrid",
        client_factory=lambda _p: client,
    )
    assert any(i.code == "bm25_empty" for i in report.errors)


def test_code_collection_empty_is_error(tmp_path: Path) -> None:
    """Пустая code-коллекция — fatal; пустая metadata — warning."""
    chroma = tmp_path / "chroma"
    _chroma_ok(chroma)
    bm25 = tmp_path / "bm25"
    _write_bm25(bm25)
    client = _FakeClient({"code_proc": 0, "metadata_xml": 0})
    report = validate_runtime(
        chroma_dir=chroma,
        bm25_dir=bm25,
        code_collection="code_proc",
        metadata_collection="metadata_xml",
        search_default_mode="hybrid",
        client_factory=lambda _p: client,
    )
    assert any(i.code == "code_collection_empty" for i in report.errors)
    assert any(i.code == "metadata_collection_empty" for i in report.warnings)
    assert not any(i.code.startswith("metadata_") for i in report.errors)


def test_metadata_missing_is_warning(tmp_path: Path) -> None:
    """Отсутствующая metadata-коллекция — warning, не fatal сама по себе."""
    chroma = tmp_path / "chroma"
    _chroma_ok(chroma)
    bm25 = tmp_path / "bm25"
    _write_bm25(bm25)
    client = _FakeClient({"code_proc": 3}, missing={"metadata_xml"})
    report = validate_runtime(
        chroma_dir=chroma,
        bm25_dir=bm25,
        code_collection="code_proc",
        metadata_collection="metadata_xml",
        search_default_mode="hybrid",
        client_factory=lambda _p: client,
    )
    assert not any(i.code.startswith("code_") for i in report.errors)
    assert any(i.code == "metadata_collection_missing" for i in report.warnings)
    assert report.errors == [] or all(
        i.code.startswith("bm25_") is False for i in report.errors
    )
    assert not report.errors


def test_skip_startup_check(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """MCP_1C_SKIP_STARTUP_CHECK=1 → skipped, без обращения к client."""
    monkeypatch.setenv("MCP_1C_SKIP_STARTUP_CHECK", "1")
    calls: list[Path] = []

    def factory(path: Path) -> Any:
        calls.append(path)
        raise AssertionError("client must not be called")

    report = validate_runtime(
        chroma_dir=tmp_path / "missing",
        bm25_dir=tmp_path / "missing_bm25",
        client_factory=factory,
    )
    assert report.skipped is True
    assert report.issues == []
    assert calls == []


def test_bad_search_mode(tmp_path: Path) -> None:
    """Неизвестный SEARCH_DEFAULT_MODE → error."""
    chroma = tmp_path / "chroma"
    _chroma_ok(chroma)
    bm25 = tmp_path / "bm25"
    _write_bm25(bm25)
    client = _FakeClient({"code_proc": 1, "metadata_xml": 1})
    report = validate_runtime(
        chroma_dir=chroma,
        bm25_dir=bm25,
        code_collection="code_proc",
        metadata_collection="metadata_xml",
        search_default_mode="nope",
        client_factory=lambda _p: client,
    )
    assert any(i.code == "bad_search_mode" for i in report.errors)


def test_ok_report(tmp_path: Path) -> None:
    """Валидные пути и ненулевые коллекции → без errors."""
    chroma = tmp_path / "chroma"
    _chroma_ok(chroma)
    bm25 = tmp_path / "bm25"
    _write_bm25(bm25, n_docs=42)
    client = _FakeClient({"code_proc": 10, "metadata_xml": 5})
    report = validate_runtime(
        chroma_dir=chroma,
        bm25_dir=bm25,
        code_collection="code_proc",
        metadata_collection="metadata_xml",
        search_default_mode="hybrid",
        client_factory=lambda _p: client,
    )
    assert report.skipped is False
    assert report.errors == []
    assert report.warnings == []


def test_format_validation_error_actionable(tmp_path: Path) -> None:
    """format_validation_error содержит путь, факт и подсказку."""
    chroma = tmp_path / "chroma"
    chroma.mkdir()
    bm25 = tmp_path / "bm25"
    _write_bm25(bm25)
    report = validate_runtime(
        chroma_dir=chroma,
        bm25_dir=bm25,
        search_default_mode="hybrid",
        client_factory=lambda _p: (_ for _ in ()).throw(AssertionError("no")),
    )
    text = format_validation_error(
        RuntimeValidationError(report.errors, report.warnings)
    )
    assert "chroma" in text.lower() or str(chroma) in text
    assert "исправление:" in text
    assert "MCP_1C_SKIP_STARTUP_CHECK=1" in text
    assert isinstance(report, ValidationReport)


def test_server_main_exits_2_on_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """server.main печатает ошибку в stderr и выходит с кодом 2."""
    import mcp_1c_rag.startup as startup_mod
    from mcp_1c_rag import server
    from mcp_1c_rag.startup import ValidationIssue

    bad = ValidationReport(
        issues=[
            ValidationIssue(
                level="error",
                code="chroma_missing",
                message="каталог Chroma не существует",
                path="/no/chroma",
                actual="отсутствует",
                hint="задайте MCP_1C_CHROMA_DIR",
            )
        ]
    )
    monkeypatch.setattr(startup_mod, "validate_runtime", lambda: bad)

    ran = {"mcp": False}

    def fake_run(**_kwargs: Any) -> None:
        ran["mcp"] = True

    monkeypatch.setattr(server.mcp, "run", fake_run)
    with pytest.raises(SystemExit) as exc_info:
        server.main()
    assert exc_info.value.code == 2
    assert ran["mcp"] is False


def test_server_main_skip(monkeypatch: pytest.MonkeyPatch) -> None:
    """При skipped — сразу mcp.run без ошибок."""
    import mcp_1c_rag.startup as startup_mod
    from mcp_1c_rag import server

    monkeypatch.setattr(
        startup_mod, "validate_runtime", lambda: ValidationReport(skipped=True)
    )
    ran = {"mcp": False}

    def fake_run(**_kwargs: Any) -> None:
        ran["mcp"] = True

    monkeypatch.setattr(server.mcp, "run", fake_run)
    server.main()
    assert ran["mcp"] is True


def test_no_heavy_imports() -> None:
    """Импорт startup не тянет sentence_transformers / torch / flashrank."""
    script = f"""
import sys
from pathlib import Path
root = Path(r'''{ROOT}''')
sys.path.insert(0, str(root / "src"))
for name in list(sys.modules):
    if name == "mcp_1c_rag" or name.startswith("mcp_1c_rag."):
        del sys.modules[name]
    if name in {{
        "sentence_transformers",
        "torch",
        "flashrank",
    }} or name.startswith(("sentence_transformers.", "torch.", "flashrank.")):
        del sys.modules[name]
import mcp_1c_rag.startup as s
forbidden = ("sentence_transformers", "torch", "flashrank")
loaded = [n for n in forbidden if n in sys.modules]
assert not loaded, loaded
assert s.validate_runtime(skip=True).skipped
print("ok")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )
    assert result.returncode == 0, result.stderr or result.stdout
    assert "ok" in result.stdout


def test_main_unknown_collection_no_traceback() -> None:
    """python -m mcp_1c_rag с плохим env → exit 2, без traceback."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    env["MCP_1C_CODE_COLLECTION"] = "no_such_collection_xyz"
    env["MCP_1C_SKIP_STARTUP_CHECK"] = "1"
    result = subprocess.run(
        [sys.executable, "-m", "mcp_1c_rag"],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        env=env,
    )
    assert result.returncode == 2
    assert "Traceback" not in (result.stderr or "")
    assert "no_such_collection_xyz" in (result.stderr or "")
