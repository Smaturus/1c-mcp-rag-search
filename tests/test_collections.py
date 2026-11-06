"""Тесты чистого реестра коллекций (без Chroma/ML)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mcp_1c_rag.collections import (  # noqa: E402
    DEFAULT_CODE_COLLECTION,
    DEFAULT_METADATA_COLLECTION,
    SPECS,
    CollectionSpec,
    UnknownCollectionError,
    all_specs,
    canonical_name,
    get_spec,
    id_prefix,
    validate_collection_name,
)

EXPECTED = (
    ("code_proc", "code_proc", "bsl_proc:"),
    ("metadata_xml", "metadata_xml", "metaobj:"),
    ("config_xml", "config_xml", "xml:"),
    ("code", "code", "bsl:"),
    ("metadata", "metadata", "meta:"),
)


def test_specs_unique() -> None:
    """Имена, chroma_name и id_prefix уникальны и непусты."""
    names = [s.name for s in SPECS]
    chroma = [s.chroma_name for s in SPECS]
    prefixes = [s.id_prefix for s in SPECS]
    assert all(names) and len(names) == len(set(names))
    assert all(chroma) and len(chroma) == len(set(chroma))
    assert all(prefixes) and len(prefixes) == len(set(prefixes))
    for spec in SPECS:
        assert isinstance(spec, CollectionSpec)
        assert spec.name.strip() == spec.name
        assert spec.id_prefix.endswith(":")


@pytest.mark.parametrize(
    ("name", "chroma_name", "prefix"),
    EXPECTED,
)
def test_expected_mapping(name: str, chroma_name: str, prefix: str) -> None:
    """Канон совпадает с текущими префиксами индексатора."""
    spec = get_spec(name)
    assert spec.name == name
    assert spec.chroma_name == chroma_name
    assert spec.id_prefix == prefix


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (" CODE_PROC ", "code_proc"),
        ("Code_Proc", "code_proc"),
        (" metadata ", "metadata"),
        ("META", "metadata"),
        ("Metadata_XML", "metadata_xml"),
        ("CONFIG_XML", "config_xml"),
        ("Code", "code"),
    ],
)
def test_aliases_case_insensitive(raw: str, expected: str) -> None:
    """Резолв strip/lower по name, chroma_name и aliases."""
    assert get_spec(raw).name == expected
    assert validate_collection_name(raw) == expected


def test_unknown_collection_raises() -> None:
    """Неизвестное имя — UnknownCollectionError с именем и списком доступных."""
    bad = "no_such_collection_xyz"
    with pytest.raises(UnknownCollectionError) as exc_info:
        get_spec(bad)
    message = str(exc_info.value)
    assert bad in message
    for name, _, _ in EXPECTED:
        assert name in message
    assert issubclass(UnknownCollectionError, ValueError)
    with pytest.raises(ValueError):
        validate_collection_name(bad)


def test_canonical_and_id_prefix_helpers() -> None:
    """Хелперы canonical_name / id_prefix / validate_collection_name."""
    assert canonical_name("code_proc") == "code_proc"
    assert id_prefix("code_proc") == "bsl_proc:"
    assert id_prefix("metadata_xml") == "metaobj:"
    assert id_prefix("config_xml") == "xml:"
    assert id_prefix("code") == "bsl:"
    assert id_prefix("metadata") == "meta:"
    assert validate_collection_name(" CODE ") == "code"


def test_defaults_constants() -> None:
    """Дефолты объявлены как code_proc / metadata_xml (ещё без config)."""
    assert DEFAULT_CODE_COLLECTION == "code_proc"
    assert DEFAULT_METADATA_COLLECTION == "metadata_xml"
    assert get_spec(DEFAULT_CODE_COLLECTION).role == "code"
    assert get_spec(DEFAULT_METADATA_COLLECTION).role == "metadata"


def test_all_specs_stable() -> None:
    """Порядок и состав SPECS предсказуемы."""
    specs = all_specs()
    assert specs is SPECS
    assert tuple(s.name for s in specs) == tuple(row[0] for row in EXPECTED)
    assert len(specs) == 5
    roles = {s.name: s.role for s in specs}
    assert roles["code_proc"] == "code"
    assert roles["metadata_xml"] == "metadata"
    assert roles["config_xml"] == "aux"
    assert roles["code"] == "aux"
    assert roles["metadata"] == "aux"


def test_no_heavy_imports() -> None:
    """Импорт collections не тянет chromadb / ST / torch / config."""
    script = f"""
import sys
from pathlib import Path
root = Path(r'''{ROOT}''')
sys.path.insert(0, str(root / "src"))
# drop already-loaded heavy / project modules if present
for name in list(sys.modules):
    if name == "mcp_1c_rag" or name.startswith("mcp_1c_rag."):
        del sys.modules[name]
    if name in {{
        "chromadb",
        "sentence_transformers",
        "torch",
        "mcp_1c_rag.config",
    }} or name.startswith(("chromadb.", "sentence_transformers.", "torch.")):
        del sys.modules[name]
import mcp_1c_rag.collections as c
forbidden = (
    "chromadb",
    "sentence_transformers",
    "torch",
    "mcp_1c_rag.config",
    "mcp_1c_rag.store",
    "mcp_1c_rag.embeddings",
)
loaded = [name for name in forbidden if name in sys.modules]
assert not loaded, loaded
assert c.id_prefix("code_proc") == "bsl_proc:"
print("ok")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    assert result.returncode == 0, result.stderr or result.stdout
    assert "ok" in result.stdout
