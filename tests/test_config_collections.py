"""Интеграция registry с config: дефолты и резолв env."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mcp_1c_rag.collections import (  # noqa: E402
    DEFAULT_CODE_COLLECTION,
    DEFAULT_METADATA_COLLECTION,
    UnknownCollectionError,
)


def _run_config_probe(env: dict[str, str], body: str) -> subprocess.CompletedProcess[str]:
    """Импорт config в чистом subprocess с заданным env."""
    full_env = os.environ.copy()
    full_env.pop("MCP_1C_CODE_COLLECTION", None)
    full_env.pop("MCP_1C_METADATA_COLLECTION", None)
    full_env.update(env)
    full_env["PYTHONPATH"] = str(ROOT / "src")
    script = f"""
from mcp_1c_rag import config
{body}
"""
    return subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        env=full_env,
    )


def test_config_defaults_match_registry() -> None:
    """Без env дефолты config == registry (code_proc / metadata_xml)."""
    result = _run_config_probe(
        {},
        (
            f"assert config.COLLECTION_CODE == {DEFAULT_CODE_COLLECTION!r}\n"
            f"assert config.COLLECTION_METADATA == {DEFAULT_METADATA_COLLECTION!r}\n"
            "print('ok')"
        ),
    )
    assert result.returncode == 0, result.stderr or result.stdout
    assert "ok" in result.stdout


@pytest.mark.parametrize(
    ("env_key", "env_value", "attr", "expected"),
    [
        ("MCP_1C_CODE_COLLECTION", "code", "COLLECTION_CODE", "code"),
        ("MCP_1C_CODE_COLLECTION", "CODE_PROC", "COLLECTION_CODE", "code_proc"),
        ("MCP_1C_METADATA_COLLECTION", "meta", "COLLECTION_METADATA", "metadata"),
        ("MCP_1C_METADATA_COLLECTION", "metadata", "COLLECTION_METADATA", "metadata"),
        (
            "MCP_1C_METADATA_COLLECTION",
            "metadata_xml",
            "COLLECTION_METADATA",
            "metadata_xml",
        ),
    ],
)
def test_config_env_aliases_resolve(
    env_key: str, env_value: str, attr: str, expected: str
) -> None:
    """Env alias/регистр резолвится в canonical через registry."""
    result = _run_config_probe(
        {env_key: env_value},
        f"assert config.{attr} == {expected!r}\nprint('ok')",
    )
    assert result.returncode == 0, result.stderr or result.stdout
    assert "ok" in result.stdout


def test_config_unknown_env_raises() -> None:
    """Неизвестное env-имя коллекции → UnknownCollectionError при импорте config."""
    result = _run_config_probe(
        {"MCP_1C_CODE_COLLECTION": "no_such_collection_xyz"},
        "print(config.COLLECTION_CODE)",
    )
    assert result.returncode != 0
    combined = (result.stderr or "") + (result.stdout or "")
    assert "no_such_collection_xyz" in combined
    assert "UnknownCollectionError" in combined or "неизвестная коллекция" in combined
    assert issubclass(UnknownCollectionError, ValueError)
