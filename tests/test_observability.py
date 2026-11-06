"""Unit-тесты observability: stderr logs, privacy, metrics, health."""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mcp_1c_rag import observability as obs  # noqa: E402
from mcp_1c_rag.startup import ValidationIssue, ValidationReport  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_obs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MCP_1C_LOG_JSON", raising=False)
    monkeypatch.delenv("MCP_1C_LOG_QUERIES", raising=False)
    monkeypatch.setenv("MCP_1C_LOG_LEVEL", "INFO")
    obs.reset_metrics()
    obs.set_startup_report(None)
    obs.setup_logging(level="INFO", json_logs=False, log_queries=False)


def test_logging_goes_to_stderr_not_stdout(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    err = io.StringIO()
    monkeypatch.setattr(sys, "stderr", err)
    obs.setup_logging(level="INFO", json_logs=False, log_queries=False)
    obs.log_event("ping", tool="search", latency_ms=1.5, ok=True)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "event=ping" in err.getvalue()
    assert "tool=search" in err.getvalue()


def test_json_mode_one_valid_json_line(monkeypatch: pytest.MonkeyPatch) -> None:
    err = io.StringIO()
    monkeypatch.setattr(sys, "stderr", err)
    obs.setup_logging(level="INFO", json_logs=True, log_queries=False)
    obs.log_event("tool_request", tool="search", latency_ms=12.3, ok=True)
    line = err.getvalue().strip()
    assert "\n" not in line
    payload = json.loads(line)
    assert payload["event"] == "tool_request"
    assert payload["tool"] == "search"
    assert payload["latency_ms"] == 12.3


def test_default_privacy_no_query_text(monkeypatch: pytest.MonkeyPatch) -> None:
    err = io.StringIO()
    monkeypatch.setattr(sys, "stderr", err)
    obs.setup_logging(level="INFO", json_logs=True, log_queries=False)
    fields = obs.query_privacy_fields("секретный НДС")
    assert "query" not in fields
    assert fields["query_chars"] == len("секретный НДС")
    assert len(fields["query_hash"]) == 16
    obs.log_event("tool_request", tool="search", **fields)
    payload = json.loads(err.getvalue().strip())
    assert "query" not in payload
    assert "query_chars" in payload
    assert "query_hash" in payload
    assert "секретный" not in err.getvalue()


def test_log_queries_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MCP_1C_LOG_QUERIES", "1")
    err = io.StringIO()
    monkeypatch.setattr(sys, "stderr", err)
    obs.setup_logging(level="INFO", json_logs=True, log_queries=None)
    fields = obs.query_privacy_fields("видимый текст")
    assert fields.get("query") == "видимый текст"
    obs.log_event("tool_request", tool="search", **fields)
    assert "видимый текст" in err.getvalue()


def test_counters_latency_snapshot_and_reset() -> None:
    obs.record_request("search", latency_ms=10.0, ok=True, empty=False, result_count=3)
    obs.record_request("search", latency_ms=30.0, ok=True, empty=True, result_count=0)
    obs.record_request(
        "search", latency_ms=5.0, ok=False, error_type="ValueError", result_count=0
    )
    snap = obs.snapshot()
    assert snap["requests_total"] == 3
    assert snap["errors_total"] == 1
    assert snap["empty_results_total"] == 1
    assert snap["by_tool"]["search"]["requests"] == 3
    assert snap["by_tool"]["search"]["errors"] == 1
    assert snap["latency_ms_avg"] == pytest.approx(15.0)
    assert snap["latency_ms_max"] == pytest.approx(30.0)
    assert snap["latency_ms_min"] == pytest.approx(5.0)
    obs.reset_metrics()
    fresh = obs.snapshot()
    assert fresh["requests_total"] == 0
    assert fresh["errors_total"] == 0
    assert fresh["latency"]["count"] == 0


def test_startup_status_variants() -> None:
    assert obs.startup_status() == "unknown"
    obs.set_startup_report(ValidationReport(skipped=True))
    assert obs.startup_status() == "skipped"
    obs.set_startup_report(ValidationReport(issues=[]))
    assert obs.startup_status() == "ok"
    obs.set_startup_report(
        ValidationReport(
            issues=[
                ValidationIssue(
                    level="warning",
                    code="metadata_collection_empty",
                    message="пусто",
                )
            ]
        )
    )
    assert obs.startup_status() == "degraded"


def test_health_snapshot_without_startup() -> None:
    obs.set_startup_report(None)
    health = obs.health_snapshot()
    assert health["status"] == "unknown"
    assert "metrics" in health
    assert "config" in health
    assert health["config"]["log_queries"] is False
