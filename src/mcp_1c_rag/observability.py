"""Observability MCP: stderr logging + in-memory метрики (без нового tool).

Stdout не трогаем — занят MCP stdio. Query text по умолчанию не логируется
(``MCP_1C_LOG_QUERIES=1`` — явный opt-in).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import threading
import time
import uuid
from datetime import UTC, datetime
from typing import Any

from mcp_1c_rag.startup import ValidationReport

LOGGER_NAME = "mcp_1c_rag"
_ENV_LEVEL = "MCP_1C_LOG_LEVEL"
_ENV_JSON = "MCP_1C_LOG_JSON"
_ENV_QUERIES = "MCP_1C_LOG_QUERIES"

_lock = threading.Lock()
_startup_report: ValidationReport | None = None
_metrics: dict[str, Any] = {}
_logger: logging.Logger | None = None
_json_logs = False
_log_queries = False


def _truthy(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _parse_level(raw: str) -> int:
    name = (raw or "WARNING").strip().upper()
    return getattr(logging, name, logging.WARNING)


def _empty_metrics() -> dict[str, Any]:
    return {
        "requests_total": 0,
        "errors_total": 0,
        "empty_results_total": 0,
        "by_tool": {},
        "latency": {
            "count": 0,
            "sum_ms": 0.0,
            "min_ms": None,
            "max_ms": None,
            "last_ms": None,
        },
    }


def reset_metrics() -> None:
    """Сбросить in-memory счётчики (для тестов)."""
    global _metrics
    with _lock:
        _metrics = _empty_metrics()


def setup_logging(
    *,
    level: str | None = None,
    json_logs: bool | None = None,
    log_queries: bool | None = None,
) -> logging.Logger:
    """Настроить логгер mcp_1c_rag → только stderr."""
    global _logger, _json_logs, _log_queries
    _json_logs = (
        bool(json_logs) if json_logs is not None else _truthy(_ENV_JSON, "0")
    )
    _log_queries = (
        bool(log_queries)
        if log_queries is not None
        else _truthy(_ENV_QUERIES, "0")
    )
    lvl = _parse_level(level if level is not None else os.environ.get(_ENV_LEVEL, "WARNING"))
    logger = logging.getLogger(LOGGER_NAME)
    logger.handlers.clear()
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(lvl)
    logger.propagate = False
    _logger = logger
    if not _metrics:
        reset_metrics()
    return logger


def _get_logger() -> logging.Logger:
    if _logger is None:
        return setup_logging()
    return _logger


def query_privacy_fields(query: str) -> dict[str, Any]:
    """Поля запроса без текста по умолчанию."""
    text = query or ""
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    fields: dict[str, Any] = {
        "query_chars": len(text),
        "query_hash": digest,
    }
    if _log_queries:
        fields["query"] = text
    return fields


def new_request_id() -> str:
    """Короткий id запроса для корреляции логов."""
    return uuid.uuid4().hex[:12]


def log_event(event: str, *, level: int = logging.INFO, **fields: Any) -> None:
    """Структурированное событие в stderr (JSON или key=value)."""
    payload: dict[str, Any] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "level": logging.getLevelName(level),
        "event": event,
    }
    for key, value in fields.items():
        if value is None or key == "query" and not _log_queries:
            continue
        payload[key] = value
    logger = _get_logger()
    if _json_logs:
        message = json.dumps(payload, ensure_ascii=False, default=str)
    else:
        parts = [f"{k}={payload[k]}" for k in payload]
        message = " ".join(parts)
    logger.log(level, message)


def _tool_bucket(tool: str) -> dict[str, int]:
    by_tool: dict[str, dict[str, int]] = _metrics["by_tool"]
    if tool not in by_tool:
        by_tool[tool] = {"requests": 0, "errors": 0}
    return by_tool[tool]


def record_request(
    tool: str,
    *,
    latency_ms: float,
    ok: bool = True,
    empty: bool = False,
    error_type: str | None = None,
    **log_fields: Any,
) -> None:
    """Обновить счётчики и записать tool-событие."""
    with _lock:
        if not _metrics:
            _metrics.update(_empty_metrics())
        _metrics["requests_total"] += 1
        bucket = _tool_bucket(tool)
        bucket["requests"] += 1
        if not ok:
            _metrics["errors_total"] += 1
            bucket["errors"] += 1
        if empty:
            _metrics["empty_results_total"] += 1
        lat = _metrics["latency"]
        lat["count"] += 1
        lat["sum_ms"] += float(latency_ms)
        lat["last_ms"] = float(latency_ms)
        if lat["min_ms"] is None or latency_ms < lat["min_ms"]:
            lat["min_ms"] = float(latency_ms)
        if lat["max_ms"] is None or latency_ms > lat["max_ms"]:
            lat["max_ms"] = float(latency_ms)

    level = logging.INFO if ok else logging.ERROR
    log_event(
        "tool_request",
        level=level,
        tool=tool,
        latency_ms=round(float(latency_ms), 3),
        ok=ok,
        error_type=error_type,
        **log_fields,
    )


def snapshot() -> dict[str, Any]:
    """Снимок метрик для index_stats / health."""
    with _lock:
        if not _metrics:
            data = _empty_metrics()
        else:
            data = {
                "requests_total": _metrics["requests_total"],
                "errors_total": _metrics["errors_total"],
                "empty_results_total": _metrics["empty_results_total"],
                "by_tool": {
                    name: dict(vals) for name, vals in _metrics["by_tool"].items()
                },
                "latency": dict(_metrics["latency"]),
            }
    lat = data["latency"]
    count = int(lat["count"] or 0)
    data["latency_ms_avg"] = (
        round(float(lat["sum_ms"]) / count, 3) if count else 0.0
    )
    data["latency_ms_max"] = (
        float(lat["max_ms"]) if lat["max_ms"] is not None else 0.0
    )
    data["latency_ms_min"] = (
        float(lat["min_ms"]) if lat["min_ms"] is not None else 0.0
    )
    data["latency_ms_last"] = (
        float(lat["last_ms"]) if lat["last_ms"] is not None else 0.0
    )
    return data


def set_startup_report(report: ValidationReport | None) -> None:
    """Сохранить отчёт startup-валидации для health/index_stats."""
    global _startup_report
    with _lock:
        _startup_report = report


def startup_status() -> str:
    """ok | degraded | skipped | unknown."""
    with _lock:
        report = _startup_report
    if report is None:
        return "unknown"
    if report.skipped:
        return "skipped"
    if report.errors or report.warnings:
        return "degraded"
    return "ok"


def health_snapshot() -> dict[str, Any]:
    """Лёгкий снимок готовности + метрики + краткий config (без модели)."""
    from mcp_1c_rag.config import (
        COLLECTION_CODE,
        COLLECTION_METADATA,
        SEARCH_DEFAULT_MODE,
    )

    with _lock:
        report = _startup_report
        warnings_n = len(report.warnings) if report is not None else 0
        errors_n = len(report.errors) if report is not None else 0
        skipped = bool(report.skipped) if report is not None else False

    metrics = snapshot()
    return {
        "status": startup_status(),
        "startup_skipped": skipped,
        "startup_warnings": warnings_n,
        "startup_errors": errors_n,
        "metrics": metrics,
        "config": {
            "default_mode": SEARCH_DEFAULT_MODE,
            "collection_code": COLLECTION_CODE,
            "collection_metadata": COLLECTION_METADATA,
            "log_level": logging.getLevelName(_get_logger().level),
            "log_json": _json_logs,
            "log_queries": _log_queries,
        },
    }


def metrics_lines() -> list[str]:
    """Строки key=value для дополнения index_stats."""
    health = health_snapshot()
    metrics = health["metrics"]
    return [
        f"status={health['status']}",
        f"startup_warnings={health['startup_warnings']}",
        f"requests_total={metrics['requests_total']}",
        f"errors_total={metrics['errors_total']}",
        f"empty_results_total={metrics['empty_results_total']}",
        f"latency_ms_avg={metrics['latency_ms_avg']}",
        f"latency_ms_max={metrics['latency_ms_max']}",
    ]


def elapsed_ms(started: float) -> float:
    """Миллисекунды с момента ``time.perf_counter()``."""
    return (time.perf_counter() - started) * 1000.0


# Инициализация пустых метрик при импорте модуля.
reset_metrics()

__all__ = [
    "LOGGER_NAME",
    "elapsed_ms",
    "health_snapshot",
    "log_event",
    "metrics_lines",
    "new_request_id",
    "query_privacy_fields",
    "record_request",
    "reset_metrics",
    "set_startup_report",
    "setup_logging",
    "snapshot",
    "startup_status",
]
