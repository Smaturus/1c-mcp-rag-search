"""Фоновый прогрев поискового контура после старта MCP-сервера.

Первое обращение к коллекции Chroma пересобирает HNSW из sqlite (десятки секунд
при ``hnsw:sync_threshold=1000000``), первый запрос грузит e5 и BM25. Прогрев
выполняет это в фоне, пока клиент уже подключён. Поисковые tools ждут его
окончания (``wait_ready``), чтобы модель и сегменты не грузились параллельно.

Загрузка HNSW держит GIL всё время загрузки: главный поток в это время не
отвечает клиенту. Поэтому прогрев стартует с задержкой ``delay_s`` — за неё
stdio-клиент успевает пройти initialize и получить tools/prompts/resources list.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from mcp_1c_rag import observability as obs

Step = tuple[str, Callable[[], object]]

# Upper bound for a search tool waiting on warm-up; after it the tool proceeds anyway.
WAIT_TIMEOUT_S = 300.0


@dataclass
class WarmupState:
    """Состояние прогрева: off → running → done | partial."""

    status: str = "off"
    duration_s: float | None = None
    failed_steps: list[str] = field(default_factory=list)
    done: threading.Event = field(default_factory=threading.Event)

    def __post_init__(self) -> None:
        self.done.set()


_state = WarmupState()
_start_lock = threading.Lock()


def _load_collection(name: str) -> int:
    """Загрузить сегменты коллекции (HNSW) без get_or_create."""
    from mcp_1c_rag.store import get_client

    return int(get_client().get_collection(name).count())


def _load_bm25() -> object:
    """Загрузить глобальный BM25 в процессный контекст поиска."""
    from mcp_1c_rag.search_modes import default_context

    return default_context.bm25()


def _load_model() -> object:
    """Загрузить e5 и прогнать один запрос (первый encode тоже медленный)."""
    from mcp_1c_rag.embeddings import embed_query

    return embed_query("прогрев")


def default_steps() -> list[Step]:
    """Шаги прогрева: HNSW коллекций кода и XML-метаданных, BM25, e5."""
    from mcp_1c_rag.config import COLLECTION_CODE, COLLECTION_METADATA

    return [
        (f"chroma:{COLLECTION_CODE}", lambda: _load_collection(COLLECTION_CODE)),
        (f"chroma:{COLLECTION_METADATA}", lambda: _load_collection(COLLECTION_METADATA)),
        ("bm25", _load_bm25),
        ("embedding_model", _load_model),
    ]


def _run(steps: Sequence[Step], delay_s: float) -> None:
    """Подождать delay_s, затем выполнить шаги; сбой шага не останавливает остальные."""
    failed: list[str] = []
    if delay_s > 0:
        time.sleep(delay_s)
    started = time.perf_counter()
    try:
        for name, fn in steps:
            step_started = time.perf_counter()
            try:
                fn()
            except Exception as exc:  # noqa: BLE001 — прогрев не должен ронять сервер
                failed.append(name)
                obs.log_event(
                    "warmup_step_failed",
                    level=logging.WARNING,
                    step=name,
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
            else:
                obs.log_event(
                    "warmup_step",
                    step=name,
                    latency_ms=round(obs.elapsed_ms(step_started), 1),
                )
    finally:
        _state.failed_steps = failed
        _state.duration_s = round(time.perf_counter() - started, 1)
        _state.status = "partial" if failed else "done"
        _state.done.set()
        obs.log_event(
            "warmup",
            status=_state.status,
            duration_s=_state.duration_s,
            failed_steps=",".join(failed) or None,
        )


def start(
    steps: Sequence[Step] | None = None, *, delay_s: float = 0.0
) -> threading.Thread | None:
    """Запустить прогрев в фоновом потоке; повторный вызов во время прогрева — None.

    Статус ``running`` выставляется сразу, поэтому поисковые tools ждут и во
    время задержки ``delay_s``.
    """
    with _start_lock:
        if _state.status == "running":
            return None
        chosen = list(steps) if steps is not None else default_steps()
        _state.status = "running"
        _state.duration_s = None
        _state.failed_steps = []
        _state.done.clear()
        thread = threading.Thread(
            target=_run, args=(chosen, delay_s), name="mcp-1c-warmup", daemon=True
        )
        thread.start()
        return thread


def is_running() -> bool:
    """Идёт ли прогрев сейчас."""
    return _state.status == "running"


def wait_ready(timeout: float | None = WAIT_TIMEOUT_S) -> bool:
    """Дождаться конца прогрева; True сразу, если прогрев не запускался."""
    return _state.done.wait(timeout)


def status_lines() -> list[str]:
    """Строки key=value для index_stats."""
    lines = [f"warmup={_state.status}"]
    if _state.duration_s is not None:
        lines.append(f"warmup_duration_s={_state.duration_s}")
    if _state.failed_steps:
        lines.append(f"warmup_failed_steps={','.join(_state.failed_steps)}")
    return lines


def reset() -> None:
    """Вернуть состояние «не запускался» (для тестов)."""
    global _state
    _state = WarmupState()


__all__ = [
    "WAIT_TIMEOUT_S",
    "WarmupState",
    "default_steps",
    "is_running",
    "reset",
    "start",
    "status_lines",
    "wait_ready",
]
