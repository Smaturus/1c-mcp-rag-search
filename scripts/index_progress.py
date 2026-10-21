"""Collect indexing progress / ETA from data/index_full.log."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "data" / "index_full.log"
STATUS = ROOT / "data" / "index_status.txt"
PROGRESS_MD = ROOT / "data" / "INDEX_PROGRESS.md"
STATE = ROOT / "data" / "index_tick_state.json"

# Progress lines are every 10 minutes. Warn only after two missed intervals.
STALL_SEC = 1200

# English tokens must not appear in summary_ru (chat line).
_SUMMARY_EN_FORBIDDEN = re.compile(
    r"(?i)\b(stall|arm|hook|sentinel|legacy|finished|metadata|ETA)\b|files/min"
)

_PHASE_RU = {
    "starting": "старт",
    "code": "код",
    "metadata": "метаданные",
    "done": "готово",
}


def _read_checkpoint(phase: str) -> dict | None:
    """Files and chunks already recorded for the current phase."""
    filename = {
        "code": "index_checkpoint_code.json",
        "metadata": "index_checkpoint_metadata.json",
    }.get(phase)
    if not filename:
        return None
    path = ROOT / "data" / filename
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        completed = data.get("completed") or {}
        if not isinstance(completed, dict):
            return None
        chunks = sum(int(v) for v in completed.values()) + int(
            data.get("partial_chunks") or 0
        )
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None
    return {"mtime": path.stat().st_mtime, "files": len(completed), "chunks": chunks}


def run_is_finished(*, code_finished: bool, meta_finished: bool) -> bool:
    """The run is complete only when both code and metadata have finished."""
    return code_finished and meta_finished


def summary_ru_is_chat_safe(text: str) -> bool:
    """True if line is suitable for user chat (no common anglicisms)."""
    return not _SUMMARY_EN_FORBIDDEN.search(text)


def _read_log_text(path: Path) -> str:
    raw = path.read_bytes()
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16")
    sample = raw[: min(64, len(raw))]
    if len(sample) >= 4 and sample[1:2] == b"\x00":
        try:
            return raw.decode("utf-16-le")
        except UnicodeDecodeError:
            pass
    return raw.decode("utf-8", errors="replace")


def _load_prev_state() -> dict[str, Any]:
    if not STATE.exists():
        return {}
    try:
        return json.loads(STATE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _log_session_key(all_lines: list[str]) -> str:
    """Identity of the current indexing run (after last [restart] marker)."""
    for ln in reversed(all_lines):
        if ln.startswith("[restart]"):
            return ln
    return ""


def _session_lines(all_lines: list[str]) -> list[str]:
    """Log lines belonging to the current run only."""
    start = 0
    for i, ln in enumerate(all_lines):
        if ln.startswith("[restart]"):
            start = i + 1
    return all_lines[start:]


def _writer_alive() -> bool:
    """True when the indexer lock belongs to a live process.

    Read-only: does not open the vector base and does not delete the lock.
    """
    try:
        from mcp_1c_rag.index_guard import lock_holder_alive

        return lock_holder_alive()
    except Exception:
        return False


def collect_progress() -> dict[str, Any]:
    """Parse log into a structured progress dict."""
    all_lines: list[str] = []
    log_mtime = None
    if LOG.exists():
        log_mtime = LOG.stat().st_mtime
        text = _read_log_text(LOG)
        all_lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    log_session = _log_session_key(all_lines)
    lines = _session_lines(all_lines) if log_session else all_lines

    code_total = None
    code_done = None
    code_chunks = None
    code_elapsed = None
    meta_chunks = None
    meta_total = None
    meta_elapsed = None
    phase = "starting"
    code_finished = False
    meta_finished = False

    for ln in lines:
        m = re.search(r"\[code\] files=(\d+)", ln)
        if m:
            code_total = int(m.group(1))
            phase = "code"
        m = re.search(
            r"\[code\] (\d+)/(\d+) files, chunks_so_far=(\d+), ([0-9.]+)s", ln
        )
        if m:
            code_done = int(m.group(1))
            code_total = int(m.group(2))
            code_chunks = int(m.group(3))
            code_elapsed = float(m.group(4))
            phase = "code"
        if "[code] done" in ln:
            code_finished = True
            phase = "metadata"
            m = re.search(r"chunks=(\d+)", ln)
            if m:
                code_chunks = int(m.group(1))
        m = re.search(r"\[metadata\] .+: chunks=(\d+)", ln)
        if m:
            meta_total = int(m.group(1))
            phase = "metadata"
        m = re.search(r"\[metadata\] passed (\d+)/(\d+) pieces=(\d+)", ln)
        if m and int(m.group(1)) == int(m.group(2)) == 1:
            meta_total = int(m.group(3))
            phase = "metadata"
        m = re.search(r"\[metadata\] flushed total=(\d+), ([0-9.]+)s", ln)
        if m:
            meta_chunks = int(m.group(1))
            meta_elapsed = float(m.group(2))
            phase = "metadata"
        if "[metadata] done" in ln:
            meta_finished = True
            phase = "done" if code_finished else "metadata"
            m = re.search(r"chunks=(\d+)", ln)
            if m:
                meta_chunks = int(m.group(1))

        # Both collections must finish in this run. Metadata alone leaves code empty.
    finished = run_is_finished(
        code_finished=code_finished, meta_finished=meta_finished
    )
    code_started = code_finished or code_total is not None or code_done is not None
    need_code = meta_finished and not code_finished and not code_started
    if need_code:
        phase = "metadata"

    now = time.time()
    stall = False
    stall_sec = None
    if log_mtime is not None:
        stall_sec = int(now - log_mtime)
        stall = stall_sec >= STALL_SEC and not finished

    # A live writer is not a hang. The 10-minute line uses the journal and the
    # checkpoint; it must not open the base and must not stop the writer.
    writer_alive = _writer_alive()
    if writer_alive:
        stall = False
        stall_sec = None
    snap = _read_checkpoint(phase)
    if snap is not None and (now - float(snap["mtime"])) < STALL_SEC:
        stall = False
        stall_sec = None
    if phase == "code" and snap is not None:
        if code_done is None or int(snap["files"]) > code_done:
            code_done = int(snap["files"])
        if code_chunks is None or int(snap["chunks"]) > (code_chunks or 0):
            code_chunks = int(snap["chunks"])
        if code_elapsed is not None and log_mtime is not None:
            code_elapsed = float(code_elapsed) + max(0.0, now - log_mtime)

    eta_code_min = None
    rate_files_min = None
    pct = None
    if code_total and code_done and code_elapsed and code_done > 0:
        rate = code_done / code_elapsed
        rate_files_min = rate * 60
        remain = max(code_total - code_done, 0)
        eta_code_min = (remain / rate) / 60 if rate > 0 else None
        pct = 100.0 * code_done / code_total

    # Journal and checkpoint only. Opening the base from this reader
    # collides with the writer's snapshot and aborts the run.
    code_in_db, meta_in_db = None, None

    prev = _load_prev_state()
    same_session = prev.get("log_session") == log_session
    prev_done = prev.get("code_done") if same_session else None
    delta_files = None
    if code_done is not None and isinstance(prev_done, int):
        delta_files = code_done - prev_done
        if delta_files < 0:
            delta_files = None

    prev_meta = prev.get("meta_chunks") if same_session else None
    delta_meta = None
    if meta_chunks is not None and isinstance(prev_meta, int):
        delta_meta = meta_chunks - prev_meta
        if delta_meta < 0:
            delta_meta = None

    meta_pct = None
    if meta_total and meta_chunks is not None and meta_total > 0:
        meta_pct = 100.0 * meta_chunks / meta_total

    clock = time.strftime("%H:%M")
    summary_ru = _format_summary_ru(
        clock=clock,
        phase=phase,
        finished=finished,
        code_done=code_done,
        code_total=code_total,
        pct=pct,
        code_chunks=code_chunks,
        code_in_db=code_in_db,
        rate_files_min=rate_files_min,
        eta_code_min=eta_code_min,
        meta_chunks=meta_chunks,
        meta_total=meta_total,
        meta_pct=meta_pct,
        delta_files=delta_files,
        delta_meta=delta_meta,
        stall=stall,
        stall_sec=stall_sec,
        need_code=need_code,
    )

    return {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "log_session": log_session,
        "code_in_db": code_in_db,
        "meta_in_db": meta_in_db,
        "phase": phase,
        "finished": finished,
        "code_done": code_done,
        "code_total": code_total,
        "code_chunks": code_chunks,
        "code_elapsed_sec": code_elapsed,
        "pct": pct,
        "rate_files_min": rate_files_min,
        "eta_code_min": eta_code_min,
        "meta_chunks": meta_chunks,
        "meta_total": meta_total,
        "meta_pct": meta_pct,
        "meta_elapsed_sec": meta_elapsed,
        "delta_files": delta_files,
        "delta_meta": delta_meta,
        "stall": stall,
        "stall_sec": stall_sec,
        "last_log": (lines[-1][:200] if lines else all_lines[-1][:200] if all_lines else ""),
        "summary_ru": summary_ru,
    }


def _format_summary_ru(
    *,
    clock: str,
    phase: str,
    finished: bool,
    code_done: int | None,
    code_total: int | None,
    pct: float | None,
    code_chunks: int | None,
    code_in_db: int | None,
    rate_files_min: float | None,
    eta_code_min: float | None,
    meta_chunks: int | None,
    meta_total: int | None,
    meta_pct: float | None,
    delta_files: int | None,
    delta_meta: int | None,
    stall: bool,
    stall_sec: int | None,
    need_code: bool = False,
) -> str:
    """Chat line always starts with local hh:mm."""
    if finished:
        return (
            f"{clock} Индексация ЗАВЕРШЕНА. чанков кода={code_chunks}, "
            f"чанков метаданных={meta_chunks}."
        )

    parts: list[str] = []
    if code_done is not None and code_total is not None and pct is not None:
        parts.append(f"код {code_done}/{code_total} ({pct:.1f}%)")
    elif code_in_db is not None and code_in_db == 0 and phase == "metadata":
        parts.append("код в базе 0 (нужна переиндексация)")
    if code_chunks is not None and code_done is not None:
        parts.append(f"чанков ~{code_chunks}")
    if rate_files_min is not None and phase == "code":
        parts.append(f"~{rate_files_min:.0f} файл/мин")
    if eta_code_min is not None and phase == "code":
        parts.append(f"оценка до конца кода ~{eta_code_min:.0f} мин")
    if phase == "metadata":
        if meta_chunks is not None and meta_total is not None and meta_pct is not None:
            parts.append(
                f"метаданные {meta_chunks}/{meta_total} ({meta_pct:.1f}%)"
            )
        elif meta_chunks is not None:
            parts.append(f"метаданные, записано={meta_chunks}")
    if phase == "code" and delta_files is not None:
        parts.append(f"за интервал {delta_files:+d} файлов")
    if phase == "metadata" and delta_meta is not None and delta_meta >= 0:
        parts.append(f"за интервал {delta_meta:+d} чанков")
    if need_code:
        parts.append("код в базе ещё не записан")
    if stall:
        parts.append(
            f"ВНИМАНИЕ: лог не обновлялся ~{stall_sec} с (возможно зависание)"
        )

    if not parts:
        phase_ru = _PHASE_RU.get(phase, phase)
        body = f"Индекс: фаза {phase_ru}"
    else:
        body = "Индекс: " + ", ".join(parts)
    remain_pct = None
    if phase == "metadata" and meta_pct is not None:
        remain_pct = max(0.0, 100.0 - meta_pct)
    elif phase == "code" and pct is not None:
        remain_pct = max(0.0, 100.0 - pct)
    line = f"{clock} {body}"
    if remain_pct is not None:
        line += f", осталось {remain_pct:.1f}%"
    return line


def write_status_files(info: dict[str, Any]) -> str:
    """Write status/md/state; return plain multi-line status text."""
    lines = [
        f"time={info['time']}",
        f"phase={info['phase']}",
        f"finished={info['finished']}",
        f"summary_ru={info['summary_ru']}",
    ]
    if info.get("code_done") is not None:
        lines.append(
            f"code: {info['code_done']}/{info['code_total']} "
            f"({info['pct']:.1f}%), chunks~{info['code_chunks']}, "
            f"rate={info['rate_files_min']:.1f} files/min"
            if info.get("pct") is not None and info.get("rate_files_min") is not None
            else f"code: {info['code_done']}/{info['code_total']}"
        )
    if info.get("eta_code_min") is not None and info["phase"] == "code":
        lines.append(f"ETA code remaining ~ {info['eta_code_min']:.0f} min")
    if info.get("last_log"):
        lines.append("last_log: " + info["last_log"])

    report = "\n".join(lines) + "\n"
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(report, encoding="utf-8")
    PROGRESS_MD.write_text(
        "# Прогресс индексации MCP 1С\n\n"
        f"Обновлено: **{info['time']}**\n\n"
        f"**Для чата:** {info['summary_ru']}\n\n```\n{report}```\n",
        encoding="utf-8",
    )

    prev = _load_prev_state()
    prev.update(
        {
            "last_tick_iso": info["time"],
            "last_tick_unix": time.time(),
            "finished": info["finished"],
            "phase": info["phase"],
            "code_done": info.get("code_done"),
            "code_total": info.get("code_total"),
            "meta_chunks": info.get("meta_chunks"),
            "meta_total": info.get("meta_total"),
            "log_session": info.get("log_session", ""),
            "summary_ru": info["summary_ru"],
            "mechanism": "single-shell-python-loop",
        }
    )
    STATE.write_text(json.dumps(prev, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    info = collect_progress()
    report = write_status_files(info)
    try:
        import sys

        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass
    print(report, end="")


if __name__ == "__main__":
    main()
