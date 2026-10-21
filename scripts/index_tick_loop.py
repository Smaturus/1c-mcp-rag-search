"""
Single-shell index progress loop (Cursor local best practice).

Per Cursor loop skill (monitored shell output):
  - ONE background shell, stdout attached to the terminal
  - notify_on_output on ^AGENT_LOOP_TICK_index
  - quiet loop: emit sentinel JSON only (summary_ru inside for chat)
  - no PowerShell while/Sleep, no Start-Process Hidden, no second watcher
  - PID lock prevents duplicate loops

While the indexer writes, this loop asks Windows not to idle-sleep: a sleeping
PC cannot deliver chat ticks. After resume, a wall-clock jump emits at once.
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from index_progress import collect_progress, write_status_files  # noqa: E402
from mcp_1c_homework.index_guard import lock_holder_alive  # noqa: E402
from mcp_1c_homework.keep_awake import allow_sleep, prevent_sleep  # noqa: E402
from process_alive import process_alive  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "data" / "index_tick.pid"
LAST_REPORT = ROOT / "data" / "index_chat_last_report.json"
ERR_LOG = ROOT / "data" / "index_tick_errors.log"
SENTINEL = "AGENT_LOOP_TICK_index"
LAST_LINE = ROOT / "data" / "index_tick_last_line.txt"
# If wall clock jumps by more than this past the interval, treat as wake/resume.
WAKE_SLACK_SEC = 45
# After this many failed stdout writes in a row, exit so a fresh shell can be armed.
_STDOUT_FAILS = 0
_STDOUT_FAIL_LIMIT = 2


def _log_error(exc: BaseException) -> None:
    ERR_LOG.parent.mkdir(parents=True, exist_ok=True)
    with ERR_LOG.open("a", encoding="utf-8") as fh:
        fh.write(time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
        fh.write("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
        fh.write("\n")


def _write_sentinel_line(line: str) -> bool:
    """Write one notify line. Cursor watches stdout; Windows pipes die after long runs.

    Always mirrors the line to a file. Returns False when stdout is broken.
    """
    global _STDOUT_FAILS
    LAST_LINE.parent.mkdir(parents=True, exist_ok=True)
    try:
        LAST_LINE.write_text(line + "\n", encoding="utf-8")
    except OSError:
        pass
    payload = (line + "\n").encode("utf-8", errors="replace")
    for _ in range(2):
        try:
            buf = getattr(sys.stdout, "buffer", None)
            if buf is not None:
                buf.write(payload)
                buf.flush()
            else:
                print(line, flush=True)
            _STDOUT_FAILS = 0
            return True
        except OSError as exc:
            _log_error(exc)
            try:
                sys.stdout = open(  # noqa: SIM115 — must rebind stdout after broken pipe
                    1, mode="w", encoding="utf-8", buffering=1, closefd=False
                )
            except OSError:
                pass
    _STDOUT_FAILS += 1
    return False


def acquire_lock() -> None:
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    if LOCK.exists():
        try:
            old = int(LOCK.read_text(encoding="utf-8").strip().splitlines()[0])
        except Exception:
            old = -1
        if old != os.getpid() and process_alive(old):
            print(
                f"[index-tick-loop] already running pid={old}; exit",
                file=sys.stderr,
                flush=True,
            )
            raise SystemExit(2)
    LOCK.write_text(f"{os.getpid()}\n", encoding="utf-8")

    def _clear() -> None:
        try:
            if LOCK.exists() and LOCK.read_text(encoding="utf-8").startswith(
                str(os.getpid())
            ):
                LOCK.unlink(missing_ok=True)
        except Exception:
            pass
        allow_sleep()

    atexit.register(_clear)


def _sync_keep_awake() -> None:
    """While a live indexer holds the write lock, do not let the PC idle-sleep."""
    if lock_holder_alive():
        prevent_sleep()
    else:
        allow_sleep()


def _mark_report(summary: str, finished: bool) -> None:
    LAST_REPORT.parent.mkdir(parents=True, exist_ok=True)
    LAST_REPORT.write_text(
        json.dumps(
            {
                "unix": time.time(),
                "summary_ru": summary,
                "finished": finished,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def emit_tick() -> bool:
    """Write status files and print one sentinel line. Returns finished."""
    _sync_keep_awake()
    info = collect_progress()
    write_status_files(info)
    finished = bool(info["finished"])
    summary = info["summary_ru"]
    _mark_report(summary, finished)
    prompt = (
        "Напиши пользователю в чат одну строку (только по-русски, без английских слов): "
        f"{summary} "
        "Не предлагай открывать файлы. "
        "Если индексация завершена — останови фоновый цикл оповещений и обнови progress.md."
    )
    payload = {
        "prompt": prompt,
        "summary_ru": summary,
        "finished": finished,
        "phase": info["phase"],
        "code_done": info.get("code_done"),
        "code_total": info.get("code_total"),
    }
    line = f"{SENTINEL} {json.dumps(payload, ensure_ascii=False)}"
    if not _write_sentinel_line(line):
        raise OSError("stdout pipe broken for progress notify")
    if finished:
        _write_sentinel_line("AGENT_LOOP_TICK_index_DONE")
        allow_sleep()
    return finished


def emit_tick_safe() -> bool | None:
    """Like emit_tick, but never raises. None means the tick failed."""
    try:
        return emit_tick()
    except Exception as exc:  # noqa: BLE001 — keep the loop alive
        _log_error(exc)
        fallback = (
            time.strftime("%H:%M")
            + " Индекс: строка прогресса временно недоступна, запись не останавливалась"
        )
        try:
            _mark_report(fallback, False)
            payload = {
                "prompt": (
                    "Напиши пользователю в чат одну строку (только по-русски): "
                    f"{fallback}"
                ),
                "summary_ru": fallback,
                "finished": False,
                "phase": "unknown",
                "code_done": None,
                "code_total": None,
            }
            line = f"{SENTINEL} {json.dumps(payload, ensure_ascii=False)}"
            _write_sentinel_line(line)
        except Exception as nested:  # noqa: BLE001
            _log_error(nested)
        if _STDOUT_FAILS >= _STDOUT_FAIL_LIMIT:
            # Fresh Cursor shell + notify is required; staying alive is useless.
            raise SystemExit(3)
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=int, default=600)
    parser.add_argument(
        "--once",
        action="store_true",
        help="Emit one tick and exit (smoke test)",
    )
    args = parser.parse_args()

    try:
        sys.stdout.reconfigure(line_buffering=True, encoding="utf-8")  # type: ignore
    except Exception:
        pass

    acquire_lock()
    _sync_keep_awake()

    if args.once:
        result = emit_tick_safe()
        return 0 if result is not False else 0

    interval = max(args.interval, 1)
    last_emit = 0.0
    last_wall = time.time()

    def wait_for_next_deadline() -> None:
        nonlocal last_emit, last_wall
        while True:
            _sync_keep_awake()
            now = time.time()
            # Sleep/hibernate: wall clock jumps while this process was frozen.
            if now - last_wall > WAKE_SLACK_SEC:
                last_wall = now
                return
            last_wall = now
            if last_emit <= 0 or (now - last_emit) >= interval:
                return
            time.sleep(min(5.0, max(0.5, interval - (now - last_emit))))

    first = emit_tick_safe()
    if first is True:
        return 0
    last_emit = time.time()
    last_wall = last_emit

    while True:
        wait_for_next_deadline()
        result = emit_tick_safe()
        if result is True:
            return 0
        last_emit = time.time()
        last_wall = last_emit


if __name__ == "__main__":
    raise SystemExit(main())
