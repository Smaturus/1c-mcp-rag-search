"""
One-shot 10-minute wake for index progress → Cursor chat.

Unlike the infinite loop (dies with the agent session), this process:
  sleep(interval) → print AGENT_LOOP_TICK_index … → exit

Agent on wake: post summary_ru to chat, then start this script again
(background Shell + notify_on_output ^AGENT_LOOP_TICK_index).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from index_progress import collect_progress, write_status_files  # noqa: E402
from process_alive import process_alive  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "data" / "index_chat_arm.pid"
LAST_REPORT = ROOT / "data" / "index_chat_last_report.json"
SENTINEL = "AGENT_LOOP_TICK_index"


def acquire_lock() -> None:
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    if LOCK.exists():
        try:
            old = int(LOCK.read_text(encoding="utf-8").strip().splitlines()[0])
        except Exception:
            old = -1
        if old != os.getpid() and process_alive(old):
            print(
                f"[index-chat-arm] already armed pid={old}; exit",
                file=sys.stderr,
                flush=True,
            )
            raise SystemExit(2)
    LOCK.write_text(f"{os.getpid()}\n", encoding="utf-8")


def clear_lock() -> None:
    try:
        if LOCK.exists() and LOCK.read_text(encoding="utf-8").startswith(
            str(os.getpid())
        ):
            LOCK.unlink(missing_ok=True)
    except Exception:
        pass


def mark_report(summary: str, finished: bool) -> None:
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


def emit() -> bool:
    info = collect_progress()
    write_status_files(info)
    finished = bool(info["finished"])
    summary = info["summary_ru"]
    payload = {
        "prompt": (
            "Post to the USER CHAT this exact brief RU line: "
            f"{summary} "
            "Then re-arm: background Shell "
            "`python -u scripts/index_chat_arm_once.py --interval 600` "
            "with notify_on_output pattern ^AGENT_LOOP_TICK_index. "
            "If finished=true: stop arm, update progress.md."
        ),
        "summary_ru": summary,
        "finished": finished,
        "phase": info["phase"],
        "code_done": info.get("code_done"),
        "code_total": info.get("code_total"),
        "mechanism": "one-shot-arm",
    }
    print(f"{SENTINEL} {json.dumps(payload, ensure_ascii=False)}", flush=True)
    if finished:
        print("AGENT_LOOP_TICK_index_DONE", flush=True)
    return finished


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=int, default=600)
    parser.add_argument(
        "--now",
        action="store_true",
        help="Emit immediately (no sleep) then exit",
    )
    args = parser.parse_args()

    try:
        sys.stdout.reconfigure(line_buffering=True, encoding="utf-8")  # type: ignore
    except Exception:
        pass

    acquire_lock()
    try:
        if not args.now:
            time.sleep(max(args.interval, 1))
        finished = emit()
        return 0
    finally:
        clear_lock()


if __name__ == "__main__":
    raise SystemExit(main())
