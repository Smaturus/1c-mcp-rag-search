"""Control helper: status / stop / claim for index chat arm (and legacy loop)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from index_chat_owner import (  # noqa: E402
    claim_from_last,
    clear_owner,
    owner_conversation_id,
    read_last_conversation_id,
    read_owner,
)
from process_alive import process_alive  # noqa: E402

ARM_LOCK = ROOT / "data" / "index_chat_arm.pid"
LEGACY_LOCK = ROOT / "data" / "index_tick.pid"
STATE = ROOT / "data" / "index_tick_state.json"
LAST = ROOT / "data" / "index_chat_last_report.json"

LOOP_CMD = (
    r"D:\MCP_1С_homework\.venv\Scripts\python.exe -u "
    r"D:\MCP_1С_homework\scripts\index_tick_loop.py --interval 600"
)


def _alive(pid: int) -> bool:
    return process_alive(pid)


def _read_pid(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        return int(path.read_text(encoding="utf-8").strip().splitlines()[0])
    except Exception:
        return None


def _kill(pid: int) -> None:
    if sys.platform == "win32":
        os.system(f"taskkill /PID {pid} /F >NUL 2>&1")
    else:
        os.kill(pid, 15)


def cmd_status() -> int:
    arm_pid = _read_pid(ARM_LOCK)
    leg_pid = _read_pid(LEGACY_LOCK)
    arm_ok = bool(arm_pid and _alive(arm_pid))
    leg_ok = bool(leg_pid and _alive(leg_pid))
    own = read_owner()
    last_cid = read_last_conversation_id()
    owner_cid = owner_conversation_id()
    print(f"arm_pid={arm_pid} arm_alive={arm_ok}")
    print(f"legacy_loop_pid={leg_pid} legacy_alive={leg_ok}")
    print(f"owner={own}")
    print(f"last_conversation_id={last_cid}")
    print(
        f"this_submit_is_owner="
        f"{bool(owner_cid and last_cid and owner_cid == last_cid)}"
    )
    if LAST.exists():
        try:
            print(f"last_report={json.loads(LAST.read_text(encoding='utf-8'))}")
        except Exception:
            pass
    if STATE.exists():
        try:
            st = json.loads(STATE.read_text(encoding="utf-8"))
            print(f"last_summary={st.get('summary_ru', '')}")
            print(f"finished={st.get('finished')}")
        except Exception:
            pass
    print(f"start_cmd={LOOP_CMD}")
    return 0 if (leg_ok or arm_ok) else 1


def cmd_stop() -> int:
    for path in (ARM_LOCK, LEGACY_LOCK):
        pid = _read_pid(path)
        if pid and _alive(pid):
            _kill(pid)
            print(f"killed {pid} ({path.name})")
        path.unlink(missing_ok=True)
    return 0


def cmd_claim(note: str) -> int:
    try:
        payload = claim_from_last(note=note)
    except RuntimeError as exc:
        print(f"claim_failed={exc}", file=sys.stderr)
        return 2
    print(f"claimed={json.dumps(payload, ensure_ascii=False)}")
    return 0


def cmd_unclaim() -> int:
    clear_owner()
    print("owner_cleared=True")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("status", "stop", "claim", "unclaim"))
    p.add_argument(
        "--note",
        default="reindex",
        help="optional label stored with claim",
    )
    args = p.parse_args()
    if args.action == "status":
        return cmd_status()
    if args.action == "stop":
        return cmd_stop()
    if args.action == "claim":
        return cmd_claim(args.note)
    return cmd_unclaim()


if __name__ == "__main__":
    raise SystemExit(main())
