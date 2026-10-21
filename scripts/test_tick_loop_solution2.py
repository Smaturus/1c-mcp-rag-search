"""
Integration test: persistent index_tick_loop (solution 2).
Uses 15s interval; does not touch index_full.log content.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv" / "Scripts" / "python.exe"
LOOP = ROOT / "scripts" / "index_tick_loop.py"
CTL = ROOT / "scripts" / "index_tick_ctl.py"
LOCK = ROOT / "data" / "index_tick.pid"
REPORT = ROOT / "data" / "test_tick_loop_report.json"
INTERVAL = 15

sys.path.insert(0, str(ROOT / "scripts"))
from index_progress import summary_ru_is_chat_safe  # noqa: E402


def run(args: list[str], *, timeout: float = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(PY), "-u", *args],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )


def stop_all() -> None:
    run([str(CTL), "stop"], timeout=30)


def scenario_once() -> dict:
    stop_all()
    r = run([str(LOOP), "--once"], timeout=60)
    ok = r.returncode == 0 and r.stdout.startswith("AGENT_LOOP_TICK_index ")
    payload = {}
    if ok:
        payload = json.loads(r.stdout.split(" ", 1)[1].strip())
    ru = payload.get("summary_ru", "")
    no_en = summary_ru_is_chat_safe(ru)
    return {
        "name": "once",
        "ok": ok and no_en and bool(ru),
        "returncode": r.returncode,
        "summary_sample": ru[:120],
        "no_english_mix": no_en,
    }


def scenario_duplicate_loop() -> dict:
    stop_all()
    proc = subprocess.Popen(
        [str(PY), "-u", str(LOOP), "--interval", str(INTERVAL)],
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    time.sleep(2)
    r2 = run([str(LOOP), "--interval", str(INTERVAL)], timeout=10)
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
    stop_all()
    return {
        "name": "duplicate_loop",
        "ok": r2.returncode == 2 and "already running" in (r2.stderr or ""),
        "second_returncode": r2.returncode,
        "stderr": (r2.stderr or "").strip()[:200],
    }


def scenario_stale_lock() -> dict:
    stop_all()
    LOCK.write_text("999999\n", encoding="utf-8")
    r = run([str(LOOP), "--once"], timeout=60)
    stop_all()
    return {
        "name": "stale_lock",
        "ok": r.returncode == 0 and "AGENT_LOOP_TICK_index" in r.stdout,
        "returncode": r.returncode,
    }


def scenario_interval_ticks() -> dict:
    """Background loop 15s: expect >=2 ticks in ~35s."""
    stop_all()
    proc = subprocess.Popen(
        [str(PY), "-u", str(LOOP), "--interval", str(INTERVAL)],
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    ticks: list[str] = []
    deadline = time.time() + 38
    while time.time() < deadline and proc.poll() is None:
        line = proc.stdout.readline() if proc.stdout else ""
        if line.startswith("AGENT_LOOP_TICK_index "):
            ticks.append(line.strip())
        elif not line and proc.poll() is not None:
            break
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
    stop_all()
    return {
        "name": "interval_15s",
        "ok": len(ticks) >= 2,
        "tick_count": len(ticks),
        "interval_sec": INTERVAL,
    }


def scenario_finished_exit() -> dict:
    """Mock finished=True → loop exits after first tick post-sleep."""
    stop_all()
    dbg = ROOT / "scripts" / "_debug_finished_tick.py"
    r = run([str(dbg)], timeout=15)
    stop_all()
    return {
        "name": "finished_exit",
        "ok": r.returncode == 0
        and "AGENT_LOOP_TICK_index_DONE" in r.stdout
        and "ЗАВЕРШЕНА" in r.stdout,
        "returncode": r.returncode,
    }


def scenario_stop_hook_overdue() -> dict:
    """Simulate overdue last_report → stop-hook should emit followup only for owner."""
    stop_all()
    time.sleep(0.5)
    hook = ROOT / ".cursor" / "hooks" / "index_chat_stop.py"
    owner = ROOT / "data" / "index_chat_owner.json"
    last = ROOT / "data" / "index_chat_last_report.json"
    owner_cid = "test-owner-conversation-id"
    owner.write_text(
        json.dumps(
            {
                "conversation_id": owner_cid,
                "claimed_unix": time.time(),
                "note": "test",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    last.write_text(
        json.dumps(
            {"unix": time.time() - 700, "summary_ru": "тест", "finished": False},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    # Foreign chat must stay silent
    foreign = subprocess.run(
        [str(PY), str(hook)],
        input=json.dumps(
            {"status": "completed", "conversation_id": "other-chat-id"}
        ),
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
    )
    foreign_out = (foreign.stdout or "").strip()
    foreign_silent = "followup_message" not in foreign_out

    stdin = json.dumps({"status": "completed", "conversation_id": owner_cid})
    r = subprocess.run(
        [str(PY), str(hook)],
        input=stdin,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
    )
    out = (r.stdout or "").strip()
    has_followup = "followup_message" in out and "index_tick_loop" in out
    return {
        "name": "stop_hook_overdue",
        "ok": r.returncode == 0 and has_followup and foreign.returncode == 0 and foreign_silent,
        "stdout_head": out[:200],
        "foreign_silent": foreign_silent,
        "ticker_alive_after_stop": LOCK.exists(),
    }


def main() -> int:
    results = [
        scenario_once(),
        scenario_duplicate_loop(),
        scenario_stale_lock(),
        scenario_interval_ticks(),
        scenario_finished_exit(),
        scenario_stop_hook_overdue(),
    ]
    all_ok = all(r["ok"] for r in results)
    payload = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "interval_test_sec": INTERVAL,
        "all_ok": all_ok,
        "results": results,
    }
    REPORT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for r in results:
        mark = "OK" if r["ok"] else "FAIL"
        line = f"[{mark}] {r['name']}: {json.dumps(r, ensure_ascii=True)}\n"
        sys.stdout.buffer.write(line.encode("utf-8", errors="replace"))
    print(f"report={REPORT}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
