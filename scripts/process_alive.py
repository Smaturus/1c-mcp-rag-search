"""Reliable process liveness for Windows and POSIX.

On Windows, OpenProcess alone is not enough: a handle can exist for a process
that has already exited. Check STILL_ACTIVE (259) via GetExitCodeProcess.
"""

from __future__ import annotations

import os
import sys


def process_alive(pid: int) -> bool:
    """Return True only if ``pid`` is a running process."""
    if pid <= 0:
        return False
    if sys.platform != "win32":
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True
    import ctypes

    handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
    if not handle:
        return False
    try:
        exit_code = ctypes.c_ulong()
        ok = ctypes.windll.kernel32.GetExitCodeProcess(
            handle, ctypes.byref(exit_code)
        )
        return bool(ok) and exit_code.value == 259
    finally:
        ctypes.windll.kernel32.CloseHandle(handle)
