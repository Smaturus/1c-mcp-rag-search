"""Ask Windows not to sleep while long indexing runs.

While the machine is already asleep, no process can post to chat. Keep the
system awake for the duration of the writer, and clear the request on exit.
"""

from __future__ import annotations

import atexit
import sys

_ES_CONTINUOUS = 0x80000000
_ES_SYSTEM_REQUIRED = 0x00000001
_ES_AWAYMODE_REQUIRED = 0x00000040
_armed = False


def _set_thread_execution_state(flags: int) -> None:
    """Call the Windows API; kept separate so mypy is happy on non-Windows CI."""
    import ctypes

    # ``ctypes.windll`` exists only on Windows; getattr avoids a hard attribute
    # reference that fails type checking on Linux/macOS.
    getattr(ctypes, "windll").kernel32.SetThreadExecutionState(flags)


def prevent_sleep() -> None:
    """Hold an execution state request so the PC should not idle-sleep."""
    global _armed
    if sys.platform != "win32":
        return

    flags = _ES_CONTINUOUS | _ES_SYSTEM_REQUIRED | _ES_AWAYMODE_REQUIRED
    _set_thread_execution_state(flags)
    if not _armed:
        atexit.register(allow_sleep)
        _armed = True


def allow_sleep() -> None:
    """Clear the execution state request."""
    global _armed
    if sys.platform != "win32":
        return

    _set_thread_execution_state(_ES_CONTINUOUS)
    _armed = False
