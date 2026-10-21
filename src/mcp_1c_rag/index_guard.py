"""One writer for the vector base, and a checkpoint to continue after a stop.

The running indexer creates ``data/index.lock``. Search and a second indexer
refuse to open the base while that process is alive. Checkpoints are per
target (code / metadata) and record only batches that were successfully
written, so a later run can skip finished files and the tail of a broken file.
"""

from __future__ import annotations

import contextlib
import json
import os
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from mcp_1c_rag.config import DATA_DIR

# Chat progress is every 10 minutes. The writer must emit a line at least that often.
PROGRESS_INTERVAL_SEC = 600


def progress_due(last_emit: float, now: float, *, interval: float = PROGRESS_INTERVAL_SEC) -> bool:
    """True when the next progress line is due."""
    return (now - last_emit) >= interval


class IndexBusy(RuntimeError):
    """Raised when another live process already owns the vector base."""


def pid_alive(pid: int) -> bool:
    """Return True if ``pid`` is a running process."""
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        from typing import Any, cast

        # ``ctypes.windll`` exists only on Windows; casting the module to Any
        # keeps this file importable and type-checkable on Linux/macOS runners.
        kernel32 = cast("Any", ctypes).windll.kernel32
        handle = kernel32.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        try:
            exit_code = ctypes.c_ulong()
            ok = kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
            # 259 == STILL_ACTIVE. A handle alone is not proof the process runs.
            return bool(ok) and exit_code.value == 259
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def lock_path(directory: Path | None = None) -> Path:
    """Path of the single-writer lock file."""
    return (directory or DATA_DIR) / "index.lock"


def checkpoint_path(target: str, directory: Path | None = None) -> Path:
    """Path of the checkpoint for one index target."""
    safe = {"code": "code", "metadata": "metadata"}.get(target, target)
    return (directory or DATA_DIR) / f"index_checkpoint_{safe}.json"


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _holder_pids(data: dict) -> list[int]:
    pids: list[int] = []
    primary = int(data.get("pid") or 0)
    if primary > 0:
        pids.append(primary)
    for extra in data.get("extra_pids") or []:
        try:
            pid = int(extra)
        except (TypeError, ValueError):
            continue
        if pid > 0 and pid not in pids:
            pids.append(pid)
    return pids


def _drop_lock(path: Path) -> None:
    with contextlib.suppress(OSError):
        path.unlink(missing_ok=True)


def assert_output_not_inside_sources(output: Path, sources: list[Path]) -> None:
    """Refuse to write when the output directory sits inside a source tree.

    Source trees are read-only. A mistaken path must not create, edit, or
    delete anything under them.
    """
    out = output.resolve()
    for source in sources:
        try:
            src = source.resolve()
        except OSError:
            continue
        if not src.exists():
            continue
        if out == src or out.is_relative_to(src):
            raise RuntimeError(
                f"Запись в {out} запрещена: этот каталог внутри источника {src}."
            )


def lock_holder_alive(path: Path | None = None) -> bool:
    """True if the lock names a live process. Does not delete the lock.

    The progress reader must use this. Removing a lock is a writer's decision:
    a reader that deletes it lets a second process open the base mid-snapshot.
    """
    path = path or lock_path()
    data = _read_json(path)
    if not data:
        return False
    return any(pid_alive(pid) for pid in _holder_pids(data))


def live_holder_pids(path: Path | None = None) -> list[int]:
    """Return live pids from the lock. A stale lock is removed."""
    path = path or lock_path()
    data = _read_json(path)
    if not data:
        if path.exists():
            _drop_lock(path)
        return []
    alive = [pid for pid in _holder_pids(data) if pid_alive(pid)]
    if not alive:
        _drop_lock(path)
    return alive


def foreign_writer_pids(path: Path | None = None) -> list[int]:
    """Live lock holders other than this process."""
    me = os.getpid()
    return [pid for pid in live_holder_pids(path) if pid != me]


def refuse_if_foreign_writer(path: Path | None = None) -> None:
    """Raise IndexBusy when another live indexer owns the base."""
    foreign = foreign_writer_pids(path)
    if not foreign:
        return
    pid = foreign[0]
    raise IndexBusy(
        f"Базу сейчас пишет процесс {pid}. "
        "Поиск и второй прогон отключены, пока запись не закончится."
    )


def _write_lock_exclusive(path: Path, payload: dict) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    try:
        fd = os.open(str(path), flags)
    except FileExistsError:
        return False
    try:
        os.write(fd, json.dumps(payload, ensure_ascii=False).encode("utf-8"))
    finally:
        os.close(fd)
    return True


def acquire_lock(target: str, path: Path | None = None) -> None:
    """Take the single-writer lock or raise IndexBusy."""
    path = path or lock_path()
    payload = {
        "pid": os.getpid(),
        "extra_pids": [],
        "target": target,
        "unix": time.time(),
    }
    for _ in range(3):
        foreign = foreign_writer_pids(path)
        if foreign:
            raise IndexBusy(
                f"Базу сейчас пишет процесс {foreign[0]}. "
                "Второй прогон не запущен."
            )
        if os.getpid() in live_holder_pids(path):
            return
        if _write_lock_exclusive(path, payload):
            return
    raise IndexBusy("Не удалось занять блокировку записи индекса.")


def adopt_running_writer(
    pids: list[int],
    target: str,
    path: Path | None = None,
) -> None:
    """Record an indexer that is already running (it cannot lock itself)."""
    path = path or lock_path()
    alive = [pid for pid in pids if pid_alive(pid)]
    if not alive:
        raise IndexBusy("Нет живого процесса индексации для блокировки.")
    foreign = [pid for pid in foreign_writer_pids(path) if pid not in alive]
    if foreign:
        raise IndexBusy(
            f"Блокировка уже принадлежит процессу {foreign[0]}."
        )
    _drop_lock(path)
    payload = {
        "pid": alive[0],
        "extra_pids": alive[1:],
        "target": target,
        "unix": time.time(),
        "adopted": True,
    }
    if not _write_lock_exclusive(path, payload):
        raise IndexBusy("Не удалось записать блокировку текущего прогона.")


def release_lock(path: Path | None = None) -> None:
    """Remove the lock if this process owns it. A foreign lock is left in place."""
    path = path or lock_path()
    data = _read_json(path)
    if not data:
        return
    if int(data.get("pid") or 0) != os.getpid():
        return
    _drop_lock(path)


@dataclass
class IndexProgress:
    """Chunks known to be durable, plus the unfinished file tail."""

    target: str
    completed: dict[str, int] = field(default_factory=dict)
    partial_source: str | None = None
    partial_chunks: int = 0
    started_unix: float | None = None

    def stored_chunks(self) -> int:
        """Chunks already written and recorded."""
        return sum(self.completed.values()) + self.partial_chunks

    def skip_chunks(self, source: str) -> int:
        """How many leading chunks of ``source`` are already stored."""
        if source == self.partial_source:
            return self.partial_chunks
        return 0

    def apply_batch(
        self,
        batch_sources: list[str],
        current: str | None,
        current_finished: bool,
    ) -> None:
        """Fold one successful batch into completed files or the open tail."""
        counts = Counter(batch_sources)
        for source, count in counts.items():
            finishing = bool(current_finished) or source != current
            if finishing:
                add = count
                if self.partial_source == source:
                    add += self.partial_chunks
                    self.partial_source = None
                    self.partial_chunks = 0
                self.completed[source] = self.completed.get(source, 0) + add
                continue
            if self.partial_source not in (None, source):
                prev = self.partial_source
                if prev is not None:
                    self.completed[prev] = (
                        self.completed.get(prev, 0) + self.partial_chunks
                    )
                self.partial_chunks = 0
            self.partial_source = source
            self.partial_chunks += count

    def mark_source_fully_flushed(self, source: str) -> None:
        """Mark a file whose chunks are all in the base (nothing left in memory)."""
        if self.partial_source == source:
            self.completed[source] = (
                self.completed.get(source, 0) + self.partial_chunks
            )
            self.partial_source = None
            self.partial_chunks = 0
        elif source not in self.completed:
            self.completed[source] = 0


def save_progress(progress: IndexProgress, path: Path | None = None) -> None:
    """Atomically store a checkpoint."""
    path = path or checkpoint_path(progress.target)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "target": progress.target,
        "completed": progress.completed,
        "partial_source": progress.partial_source,
        "partial_chunks": progress.partial_chunks,
        "started_unix": progress.started_unix,
        "unix": time.time(),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def load_progress(target: str, path: Path | None = None) -> IndexProgress | None:
    """Load a checkpoint for ``target``. Missing or other-target files yield None."""
    path = path or checkpoint_path(target)
    data = _read_json(path)
    if not data or data.get("target") != target:
        return None
    completed_raw = data.get("completed") or {}
    if not isinstance(completed_raw, dict):
        return None
    completed: dict[str, int] = {}
    for key, value in completed_raw.items():
        try:
            completed[str(key)] = int(value)
        except (TypeError, ValueError):
            return None
    partial = data.get("partial_source")
    partial_source = str(partial) if partial else None
    try:
        partial_chunks = int(data.get("partial_chunks") or 0)
    except (TypeError, ValueError):
        return None
    started_raw = data.get("started_unix")
    started_unix: float | None
    try:
        started_unix = float(started_raw) if started_raw else None
    except (TypeError, ValueError):
        started_unix = None
    return IndexProgress(
        target=target,
        completed=completed,
        partial_source=partial_source,
        partial_chunks=partial_chunks,
        started_unix=started_unix,
    )


def empty_progress(target: str) -> IndexProgress:
    """A checkpoint with nothing stored yet."""
    return IndexProgress(target=target)
