"""Lock and checkpoint tests. They do not open the vector base."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from index_chat_owner import (  # noqa: E402
    STUB_CONVERSATION_ID,
    extract_conversation_id,
    is_real_conversation_id,
)
from index_progress import run_is_finished  # noqa: E402
from mcp_1c_homework.index_guard import (  # noqa: E402
    IndexBusy,
    IndexProgress,
    acquire_lock,
    adopt_running_writer,
    assert_output_not_inside_sources,
    foreign_writer_pids,
    load_progress,
    lock_holder_alive,
    progress_due,  # noqa: E402
    release_lock,
    save_progress,
)
from process_alive import process_alive  # noqa: E402


def test_batch_tail_stays_partial() -> None:
    progress = IndexProgress(target="metadata")
    progress.apply_batch(["report.txt"] * 64, "report.txt", False)
    assert progress.completed == {}
    assert progress.partial_source == "report.txt"
    assert progress.partial_chunks == 64
    assert progress.skip_chunks("report.txt") == 64


def test_finished_file_folds_partial() -> None:
    progress = IndexProgress(
        target="metadata", partial_source="report.txt", partial_chunks=64
    )
    progress.apply_batch(["report.txt"] * 36, "report.txt", True)
    assert progress.completed["report.txt"] == 100
    assert progress.partial_source is None
    assert progress.stored_chunks() == 100


def test_batch_closes_previous_file() -> None:
    progress = IndexProgress(target="code")
    progress.apply_batch(
        ["a.bsl"] * 30 + ["b.bsl"] * 34,
        "b.bsl",
        False,
    )
    assert progress.completed["a.bsl"] == 30
    assert progress.partial_source == "b.bsl"
    assert progress.partial_chunks == 34


def test_exact_batch_promotes_when_buffer_empty() -> None:
    progress = IndexProgress(target="code")
    progress.apply_batch(["a.bsl"] * 64, "a.bsl", False)
    progress.mark_source_fully_flushed("a.bsl")
    assert progress.completed["a.bsl"] == 64
    assert progress.partial_chunks == 0


def test_checkpoint_roundtrip(tmp: Path) -> None:
    progress = IndexProgress(
        target="code",
        completed={"a.bsl": 3},
        partial_source="b.bsl",
        partial_chunks=10,
    )
    path = tmp / "index_checkpoint_code.json"
    save_progress(progress, path)
    loaded = load_progress("code", path)
    assert loaded is not None
    assert loaded.skip_chunks("b.bsl") == 10
    assert loaded.completed["a.bsl"] == 3


def test_lock_blocks_foreign_and_ignores_stale(tmp: Path) -> None:
    path = tmp / "index.lock"
    acquire_lock("metadata", path)
    assert foreign_writer_pids(path) == []
    release_lock(path)
    assert not path.exists()

    path.write_text('{"pid": 99999999, "extra_pids": [], "target": "code"}', encoding="utf-8")
    acquire_lock("code", path)
    release_lock(path)

    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        adopt_running_writer([proc.pid], "metadata", path)
        assert proc.pid in foreign_writer_pids(path)
        try:
            acquire_lock("code", path)
            raise AssertionError("second writer was allowed")
        except IndexBusy:
            pass
        release_lock(path)
        assert path.exists()
    finally:
        proc.kill()
        proc.wait(timeout=10)
    deadline = time.time() + 5
    while time.time() < deadline and foreign_writer_pids(path):
        time.sleep(0.1)
    assert foreign_writer_pids(path) == []


def test_output_inside_source_is_refused(tmp: Path) -> None:
    source = tmp / "source"
    source.mkdir()
    nested = source / "out"
    nested.mkdir()
    try:
        assert_output_not_inside_sources(nested, [source])
        raise AssertionError("write inside source was allowed")
    except RuntimeError:
        pass
    outside = tmp / "outside"
    outside.mkdir()
    assert_output_not_inside_sources(outside, [source])
    nested.rmdir()
    source.rmdir()
    outside.rmdir()


def test_lock_peek_does_not_drop_stale_lock(tmp: Path) -> None:
    path = tmp / "index.lock"
    path.write_text(
        '{"pid": 99999999, "extra_pids": [], "target": "code"}',
        encoding="utf-8",
    )
    assert lock_holder_alive(path) is False
    assert path.exists()


def test_progress_due_every_ten_minutes() -> None:
    assert progress_due(0.0, 599.0) is False
    assert progress_due(0.0, 600.0) is True


def test_stub_conversation_is_not_real() -> None:
    assert is_real_conversation_id(STUB_CONVERSATION_ID) is False
    assert is_real_conversation_id("abc-real-id") is True
    assert extract_conversation_id({"conversationId": "cid-1"}) == "cid-1"
    assert process_alive(99999999) is False
    assert process_alive(os.getpid()) is True


def test_metadata_alone_is_not_finished() -> None:
    assert run_is_finished(code_finished=False, meta_finished=True) is False
    assert run_is_finished(code_finished=True, meta_finished=False) is False
    assert run_is_finished(code_finished=True, meta_finished=True) is True


def main() -> int:
    tmp = ROOT / "data" / "_index_guard_test"
    if tmp.exists():
        for child in tmp.iterdir():
            if child.is_dir():
                for nested in child.iterdir():
                    nested.rmdir() if nested.is_dir() else nested.unlink()
                child.rmdir()
            else:
                child.unlink()
    tmp.mkdir(parents=True, exist_ok=True)
    test_batch_tail_stays_partial()
    test_finished_file_folds_partial()
    test_batch_closes_previous_file()
    test_exact_batch_promotes_when_buffer_empty()
    test_checkpoint_roundtrip(tmp)
    test_lock_blocks_foreign_and_ignores_stale(tmp)
    test_output_inside_source_is_refused(tmp)
    test_lock_peek_does_not_drop_stale_lock(tmp)
    test_metadata_alone_is_not_finished()
    test_progress_due_every_ten_minutes()
    test_stub_conversation_is_not_real()
    for child in tmp.iterdir():
        child.unlink()
    tmp.rmdir()
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
