"""Start one full reindex and append its output to data/index_full.log as UTF-8.

The parent waits so the writer is not left detached. Progress ticks must not
use this script to stop the writer.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "data" / "index_full.log"


def main() -> int:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8", buffering=1) as log:
        log.write(
            "\n[restart] 2026-10-20 пересчёт с нуля: сначала метаданные, затем код\n"
        )
        log.flush()
        proc = subprocess.Popen(
            [
                sys.executable,
                "-u",
                "-m",
                "mcp_1c_rag.indexer",
                "--target",
                "all",
                "--reset",
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            cwd=str(ROOT),
        )
        return proc.wait()


if __name__ == "__main__":
    raise SystemExit(main())
