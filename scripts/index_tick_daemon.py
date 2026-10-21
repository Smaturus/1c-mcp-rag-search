"""Deprecated: use index_tick_loop.py (single-shell Cursor loop)."""

from __future__ import annotations

import sys

print(
    "DEPRECATED: use scripts/index_tick_loop.py --interval 600. "
    "See .cursor/rules/index-progress-ticks.mdc",
    file=sys.stderr,
)
raise SystemExit(3)
