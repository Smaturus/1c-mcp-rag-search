"""Re-export for scripts; implementation lives in the package."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mcp_1c_rag.keep_awake import allow_sleep, prevent_sleep  # noqa: E402

__all__ = ["allow_sleep", "prevent_sleep"]
