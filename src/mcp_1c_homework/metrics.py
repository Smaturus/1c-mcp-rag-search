"""Vector similarity helpers (cosine already via Chroma; Pearson for rerank)."""

from __future__ import annotations

import math
from collections.abc import Sequence


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def pearson_correlation(a: Sequence[float], b: Sequence[float]) -> float:
    """
    Pearson correlation = cosine after per-vector mean centering.

    Returns value in [-1, 1]; 0 if a vector is constant (zero variance).
    """
    if len(a) != len(b) or not a:
        return 0.0
    ma, mb = _mean(a), _mean(b)
    num = 0.0
    da = 0.0
    db = 0.0
    for x, y in zip(a, b, strict=False):
        xa = x - ma
        yb = y - mb
        num += xa * yb
        da += xa * xa
        db += yb * yb
    denom = math.sqrt(da) * math.sqrt(db)
    if denom < 1e-12:
        return 0.0
    return num / denom
