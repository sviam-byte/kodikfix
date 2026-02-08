from __future__ import annotations

from typing import Iterable

import numpy as np

from .config import JUMP_ALPHA, JUMP_FRACTION_FALLBACK


def compute_null_threshold(jumps: Iterable[float]) -> float:
    """Return the phase-transition threshold from null-model jump samples."""
    samples = [float(v) for v in jumps if np.isfinite(v)]
    if not samples:
        return float(JUMP_FRACTION_FALLBACK)
    return float(np.quantile(samples, 1.0 - JUMP_ALPHA))
