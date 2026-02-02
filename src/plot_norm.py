"""Normalization helpers for plotting curves."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def normalize_series(s: pd.Series, mode: str) -> pd.Series:
    """
    Normalize a numeric series for plotting.
    Modes:
      - "none": no change
      - "rel0": y / y0 (if y0==0 -> keep as-is)
      - "delta0": y - y0
      - "minmax": (y - min) / (max - min)
      - "zscore": (y - mean) / std
    """
    if s is None:
        return s
    mode = (mode or "none").lower().strip()
    x = pd.to_numeric(s, errors="coerce")
    if mode == "none":
        return x

    if x.dropna().empty:
        return x

    y0 = x.dropna().iloc[0]

    if mode == "rel0":
        if y0 == 0 or not np.isfinite(y0):
            return x
        return x / float(y0)

    if mode == "delta0":
        if not np.isfinite(y0):
            return x
        return x - float(y0)

    if mode == "minmax":
        mn = float(x.min(skipna=True))
        mx = float(x.max(skipna=True))
        if not (np.isfinite(mn) and np.isfinite(mx)) or mn == mx:
            return x * 0.0
        return (x - mn) / (mx - mn)

    if mode == "zscore":
        mu = float(x.mean(skipna=True))
        sd = float(x.std(skipna=True))
        if not (np.isfinite(mu) and np.isfinite(sd)) or sd == 0:
            return x * 0.0
        return (x - mu) / sd

    return x


def normalize_df(
    df: pd.DataFrame,
    y_col: str,
    mode: str,
    out_col: Optional[str] = None,
) -> pd.DataFrame:
    """Return a copy with normalized column."""
    if df is None or df.empty or y_col not in df.columns:
        return df
    d = df.copy()
    out = out_col or y_col
    d[out] = normalize_series(d[y_col], mode)
    return d
