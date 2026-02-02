from __future__ import annotations

import numpy as np
import pandas as pd

AUC_TRAP = getattr(np, "trapezoid", None) or getattr(np, "trapz")


def auto_y_range(series: pd.Series, pad_frac: float = 0.08):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return None
    y0, y1 = float(s.min()), float(s.max())
    if not (np.isfinite(y0) and np.isfinite(y1)):
        return None
    if y0 == y1:
        eps = 1e-6 if y0 == 0 else abs(y0) * 0.05
        return [y0 - eps, y1 + eps]
    pad = (y1 - y0) * pad_frac
    return [y0 - pad, y1 + pad]


def apply_plot_defaults(fig, height: int = 780, y_range=None):
    fig.update_layout(height=height)
    fig.update_xaxes(showgrid=True, zeroline=False)
    fig.update_yaxes(showgrid=True, zeroline=True)
    if y_range is not None:
        fig.update_yaxes(range=y_range)
    return fig


def forward_fill_heavy(df_hist: pd.DataFrame) -> pd.DataFrame:
    df = df_hist.copy()
    for col in ["l2_lcc", "mod", "H_tri"]:
        if col in df.columns:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan).ffill()
    return df
