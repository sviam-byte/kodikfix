from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def _as_metric_list(metrics: Sequence[str] | None) -> list[str]:
    """Normalize scalar metric selection to list of non-empty names."""
    if metrics is None:
        return []
    return [str(m) for m in metrics if str(m)]


def _coerce_float(v) -> float:
    """Convert value to finite float or NaN when conversion is invalid."""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return float(x) if np.isfinite(x) else float("nan")


def scalar_error(
    value: float,
    target: float,
    *,
    scale: float | None = None,
    absolute: bool = True,
) -> float:
    """Compute scaled scalar error for one metric value vs target."""
    x = _coerce_float(value)
    t = _coerce_float(target)
    s = _coerce_float(scale if scale is not None else 1.0)

    if not np.isfinite(x) or not np.isfinite(t):
        return float("nan")
    if not np.isfinite(s) or s <= 1e-12:
        s = 1.0

    err = (x - t) / s
    return float(abs(err) if absolute else err)


def annotate_scalar_errors(
    traj_df: pd.DataFrame,
    *,
    target_vector: dict[str, float],
    metrics: Sequence[str],
    scales: dict[str, float] | None = None,
    absolute: bool = True,
) -> pd.DataFrame:
    """Annotate trajectory with per-metric scalar error columns."""
    out = traj_df.copy()
    metric_list = _as_metric_list(metrics)

    for m in metric_list:
        errs = []
        target = target_vector.get(m, np.nan)
        scale = (scales or {}).get(m, 1.0)
        for _, row in out.iterrows():
            errs.append(
                scalar_error(
                    row.get(m, np.nan),
                    target,
                    scale=scale,
                    absolute=absolute,
                )
            )
        out[f"{m}__scalar_error"] = errs

    return out


def find_best_scalar_match(
    traj_df: pd.DataFrame,
    *,
    target_vector: dict[str, float],
    metrics: Sequence[str],
    scales: dict[str, float] | None = None,
    absolute: bool = True,
) -> dict[str, dict]:
    """Find best trajectory step independently for each scalar metric."""
    scored = annotate_scalar_errors(
        traj_df,
        target_vector=target_vector,
        metrics=metrics,
        scales=scales,
        absolute=absolute,
    )

    out: dict[str, dict] = {}
    for m in _as_metric_list(metrics):
        ecol = f"{m}__scalar_error"
        if ecol not in scored.columns:
            out[m] = {
                "best_step": None,
                "best_damage_frac": np.nan,
                "best_scalar_error": np.nan,
                "best_value": np.nan,
            }
            continue

        valid = scored[np.isfinite(pd.to_numeric(scored[ecol], errors="coerce"))].copy()
        if valid.empty:
            out[m] = {
                "best_step": None,
                "best_damage_frac": np.nan,
                "best_scalar_error": np.nan,
                "best_value": np.nan,
            }
            continue

        idx = int(valid[ecol].astype(float).idxmin())
        row = valid.loc[idx]

        out[m] = {
            "best_step": int(row["step"]) if "step" in row and np.isfinite(row["step"]) else None,
            "best_damage_frac": float(row.get("damage_frac", np.nan)),
            "best_scalar_error": float(row[ecol]),
            "best_value": float(row.get(m, np.nan)),
        }

    return out


def build_scalar_subject_results(
    subject_results: pd.DataFrame,
    *,
    metrics: Sequence[str],
) -> pd.DataFrame:
    """Normalize scalar subject results into a stable tidy schema."""
    if subject_results is None or subject_results.empty:
        return pd.DataFrame(
            columns=[
                "subject_idx",
                "attack_kind",
                "metric",
                "best_step",
                "best_damage_frac",
                "best_scalar_error",
                "best_value",
            ]
        )

    cols = [
        "subject_idx",
        "attack_kind",
        "metric",
        "best_step",
        "best_damage_frac",
        "best_scalar_error",
        "best_value",
    ]
    out = subject_results.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
    return out[cols].copy()


def build_scalar_winners(scalar_subject_df: pd.DataFrame) -> pd.DataFrame:
    """Pick best attack for each subject x metric by minimum scalar error."""
    if scalar_subject_df is None or scalar_subject_df.empty:
        return pd.DataFrame(
            columns=[
                "subject_idx",
                "metric",
                "attack_kind",
                "best_step",
                "best_damage_frac",
                "best_scalar_error",
                "best_value",
            ]
        )

    df = scalar_subject_df.copy()
    df["best_scalar_error"] = pd.to_numeric(df["best_scalar_error"], errors="coerce")

    winners = []
    for (subject_idx, metric), sub in df.groupby(["subject_idx", "metric"], dropna=False):
        valid = sub[np.isfinite(sub["best_scalar_error"])].copy()
        if valid.empty:
            continue
        idx = int(valid["best_scalar_error"].astype(float).idxmin())
        row = valid.loc[idx].to_dict()
        winners.append(row)

    return pd.DataFrame(winners)


def build_scalar_summary(
    scalar_subject_df: pd.DataFrame,
    scalar_winners_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build metric x attack summary from scalar matching outputs."""
    cols = [
        "metric",
        "attack_kind",
        "n",
        "scalar_error_median",
        "scalar_error_mean",
        "winner_count",
        "winner_rate_within_metric",
    ]
    if scalar_subject_df is None or scalar_subject_df.empty:
        return pd.DataFrame(columns=cols)

    df = scalar_subject_df.copy()
    df["best_scalar_error"] = pd.to_numeric(df["best_scalar_error"], errors="coerce")

    win_counts = {}
    metric_totals = {}
    if scalar_winners_df is not None and not scalar_winners_df.empty:
        tmp = scalar_winners_df.copy()
        metric_totals = tmp.groupby("metric").size().to_dict()
        win_counts = tmp.groupby(["metric", "attack_kind"]).size().to_dict()

    rows = []
    for (metric, attack_kind), sub in df.groupby(["metric", "attack_kind"], dropna=False):
        err = pd.to_numeric(sub["best_scalar_error"], errors="coerce").dropna()
        wc = int(win_counts.get((metric, attack_kind), 0))
        total = int(metric_totals.get(metric, 0))
        rows.append(
            {
                "metric": metric,
                "attack_kind": attack_kind,
                "n": int(len(sub)),
                "scalar_error_median": float(err.median()) if not err.empty else np.nan,
                "scalar_error_mean": float(err.mean()) if not err.empty else np.nan,
                "winner_count": wc,
                "winner_rate_within_metric": float(wc / max(1, total)),
            }
        )

    out = pd.DataFrame(rows, columns=cols)
    if not out.empty:
        out = out.sort_values(
            ["metric", "winner_count", "scalar_error_median", "attack_kind"],
            ascending=[True, False, True, True],
        ).reset_index(drop=True)
    return out
