from __future__ import annotations

"""Inference helpers for phenotype scalar/family claim readiness tables."""

from itertools import combinations
from math import comb

import numpy as np
import pandas as pd


def _binom_two_sided_pvalue(n_pos: int, n_neg: int) -> float:
    """Exact two-sided sign-test p-value under p=0.5."""
    n = int(n_pos + n_neg)
    if n <= 0:
        return float("nan")
    k = int(min(n_pos, n_neg))
    probs = [comb(n, i) for i in range(0, k + 1)]
    p = float(sum(probs)) / float(2 ** n)
    return min(1.0, 2.0 * p)


def benjamini_hochberg(p_values: list[float] | np.ndarray | pd.Series) -> list[float]:
    """Benjamini-Hochberg FDR correction with NaN-safe handling."""
    vals = [float(x) if pd.notna(x) else np.nan for x in list(p_values)]
    n = len(vals)
    order = sorted([(v, i) for i, v in enumerate(vals) if np.isfinite(v)], key=lambda t: t[0])
    q = [np.nan] * n
    if not order:
        return q
    m = len(order)
    prev = 1.0
    for rank_rev, (pval, idx) in enumerate(reversed(order), start=1):
        rank = m - rank_rev + 1
        cur = min(prev, pval * m / max(1, rank))
        prev = cur
        q[idx] = float(min(1.0, max(0.0, cur)))
    return q


def cliffs_delta(a, b) -> float:
    """Compute Cliff's delta effect size (a vs b)."""
    a = np.asarray([x for x in a if np.isfinite(x)], dtype=float)
    b = np.asarray([x for x in b if np.isfinite(x)], dtype=float)
    if a.size == 0 or b.size == 0:
        return float("nan")
    gt = 0
    lt = 0
    for x in a:
        gt += int(np.sum(x > b))
        lt += int(np.sum(x < b))
    return float((gt - lt) / (a.size * b.size))


def _paired_table(df: pd.DataFrame, group_cols: list[str], value_col: str) -> pd.DataFrame:
    """Build per-group paired attack comparison table with FDR/effect size."""
    rows = []
    cols = group_cols + [
        "attack_a", "attack_b", "n_pairs", "median_diff", "mean_diff",
        "a_better_count", "b_better_count", "sign_test_p", "cliffs_delta", "fdr_q_value",
    ]
    if df is None or df.empty:
        return pd.DataFrame(columns=cols)

    work = df.copy()
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    work = work.dropna(subset=[value_col])
    if work.empty:
        return pd.DataFrame(columns=cols)

    idx_col = "subject_id" if "subject_id" in work.columns else "subject_idx"
    for key, sub in work.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        pivot = sub.pivot_table(index=idx_col, columns="attack_kind", values=value_col, aggfunc="first")
        attacks = [c for c in pivot.columns if str(c)]
        for a, b in combinations(attacks, 2):
            pair = pivot[[a, b]].dropna()
            if pair.empty:
                continue
            diff = pd.to_numeric(pair[a], errors="coerce") - pd.to_numeric(pair[b], errors="coerce")
            diff = diff[np.isfinite(diff)]
            if diff.empty:
                continue
            a_better = int((diff < 0).sum())
            b_better = int((diff > 0).sum())
            row = {col: val for col, val in zip(group_cols, key)}
            row.update({
                "attack_a": a,
                "attack_b": b,
                "n_pairs": int(len(diff)),
                "median_diff": float(diff.median()),
                "mean_diff": float(diff.mean()),
                "a_better_count": a_better,
                "b_better_count": b_better,
                "sign_test_p": _binom_two_sided_pvalue(a_better, b_better),
                "cliffs_delta": cliffs_delta(pair[a].to_numpy(), pair[b].to_numpy()),
            })
            rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=cols)
    out["fdr_q_value"] = benjamini_hochberg(out["sign_test_p"].tolist())
    return out.sort_values(group_cols + ["fdr_q_value", "sign_test_p", "attack_a", "attack_b"]).reset_index(drop=True)


def build_scalar_inference_table(result: dict) -> pd.DataFrame:
    """Inference table per metric x family for scalar errors."""
    scalar_subject = result.get("scalar_subject_results", pd.DataFrame())
    metric_family_map = result.get("metric_family_map", {}) or {}
    if scalar_subject is None or scalar_subject.empty:
        return pd.DataFrame()
    work = scalar_subject.copy()
    if "metric_family" not in work.columns:
        work["metric_family"] = work["metric"].map(lambda x: metric_family_map.get(x, x))
    return _paired_table(work, ["metric", "metric_family"], "best_scalar_error")


def build_family_inference_table(result: dict) -> pd.DataFrame:
    """Inference table aggregated by metric family."""
    scalar_subject = result.get("scalar_subject_results", pd.DataFrame())
    metric_family_map = result.get("metric_family_map", {}) or {}
    if scalar_subject is None or scalar_subject.empty:
        return pd.DataFrame()
    work = scalar_subject.copy()
    if "metric_family" not in work.columns:
        work["metric_family"] = work["metric"].map(lambda x: metric_family_map.get(x, x))
    work["best_scalar_error"] = pd.to_numeric(work["best_scalar_error"], errors="coerce")
    idx_col = "subject_id" if "subject_id" in work.columns else "subject_idx"
    agg = work.groupby([idx_col, "attack_kind", "metric_family"], dropna=False)["best_scalar_error"].median().reset_index()
    return _paired_table(agg, ["metric_family"], "best_scalar_error")


def build_core_claim_readiness(result: dict, suite: dict | None = None) -> pd.DataFrame:
    """Compute lightweight readiness score for claiming robust phenotype effects."""
    warning_flags = suite.get("warning_flags") if suite is not None else None
    if warning_flags is None:
        warning_flags = result.get("warning_flags", pd.DataFrame())
    if warning_flags is None or not isinstance(warning_flags, pd.DataFrame) or warning_flags.empty:
        warning_flags = pd.DataFrame(columns=["flag", "value", "status", "details"])

    active_flags = pd.DataFrame()
    if not warning_flags.empty:
        if "value" in warning_flags.columns:
            vals = warning_flags["value"].astype(str).str.lower()
            active_flags = warning_flags[vals.isin(["true", "1"])]
        elif "is_triggered" in warning_flags.columns:
            vals = warning_flags["is_triggered"].astype(str).str.lower()
            active_flags = warning_flags[vals.isin(["true", "1"])]

    winners = result.get("winner_results", pd.DataFrame())
    summary_winners = result.get("summary_winners", pd.DataFrame())
    top_win_rate = np.nan
    if isinstance(summary_winners, pd.DataFrame) and not summary_winners.empty and "win_rate" in summary_winners.columns:
        top_win_rate = float(pd.to_numeric(summary_winners["win_rate"], errors="coerce").max())

    scalar_inf = build_scalar_inference_table(result)
    fam_inf = build_family_inference_table(result)
    robust_scalar_hits = int((pd.to_numeric(scalar_inf.get("fdr_q_value", pd.Series(dtype=float)), errors="coerce") <= 0.10).sum()) if isinstance(scalar_inf, pd.DataFrame) and not scalar_inf.empty else 0
    robust_family_hits = int((pd.to_numeric(fam_inf.get("fdr_q_value", pd.Series(dtype=float)), errors="coerce") <= 0.10).sum()) if isinstance(fam_inf, pd.DataFrame) and not fam_inf.empty else 0
    has_subject_winners = isinstance(winners, pd.DataFrame) and not winners.empty

    score = 0
    score += 1 if has_subject_winners else 0
    score += 1 if np.isfinite(top_win_rate) and top_win_rate >= 0.45 else 0
    score += 1 if robust_scalar_hits > 0 else 0
    score += 1 if robust_family_hits > 0 else 0
    score += 1 if active_flags.empty else 0
    status = "strong" if score >= 5 else ("moderate" if score >= 3 else "weak")
    return pd.DataFrame([{
        "claim_status": status,
        "readiness_score": int(score),
        "max_score": 5,
        "has_subject_winners": bool(has_subject_winners),
        "top_win_rate": float(top_win_rate) if np.isfinite(top_win_rate) else np.nan,
        "robust_scalar_hits_q10": int(robust_scalar_hits),
        "robust_family_hits_q10": int(robust_family_hits),
        "active_warning_flags": int(len(active_flags)),
        "active_flag_names": ", ".join(active_flags["flag"].astype(str).tolist()) if not active_flags.empty and "flag" in active_flags.columns else "",
    }])
