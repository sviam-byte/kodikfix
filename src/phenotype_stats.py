from __future__ import annotations

from itertools import combinations
from math import erf

import numpy as np
import pandas as pd


def _safe_num(s: pd.Series) -> pd.Series:
    """Convert input series to numeric with NaN fallback."""
    return pd.to_numeric(s, errors="coerce")


def _normal_cdf(x: float) -> float:
    """Standard normal CDF approximation via erf."""
    return 0.5 * (1.0 + erf(float(x) / np.sqrt(2.0)))


def _rankdata_average(a: np.ndarray) -> np.ndarray:
    """Compute average ranks with stable tie handling."""
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and a[order[j + 1]] == a[order[i]]:
            j += 1
        avg_rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def _chi2_sf_approx(x: float, df: int) -> float:
    """Approximate chi-square survival function using Wilson-Hilferty transform."""
    if df <= 0:
        return np.nan
    x = float(max(0.0, x))
    k = float(df)
    z = ((x / k) ** (1.0 / 3.0) - (1.0 - 2.0 / (9.0 * k))) / np.sqrt(2.0 / (9.0 * k))
    return float(max(0.0, min(1.0, 1.0 - _normal_cdf(z))))


def _binom_two_sided_pvalue(n_pos: int, n_neg: int) -> float:
    """Compute exact two-sided sign-test p-value under p=0.5."""
    n = int(n_pos + n_neg)
    if n <= 0:
        return np.nan
    k = int(min(n_pos, n_neg))
    from math import comb

    p_lo = sum(comb(n, i) for i in range(0, k + 1)) / (2**n)
    return float(min(1.0, 2.0 * p_lo))


def friedman_test_from_subject_results(subject_df: pd.DataFrame) -> dict:
    """Run Friedman test approximation over per-subject best_distance by attack kind."""
    if subject_df is None or subject_df.empty:
        return {"test": "friedman", "n_subjects": 0, "k_attacks": 0, "statistic": np.nan, "pvalue": np.nan}

    df = subject_df.copy()
    df["best_distance"] = _safe_num(df["best_distance"])
    pivot = df.pivot_table(index="subject_id" if "subject_id" in df.columns else "subject_idx", columns="attack_kind", values="best_distance", aggfunc="first")
    pivot = pivot.dropna(axis=0, how="any")
    n, k = pivot.shape
    if n < 2 or k < 2:
        return {"test": "friedman", "n_subjects": int(n), "k_attacks": int(k), "statistic": np.nan, "pvalue": np.nan}

    ranks = np.vstack([_rankdata_average(row.values.astype(float)) for _, row in pivot.iterrows()])
    rj = ranks.sum(axis=0)
    stat = (12.0 / (n * k * (k + 1.0))) * np.sum(rj**2) - 3.0 * n * (k + 1.0)
    p = _chi2_sf_approx(stat, k - 1)
    return {"test": "friedman", "n_subjects": int(n), "k_attacks": int(k), "statistic": float(stat), "pvalue": float(p)}


def pairwise_wilcoxon_like(subject_df: pd.DataFrame) -> pd.DataFrame:
    """Compute dependency-light paired comparisons using sign-test style stats."""
    columns = ["attack_a", "attack_b", "n_pairs", "median_diff_a_minus_b", "a_better_count", "b_better_count", "ties_count", "sign_test_pvalue"]
    if subject_df is None or subject_df.empty:
        return pd.DataFrame(columns=columns)

    df = subject_df.copy()
    df["best_distance"] = _safe_num(df["best_distance"])
    pivot = df.pivot_table(index="subject_id" if "subject_id" in df.columns else "subject_idx", columns="attack_kind", values="best_distance", aggfunc="first")
    attacks = list(pivot.columns)
    rows = []
    for a, b in combinations(attacks, 2):
        sub = pivot[[a, b]].dropna()
        if sub.empty:
            rows.append({"attack_a": a, "attack_b": b, "n_pairs": 0, "median_diff_a_minus_b": np.nan, "a_better_count": 0, "b_better_count": 0, "ties_count": 0, "sign_test_pvalue": np.nan})
            continue
        diff = sub[a].astype(float) - sub[b].astype(float)
        a_better = int((diff < 0).sum())
        b_better = int((diff > 0).sum())
        ties = int((diff == 0).sum())
        rows.append({
            "attack_a": a,
            "attack_b": b,
            "n_pairs": int(len(sub)),
            "median_diff_a_minus_b": float(diff.median()) if len(diff) else np.nan,
            "a_better_count": a_better,
            "b_better_count": b_better,
            "ties_count": ties,
            "sign_test_pvalue": _binom_two_sided_pvalue(a_better, b_better),
        })
    out = pd.DataFrame(rows, columns=columns)
    if not out.empty:
        out = out.sort_values(["sign_test_pvalue", "attack_a", "attack_b"], ascending=[True, True, True]).reset_index(drop=True)
    return out


def matched_severity_pairwise_stats(subject_df: pd.DataFrame, *, severity_col: str = "best_delta_density", gap_quantile: float = 0.5, max_gap_floor: float = 1e-9) -> pd.DataFrame:
    """Pairwise comparisons restricted to subject rows with matched severity."""
    columns = ["severity_col", "attack_a", "attack_b", "max_allowed_gap", "n_pairs_total", "n_pairs_matched", "median_abs_gap", "median_diff_a_minus_b", "a_better_count", "b_better_count", "ties_count", "sign_test_pvalue"]
    if subject_df is None or subject_df.empty or severity_col not in subject_df.columns:
        return pd.DataFrame(columns=columns)

    df = subject_df.copy()
    df["best_distance"] = _safe_num(df.get("best_distance", pd.Series(dtype=float)))
    df[severity_col] = _safe_num(df.get(severity_col, pd.Series(dtype=float)))
    idx_col = "subject_id" if "subject_id" in df.columns else "subject_idx"
    dist_pivot = df.pivot_table(index=idx_col, columns="attack_kind", values="best_distance", aggfunc="first")
    sev_pivot = df.pivot_table(index=idx_col, columns="attack_kind", values=severity_col, aggfunc="first")
    attacks = [a for a in dist_pivot.columns if a in sev_pivot.columns]

    rows = []
    for a, b in combinations(attacks, 2):
        sub = pd.DataFrame({"a_dist": dist_pivot[a], "b_dist": dist_pivot[b], "a_sev": sev_pivot[a], "b_sev": sev_pivot[b]}).dropna()
        if sub.empty:
            rows.append({"severity_col": severity_col, "attack_a": a, "attack_b": b, "max_allowed_gap": np.nan, "n_pairs_total": 0, "n_pairs_matched": 0, "median_abs_gap": np.nan, "median_diff_a_minus_b": np.nan, "a_better_count": 0, "b_better_count": 0, "ties_count": 0, "sign_test_pvalue": np.nan})
            continue

        sub["abs_gap"] = (sub["a_sev"] - sub["b_sev"]).abs()
        thr = float(sub["abs_gap"].quantile(float(gap_quantile))) if len(sub) else float("nan")
        if not np.isfinite(thr):
            thr = float(max_gap_floor)
        thr = max(float(thr), float(max_gap_floor))
        matched = sub[sub["abs_gap"] <= thr].copy()
        diff = matched["a_dist"].astype(float) - matched["b_dist"].astype(float) if not matched.empty else pd.Series(dtype=float)
        a_better = int((diff < 0).sum()) if not diff.empty else 0
        b_better = int((diff > 0).sum()) if not diff.empty else 0
        ties = int((diff == 0).sum()) if not diff.empty else 0
        rows.append({
            "severity_col": severity_col,
            "attack_a": a,
            "attack_b": b,
            "max_allowed_gap": thr,
            "n_pairs_total": int(len(sub)),
            "n_pairs_matched": int(len(matched)),
            "median_abs_gap": float(sub["abs_gap"].median()) if not sub.empty else np.nan,
            "median_diff_a_minus_b": float(diff.median()) if not diff.empty else np.nan,
            "a_better_count": a_better,
            "b_better_count": b_better,
            "ties_count": ties,
            "sign_test_pvalue": _binom_two_sided_pvalue(a_better, b_better),
        })

    out = pd.DataFrame(rows, columns=columns)
    if not out.empty:
        out = out.sort_values(["severity_col", "sign_test_pvalue", "attack_a", "attack_b"], ascending=[True, True, True, True]).reset_index(drop=True)
    return out


def permutation_style_winner_test(winners_df: pd.DataFrame, *, total_subjects: int | None = None) -> pd.DataFrame:
    """Build winner-count table against equal-share null baseline."""
    cols = ["attack_kind", "win_count", "expected_under_null", "excess_vs_null"]
    if winners_df is None or winners_df.empty:
        return pd.DataFrame(columns=cols)
    df = winners_df.copy()
    if total_subjects is None:
        total_subjects = int(df["subject_id"].nunique()) if "subject_id" in df.columns else (int(df["subject_idx"].nunique()) if "subject_idx" in df.columns else int(len(df)))

    counts = df["attack_kind"].value_counts(dropna=False)
    k = int(len(counts))
    expected = float(total_subjects / max(1, k))
    rows = [{"attack_kind": attack_kind, "win_count": int(win_count), "expected_under_null": expected, "excess_vs_null": float(win_count - expected)} for attack_kind, win_count in counts.items()]
    out = pd.DataFrame(rows, columns=cols)
    return out.sort_values(["win_count", "attack_kind"], ascending=[False, True]).reset_index(drop=True)


def build_stats_tables(result: dict) -> dict[str, pd.DataFrame]:
    """Build all statistics tables used in phenotype report outputs."""
    subject_df = result.get("subject_results", pd.DataFrame())
    winners_df = result.get("winner_results", pd.DataFrame())

    fried_df = pd.DataFrame([friedman_test_from_subject_results(subject_df)])
    if isinstance(subject_df, pd.DataFrame) and not subject_df.empty and "subject_id" in subject_df.columns:
        total_subjects = int(subject_df["subject_id"].nunique())
    elif isinstance(subject_df, pd.DataFrame) and not subject_df.empty and "subject_idx" in subject_df.columns:
        total_subjects = int(subject_df["subject_idx"].nunique())
    else:
        total_subjects = 0

    return {
        "stats_overall": fried_df,
        "stats_pairwise": pairwise_wilcoxon_like(subject_df),
        "stats_pairwise_matched_delta_density": matched_severity_pairwise_stats(subject_df, severity_col="best_delta_density"),
        "stats_pairwise_matched_delta_total_weight": matched_severity_pairwise_stats(subject_df, severity_col="best_delta_total_weight"),
        "stats_winners": permutation_style_winner_test(winners_df, total_subjects=total_subjects),
    }
