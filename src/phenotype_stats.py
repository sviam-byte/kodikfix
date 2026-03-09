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
    p = min(1.0, 2.0 * p_lo)
    return float(p)


def friedman_test_from_subject_results(subject_df: pd.DataFrame) -> dict:
    """Run Friedman test approximation over per-subject best_distance by attack kind."""
    if subject_df is None or subject_df.empty:
        return {
            "test": "friedman",
            "n_subjects": 0,
            "k_attacks": 0,
            "statistic": np.nan,
            "pvalue": np.nan,
        }

    df = subject_df.copy()
    df["best_distance"] = _safe_num(df["best_distance"])

    pivot = df.pivot_table(
        index="subject_idx",
        columns="attack_kind",
        values="best_distance",
        aggfunc="first",
    )
    pivot = pivot.dropna(axis=0, how="any")

    n, k = pivot.shape
    if n < 2 or k < 2:
        return {
            "test": "friedman",
            "n_subjects": int(n),
            "k_attacks": int(k),
            "statistic": np.nan,
            "pvalue": np.nan,
        }

    ranks = np.vstack([_rankdata_average(row.values.astype(float)) for _, row in pivot.iterrows()])
    Rj = ranks.sum(axis=0)
    stat = (12.0 / (n * k * (k + 1.0))) * np.sum(Rj**2) - 3.0 * n * (k + 1.0)
    p = _chi2_sf_approx(stat, k - 1)

    return {
        "test": "friedman",
        "n_subjects": int(n),
        "k_attacks": int(k),
        "statistic": float(stat),
        "pvalue": float(p),
    }


def pairwise_wilcoxon_like(subject_df: pd.DataFrame) -> pd.DataFrame:
    """Compute dependency-light paired comparisons using sign-test style stats."""
    columns = [
        "attack_a",
        "attack_b",
        "n_pairs",
        "median_diff_a_minus_b",
        "a_better_count",
        "b_better_count",
        "ties_count",
        "sign_test_pvalue",
    ]

    if subject_df is None or subject_df.empty:
        return pd.DataFrame(columns=columns)

    df = subject_df.copy()
    df["best_distance"] = _safe_num(df["best_distance"])

    pivot = df.pivot_table(
        index="subject_idx",
        columns="attack_kind",
        values="best_distance",
        aggfunc="first",
    )

    attacks = list(pivot.columns)
    rows = []

    for a, b in combinations(attacks, 2):
        sub = pivot[[a, b]].dropna()
        if sub.empty:
            rows.append(
                {
                    "attack_a": a,
                    "attack_b": b,
                    "n_pairs": 0,
                    "median_diff_a_minus_b": np.nan,
                    "a_better_count": 0,
                    "b_better_count": 0,
                    "ties_count": 0,
                    "sign_test_pvalue": np.nan,
                }
            )
            continue

        diff = sub[a].astype(float) - sub[b].astype(float)
        a_better = int((diff < 0).sum())
        b_better = int((diff > 0).sum())
        ties = int((diff == 0).sum())
        p = _binom_two_sided_pvalue(a_better, b_better)

        rows.append(
            {
                "attack_a": a,
                "attack_b": b,
                "n_pairs": int(len(sub)),
                "median_diff_a_minus_b": float(diff.median()) if len(diff) else np.nan,
                "a_better_count": a_better,
                "b_better_count": b_better,
                "ties_count": ties,
                "sign_test_pvalue": p,
            }
        )

    out = pd.DataFrame(rows, columns=columns)
    if not out.empty:
        out = out.sort_values(["sign_test_pvalue", "attack_a", "attack_b"], ascending=[True, True, True]).reset_index(drop=True)
    return out


def permutation_style_winner_test(winners_df: pd.DataFrame, *, total_subjects: int | None = None) -> pd.DataFrame:
    """Build winner-count table against equal-share null baseline."""
    cols = ["attack_kind", "win_count", "expected_under_null", "excess_vs_null"]
    if winners_df is None or winners_df.empty:
        return pd.DataFrame(columns=cols)

    df = winners_df.copy()
    if total_subjects is None:
        total_subjects = int(df["subject_idx"].nunique()) if "subject_idx" in df.columns else int(len(df))

    counts = df["attack_kind"].value_counts(dropna=False)
    k = int(len(counts))
    expected = float(total_subjects / max(1, k))

    rows = []
    for attack_kind, win_count in counts.items():
        rows.append(
            {
                "attack_kind": attack_kind,
                "win_count": int(win_count),
                "expected_under_null": expected,
                "excess_vs_null": float(win_count - expected),
            }
        )

    out = pd.DataFrame(rows, columns=cols)
    return out.sort_values(["win_count", "attack_kind"], ascending=[False, True]).reset_index(drop=True)


def build_stats_tables(result: dict) -> dict[str, pd.DataFrame]:
    """Build all statistics tables used in phenotype report outputs."""
    subject_df = result.get("subject_results", pd.DataFrame())
    winners_df = result.get("winner_results", pd.DataFrame())

    fried = friedman_test_from_subject_results(subject_df)
    fried_df = pd.DataFrame([fried])

    if isinstance(subject_df, pd.DataFrame) and not subject_df.empty and "subject_idx" in subject_df.columns:
        total_subjects = int(subject_df["subject_idx"].nunique())
    else:
        total_subjects = 0

    return {
        "stats_overall": fried_df,
        "stats_pairwise": pairwise_wilcoxon_like(subject_df),
        "stats_winners": permutation_style_winner_test(winners_df, total_subjects=total_subjects),
    }
