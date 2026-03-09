from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .phenotype_stats import build_stats_tables


def _safe_numeric(s: pd.Series) -> pd.Series:
    """Convert Series to numeric with NaN for non-parsable values."""
    return pd.to_numeric(s, errors="coerce")


def build_attack_summary(subject_df: pd.DataFrame) -> pd.DataFrame:
    """Build per-attack compact summary over all subject best matches."""
    if subject_df is None or subject_df.empty:
        return pd.DataFrame(
            columns=[
                "attack_kind",
                "n",
                "best_distance_median",
                "best_distance_mean",
                "best_distance_min",
                "best_distance_max",
                "best_step_median",
                "best_damage_frac_median",
            ]
        )

    df = subject_df.copy()
    if "best_distance" in df.columns:
        df["best_distance"] = _safe_numeric(df["best_distance"])
    if "best_step" in df.columns:
        df["best_step"] = _safe_numeric(df["best_step"])
    if "best_damage_frac" in df.columns:
        df["best_damage_frac"] = _safe_numeric(df["best_damage_frac"])

    rows = []
    for attack_kind, sub in df.groupby("attack_kind", dropna=False):
        dist = _safe_numeric(sub.get("best_distance", pd.Series(dtype=float))).dropna()
        step = _safe_numeric(sub.get("best_step", pd.Series(dtype=float))).dropna()
        dmg = _safe_numeric(sub.get("best_damage_frac", pd.Series(dtype=float))).dropna()

        rows.append(
            {
                "attack_kind": attack_kind,
                "n": int(len(sub)),
                "best_distance_median": float(dist.median()) if not dist.empty else np.nan,
                "best_distance_mean": float(dist.mean()) if not dist.empty else np.nan,
                "best_distance_min": float(dist.min()) if not dist.empty else np.nan,
                "best_distance_max": float(dist.max()) if not dist.empty else np.nan,
                "best_step_median": float(step.median()) if not step.empty else np.nan,
                "best_damage_frac_median": float(dmg.median()) if not dmg.empty else np.nan,
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty and "best_distance_median" in out.columns:
        out = out.sort_values(["best_distance_median", "attack_kind"], ascending=[True, True]).reset_index(drop=True)
    return out


def build_winner_summary(winners_df: pd.DataFrame, *, total_subjects: int | None = None) -> pd.DataFrame:
    """Build per-attack compact summary over winner rows only."""
    if winners_df is None or winners_df.empty:
        return pd.DataFrame(
            columns=[
                "attack_kind",
                "win_count",
                "win_rate",
                "winning_distance_median",
                "winning_step_median",
                "winning_damage_frac_median",
            ]
        )

    df = winners_df.copy()
    if total_subjects is None:
        total_subjects = int(df["subject_idx"].nunique()) if "subject_idx" in df.columns else int(len(df))

    if "best_distance" in df.columns:
        df["best_distance"] = _safe_numeric(df["best_distance"])
    if "best_step" in df.columns:
        df["best_step"] = _safe_numeric(df["best_step"])
    if "best_damage_frac" in df.columns:
        df["best_damage_frac"] = _safe_numeric(df["best_damage_frac"])

    rows = []
    for attack_kind, sub in df.groupby("attack_kind", dropna=False):
        dist = _safe_numeric(sub.get("best_distance", pd.Series(dtype=float))).dropna()
        step = _safe_numeric(sub.get("best_step", pd.Series(dtype=float))).dropna()
        dmg = _safe_numeric(sub.get("best_damage_frac", pd.Series(dtype=float))).dropna()

        rows.append(
            {
                "attack_kind": attack_kind,
                "win_count": int(len(sub)),
                "win_rate": float(len(sub) / max(1, int(total_subjects))),
                "winning_distance_median": float(dist.median()) if not dist.empty else np.nan,
                "winning_step_median": float(step.median()) if not step.empty else np.nan,
                "winning_damage_frac_median": float(dmg.median()) if not dmg.empty else np.nan,
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(
            ["win_count", "winning_distance_median", "attack_kind"],
            ascending=[False, True, True],
        ).reset_index(drop=True)
    return out


def build_target_vector_df(target_vector: dict) -> pd.DataFrame:
    """Convert target vector mapping to tabular view."""
    if not target_vector:
        return pd.DataFrame(columns=["metric", "target_value"])
    return pd.DataFrame([{"metric": str(k), "target_value": v} for k, v in target_vector.items()])


def build_scales_df(scales: dict) -> pd.DataFrame:
    """Convert scales mapping to tabular view."""
    if not scales:
        return pd.DataFrame(columns=["metric", "scale"])
    return pd.DataFrame([{"metric": str(k), "scale": v} for k, v in scales.items()])


def build_paper_ready_summary(result: dict) -> dict[str, pd.DataFrame]:
    """Build compact, stats, and scalar analysis tables for phenotype reports."""
    subject_df = result.get("subject_results", pd.DataFrame())
    winners_df = result.get("winner_results", pd.DataFrame())
    scalar_subject_df = result.get("scalar_subject_results", pd.DataFrame())
    scalar_winners_df = result.get("scalar_winners", pd.DataFrame())
    scalar_summary_df = result.get("scalar_summary", pd.DataFrame())

    total_subjects = 0
    if isinstance(subject_df, pd.DataFrame) and not subject_df.empty and "subject_idx" in subject_df.columns:
        total_subjects = int(subject_df["subject_idx"].nunique())

    stats_pack = build_stats_tables(result)

    return {
        "summary_attack": build_attack_summary(subject_df),
        "summary_winners": build_winner_summary(winners_df, total_subjects=total_subjects),
        "target_vector": build_target_vector_df(result.get("target_vector", {})),
        "scales": build_scales_df(result.get("scales", {})),
        "stats_overall": stats_pack["stats_overall"],
        "stats_pairwise": stats_pack["stats_pairwise"],
        "stats_winners": stats_pack["stats_winners"],
        "scalar_subject_results": scalar_subject_df if isinstance(scalar_subject_df, pd.DataFrame) else pd.DataFrame(),
        "scalar_winners": scalar_winners_df if isinstance(scalar_winners_df, pd.DataFrame) else pd.DataFrame(),
        "scalar_summary": scalar_summary_df if isinstance(scalar_summary_df, pd.DataFrame) else pd.DataFrame(),
    }


def export_phenotype_match_excel(
    result: dict,
    out_path: str | Path,
) -> Path:
    """Export a full workbook with raw trajectory tables and summary/stat sheets."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    winners_df = result.get("winner_results", pd.DataFrame())
    subject_df = result.get("subject_results", pd.DataFrame())
    traj_df = result.get("trajectory_results", pd.DataFrame())
    extra = build_paper_ready_summary(result)

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        (winners_df if isinstance(winners_df, pd.DataFrame) else pd.DataFrame()).to_excel(
            writer, index=False, sheet_name="winners"
        )
        (subject_df if isinstance(subject_df, pd.DataFrame) else pd.DataFrame()).to_excel(
            writer, index=False, sheet_name="subject_results"
        )
        (traj_df if isinstance(traj_df, pd.DataFrame) else pd.DataFrame()).to_excel(
            writer, index=False, sheet_name="trajectories"
        )
        extra["summary_attack"].to_excel(writer, index=False, sheet_name="summary_attack")
        extra["summary_winners"].to_excel(writer, index=False, sheet_name="summary_winners")
        extra["target_vector"].to_excel(writer, index=False, sheet_name="target_vector")
        extra["scales"].to_excel(writer, index=False, sheet_name="scales")
        extra["stats_overall"].to_excel(writer, index=False, sheet_name="stats_overall")
        extra["stats_pairwise"].to_excel(writer, index=False, sheet_name="stats_pairwise")
        extra["stats_winners"].to_excel(writer, index=False, sheet_name="stats_winners")
        extra["scalar_subject_results"].to_excel(writer, index=False, sheet_name="scalar_subject")
        extra["scalar_winners"].to_excel(writer, index=False, sheet_name="scalar_winners")
        extra["scalar_summary"].to_excel(writer, index=False, sheet_name="scalar_summary")

    return out_path
