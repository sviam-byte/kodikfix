from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .phenotype_claims import (
    build_core_claim_readiness,
    build_family_inference_table,
    build_scalar_inference_table,
)
from .phenotype_stats import build_stats_tables


def _safe_numeric(s: pd.Series) -> pd.Series:
    """Convert values to numeric and coerce invalid entries to NaN."""
    return pd.to_numeric(s, errors="coerce")


def build_attack_summary(subject_df: pd.DataFrame) -> pd.DataFrame:
    """Build per-attack compact summary over all subject best matches."""
    if subject_df is None or subject_df.empty:
        return pd.DataFrame(columns=["attack_kind", "n", "best_distance_median", "best_distance_mean", "best_distance_min", "best_distance_max", "best_step_median", "best_damage_frac_median"])
    df = subject_df.copy()
    for col in ["best_distance", "best_step", "best_damage_frac"]:
        if col in df.columns:
            df[col] = _safe_numeric(df[col])
    rows = []
    for attack_kind, sub in df.groupby("attack_kind", dropna=False):
        dist = _safe_numeric(sub.get("best_distance", pd.Series(dtype=float))).dropna()
        step = _safe_numeric(sub.get("best_step", pd.Series(dtype=float))).dropna()
        dmg = _safe_numeric(sub.get("best_damage_frac", pd.Series(dtype=float))).dropna()
        rows.append({
            "attack_kind": attack_kind,
            "n": int(len(sub)),
            "best_distance_median": float(dist.median()) if not dist.empty else np.nan,
            "best_distance_mean": float(dist.mean()) if not dist.empty else np.nan,
            "best_distance_min": float(dist.min()) if not dist.empty else np.nan,
            "best_distance_max": float(dist.max()) if not dist.empty else np.nan,
            "best_step_median": float(step.median()) if not step.empty else np.nan,
            "best_damage_frac_median": float(dmg.median()) if not dmg.empty else np.nan,
        })
    out = pd.DataFrame(rows)
    if not out.empty and "best_distance_median" in out.columns:
        out = out.sort_values(["best_distance_median", "attack_kind"], ascending=[True, True]).reset_index(drop=True)
    return out


def build_winner_summary(winners_df: pd.DataFrame, *, total_subjects: int | None = None) -> pd.DataFrame:
    """Build per-attack compact summary over winner rows only."""
    if winners_df is None or winners_df.empty:
        return pd.DataFrame(columns=["attack_kind", "win_count", "win_rate", "winning_distance_median", "winning_step_median", "winning_damage_frac_median"])
    df = winners_df.copy()
    if total_subjects is None:
        if "subject_id" in df.columns:
            total_subjects = int(df["subject_id"].nunique())
        elif "subject_idx" in df.columns:
            total_subjects = int(df["subject_idx"].nunique())
        else:
            total_subjects = int(len(df))
    for col in ["best_distance", "best_step", "best_damage_frac"]:
        if col in df.columns:
            df[col] = _safe_numeric(df[col])
    rows = []
    for attack_kind, sub in df.groupby("attack_kind", dropna=False):
        rows.append({
            "attack_kind": attack_kind,
            "win_count": int(len(sub)),
            "win_rate": float(len(sub) / max(1, total_subjects)),
            "winning_distance_median": float(sub["best_distance"].median()) if "best_distance" in sub.columns else np.nan,
            "winning_step_median": float(sub["best_step"].median()) if "best_step" in sub.columns else np.nan,
            "winning_damage_frac_median": float(sub["best_damage_frac"].median()) if "best_damage_frac" in sub.columns else np.nan,
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["win_count", "winning_distance_median", "attack_kind"], ascending=[False, True, True]).reset_index(drop=True)
    return out


def build_target_vector_df(target_vector: dict) -> pd.DataFrame:
    """Convert target-vector mapping to a tabular dataframe."""
    if not target_vector:
        return pd.DataFrame(columns=["metric", "target_value"])
    return pd.DataFrame([{"metric": str(k), "target_value": v} for k, v in target_vector.items()])


def build_scales_df(scales: dict) -> pd.DataFrame:
    """Convert scale mapping to a tabular dataframe."""
    if not scales:
        return pd.DataFrame(columns=["metric", "scale"])
    return pd.DataFrame([{"metric": str(k), "scale": v} for k, v in scales.items()])


def build_metric_families_df(metric_families: dict) -> pd.DataFrame:
    """Flatten metric-family mapping to long format."""
    if not metric_families:
        return pd.DataFrame(columns=["metric_family", "metric"])
    rows = []
    for fam, metrics in metric_families.items():
        for metric in metrics:
            rows.append({"metric_family": str(fam), "metric": str(metric)})
    return pd.DataFrame(rows)


def build_family_summary(scalar_summary_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate scalar summaries to metric-family level."""
    cols = ["metric_family", "attack_kind", "n_metrics", "winner_count_sum", "winner_rate_mean", "scalar_error_median_mean"]
    if scalar_summary_df is None or scalar_summary_df.empty or "metric_family" not in scalar_summary_df.columns:
        return pd.DataFrame(columns=cols)
    df = scalar_summary_df.copy()
    rows = []
    for (metric_family, attack_kind), sub in df.groupby(["metric_family", "attack_kind"], dropna=False):
        rows.append({
            "metric_family": metric_family,
            "attack_kind": attack_kind,
            "n_metrics": int(sub["metric"].nunique()) if "metric" in sub.columns else int(len(sub)),
            "winner_count_sum": int(pd.to_numeric(sub.get("winner_count", 0), errors="coerce").fillna(0).sum()),
            "winner_rate_mean": float(pd.to_numeric(sub.get("winner_rate_within_metric", np.nan), errors="coerce").mean()),
            "scalar_error_median_mean": float(pd.to_numeric(sub.get("scalar_error_median", np.nan), errors="coerce").mean()),
        })
    out = pd.DataFrame(rows, columns=cols)
    if not out.empty:
        out = out.sort_values(["metric_family", "winner_count_sum", "scalar_error_median_mean", "attack_kind"], ascending=[True, False, True, True]).reset_index(drop=True)
    return out


def build_warning_flags(
    result: dict,
    *,
    density_control_result: dict | None = None,
    modularity_sensitivity_summary: pd.DataFrame | None = None,
    target_stability_summary: pd.DataFrame | None = None,
    null_severity_density_summary: pd.DataFrame | None = None,
    null_severity_total_weight_summary: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Generate high-level warning flags for robustness/instability diagnostics."""
    rows = []
    subject_df = result.get("subject_results", pd.DataFrame()) if isinstance(result, dict) else pd.DataFrame()
    primary_summary = build_attack_summary(subject_df)
    primary_best = primary_summary.iloc[0].to_dict() if not primary_summary.empty else {}
    primary_best_attack = primary_best.get("attack_kind")
    primary_best_distance = float(primary_best.get("best_distance_median", np.nan)) if primary_best else np.nan

    density_flag = False
    density_note = ""
    if density_control_result is not None:
        density_summary = build_attack_summary(density_control_result.get("subject_results", pd.DataFrame()))
        density_best = density_summary.iloc[0].to_dict() if not density_summary.empty else {}
        density_best_attack = density_best.get("attack_kind")
        density_best_distance = float(density_best.get("best_distance_median", np.nan)) if density_best else np.nan
        density_flag = bool(primary_best_attack and density_best_attack and primary_best_attack != density_best_attack)
        if np.isfinite(primary_best_distance) and np.isfinite(density_best_distance):
            denom = max(1e-12, abs(primary_best_distance))
            if abs(density_best_distance - primary_best_distance) / denom > 0.15:
                density_flag = True
        density_note = f"primary_best={primary_best_attack}; density_control_best={density_best_attack}"
    rows.append({"flag": "density_axis_dominance_flag", "is_triggered": bool(density_flag), "details": density_note})

    module_flag = False
    module_note = ""
    if isinstance(modularity_sensitivity_summary, pd.DataFrame) and not modularity_sensitivity_summary.empty and "winner_attack" in modularity_sensitivity_summary.columns:
        winners = modularity_sensitivity_summary["winner_attack"].dropna().astype(str)
        module_flag = winners.nunique() > 1
        module_note = ", ".join(sorted(winners.unique().tolist()))
    rows.append({"flag": "module_partition_instability_flag", "is_triggered": bool(module_flag), "details": module_note})

    target_flag = False
    target_note = ""
    if isinstance(target_stability_summary, pd.DataFrame) and not target_stability_summary.empty:
        unstable = target_stability_summary[pd.to_numeric(target_stability_summary.get("win_rate_std", pd.Series(dtype=float)), errors="coerce") > 0.10]
        target_flag = not unstable.empty
        target_note = ", ".join(unstable.get("attack_kind", pd.Series(dtype=str)).astype(str).tolist())
    rows.append({"flag": "target_instability_flag", "is_triggered": bool(target_flag), "details": target_note})

    metric_families = result.get("metric_families", {}) if isinstance(result, dict) else {}
    family_sizes = [len(v) for v in metric_families.values()] if isinstance(metric_families, dict) else []
    redundancy_flag = bool(family_sizes and max(family_sizes) >= max(3, 2 * max(1, min(family_sizes))))
    rows.append({"flag": "high_metric_redundancy_flag", "is_triggered": redundancy_flag, "details": str(metric_families) if redundancy_flag else ""})

    null_flag = False
    null_details = []
    for label, df in [("delta_density", null_severity_density_summary), ("delta_total_weight", null_severity_total_weight_summary)]:
        if isinstance(df, pd.DataFrame) and not df.empty and "distance_gap_median" in df.columns:
            work = df.copy()
            if primary_best_attack is not None and "attack_kind" in work.columns:
                sub = work[work["attack_kind"].astype(str) == str(primary_best_attack)].copy()
            else:
                sub = work.copy()
            if not sub.empty:
                gap = pd.to_numeric(sub.get("distance_gap_median", pd.Series(dtype=float)), errors="coerce").iloc[0]
                pb = int(pd.to_numeric(sub.get("primary_better_count", pd.Series([0])), errors="coerce").fillna(0).iloc[0])
                nb = int(pd.to_numeric(sub.get("null_better_count", pd.Series([0])), errors="coerce").fillna(0).iloc[0])
                if (pd.notna(gap) and float(gap) >= 0.0) or nb >= pb:
                    null_flag = True
                null_details.append(f"{label}:gap={float(gap) if pd.notna(gap) else 'nan'};primary={pb};null={nb}")
    rows.append({"flag": "severity_matched_null_failure_flag", "is_triggered": bool(null_flag), "details": "; ".join(null_details)})

    return pd.DataFrame(rows)


def build_paper_ready_summary(result: dict) -> dict[str, pd.DataFrame]:
    """Build compact, stats, and scalar analysis tables for phenotype reports."""
    subject_df = result.get("subject_results", pd.DataFrame())
    winners_df = result.get("winner_results", pd.DataFrame())
    scalar_subject_df = result.get("scalar_subject_results", pd.DataFrame())
    scalar_winners_df = result.get("scalar_winners", pd.DataFrame())
    scalar_summary_df = result.get("scalar_summary", pd.DataFrame())
    total_subjects = 0
    if isinstance(subject_df, pd.DataFrame) and not subject_df.empty and "subject_id" in subject_df.columns:
        total_subjects = int(subject_df["subject_id"].nunique())
    elif isinstance(subject_df, pd.DataFrame) and not subject_df.empty and "subject_idx" in subject_df.columns:
        total_subjects = int(subject_df["subject_idx"].nunique())
    stats_pack = build_stats_tables(result)
    out = {
        "summary_attack": build_attack_summary(subject_df),
        "summary_winners": build_winner_summary(winners_df, total_subjects=total_subjects),
        "target_vector": build_target_vector_df(result.get("target_vector", {})),
        "scales": build_scales_df(result.get("scales", {})),
        "metric_families": build_metric_families_df(result.get("metric_families", {})),
        "family_summary": build_family_summary(scalar_summary_df),
        "warning_flags": build_warning_flags(result),
        "stats_overall": stats_pack["stats_overall"],
        "stats_pairwise": stats_pack["stats_pairwise"],
        "stats_pairwise_matched_delta_density": stats_pack.get("stats_pairwise_matched_delta_density", pd.DataFrame()),
        "stats_pairwise_matched_delta_total_weight": stats_pack.get("stats_pairwise_matched_delta_total_weight", pd.DataFrame()),
        "stats_winners": stats_pack["stats_winners"],
        "scalar_subject_results": scalar_subject_df if isinstance(scalar_subject_df, pd.DataFrame) else pd.DataFrame(),
        "scalar_winners": scalar_winners_df if isinstance(scalar_winners_df, pd.DataFrame) else pd.DataFrame(),
        "scalar_summary": scalar_summary_df if isinstance(scalar_summary_df, pd.DataFrame) else pd.DataFrame(),
    }
    out["scalar_inference"] = build_scalar_inference_table(result)
    out["family_inference"] = build_family_inference_table(result)
    out["core_claim_readiness"] = build_core_claim_readiness(out)
    return out


def export_phenotype_match_excel(result: dict, out_path: str | Path) -> Path:
    """Export full phenotype workbook with raw and summary/stat sheets."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    winners_df = result.get("winner_results", pd.DataFrame())
    subject_df = result.get("subject_results", pd.DataFrame())
    traj_df = result.get("trajectory_results", pd.DataFrame())
    extra = build_paper_ready_summary(result)
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        (winners_df if isinstance(winners_df, pd.DataFrame) else pd.DataFrame()).to_excel(writer, index=False, sheet_name="winners")
        (subject_df if isinstance(subject_df, pd.DataFrame) else pd.DataFrame()).to_excel(writer, index=False, sheet_name="subject_results")
        (traj_df if isinstance(traj_df, pd.DataFrame) else pd.DataFrame()).to_excel(writer, index=False, sheet_name="trajectories")
        extra["summary_attack"].to_excel(writer, index=False, sheet_name="summary_attack")
        extra["summary_winners"].to_excel(writer, index=False, sheet_name="summary_winners")
        extra["target_vector"].to_excel(writer, index=False, sheet_name="target_vector")
        extra["scales"].to_excel(writer, index=False, sheet_name="scales")
        extra["metric_families"].to_excel(writer, index=False, sheet_name="metric_families")
        extra["family_summary"].to_excel(writer, index=False, sheet_name="family_summary")
        extra["warning_flags"].to_excel(writer, index=False, sheet_name="warning_flags")
        extra["stats_overall"].to_excel(writer, index=False, sheet_name="stats_overall")
        extra["stats_pairwise"].to_excel(writer, index=False, sheet_name="stats_pairwise")
        extra["stats_pairwise_matched_delta_density"].to_excel(writer, index=False, sheet_name="stats_pair_sev_density")
        extra["stats_pairwise_matched_delta_total_weight"].to_excel(writer, index=False, sheet_name="stats_pair_sev_weight")
        extra["stats_winners"].to_excel(writer, index=False, sheet_name="stats_winners")
        extra["scalar_subject_results"].to_excel(writer, index=False, sheet_name="scalar_subject")
        extra["scalar_winners"].to_excel(writer, index=False, sheet_name="scalar_winners")
        extra["scalar_summary"].to_excel(writer, index=False, sheet_name="scalar_summary")
        extra["scalar_inference"].to_excel(writer, index=False, sheet_name="scalar_inference")
        extra["family_inference"].to_excel(writer, index=False, sheet_name="family_inference")
        extra["core_claim_readiness"].to_excel(writer, index=False, sheet_name="claim_readiness")
    return out_path
