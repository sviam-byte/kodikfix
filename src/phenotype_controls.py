from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from .phenotype_matching import compare_degradation_models, normalize_metric_families
from .phenotype_reporting import build_paper_ready_summary, build_warning_flags


def _metric_list(metrics: Sequence[str] | None) -> list[str]:
    """Normalize metrics to non-empty strings."""
    return [str(m) for m in (metrics or []) if str(m)]


def effective_metrics_for_run_type(metrics: Sequence[str], *, run_type: str, metric_families: Mapping[str, Sequence[str]] | None = None) -> list[str]:
    """Select effective metrics for the requested run type."""
    metric_list = _metric_list(metrics)
    families = normalize_metric_families(metric_list, metric_families)
    rt = str(run_type or "primary_run")
    if rt in {"primary_run", "modularity_sensitivity_run", "null_attack_run", "target_stability_run"}:
        return metric_list
    if rt in {"density_control_run", "leave_density_out_run"}:
        density_metrics = set(families.get("density", []))
        return [m for m in metric_list if m not in density_metrics]
    return metric_list


def summarize_suite_result(name: str, result: dict) -> pd.DataFrame:
    """Build tagged summary rows for a single suite result."""
    pack = build_paper_ready_summary(result)
    sw = pack.get("summary_winners", pd.DataFrame()).copy()
    sa = pack.get("summary_attack", pd.DataFrame()).copy()
    if not sw.empty:
        sw["suite_name"] = str(name)
        sw["table"] = "summary_winners"
    if not sa.empty:
        sa["suite_name"] = str(name)
        sa["table"] = "summary_attack"
    return pd.concat([sa, sw], ignore_index=True, sort=False) if (not sa.empty or not sw.empty) else pd.DataFrame()


def summarize_stability_result(name: str, summary_df: pd.DataFrame) -> pd.DataFrame:
    """Attach suite_name to stability summaries."""
    if summary_df is None or summary_df.empty:
        return pd.DataFrame()
    out = summary_df.copy()
    out["suite_name"] = str(name)
    return out


def _aggregate_bootstrap_metric(joined: pd.DataFrame, *, group_cols: list[str], value_col: str, prefix: str) -> pd.DataFrame:
    """Aggregate bootstrap metric with mean/std/CI statistics."""
    cols = [*group_cols, f"{prefix}_mean", f"{prefix}_std", f"{prefix}_ci_low", f"{prefix}_ci_high", "n_bootstrap_reps"]
    if joined is None or joined.empty or value_col not in joined.columns:
        return pd.DataFrame(columns=cols)
    rows = []
    for group_key, sub in joined.groupby(group_cols, dropna=False):
        vals = pd.to_numeric(sub.get(value_col, pd.Series(dtype=float)), errors="coerce").dropna()
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        row = {col: key for col, key in zip(group_cols, group_key, strict=False)}
        row.update(
            {
                f"{prefix}_mean": float(vals.mean()) if not vals.empty else float("nan"),
                f"{prefix}_std": float(vals.std(ddof=1)) if len(vals) >= 2 else 0.0,
                f"{prefix}_ci_low": float(vals.quantile(0.025)) if not vals.empty else float("nan"),
                f"{prefix}_ci_high": float(vals.quantile(0.975)) if not vals.empty else float("nan"),
                "n_bootstrap_reps": int(sub["bootstrap_rep"].nunique()) if "bootstrap_rep" in sub.columns else int(len(sub)),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows, columns=cols)


def _aggregate_scalar_stability(scalar_tables: list[pd.DataFrame]) -> pd.DataFrame:
    """Aggregate scalar-summary stability across bootstrap reps."""
    cols = [
        "metric", "metric_family", "attack_kind", "n_bootstrap_reps",
        "winner_rate_mean", "winner_rate_std", "winner_rate_ci_low", "winner_rate_ci_high",
        "scalar_error_median_mean", "scalar_error_median_std", "scalar_error_median_ci_low", "scalar_error_median_ci_high",
    ]
    if not scalar_tables:
        return pd.DataFrame(columns=cols)
    joined = pd.concat(scalar_tables, ignore_index=True, sort=False)
    base = _aggregate_bootstrap_metric(joined, group_cols=["metric", "metric_family", "attack_kind"], value_col="winner_rate_within_metric", prefix="winner_rate")
    agg = _aggregate_bootstrap_metric(joined, group_cols=["metric", "metric_family", "attack_kind"], value_col="scalar_error_median", prefix="scalar_error_median")
    if base.empty:
        out = agg
    elif agg.empty:
        out = base
    else:
        out = base.merge(agg, on=["metric", "metric_family", "attack_kind", "n_bootstrap_reps"], how="outer")
    if out.empty:
        return pd.DataFrame(columns=cols)
    return out.sort_values(["metric_family", "metric", "winner_rate_mean", "scalar_error_median_mean", "attack_kind"], ascending=[True, True, False, True, True]).reset_index(drop=True)


def _aggregate_family_stability(family_tables: list[pd.DataFrame]) -> pd.DataFrame:
    """Aggregate family-summary statistics across bootstrap reps."""
    if not family_tables:
        return pd.DataFrame(columns=[
            "metric_family", "attack_kind", "n_bootstrap_reps",
            "winner_count_sum_mean", "winner_count_sum_std", "winner_count_sum_ci_low", "winner_count_sum_ci_high",
            "winner_rate_mean_mean", "winner_rate_mean_std", "winner_rate_mean_ci_low", "winner_rate_mean_ci_high",
            "scalar_error_median_mean_mean", "scalar_error_median_mean_std", "scalar_error_median_mean_ci_low", "scalar_error_median_mean_ci_high",
        ])

    joined = pd.concat(family_tables, ignore_index=True, sort=False)
    base = _aggregate_bootstrap_metric(joined, group_cols=["metric_family", "attack_kind"], value_col="winner_count_sum", prefix="winner_count_sum")
    for value_col, prefix in [("winner_rate_mean", "winner_rate_mean"), ("scalar_error_median_mean", "scalar_error_median_mean")]:
        agg = _aggregate_bootstrap_metric(joined, group_cols=["metric_family", "attack_kind"], value_col=value_col, prefix=prefix)
        if base.empty:
            base = agg
        elif not agg.empty:
            base = base.merge(agg, on=["metric_family", "attack_kind", "n_bootstrap_reps"], how="outer")

    if not base.empty:
        base = base.sort_values(["metric_family", "winner_count_sum_mean", "scalar_error_median_mean_mean", "attack_kind"], ascending=[True, False, True, True]).reset_index(drop=True)
    return base


def _build_severity_matched_null_detail(primary_result: dict, null_result: dict, *, severity_col: str, label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Match null trajectories by severity to winner severities and summarize gaps."""
    cols = [
        "subject_id", "primary_attack_kind", "primary_best_distance", "primary_severity_value",
        "null_attack_kind", "null_step", "null_distance_to_target", "null_severity_value",
        "severity_abs_gap", "distance_gap_primary_minus_null", "severity_match_mode",
    ]
    winners = primary_result.get("winner_results", pd.DataFrame()) if isinstance(primary_result, dict) else pd.DataFrame()
    traj = null_result.get("trajectory_results", pd.DataFrame()) if isinstance(null_result, dict) else pd.DataFrame()
    if winners is None or winners.empty or traj is None or traj.empty or severity_col not in traj.columns:
        empty_summary = pd.DataFrame(columns=["attack_kind", "n_subjects", "primary_distance_median", "matched_null_distance_median", "distance_gap_median", "primary_better_count", "null_better_count", "severity_match_mode"])
        return pd.DataFrame(columns=cols), empty_summary

    rows = []
    for _, win in winners.iterrows():
        subject_id = str(win.get("subject_id", ""))
        winner_severity_col = severity_col if severity_col in winners.columns else (f"best_{severity_col}" if f"best_{severity_col}" in winners.columns else severity_col)
        target_val = pd.to_numeric(pd.Series([win.get(winner_severity_col, np.nan)]), errors="coerce").iloc[0]
        sub = traj[traj.get("subject_id", pd.Series(dtype=str)).astype(str) == subject_id].copy() if "subject_id" in traj.columns else traj.copy()
        if sub.empty:
            continue
        sub[severity_col] = pd.to_numeric(sub.get(severity_col, pd.Series(dtype=float)), errors="coerce")
        sub["distance_to_target"] = pd.to_numeric(sub.get("distance_to_target", pd.Series(dtype=float)), errors="coerce")
        if np.isfinite(target_val):
            sub["severity_abs_gap"] = (sub[severity_col] - float(target_val)).abs()
            sub = sub.sort_values(["severity_abs_gap", "distance_to_target"], ascending=[True, True])
        else:
            sub["severity_abs_gap"] = np.nan
            sub = sub.sort_values(["distance_to_target"], ascending=[True])

        row = sub.iloc[0]
        primary_dist = pd.to_numeric(pd.Series([win.get("best_distance", np.nan)]), errors="coerce").iloc[0]
        null_dist = pd.to_numeric(pd.Series([row.get("distance_to_target", np.nan)]), errors="coerce").iloc[0]
        rows.append({
            "subject_id": subject_id,
            "primary_attack_kind": win.get("attack_kind"),
            "primary_best_distance": float(primary_dist) if pd.notna(primary_dist) else float("nan"),
            "primary_severity_value": float(target_val) if pd.notna(target_val) else float("nan"),
            "null_attack_kind": row.get("attack_kind", "random_edges"),
            "null_step": int(row.get("step", 0)) if pd.notna(row.get("step", np.nan)) else None,
            "null_distance_to_target": float(null_dist) if pd.notna(null_dist) else float("nan"),
            "null_severity_value": float(row.get(severity_col, np.nan)) if pd.notna(row.get(severity_col, np.nan)) else float("nan"),
            "severity_abs_gap": float(row.get("severity_abs_gap", np.nan)) if pd.notna(row.get("severity_abs_gap", np.nan)) else float("nan"),
            "distance_gap_primary_minus_null": float(primary_dist - null_dist) if pd.notna(primary_dist) and pd.notna(null_dist) else float("nan"),
            "severity_match_mode": str(label),
        })

    detail = pd.DataFrame(rows, columns=cols)
    if detail.empty:
        empty_summary = pd.DataFrame(columns=["attack_kind", "n_subjects", "primary_distance_median", "matched_null_distance_median", "distance_gap_median", "primary_better_count", "null_better_count", "severity_match_mode"])
        return detail, empty_summary

    summary_rows = []
    for attack_kind, sub in detail.groupby("primary_attack_kind", dropna=False):
        diff = pd.to_numeric(sub.get("distance_gap_primary_minus_null", pd.Series(dtype=float)), errors="coerce")
        summary_rows.append({
            "attack_kind": attack_kind,
            "n_subjects": int(len(sub)),
            "primary_distance_median": float(pd.to_numeric(sub.get("primary_best_distance", pd.Series(dtype=float)), errors="coerce").median()),
            "matched_null_distance_median": float(pd.to_numeric(sub.get("null_distance_to_target", pd.Series(dtype=float)), errors="coerce").median()),
            "distance_gap_median": float(diff.median()) if not diff.empty else float("nan"),
            "primary_better_count": int((diff < 0).sum()),
            "null_better_count": int((diff > 0).sum()),
            "severity_match_mode": str(label),
        })
    summary = pd.DataFrame(summary_rows).sort_values(["distance_gap_median", "attack_kind"], ascending=[True, True]).reset_index(drop=True)
    return detail, summary


def run_modularity_sensitivity_suite(*, hc_graphs, sz_group_metrics_df: pd.DataFrame, hc_baseline_metrics_df: pd.DataFrame, metrics: Sequence[str], metric_families: Mapping[str, Sequence[str]] | None, attack_kinds: Sequence[str], resolutions: Sequence[float], recompute_options: Sequence[bool], compare_kwargs: Mapping) -> dict:
    """Run inter/intra-module attacks across modularity settings."""
    rows = []
    detail_tables: dict[str, pd.DataFrame] = {}
    use_attacks = [a for a in attack_kinds if a in {"inter_module_removal", "intra_module_removal"}] or ["inter_module_removal", "intra_module_removal"]
    for resolution in resolutions:
        for recompute in recompute_options:
            result = compare_degradation_models(
                hc_graphs=hc_graphs,
                sz_group_metrics_df=sz_group_metrics_df,
                hc_baseline_metrics_df=hc_baseline_metrics_df,
                attack_kinds=use_attacks,
                metrics=metrics,
                metric_families=metric_families,
                module_resolution=float(resolution),
                recompute_modules=bool(recompute),
                **dict(compare_kwargs),
            )
            key = f"modres_{float(resolution):.2f}__recompute_{int(bool(recompute))}"
            summary = build_paper_ready_summary(result)
            detail_tables[f"{key}_summary_attack.csv"] = summary["summary_attack"]
            detail_tables[f"{key}_summary_winners.csv"] = summary["summary_winners"]
            winners = result.get("winner_results", pd.DataFrame())
            subjects = result.get("subject_results", pd.DataFrame())
            if winners is None or winners.empty:
                continue
            winner_counts = winners["attack_kind"].value_counts(dropna=False).to_dict()
            median_best_distance = pd.to_numeric(subjects.get("best_distance", pd.Series(dtype=float)), errors="coerce").median() if isinstance(subjects, pd.DataFrame) and not subjects.empty else float("nan")
            rows.append({
                "module_resolution": float(resolution),
                "recompute_modules": bool(recompute),
                "n_subjects": int(winners["subject_id"].nunique()) if "subject_id" in winners.columns else int(len(winners)),
                "winner_attack": max(winner_counts.items(), key=lambda kv: kv[1])[0] if winner_counts else None,
                "winner_count": max(winner_counts.values()) if winner_counts else 0,
                "median_best_distance": float(median_best_distance) if pd.notna(median_best_distance) else float("nan"),
                "winner_counts_json": pd.Series(winner_counts).to_json(force_ascii=False),
            })
    summary = pd.DataFrame(rows).sort_values(["median_best_distance", "module_resolution", "recompute_modules"], ascending=[True, True, True]).reset_index(drop=True) if rows else pd.DataFrame()
    return {"summary": summary, "detail_tables": detail_tables}


def _bootstrap_target_df(sz_group_metrics_df: pd.DataFrame, *, seed: int, replicate: int) -> pd.DataFrame:
    """Bootstrap SZ table rows for target-stability checks."""
    if sz_group_metrics_df is None or sz_group_metrics_df.empty:
        return pd.DataFrame(columns=list(sz_group_metrics_df.columns) if isinstance(sz_group_metrics_df, pd.DataFrame) else [])
    return sz_group_metrics_df.sample(n=len(sz_group_metrics_df), replace=True, random_state=int(seed) + int(replicate)).reset_index(drop=True)


def run_target_stability_suite(*, hc_graphs, sz_group_metrics_df: pd.DataFrame, hc_baseline_metrics_df: pd.DataFrame, metrics: Sequence[str], metric_families: Mapping[str, Sequence[str]] | None, attack_kinds: Sequence[str], compare_kwargs: Mapping, n_bootstrap: int = 16, seed: int = 42) -> dict:
    """Run bootstrap target stability and aggregate winners/attacks/families."""
    replicate_rows = []
    detail_tables: dict[str, pd.DataFrame] = {}
    winner_tables: list[pd.DataFrame] = []
    attack_tables: list[pd.DataFrame] = []
    family_tables: list[pd.DataFrame] = []
    scalar_tables: list[pd.DataFrame] = []
    for rep in range(int(max(1, n_bootstrap))):
        boot_sz = _bootstrap_target_df(sz_group_metrics_df, seed=int(seed), replicate=rep)
        result = compare_degradation_models(
            hc_graphs=hc_graphs,
            sz_group_metrics_df=boot_sz,
            hc_baseline_metrics_df=hc_baseline_metrics_df,
            attack_kinds=attack_kinds,
            metrics=metrics,
            metric_families=metric_families,
            **dict(compare_kwargs),
        )
        pack = build_paper_ready_summary(result)
        sa = pack["summary_attack"].copy()
        sw = pack["summary_winners"].copy()
        fam = pack["family_summary"].copy()
        scalar = pack["scalar_summary"].copy()
        sa["bootstrap_rep"] = int(rep)
        sw["bootstrap_rep"] = int(rep)
        fam["bootstrap_rep"] = int(rep)
        scalar["bootstrap_rep"] = int(rep)
        detail_tables[f"target_bootstrap_{rep:03d}_summary_attack.csv"] = sa
        detail_tables[f"target_bootstrap_{rep:03d}_summary_winners.csv"] = sw
        detail_tables[f"target_bootstrap_{rep:03d}_family_summary.csv"] = fam
        detail_tables[f"target_bootstrap_{rep:03d}_scalar_summary.csv"] = scalar
        if not sw.empty:
            winner_tables.append(sw)
        if not sa.empty:
            attack_tables.append(sa)
        if not fam.empty:
            family_tables.append(fam)
        if not scalar.empty:
            scalar_tables.append(scalar)

        winners = result.get("winner_results", pd.DataFrame())
        winner_counts = winners["attack_kind"].value_counts(dropna=False).to_dict() if isinstance(winners, pd.DataFrame) and not winners.empty else {}
        replicate_rows.append({
            "bootstrap_rep": int(rep),
            "n_subjects": int(winners["subject_id"].nunique()) if isinstance(winners, pd.DataFrame) and not winners.empty and "subject_id" in winners.columns else int(len(winners)) if isinstance(winners, pd.DataFrame) else 0,
            "winner_attack": max(winner_counts.items(), key=lambda kv: kv[1])[0] if winner_counts else None,
            "winner_count": max(winner_counts.values()) if winner_counts else 0,
            "winner_counts_json": pd.Series(winner_counts).to_json(force_ascii=False),
            "median_best_distance": float(pd.to_numeric(result.get("subject_results", pd.DataFrame()).get("best_distance", pd.Series(dtype=float)), errors="coerce").median()) if isinstance(result.get("subject_results", pd.DataFrame()), pd.DataFrame) else float("nan"),
        })

    summary = pd.DataFrame(replicate_rows)
    winner_stability_summary = pd.DataFrame()
    if winner_tables:
        joined = pd.concat(winner_tables, ignore_index=True, sort=False)
        winner_stability_summary = _aggregate_bootstrap_metric(joined, group_cols=["attack_kind"], value_col="win_rate", prefix="win_rate")
        if not winner_stability_summary.empty:
            winner_stability_summary = winner_stability_summary.sort_values(["win_rate_mean", "attack_kind"], ascending=[False, True]).reset_index(drop=True)

    attack_distance_stability_summary = pd.DataFrame()
    if attack_tables:
        joined_attack = pd.concat(attack_tables, ignore_index=True, sort=False)
        attack_distance_stability_summary = _aggregate_bootstrap_metric(joined_attack, group_cols=["attack_kind"], value_col="best_distance_median", prefix="best_distance_median")
        if not attack_distance_stability_summary.empty:
            attack_distance_stability_summary = attack_distance_stability_summary.sort_values(["best_distance_median_mean", "attack_kind"], ascending=[True, True]).reset_index(drop=True)

    family_stability_summary = _aggregate_family_stability(family_tables)
    scalar_stability_summary = _aggregate_scalar_stability(scalar_tables)
    return {
        "summary": summary,
        "winner_stability_summary": winner_stability_summary,
        "attack_distance_stability_summary": attack_distance_stability_summary,
        "family_stability_summary": family_stability_summary,
        "scalar_stability_summary": scalar_stability_summary,
        "detail_tables": detail_tables,
    }


def run_control_suite(*, hc_graphs, sz_group_metrics_df: pd.DataFrame, hc_baseline_metrics_df: pd.DataFrame, metrics: Sequence[str], attack_kinds: Sequence[str], metric_families: Mapping[str, Sequence[str]] | None, compare_kwargs: Mapping, modularity_resolutions: Sequence[float] = (0.5, 1.0, 1.5), modularity_recompute_options: Sequence[bool] = (False, True), target_bootstrap_reps: int = 16) -> dict:
    """Run full control suite and return all control summaries/details."""
    metric_list = _metric_list(metrics)
    families = normalize_metric_families(metric_list, metric_families)

    primary_result = compare_degradation_models(hc_graphs=hc_graphs, sz_group_metrics_df=sz_group_metrics_df, hc_baseline_metrics_df=hc_baseline_metrics_df, attack_kinds=attack_kinds, metrics=metric_list, metric_families=families, **dict(compare_kwargs))
    density_metrics = effective_metrics_for_run_type(metric_list, run_type="density_control_run", metric_families=families)
    density_result = compare_degradation_models(hc_graphs=hc_graphs, sz_group_metrics_df=sz_group_metrics_df, hc_baseline_metrics_df=hc_baseline_metrics_df, attack_kinds=attack_kinds, metrics=density_metrics, metric_families=families, **dict(compare_kwargs))
    null_result = compare_degradation_models(hc_graphs=hc_graphs, sz_group_metrics_df=sz_group_metrics_df, hc_baseline_metrics_df=hc_baseline_metrics_df, attack_kinds=["random_edges"], metrics=metric_list, metric_families=families, **dict(compare_kwargs))

    null_density_detail, null_density_summary = _build_severity_matched_null_detail(primary_result, null_result, severity_col="delta_density", label="match_delta_density")
    null_weight_detail, null_weight_summary = _build_severity_matched_null_detail(primary_result, null_result, severity_col="delta_total_weight", label="match_delta_total_weight")

    modularity_suite = run_modularity_sensitivity_suite(hc_graphs=hc_graphs, sz_group_metrics_df=sz_group_metrics_df, hc_baseline_metrics_df=hc_baseline_metrics_df, metrics=metric_list, metric_families=families, attack_kinds=attack_kinds, resolutions=modularity_resolutions, recompute_options=modularity_recompute_options, compare_kwargs=compare_kwargs)
    target_stability_suite = run_target_stability_suite(hc_graphs=hc_graphs, sz_group_metrics_df=sz_group_metrics_df, hc_baseline_metrics_df=hc_baseline_metrics_df, metrics=metric_list, metric_families=families, attack_kinds=attack_kinds, compare_kwargs=compare_kwargs, n_bootstrap=int(target_bootstrap_reps), seed=int(compare_kwargs.get("seed", 42)))

    suite_summary = pd.concat([
        summarize_suite_result("primary_run", primary_result),
        summarize_suite_result("density_control_run", density_result),
        summarize_suite_result("null_attack_run", null_result),
        summarize_stability_result("target_stability_run", target_stability_suite["winner_stability_summary"]),
        summarize_stability_result("target_distance_stability_run", target_stability_suite["attack_distance_stability_summary"]),
        summarize_stability_result("family_stability_run", target_stability_suite["family_stability_summary"]),
        summarize_stability_result("scalar_stability_run", target_stability_suite["scalar_stability_summary"]),
    ], ignore_index=True, sort=False)

    warning_flags = build_warning_flags(
        primary_result,
        density_control_result=density_result,
        modularity_sensitivity_summary=modularity_suite["summary"],
        target_stability_summary=target_stability_suite["winner_stability_summary"],
        null_severity_density_summary=null_density_summary,
        null_severity_total_weight_summary=null_weight_summary,
    )

    return {
        "primary_result": primary_result,
        "density_control_result": density_result,
        "null_attack_result": null_result,
        "modularity_sensitivity_summary": modularity_suite["summary"],
        "modularity_detail_tables": modularity_suite["detail_tables"],
        "target_stability_summary": target_stability_suite["summary"],
        "target_winner_stability_summary": target_stability_suite["winner_stability_summary"],
        "target_attack_distance_stability_summary": target_stability_suite["attack_distance_stability_summary"],
        "target_family_stability_summary": target_stability_suite["family_stability_summary"],
        "target_scalar_stability_summary": target_stability_suite["scalar_stability_summary"],
        "target_stability_detail_tables": target_stability_suite["detail_tables"],
        "null_severity_density_detail": null_density_detail,
        "null_severity_density_summary": null_density_summary,
        "null_severity_total_weight_detail": null_weight_detail,
        "null_severity_total_weight_summary": null_weight_summary,
        "suite_summary": suite_summary,
        "warning_flags": warning_flags,
        "suite_metrics": {
            "primary_run": metric_list,
            "density_control_run": density_metrics,
            "null_attack_run": metric_list,
            "target_stability_run": metric_list,
        },
    }
