from __future__ import annotations

from collections.abc import Mapping, Sequence

import networkx as nx
import numpy as np
import pandas as pd

from .degradation import prepare_module_info, run_degradation_trajectory
from .phenotype_scalar import (
    build_scalar_subject_results,
    build_scalar_summary,
    build_scalar_winners,
    find_best_scalar_match,
)

DEFAULT_PROFILE_METRICS = [
    "density",
    "clustering",
    "mod",
    "l2_lcc",
    "H_rw",
    "fragility_H",
    "eff_w",
    "lcc_frac",
]

DEFAULT_METRIC_FAMILIES = {
    "density": ["density", "avg_degree", "beta", "beta_red", "clustering"],
    "integration": ["algebraic_connectivity", "l2_lcc", "eff_w", "tau_relax", "lcc_frac"],
    "modularity": ["mod"],
    "entropy_fragility": ["H_rw", "fragility_H", "H_evo", "fragility_evo"],
}


def _as_metric_list(metrics: Sequence[str] | None) -> list[str]:
    if metrics is None:
        return list(DEFAULT_PROFILE_METRICS)
    return [str(m) for m in metrics if str(m)]


def _coerce_float(v) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return float(x) if np.isfinite(x) else float("nan")


def normalize_metric_families(
    metrics: Sequence[str] | None,
    metric_families: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, list[str]]:
    metric_list = _as_metric_list(metrics)
    families_src = dict(DEFAULT_METRIC_FAMILIES)
    if metric_families:
        for fam, fam_metrics in metric_families.items():
            families_src[str(fam)] = [str(m) for m in fam_metrics if str(m)]

    out: dict[str, list[str]] = {}
    assigned: set[str] = set()
    for fam, fam_metrics in families_src.items():
        kept = [m for m in fam_metrics if m in metric_list]
        if kept:
            out[str(fam)] = kept
            assigned.update(kept)

    leftovers = [m for m in metric_list if m not in assigned]
    for m in leftovers:
        out[f"singleton::{m}"] = [m]
    return out


def build_group_target_vector(group_df: pd.DataFrame, *, metrics: Sequence[str] | None = None) -> dict[str, float]:
    metric_list = _as_metric_list(metrics)
    out: dict[str, float] = {}
    for m in metric_list:
        if m not in group_df.columns:
            out[m] = float("nan")
            continue
        arr = pd.to_numeric(group_df[m], errors="coerce").to_numpy(dtype=float)
        arr = arr[np.isfinite(arr)]
        out[m] = float(np.median(arr)) if arr.size else float("nan")
    return out


def build_scale_vector(
    baseline_df: pd.DataFrame,
    *,
    metrics: Sequence[str] | None = None,
    eps: float = 1e-8,
) -> dict[str, float]:
    metric_list = _as_metric_list(metrics)
    scales: dict[str, float] = {}
    for m in metric_list:
        if m not in baseline_df.columns:
            scales[m] = 1.0
            continue
        arr = pd.to_numeric(baseline_df[m], errors="coerce").to_numpy(dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            scales[m] = 1.0
            continue
        s = float(np.nanstd(arr, ddof=1)) if arr.size >= 2 else 0.0
        if (not np.isfinite(s)) or s < eps:
            mad = float(np.nanmedian(np.abs(arr - np.nanmedian(arr))))
            s = 1.4826 * mad if np.isfinite(mad) and mad >= eps else 1.0
        scales[m] = float(max(s, eps))
    return scales


def _compute_row_diffs(
    row: dict | pd.Series,
    target_vector: Mapping[str, float],
    *,
    metrics: Sequence[str] | None = None,
    scales: Mapping[str, float] | None = None,
) -> tuple[list[str], dict[str, float]]:
    metric_list = _as_metric_list(metrics)
    used: list[str] = []
    diffs: dict[str, float] = {}
    for m in metric_list:
        xv = _coerce_float(row.get(m, np.nan))
        tv = _coerce_float(target_vector.get(m, np.nan))
        sv = _coerce_float((scales or {}).get(m, 1.0))
        if not np.isfinite(xv) or not np.isfinite(tv):
            continue
        if (not np.isfinite(sv)) or sv <= 1e-12:
            sv = 1.0
        used.append(m)
        diffs[m] = float((xv - tv) / sv)
    return used, diffs


def compute_profile_distance(
    row: dict | pd.Series,
    target_vector: dict[str, float],
    *,
    metrics: Sequence[str] | None = None,
    scales: dict[str, float] | None = None,
    distance_mode: str = "raw",
    metric_families: Mapping[str, Sequence[str]] | None = None,
) -> dict:
    used, diffs = _compute_row_diffs(row, target_vector, metrics=metrics, scales=scales)
    if not used:
        return {"distance": float("nan"), "used_metrics": [], "n_used_metrics": 0, "distance_mode": str(distance_mode), "family_distances": {}}

    if str(distance_mode) == "family_balanced":
        families = normalize_metric_families(used, metric_families)
        family_distances: dict[str, float] = {}
        family_values = []
        for fam, fam_metrics in families.items():
            fam_diffs = [diffs[m] for m in fam_metrics if m in diffs]
            if not fam_diffs:
                continue
            fam_dist = float(np.sqrt(np.sum(np.square(np.asarray(fam_diffs, dtype=float)))))
            family_distances[str(fam)] = fam_dist
            family_values.append(fam_dist)
        d = float(np.mean(family_values)) if family_values else float("nan")
        return {"distance": d, "used_metrics": used, "n_used_metrics": int(len(used)), "distance_mode": "family_balanced", "family_distances": family_distances}

    d = float(np.sqrt(np.sum(np.square(np.asarray([diffs[m] for m in used], dtype=float)))))
    return {"distance": d, "used_metrics": used, "n_used_metrics": int(len(used)), "distance_mode": "raw", "family_distances": {}}


def _append_severity_deltas(traj_df: pd.DataFrame) -> pd.DataFrame:
    out = traj_df.copy()
    if out.empty:
        return out
    baseline = out.iloc[0]
    baseline_weight = _coerce_float(baseline.get("E", np.nan))
    for col in ["density", "eff_w", "lcc_frac", "E", "total_weight"]:
        if col not in out.columns:
            continue
        base_val = _coerce_float(baseline.get(col, np.nan))
        if not np.isfinite(base_val):
            continue
        out[f"delta_{col}"] = pd.to_numeric(out[col], errors="coerce") - float(base_val)
    if "E" in out.columns and np.isfinite(baseline_weight) and baseline_weight > 0:
        out["removed_edge_fraction_from_baseline"] = 1.0 - (pd.to_numeric(out["E"], errors="coerce") / float(baseline_weight))
    else:
        out["removed_edge_fraction_from_baseline"] = np.nan
    return out


def annotate_trajectory_distance_to_target(
    traj_df: pd.DataFrame,
    *,
    target_vector: dict[str, float],
    metrics: Sequence[str] | None = None,
    scales: dict[str, float] | None = None,
    distance_mode: str = "raw",
    metric_families: Mapping[str, Sequence[str]] | None = None,
) -> pd.DataFrame:
    out = _append_severity_deltas(traj_df)
    dists = []
    n_used = []
    used_metrics_text = []
    normalized_families = normalize_metric_families(metrics, metric_families)
    family_cols = {f"family_dist__{fam}": [] for fam in normalized_families}

    for _, row in out.iterrows():
        info = compute_profile_distance(row, target_vector=target_vector, metrics=metrics, scales=scales, distance_mode=distance_mode, metric_families=normalized_families)
        dists.append(float(info["distance"]))
        n_used.append(int(info["n_used_metrics"]))
        used_metrics_text.append(",".join(info.get("used_metrics", [])))
        family_distances = info.get("family_distances", {}) or {}
        for fam in normalized_families:
            family_cols[f"family_dist__{fam}"].append(float(family_distances.get(fam, np.nan)))

    out["distance_to_target"] = dists
    out["n_used_metrics"] = n_used
    out["distance_mode"] = str(distance_mode)
    out["used_metrics"] = used_metrics_text
    for col, values in family_cols.items():
        out[col] = values
    return out


def find_best_match_to_target(
    traj_df: pd.DataFrame,
    *,
    target_vector: dict[str, float],
    metrics: Sequence[str] | None = None,
    scales: dict[str, float] | None = None,
    distance_mode: str = "raw",
    metric_families: Mapping[str, Sequence[str]] | None = None,
) -> dict:
    scored = annotate_trajectory_distance_to_target(
        traj_df,
        target_vector=target_vector,
        metrics=metrics,
        scales=scales,
        distance_mode=distance_mode,
        metric_families=metric_families,
    )
    if scored.empty or "distance_to_target" not in scored.columns:
        return {"best_step": None, "best_damage_frac": float("nan"), "best_distance": float("nan"), "n_used_metrics": 0, "scored_trajectory": scored, "distance_mode": str(distance_mode)}
    valid = scored[np.isfinite(pd.to_numeric(scored["distance_to_target"], errors="coerce"))].copy()
    if valid.empty:
        return {"best_step": None, "best_damage_frac": float("nan"), "best_distance": float("nan"), "n_used_metrics": 0, "scored_trajectory": scored, "distance_mode": str(distance_mode)}
    idx = int(valid["distance_to_target"].astype(float).idxmin())
    row = valid.loc[idx]
    return {
        "best_step": int(row["step"]) if "step" in row and np.isfinite(row["step"]) else None,
        "best_damage_frac": float(row.get("damage_frac", np.nan)),
        "best_distance": float(row["distance_to_target"]),
        "n_used_metrics": int(row.get("n_used_metrics", 0)),
        "scored_trajectory": scored,
        "best_row": row.to_dict(),
        "distance_mode": str(distance_mode),
    }


def compare_degradation_models(
    hc_graphs: Sequence[nx.Graph],
    *,
    sz_group_metrics_df: pd.DataFrame,
    hc_baseline_metrics_df: pd.DataFrame,
    attack_kinds: Sequence[str],
    metrics: Sequence[str] | None = None,
    steps: int = 12,
    frac: float = 0.5,
    seed: int = 42,
    eff_sources_k: int = 16,
    compute_heavy_every: int = 2,
    compute_curvature: bool = False,
    curvature_sample_edges: int = 80,
    noise_sigma_max: float = 0.5,
    keep_density_from_baseline: bool = True,
    recompute_modules: bool = False,
    module_resolution: float = 1.0,
    removal_mode: str = "random",
    fast_mode: bool = False,
    subject_ids: Sequence[str] | None = None,
    subject_metadata: pd.DataFrame | None = None,
    distance_mode: str = "raw",
    metric_families: Mapping[str, Sequence[str]] | None = None,
) -> dict:
    metric_list = _as_metric_list(metrics)
    normalized_families = normalize_metric_families(metric_list, metric_families)
    target_vector = build_group_target_vector(sz_group_metrics_df, metrics=metric_list)
    scales = build_scale_vector(hc_baseline_metrics_df, metrics=metric_list)
    if subject_ids is None:
        subject_ids = [f"hc_{i:04d}" for i in range(len(hc_graphs))]
    subject_ids = [str(x) for x in subject_ids]
    if len(subject_ids) != len(hc_graphs):
        raise ValueError("Length of subject_ids must match length of hc_graphs")

    meta_df = pd.DataFrame()
    if isinstance(subject_metadata, pd.DataFrame) and not subject_metadata.empty:
        meta_df = subject_metadata.copy()
        if "subject_id" not in meta_df.columns:
            raise ValueError("subject_metadata must include a subject_id column")
        meta_df["subject_id"] = meta_df["subject_id"].astype(str)

    subject_rows: list[dict] = []
    all_traj: list[pd.DataFrame] = []
    scalar_rows: list[dict] = []
    for subj_idx, (subject_id, G) in enumerate(zip(subject_ids, hc_graphs, strict=False)):
        G = nx.Graph(G)
        module_info = None
        if any(k in {"inter_module_removal", "intra_module_removal"} for k in attack_kinds):
            module_info = prepare_module_info(G, seed=int(seed) + subj_idx, resolution=float(module_resolution))

        for kind in attack_kinds:
            traj_df, _aux = run_degradation_trajectory(
                G,
                kind=str(kind),
                steps=int(steps),
                frac=float(frac),
                seed=int(seed) + subj_idx,
                eff_sources_k=int(eff_sources_k),
                compute_heavy_every=int(compute_heavy_every),
                compute_curvature=bool(compute_curvature),
                curvature_sample_edges=int(curvature_sample_edges),
                metric_names=metric_list,
                noise_sigma_max=float(noise_sigma_max),
                keep_density_from_baseline=bool(keep_density_from_baseline),
                module_info=module_info,
                recompute_modules=bool(recompute_modules),
                module_resolution=float(module_resolution),
                removal_mode=str(removal_mode),
                fast_mode=bool(fast_mode),
            )
            best = find_best_match_to_target(
                traj_df,
                target_vector=target_vector,
                metrics=metric_list,
                scales=scales,
                distance_mode=distance_mode,
                metric_families=normalized_families,
            )
            scored = best["scored_trajectory"].copy()
            scored["subject_idx"] = int(subj_idx)
            scored["subject_id"] = str(subject_id)
            scored["attack_kind"] = str(kind)
            all_traj.append(scored)

            best_row = best.get("best_row", {}) or {}
            subject_rows.append({
                "subject_idx": int(subj_idx), "subject_id": str(subject_id), "attack_kind": str(kind),
                "best_step": best["best_step"], "best_damage_frac": float(best["best_damage_frac"]),
                "best_distance": float(best["best_distance"]), "n_used_metrics": int(best["n_used_metrics"]),
                "distance_mode": str(best.get("distance_mode", distance_mode)),
                "best_delta_density": float(best_row.get("delta_density", np.nan)),
                "best_delta_total_weight": float(best_row.get("delta_total_weight", best_row.get("delta_E", np.nan))),
                "best_delta_E": float(best_row.get("delta_E", np.nan)),
                "best_removed_edge_fraction_from_baseline": float(best_row.get("removed_edge_fraction_from_baseline", np.nan)),
            })

            scalar_best = find_best_scalar_match(traj_df, target_vector=target_vector, metrics=metric_list, scales=scales, absolute=True)
            for m in metric_list:
                info = scalar_best.get(m, {})
                scalar_rows.append({
                    "subject_idx": int(subj_idx), "subject_id": str(subject_id), "attack_kind": str(kind), "metric": str(m),
                    "metric_family": next((fam for fam, fam_metrics in normalized_families.items() if m in fam_metrics), "singleton"),
                    "best_step": info.get("best_step", None), "best_damage_frac": float(info.get("best_damage_frac", np.nan)),
                    "best_scalar_error": float(info.get("best_scalar_error", np.nan)), "best_value": float(info.get("best_value", np.nan)),
                })

    subject_df = pd.DataFrame(subject_rows)
    traj_df = pd.concat(all_traj, ignore_index=True) if all_traj else pd.DataFrame()
    if not meta_df.empty and not subject_df.empty:
        subject_df = subject_df.merge(meta_df, on="subject_id", how="left")
    if not meta_df.empty and not traj_df.empty:
        traj_df = traj_df.merge(meta_df, on="subject_id", how="left")

    winners = []
    if not subject_df.empty:
        for _, sub_df in subject_df.groupby("subject_id", dropna=False):
            valid = sub_df[np.isfinite(pd.to_numeric(sub_df["best_distance"], errors="coerce"))].copy()
            if valid.empty:
                continue
            idx = int(valid["best_distance"].astype(float).idxmin())
            row = valid.loc[idx].to_dict()
            row["is_subject_winner"] = True
            winners.append(row)

    winners_df = pd.DataFrame(winners)
    scalar_subject_df = build_scalar_subject_results(pd.DataFrame(scalar_rows), metrics=metric_list)
    if not meta_df.empty and not scalar_subject_df.empty:
        scalar_subject_df = scalar_subject_df.merge(meta_df, on="subject_id", how="left")
    scalar_winners_df = build_scalar_winners(scalar_subject_df)
    scalar_summary_df = build_scalar_summary(scalar_subject_df, scalar_winners_df)

    return {
        "target_vector": target_vector,
        "scales": scales,
        "subject_results": subject_df,
        "winner_results": winners_df,
        "trajectory_results": traj_df,
        "scalar_subject_results": scalar_subject_df,
        "scalar_winners": scalar_winners_df,
        "scalar_summary": scalar_summary_df,
        "metrics_used": metric_list,
        "metric_families": normalized_families,
        "distance_mode": str(distance_mode),
    }


def summarize_best_attack(result: dict) -> dict:
    winners_df = result.get("winner_results", pd.DataFrame())
    subject_df = result.get("subject_results", pd.DataFrame())
    if not isinstance(winners_df, pd.DataFrame) or winners_df.empty:
        return {"best_attack_overall": None, "n_subjects": 0, "winner_counts": {}}
    counts = winners_df["attack_kind"].value_counts(dropna=False).to_dict()
    best_attack_overall = max(counts.items(), key=lambda kv: kv[1])[0] if counts else None
    n_subjects = 0
    if isinstance(subject_df, pd.DataFrame) and not subject_df.empty and "subject_id" in subject_df.columns:
        n_subjects = int(subject_df["subject_id"].nunique())
    elif isinstance(subject_df, pd.DataFrame) and not subject_df.empty and "subject_idx" in subject_df.columns:
        n_subjects = int(subject_df["subject_idx"].nunique())
    elif isinstance(winners_df, pd.DataFrame) and "subject_id" in winners_df.columns:
        n_subjects = int(winners_df["subject_id"].nunique())
    return {"best_attack_overall": best_attack_overall, "n_subjects": int(n_subjects), "winner_counts": counts}
