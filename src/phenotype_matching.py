from __future__ import annotations

from collections.abc import Sequence

import networkx as nx
import numpy as np
import pandas as pd

from .degradation import prepare_module_info, run_degradation_trajectory
from .phenotype_scalar import (
    find_best_scalar_match,
    build_scalar_subject_results,
    build_scalar_winners,
    build_scalar_summary,
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


def build_group_target_vector(
    group_df: pd.DataFrame,
    *,
    metrics: Sequence[str] | None = None,
) -> dict[str, float]:
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


def compute_profile_distance(
    row: dict | pd.Series,
    target_vector: dict[str, float],
    *,
    metrics: Sequence[str] | None = None,
    scales: dict[str, float] | None = None,
) -> dict:
    metric_list = _as_metric_list(metrics)
    used: list[str] = []
    diffs: list[float] = []

    for m in metric_list:
        xv = _coerce_float(row.get(m, np.nan))
        tv = _coerce_float(target_vector.get(m, np.nan))
        sv = _coerce_float((scales or {}).get(m, 1.0))

        if not np.isfinite(xv) or not np.isfinite(tv):
            continue
        if (not np.isfinite(sv)) or sv <= 1e-12:
            sv = 1.0

        used.append(m)
        diffs.append((xv - tv) / sv)

    if not used:
        return {
            "distance": float("nan"),
            "used_metrics": [],
            "n_used_metrics": 0,
        }

    d = float(np.sqrt(np.sum(np.square(np.asarray(diffs, dtype=float)))))
    return {
        "distance": d,
        "used_metrics": used,
        "n_used_metrics": int(len(used)),
    }


def annotate_trajectory_distance_to_target(
    traj_df: pd.DataFrame,
    *,
    target_vector: dict[str, float],
    metrics: Sequence[str] | None = None,
    scales: dict[str, float] | None = None,
) -> pd.DataFrame:
    out = traj_df.copy()
    dists = []
    n_used = []

    for _, row in out.iterrows():
        info = compute_profile_distance(
            row,
            target_vector=target_vector,
            metrics=metrics,
            scales=scales,
        )
        dists.append(float(info["distance"]))
        n_used.append(int(info["n_used_metrics"]))

    out["distance_to_target"] = dists
    out["n_used_metrics"] = n_used
    return out


def find_best_match_to_target(
    traj_df: pd.DataFrame,
    *,
    target_vector: dict[str, float],
    metrics: Sequence[str] | None = None,
    scales: dict[str, float] | None = None,
) -> dict:
    scored = annotate_trajectory_distance_to_target(
        traj_df,
        target_vector=target_vector,
        metrics=metrics,
        scales=scales,
    )

    if scored.empty or "distance_to_target" not in scored.columns:
        return {
            "best_step": None,
            "best_damage_frac": float("nan"),
            "best_distance": float("nan"),
            "n_used_metrics": 0,
            "scored_trajectory": scored,
        }

    valid = scored[np.isfinite(pd.to_numeric(scored["distance_to_target"], errors="coerce"))].copy()
    if valid.empty:
        return {
            "best_step": None,
            "best_damage_frac": float("nan"),
            "best_distance": float("nan"),
            "n_used_metrics": 0,
            "scored_trajectory": scored,
        }

    idx = int(valid["distance_to_target"].astype(float).idxmin())
    row = valid.loc[idx]
    return {
        "best_step": int(row["step"]) if "step" in row and np.isfinite(row["step"]) else None,
        "best_damage_frac": float(row.get("damage_frac", np.nan)),
        "best_distance": float(row["distance_to_target"]),
        "n_used_metrics": int(row.get("n_used_metrics", 0)),
        "scored_trajectory": scored,
        "best_row": row.to_dict(),
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
) -> dict:
    """Run model comparison with vector-distance and scalar-metric breakdown outputs."""
    metric_list = _as_metric_list(metrics)
    target_vector = build_group_target_vector(sz_group_metrics_df, metrics=metric_list)
    scales = build_scale_vector(hc_baseline_metrics_df, metrics=metric_list)

    subject_rows: list[dict] = []
    all_traj: list[pd.DataFrame] = []
    scalar_rows: list[dict] = []

    for subj_idx, G in enumerate(hc_graphs):
        G = nx.Graph(G)
        module_info = None
        if any(k in {"inter_module_removal", "intra_module_removal"} for k in attack_kinds):
            module_info = prepare_module_info(
                G,
                seed=int(seed) + subj_idx,
                resolution=float(module_resolution),
            )

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
            )

            best = find_best_match_to_target(
                traj_df,
                target_vector=target_vector,
                metrics=metric_list,
                scales=scales,
            )
            scored = best["scored_trajectory"].copy()
            scored["subject_idx"] = int(subj_idx)
            scored["attack_kind"] = str(kind)
            all_traj.append(scored)

            subject_rows.append(
                {
                    "subject_idx": int(subj_idx),
                    "attack_kind": str(kind),
                    "best_step": best["best_step"],
                    "best_damage_frac": float(best["best_damage_frac"]),
                    "best_distance": float(best["best_distance"]),
                    "n_used_metrics": int(best["n_used_metrics"]),
                }
            )

            scalar_best = find_best_scalar_match(
                traj_df,
                target_vector=target_vector,
                metrics=metric_list,
                scales=scales,
                absolute=True,
            )
            for m in metric_list:
                info = scalar_best.get(m, {})
                scalar_rows.append(
                    {
                        "subject_idx": int(subj_idx),
                        "attack_kind": str(kind),
                        "metric": str(m),
                        "best_step": info.get("best_step", None),
                        "best_damage_frac": float(info.get("best_damage_frac", np.nan)),
                        "best_scalar_error": float(info.get("best_scalar_error", np.nan)),
                        "best_value": float(info.get("best_value", np.nan)),
                    }
                )

    subject_df = pd.DataFrame(subject_rows)
    traj_df = pd.concat(all_traj, ignore_index=True) if all_traj else pd.DataFrame()

    winners = []
    if not subject_df.empty:
        for subj_idx, sub_df in subject_df.groupby("subject_idx", dropna=False):
            valid = sub_df[np.isfinite(pd.to_numeric(sub_df["best_distance"], errors="coerce"))].copy()
            if valid.empty:
                continue
            idx = int(valid["best_distance"].astype(float).idxmin())
            row = valid.loc[idx].to_dict()
            row["is_subject_winner"] = True
            winners.append(row)

    winners_df = pd.DataFrame(winners)

    scalar_subject_df = build_scalar_subject_results(pd.DataFrame(scalar_rows), metrics=metric_list)
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
    }



def summarize_best_attack(result: dict) -> dict:
    """Build compact winner-centric summary payload for phenotype matching."""
    winners_df = result.get("winner_results", pd.DataFrame())
    subject_df = result.get("subject_results", pd.DataFrame())

    if not isinstance(winners_df, pd.DataFrame) or winners_df.empty:
        return {
            "best_attack_overall": None,
            "n_subjects": 0,
            "winner_counts": {},
        }

    counts = winners_df["attack_kind"].value_counts(dropna=False).to_dict()
    best_attack_overall = None
    if counts:
        best_attack_overall = max(counts.items(), key=lambda kv: kv[1])[0]

    n_subjects = 0
    if isinstance(subject_df, pd.DataFrame) and not subject_df.empty and "subject_idx" in subject_df.columns:
        n_subjects = int(subject_df["subject_idx"].nunique())
    elif isinstance(winners_df, pd.DataFrame) and "subject_idx" in winners_df.columns:
        n_subjects = int(winners_df["subject_idx"].nunique())

    return {
        "best_attack_overall": best_attack_overall,
        "n_subjects": int(n_subjects),
        "winner_counts": counts,
    }
