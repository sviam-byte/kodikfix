"""Batch degradation runner with optional phenotype matching.

This module supports a two-stage workflow:

1. **Degradation stage** (always): run selected attacks for each input graph and
   persist per-item trajectories.
2. **Phenotype stage** (optional): annotate saved trajectories with distance to
   an SZ target phenotype and export summary tables.

The implementation is incremental and resumable by design.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import networkx as nx
import numpy as np
import pandas as pd

from .degradation import prepare_module_info, run_degradation_trajectory
from .phenotype_matching import (
    DEFAULT_METRIC_FAMILIES,
    DEFAULT_PROFILE_METRICS,
    build_group_target_vector,
    compute_profile_distance,
    normalize_metric_families,
    resolve_metric_scales,
)
from .phenotype_reporting import (
    build_attack_summary,
    build_family_summary,
    build_metric_families_df,
    build_scales_df,
    build_target_vector_df,
    build_winner_summary,
)
from .phenotype_scalar import (
    build_scalar_subject_results,
    build_scalar_summary,
    build_scalar_winners,
    scalar_error,
)
from .phenotype_stats import build_stats_tables

# Default attack catalogue: mirrors kinds accepted by run_degradation_trajectory.
ALL_ATTACK_KINDS: list[str] = [
    "weak_edges_by_weight",
    "strong_edges_by_weight",
    "weak_positive_edges",
    "strong_negative_edges",
    "negative_edges_only",
    "negative_edges_by_magnitude",
    "mix_default",
    "mix_degree_preserving",
    "weight_noise",
    "inter_module_removal",
    "intra_module_removal",
    "random_edges",
]

# Compact default metric set for degradation trajectories (without curvature).
DEFAULT_DEGRADE_METRICS: list[str] = [
    "density",
    "clustering",
    "mod",
    "l2_lcc",
    "H_rw",
    "fragility_H",
    "eff_w",
    "lcc_frac",
    "algebraic_connectivity",
    "avg_degree",
    "assortativity",
    "tau_relax",
    "frac_negative_weight",
    "signed_balance_weight",
    "signed_std_weight",
    "signed_mean_weight",
    "signed_entropy_weight",
    "frustration_index",
    "signed_lambda_min",
    "strength_pos_mean",
    "strength_neg_mean",
]


def _item_csv_name(subject_id: str, attack_kind: str) -> str:
    """Build a safe filename for one ``(subject, attack)`` trajectory."""
    safe_sid = subject_id.replace("/", "_").replace("\\", "_").replace(" ", "_")
    return f"{safe_sid}__{attack_kind}.csv"


def _already_done(per_item_dir: Path, subject_id: str, attack_kind: str) -> bool:
    """Return ``True`` when an existing trajectory CSV is present and non-empty."""
    path = per_item_dir / _item_csv_name(subject_id, attack_kind)
    return path.exists() and path.stat().st_size > 50


def _coerce_float(value) -> float:
    """Convert to finite float; return NaN for missing/non-finite values."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return numeric if np.isfinite(numeric) else float("nan")


def run_batch_degrade(
    graphs: Sequence[nx.Graph],
    *,
    subject_ids: Sequence[str],
    attack_kinds: Sequence[str] | None = None,
    out_dir: str | Path,
    steps: int = 12,
    frac: float = 0.5,
    seed: int = 42,
    eff_sources_k: int = 16,
    compute_heavy_every: int = 2,
    metric_names: Sequence[str] | None = None,
    noise_sigma_max: float = 0.5,
    keep_density_from_baseline: bool = True,
    recompute_modules: bool = False,
    module_resolution: float = 1.0,
    removal_mode: str = "random",
    fast_mode: bool = False,
    skip_existing: bool = True,
    start_from: int = 0,
    sz_group_metrics_df: pd.DataFrame | None = None,
    hc_baseline_metrics_df: pd.DataFrame | None = None,
    phenotype_metrics: Sequence[str] | None = None,
    distance_mode: str = "raw",
    metric_families: Mapping[str, Sequence[str]] | None = None,
    progress_cb: Callable[[int, int, str], None] | None = None,
    return_details: bool = False,
) -> Path | dict[str, object]:
    """Run degradation batch and optionally execute phenotype pass.

    By default returns ``Path`` to ``trajectories_all.csv`` for backward
    compatibility. Set ``return_details=True`` to get a rich result dictionary.
    """
    out_path = Path(out_dir)
    per_item_dir = out_path / "per_item"
    per_item_dir.mkdir(parents=True, exist_ok=True)

    attacks = [str(kind) for kind in (attack_kinds or ALL_ATTACK_KINDS)]
    metrics = [str(name) for name in (metric_names or DEFAULT_DEGRADE_METRICS)]

    if len(subject_ids) != len(graphs):
        raise ValueError(
            f"subject_ids length ({len(subject_ids)}) != graphs length ({len(graphs)})"
        )

    n_graphs = len(graphs)
    n_attacks = len(attacks)
    total_jobs = max(0, n_graphs - int(start_from)) * n_attacks
    done_jobs = 0
    skipped_jobs = 0
    errors: list[dict[str, str]] = []
    started_at = time.monotonic()

    needs_modules = any(kind in {"inter_module_removal", "intra_module_removal"} for kind in attacks)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "n_graphs": n_graphs,
        "start_from": int(start_from),
        "attack_kinds": attacks,
        "metric_names": metrics,
        "steps": int(steps),
        "frac": float(frac),
        "seed": int(seed),
        "skip_existing": bool(skip_existing),
        "has_phenotype_pass": isinstance(sz_group_metrics_df, pd.DataFrame)
        and not sz_group_metrics_df.empty,
    }
    (out_path / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    for subj_idx in range(int(start_from), n_graphs):
        subject_id = str(subject_ids[subj_idx])
        graph = nx.Graph(graphs[subj_idx])

        module_info = None
        if needs_modules:
            module_info = prepare_module_info(
                graph,
                seed=int(seed) + subj_idx,
                resolution=float(module_resolution),
            )

        for attack_kind in attacks:
            label = f"[{subj_idx + 1}/{n_graphs}] {subject_id} × {attack_kind}"

            if skip_existing and _already_done(per_item_dir, subject_id, attack_kind):
                skipped_jobs += 1
                done_jobs += 1
                if progress_cb is not None:
                    progress_cb(done_jobs, total_jobs, f"SKIP {label}")
                else:
                    print(f"  SKIP {label}", flush=True)
                continue

            if progress_cb is not None:
                progress_cb(done_jobs, total_jobs, label)
            else:
                print(f"  RUN  {label}", flush=True)

            try:
                traj_df, _aux = run_degradation_trajectory(
                    graph,
                    kind=attack_kind,
                    steps=int(steps),
                    frac=float(frac),
                    seed=int(seed) + subj_idx,
                    eff_sources_k=int(eff_sources_k),
                    compute_heavy_every=int(compute_heavy_every),
                    compute_curvature=False,
                    curvature_sample_edges=0,
                    metric_names=metrics,
                    noise_sigma_max=float(noise_sigma_max),
                    keep_density_from_baseline=bool(keep_density_from_baseline),
                    module_info=module_info,
                    recompute_modules=bool(recompute_modules),
                    module_resolution=float(module_resolution),
                    removal_mode=str(removal_mode),
                    fast_mode=bool(fast_mode),
                )

                traj_df["attack_kind"] = attack_kind
                traj_df["subject_idx"] = int(subj_idx)
                traj_df["subject_id"] = subject_id

                lead = ["subject_id", "subject_idx", "attack_kind"]
                rest = [col for col in traj_df.columns if col not in lead]
                traj_df = traj_df[lead + rest]

                csv_path = per_item_dir / _item_csv_name(subject_id, attack_kind)
                traj_df.to_csv(csv_path, index=False)

            except Exception as exc:  # noqa: BLE001
                message = f"{type(exc).__name__}: {exc}"
                errors.append(
                    {
                        "subject_id": subject_id,
                        "attack_kind": attack_kind,
                        "error": message,
                    }
                )
                print(f"  ERROR {label} :: {message}", flush=True)

            done_jobs += 1

    elapsed = time.monotonic() - started_at
    print(
        f"\n[batch-degrade] done: {done_jobs} jobs, "
        f"{skipped_jobs} skipped, {len(errors)} errors, "
        f"{elapsed:.1f}s elapsed",
        flush=True,
    )

    if errors:
        pd.DataFrame(errors).to_csv(out_path / "errors.csv", index=False)

    trajectories_csv = _consolidate_per_item_dir(per_item_dir, out_path)

    phenotype_result: dict[str, object] | None = None
    if isinstance(sz_group_metrics_df, pd.DataFrame) and not sz_group_metrics_df.empty:
        print("[batch-degrade] running phenotype pass ...", flush=True)
        phenotype_result = run_phenotype_pass(
            trajectories_csv=trajectories_csv,
            sz_group_metrics_df=sz_group_metrics_df,
            hc_baseline_metrics_df=hc_baseline_metrics_df,
            phenotype_metrics=phenotype_metrics,
            distance_mode=distance_mode,
            metric_families=metric_families,
            out_dir=out_path / "phenotype",
        )

    result: dict[str, object] = {
        "trajectories_csv": trajectories_csv,
        "phenotype": phenotype_result,
        "errors": errors,
        "n_done": done_jobs,
        "n_skipped": skipped_jobs,
    }
    return result if return_details else trajectories_csv


def run_phenotype_pass(
    *,
    trajectories_csv: str | Path,
    sz_group_metrics_df: pd.DataFrame,
    hc_baseline_metrics_df: pd.DataFrame | None = None,
    phenotype_metrics: Sequence[str] | None = None,
    distance_mode: str = "raw",
    metric_families: Mapping[str, Sequence[str]] | None = None,
    out_dir: str | Path | None = None,
) -> dict[str, object]:
    """Run phenotype matching on pre-computed trajectories."""
    traj_df = pd.read_csv(trajectories_csv)
    if traj_df.empty:
        print("[phenotype-pass] trajectory CSV is empty, nothing to do", flush=True)
        return {}

    requested_metrics = (
        [str(metric) for metric in phenotype_metrics if str(metric)]
        if phenotype_metrics is not None
        else list(DEFAULT_PROFILE_METRICS)
    )
    requested_metrics = [metric for metric in requested_metrics if metric in traj_df.columns]

    if not requested_metrics:
        print("[phenotype-pass] no matching metrics in trajectory CSV", flush=True)
        return {}

    if isinstance(hc_baseline_metrics_df, pd.DataFrame) and not hc_baseline_metrics_df.empty:
        resolved = resolve_metric_scales(hc_baseline_metrics_df, metrics=requested_metrics)
        metric_list = [metric for metric in (resolved.get("kept_metrics") or []) if metric in traj_df.columns]
        scales = {str(k): float(v) for k, v in (resolved.get("scales") or {}).items() if str(k) in metric_list}
        metric_audit_df = resolved.get("audit_df", pd.DataFrame())
        excluded_metrics = [str(x) for x in resolved.get("excluded_metrics", []) if str(x) in requested_metrics]
    else:
        metric_list = list(requested_metrics)
        scales = {metric: 1.0 for metric in metric_list}
        metric_audit_df = pd.DataFrame()
        excluded_metrics = []

    if not metric_list:
        print("[phenotype-pass] all candidate metrics were excluded by baseline variability audit", flush=True)
        return {}

    target_vector = build_group_target_vector(sz_group_metrics_df, metrics=metric_list)

    normalized_families = normalize_metric_families(
        metric_list,
        metric_families if metric_families is not None else DEFAULT_METRIC_FAMILIES,
    )

    distances: list[float] = []
    n_used_metrics: list[int] = []
    scalar_error_cols: dict[str, list[float]] = {
        f"{metric}__scalar_error": [] for metric in metric_list
    }

    for _, row in traj_df.iterrows():
        info = compute_profile_distance(
            row,
            target_vector=target_vector,
            metrics=metric_list,
            scales=scales,
            distance_mode=distance_mode,
            metric_families=normalized_families,
        )
        distances.append(float(info["distance"]))
        n_used_metrics.append(int(info["n_used_metrics"]))

        for metric in metric_list:
            current_value = _coerce_float(row.get(metric, np.nan))
            target_value = _coerce_float(target_vector.get(metric, np.nan))
            scale_value = _coerce_float(scales.get(metric, 1.0))
            if np.isfinite(current_value) and np.isfinite(target_value):
                error_value = scalar_error(current_value, target_value, scale=scale_value, absolute=True)
            else:
                error_value = float("nan")
            scalar_error_cols[f"{metric}__scalar_error"].append(float(error_value))

    traj_df["distance_to_target"] = distances
    traj_df["n_used_metrics"] = n_used_metrics
    for col_name, values in scalar_error_cols.items():
        traj_df[col_name] = values

    subject_rows: list[dict[str, object]] = []
    scalar_rows: list[dict[str, object]] = []

    for (subject_id, attack_kind), sub_df in traj_df.groupby(["subject_id", "attack_kind"], dropna=False):
        valid = sub_df[np.isfinite(pd.to_numeric(sub_df["distance_to_target"], errors="coerce"))]
        if valid.empty:
            continue

        best_idx = int(valid["distance_to_target"].astype(float).idxmin())
        best_row = valid.loc[best_idx]
        subject_rows.append(
            {
                "subject_id": str(subject_id),
                "subject_idx": int(best_row.get("subject_idx", 0)),
                "attack_kind": str(attack_kind),
                "best_step": int(best_row["step"])
                if "step" in best_row and np.isfinite(best_row["step"])
                else None,
                "best_damage_frac": float(best_row.get("damage_frac", np.nan)),
                "best_distance": float(best_row["distance_to_target"]),
                "n_used_metrics": int(best_row.get("n_used_metrics", 0)),
            }
        )

        for metric in metric_list:
            error_col = f"{metric}__scalar_error"
            if error_col not in sub_df.columns:
                continue
            scalar_valid = sub_df[np.isfinite(pd.to_numeric(sub_df[error_col], errors="coerce"))]
            if scalar_valid.empty:
                continue
            scalar_idx = int(scalar_valid[error_col].astype(float).idxmin())
            scalar_best = scalar_valid.loc[scalar_idx]
            scalar_rows.append(
                {
                    "subject_id": str(subject_id),
                    "subject_idx": int(scalar_best.get("subject_idx", 0)),
                    "attack_kind": str(attack_kind),
                    "metric": metric,
                    "metric_family": next(
                        (
                            family
                            for family, family_metrics in normalized_families.items()
                            if metric in family_metrics
                        ),
                        "singleton",
                    ),
                    "best_step": int(scalar_best["step"])
                    if "step" in scalar_best and np.isfinite(scalar_best["step"])
                    else None,
                    "best_damage_frac": float(scalar_best.get("damage_frac", np.nan)),
                    "best_scalar_error": float(scalar_best[error_col]),
                    "best_value": float(scalar_best.get(metric, np.nan)),
                }
            )

    subject_df = pd.DataFrame(subject_rows)
    scalar_subject_df = build_scalar_subject_results(pd.DataFrame(scalar_rows), metrics=metric_list)

    winners: list[dict[str, object]] = []
    if not subject_df.empty:
        for _, by_subject in subject_df.groupby("subject_id", dropna=False):
            valid = by_subject[np.isfinite(pd.to_numeric(by_subject["best_distance"], errors="coerce"))]
            if valid.empty:
                continue
            row = valid.loc[int(valid["best_distance"].astype(float).idxmin())].to_dict()
            row["is_subject_winner"] = True
            winners.append(row)

    winners_df = pd.DataFrame(winners)
    scalar_winners_df = build_scalar_winners(scalar_subject_df)
    scalar_summary_df = build_scalar_summary(scalar_subject_df, scalar_winners_df)

    result: dict[str, object] = {
        "target_vector": target_vector,
        "scales": scales,
        "subject_results": subject_df,
        "winner_results": winners_df,
        "trajectory_results": traj_df,
        "scalar_subject_results": scalar_subject_df,
        "scalar_winners": scalar_winners_df,
        "scalar_summary": scalar_summary_df,
        "metrics_requested": requested_metrics,
        "metrics_used": metric_list,
        "metrics_excluded": excluded_metrics,
        "metric_scale_audit": metric_audit_df,
        "metric_families": normalized_families,
        "distance_mode": str(distance_mode),
    }

    summary_attack = build_attack_summary(subject_df)
    total_subjects = int(subject_df["subject_id"].nunique()) if not subject_df.empty else 0
    summary_winners = build_winner_summary(winners_df, total_subjects=total_subjects)
    family_summary = build_family_summary(scalar_summary_df)
    stats_pack = build_stats_tables(result)

    tables: dict[str, object] = {
        "summary_attack": summary_attack,
        "summary_winners": summary_winners,
        "target_vector": build_target_vector_df(target_vector),
        "scales": build_scales_df(scales),
        "metric_families": build_metric_families_df(normalized_families),
        "family_summary": family_summary,
        "subject_results": subject_df,
        "winner_results": winners_df,
        "scalar_subject_results": scalar_subject_df,
        "scalar_winners": scalar_winners_df,
        "scalar_summary": scalar_summary_df,
        **stats_pack,
    }

    if out_dir is not None:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        traj_df.to_csv(out_path / "trajectories_annotated.csv", index=False)
        for name, table in tables.items():
            if isinstance(table, pd.DataFrame):
                table.to_csv(out_path / f"{name}.csv", index=False)

        try:
            with pd.ExcelWriter(out_path / "phenotype_results.xlsx", engine="openpyxl") as writer:
                for name, table in tables.items():
                    if isinstance(table, pd.DataFrame) and not table.empty:
                        table.to_excel(writer, index=False, sheet_name=name[:31])
        except Exception as exc:  # noqa: BLE001
            print(f"[phenotype-pass] XLSX export failed: {exc}", flush=True)

    return {**result, **tables}


def _consolidate_per_item_dir(per_item_dir: Path, out_dir: Path) -> Path:
    """Concatenate all per-item CSVs into a single master trajectory table."""
    csv_files = sorted(per_item_dir.glob("*.csv"))
    master_path = out_dir / "trajectories_all.csv"

    if not csv_files:
        pd.DataFrame().to_csv(master_path, index=False)
        return master_path

    chunks: list[pd.DataFrame] = []
    for path in csv_files:
        try:
            df = pd.read_csv(path)
        except Exception:  # noqa: BLE001
            print(f"  WARN: could not read {path.name}, skipping", flush=True)
            continue
        if not df.empty:
            chunks.append(df)

    master_df = pd.concat(chunks, ignore_index=True, sort=False) if chunks else pd.DataFrame()
    master_df.to_csv(master_path, index=False)
    print(f"[batch-degrade] consolidated {len(chunks)} files -> {master_path}", flush=True)
    print(f"[batch-degrade] total rows: {len(master_df)}", flush=True)
    return master_path


def consolidate(out_dir: str | Path) -> Path:
    """Standalone helper to rebuild ``trajectories_all.csv`` from ``per_item``."""
    out_path = Path(out_dir)
    return _consolidate_per_item_dir(out_path / "per_item", out_path)
