from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .metric_registry import (
    describe_metrics_for_regime,
    split_metrics_by_regime,
)
from .phenotype_matching import normalize_metric_families

ATTACK_FAMILY_MAP: dict[str, str] = {
    "weight_noise": "weight",
    "weight_noise_pure": "weight",
    "weak_edges_by_weight": "topology",
    "strong_edges_by_weight": "topology",
    "weak_positive_edges": "topology",
    "strong_negative_edges": "topology",
    "negative_edges_only": "topology",
    "negative_edges_by_magnitude": "topology",
    "random_edges": "topology",
    "inter_module_removal": "topology",
    "intra_module_removal": "topology",
    "mix_default": "topology",
    "mix_degree_preserving": "topology",
}


def _as_metric_list(metrics: Sequence[str] | None) -> list[str]:
    return [str(m) for m in (metrics or []) if str(m)]


def _safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _nan_rate(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns or len(df) == 0:
        return float("nan")
    return float(_safe_num(df[col]).isna().mean())


def _find_subject_duplicates(df: pd.DataFrame, subject_id_col: str) -> list[str]:
    if subject_id_col not in df.columns:
        return []
    vc = df[subject_id_col].astype(str).value_counts(dropna=False)
    return [str(idx) for idx, count in vc.items() if int(count) > 1 and str(idx) != "nan"]


def _correlated_pairs(df: pd.DataFrame, metrics: Sequence[str], threshold: float = 0.95) -> list[dict]:
    cols = [m for m in metrics if m in df.columns]
    if len(cols) < 2:
        return []
    corr = df[cols].apply(pd.to_numeric, errors="coerce").corr(method="spearman", min_periods=3)
    out = []
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            val = corr.loc[a, b]
            if pd.notna(val) and abs(float(val)) >= float(threshold):
                out.append({"metric_a": a, "metric_b": b, "spearman_r": float(val)})
    out.sort(key=lambda x: abs(x["spearman_r"]), reverse=True)
    return out


def build_run_manifest(*, run_type: str, hc_paths: Sequence[str] | None, sz_metrics_path: str | None, hc_baseline_metrics_path: str | None, metric_list: Sequence[str], metric_families: Mapping[str, Sequence[str]] | None, attack_kinds: Sequence[str], steps: int, frac: float, seed: int, distance_mode: str, module_resolution: float, recompute_modules: bool, removal_mode: str, graph_regime: str = "full_weighted_signed_hybrid", weight_policy: str = "", sign_policy: str = "", notes: str = "") -> dict:
    """Build serialized run manifest with regime and attack-family metadata."""
    return {
        "run_type": str(run_type),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hc_inputs": [str(x) for x in (hc_paths or [])],
        "sz_metrics_path": str(sz_metrics_path or ""),
        "hc_baseline_metrics_path": str(hc_baseline_metrics_path or ""),
        "metric_list": [str(x) for x in metric_list],
        "metric_families": normalize_metric_families(metric_list, metric_families),
        "attack_kinds": [str(x) for x in attack_kinds],
        "attack_families": {str(x): ATTACK_FAMILY_MAP.get(str(x), "unknown") for x in attack_kinds},
        "steps": int(steps),
        "frac": float(frac),
        "seed": int(seed),
        "distance_mode": str(distance_mode),
        "module_resolution": float(module_resolution),
        "recompute_modules": bool(recompute_modules),
        "removal_mode": str(removal_mode),
        "graph_regime": str(graph_regime),
        "weight_policy": str(weight_policy),
        "sign_policy": str(sign_policy),
        "notes": str(notes or ""),
    }


def run_phenotype_preflight(*, sz_group_metrics_df: pd.DataFrame, hc_baseline_metrics_df: pd.DataFrame, metrics: Sequence[str], subject_ids: Sequence[str] | None = None, subject_id_col: str = "subject_id", metric_families: Mapping[str, Sequence[str]] | None = None, graph_regime: str = "full_weighted_signed_hybrid", attack_kinds: Sequence[str] | None = None) -> dict:
    """Run consistency checks before HC->SZ phenotype matching."""
    metric_list = _as_metric_list(metrics)
    normalized_families = normalize_metric_families(metric_list, metric_families)
    common_metrics = [m for m in metric_list if m in sz_group_metrics_df.columns and m in hc_baseline_metrics_df.columns]
    missing_in_sz = [m for m in metric_list if m not in sz_group_metrics_df.columns]
    missing_in_hc = [m for m in metric_list if m not in hc_baseline_metrics_df.columns]
    warnings, fatal_errors = [], []
    if not common_metrics:
        fatal_errors.append("No common metrics between SZ metrics table and HC baseline metrics table.")
    if missing_in_sz:
        warnings.append(f"Missing in SZ metrics table: {', '.join(missing_in_sz)}")
    if missing_in_hc:
        warnings.append(f"Missing in HC baseline metrics table: {', '.join(missing_in_hc)}")

    duplicate_subject_ids = []
    if subject_ids is not None:
        vc = pd.Series([str(x) for x in subject_ids], dtype="string").value_counts(dropna=False)
        duplicate_subject_ids = [str(idx) for idx, count in vc.items() if int(count) > 1 and str(idx) != "<NA>"]
        if duplicate_subject_ids:
            fatal_errors.append(f"Duplicate subject_ids in HC input list: {', '.join(duplicate_subject_ids[:10])}")

    hc_duplicates = _find_subject_duplicates(hc_baseline_metrics_df, subject_id_col)
    sz_duplicates = _find_subject_duplicates(sz_group_metrics_df, subject_id_col)
    if hc_duplicates:
        warnings.append(f"Duplicate HC table subject ids: {', '.join(hc_duplicates[:10])}")
    if sz_duplicates:
        warnings.append(f"Duplicate SZ table subject ids: {', '.join(sz_duplicates[:10])}")

    scales, near_zero_scales = {}, []
    for m in common_metrics:
        arr = _safe_num(hc_baseline_metrics_df[m]).to_numpy(dtype=float)
        arr = arr[np.isfinite(arr)]
        s = float(np.nanstd(arr, ddof=1)) if arr.size >= 2 else (0.0 if arr.size == 1 else float("nan"))
        scales[m] = s
        if (not np.isfinite(s)) or s < 1e-8:
            near_zero_scales.append(m)
    if near_zero_scales:
        warnings.append(f"Near-zero or invalid HC scales for metrics: {', '.join(near_zero_scales)}")

    nan_summary = [{"metric": m, "nan_rate_sz": _nan_rate(sz_group_metrics_df, m), "nan_rate_hc": _nan_rate(hc_baseline_metrics_df, m)} for m in metric_list]
    highly_correlated = _correlated_pairs(hc_baseline_metrics_df, common_metrics)
    if highly_correlated:
        warnings.append("Highly correlated HC baseline metrics detected: " + "; ".join(f"{x['metric_a']}~{x['metric_b']} (r={x['spearman_r']:.3f})" for x in highly_correlated[:5]))

    family_counts = {fam: len(vals) for fam, vals in normalized_families.items()}
    if family_counts and max(family_counts.values()) >= max(3, 2 * max(1, min(family_counts.values()))):
        warnings.append("Metric families are imbalanced; consider family_balanced distance mode.")

    regime = str(graph_regime or "")
    regime_metric_info = describe_metrics_for_regime(regime)
    metric_split = split_metrics_by_regime(metric_list, regime)

    invalid_for_regime = list(metric_split["invalid"])
    discouraged_metrics = list(metric_split["discouraged"])
    guardrail_metrics = list(metric_split["guardrail"])

    if invalid_for_regime:
        warnings.append(
            "Метрики невалидны для выбранного graph_regime и будут исключены из phenotype-distance: "
            + ", ".join(invalid_for_regime)
        )

    if discouraged_metrics:
        warnings.append(
            "Метрики нежелательны для выбранного graph_regime и, как правило, вырождаются или слабоинформативны: "
            + ", ".join(discouraged_metrics)
        )

    attack_list = [str(x) for x in (attack_kinds or []) if str(x)]
    attack_families = {k: ATTACK_FAMILY_MAP.get(k, "unknown") for k in attack_list}
    if regime.startswith("full_weighted"):
        topology_first = [k for k, fam in attack_families.items() if fam == "topology"]
        weight_first = [k for k, fam in attack_families.items() if fam == "weight"]
        if topology_first and not weight_first:
            warnings.append(
                "Full weighted regime selected, but only topology attacks are present. "
                "Add weight_noise/weight_noise_pure or another weight attack before launching phenotype matching."
            )

    return {
        "ok": len(fatal_errors) == 0,
        "fatal_errors": fatal_errors,
        "warnings": warnings,
        "metrics_requested": metric_list,
        "metrics_effective": common_metrics,
        "missing_in_sz": missing_in_sz,
        "missing_in_hc": missing_in_hc,
        "near_zero_scales": near_zero_scales,
        "duplicate_subject_ids": duplicate_subject_ids,
        "metric_nan_summary": nan_summary,
        "highly_correlated_pairs": highly_correlated,
        "metric_families": normalized_families,
        "n_sz_rows": int(len(sz_group_metrics_df)),
        "n_hc_baseline_rows": int(len(hc_baseline_metrics_df)),
        "graph_regime": regime,
        "attack_families": attack_families,
        "regime_metric_info": regime_metric_info,
        "metric_split": metric_split,
        "invalid_for_graph_regime": invalid_for_regime,
        "discouraged_for_graph_regime": discouraged_metrics,
        "guardrail_metrics": guardrail_metrics,
        "discouraged_metrics": discouraged_metrics,
        "density_estimate": float("nan"),
    }


def save_run_bundle(*, out_dir: str | Path, result: dict, manifest: dict, preflight_report: dict, extra_tables: Mapping[str, pd.DataFrame] | None = None) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "preflight_report.json").write_text(json.dumps(preflight_report, ensure_ascii=False, indent=2), encoding="utf-8")
    table_map = {
        "subject_results.csv": result.get("subject_results", pd.DataFrame()),
        "winner_results.csv": result.get("winner_results", pd.DataFrame()),
        "trajectory_results.csv": result.get("trajectory_results", pd.DataFrame()),
        "scalar_subject_results.csv": result.get("scalar_subject_results", pd.DataFrame()),
        "scalar_winners.csv": result.get("scalar_winners", pd.DataFrame()),
        "scalar_summary.csv": result.get("scalar_summary", pd.DataFrame()),
    }
    if extra_tables:
        table_map.update(extra_tables)
    for filename, df in table_map.items():
        if isinstance(df, pd.DataFrame):
            df.to_csv(out_dir / filename, index=False)
    return out_dir
