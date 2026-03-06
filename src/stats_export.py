from __future__ import annotations

from io import BytesIO
from typing import Callable
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd

from .graph_build import build_graph_from_edges, lcc_subgraph
from .metrics import calculate_metrics
from .state_models import ExperimentEntry, GraphEntry


DEFAULT_TRAJECTORY_META_COLS = {
    "step",
    "removed_frac",
    "mix_frac",
    "mix_frac_effective",
    "swap_frac_effective",
    "requested_swaps_step",
    "requested_replaced_step",
    "swaps_done_step",
    "replaced_done_step",
    "total_swaps_done",
    "total_replaced_done",
    "N",
    "E",
    "C",
}


def _safe_name(x: str) -> str:
    """Return stable display-safe stem for subject identifiers."""
    return Path(str(x)).stem.strip() or str(x)


def _infer_group(name: str, source: str = "") -> str:
    """Infer coarse clinical/control group from graph name/source heuristics."""
    txt = f"{name} {source}".lower()
    if any(tok in txt for tok in ("hc", "healthy", "control", "norm", "норма", "контроль")):
        return "HC"
    if any(tok in txt for tok in ("sz", "schizo", "patient", "case", "patient_", "пациент")):
        return "SZ"
    return "UNK"


def _subject_id_from_graph(entry: GraphEntry) -> str:
    """Use graph name as subject identifier when available."""
    return _safe_name(getattr(entry, "name", entry.id))


def _entry_to_graph(
    entry: GraphEntry,
    *,
    min_conf: float,
    min_weight: float,
    analysis_mode: str,
):
    """Build filtered networkx graph for stats export pipeline."""
    df = entry.get_filtered_edges(float(min_conf), float(min_weight))
    graph = build_graph_from_edges(df, entry.src_col, entry.dst_col, strict=True)
    if str(analysis_mode).startswith("LCC"):
        graph = lcc_subgraph(graph)
    return graph


def _metrics_to_plain_dict(metrics_obj) -> dict:
    """Normalize metric payload object to plain dict."""
    if isinstance(metrics_obj, dict):
        return dict(metrics_obj)
    if hasattr(metrics_obj, "to_dict"):
        try:
            return dict(metrics_obj.to_dict())
        except Exception:
            pass
    if hasattr(metrics_obj, "__dict__"):
        return dict(vars(metrics_obj))
    return {}


def build_subject_metrics_table(
    graphs: dict[str, GraphEntry],
    *,
    min_conf: float,
    min_weight: float,
    analysis_mode: str,
    eff_sources_k: int = 32,
    seed: int = 42,
    compute_curvature: bool = True,
    curvature_sample_edges: int = 120,
    graph_ids: list[str] | None = None,
    progress_cb: Callable[[int, int, str], None] | None = None,
    lightweight: bool = False,
) -> pd.DataFrame:
    """Build one-row-per-subject metrics table for downstream group statistics."""
    rows: list[dict] = []

    if graph_ids:
        entries = [graphs[gid] for gid in graph_ids if gid in graphs]
    else:
        entries = list(graphs.values())

    total = len(entries)
    for idx, entry in enumerate(entries, start=1):
        if progress_cb is not None:
            progress_cb(idx - 1, total, entry.name)
        try:
            graph = _entry_to_graph(
                entry,
                min_conf=min_conf,
                min_weight=min_weight,
                analysis_mode=analysis_mode,
            )
            n_nodes = graph.number_of_nodes()
            n_edges = graph.number_of_edges()
            large_graph = n_nodes > 300
            huge_graph = n_nodes > 1200 or n_edges > 8000
            met = calculate_metrics(
                graph,
                eff_sources_k=int(eff_sources_k),
                seed=int(seed),
                compute_curvature=bool(compute_curvature and not lightweight),
                curvature_sample_edges=int(curvature_sample_edges),
                compute_heavy=not (lightweight or large_graph),
                skip_spectral=bool(lightweight or huge_graph),
                diameter_samples=6 if (lightweight or large_graph) else 16,
                skip_clustering=bool(lightweight),
                skip_assortativity=bool(lightweight),
            )
            met = _metrics_to_plain_dict(met)
            row = {
                "graph_id": entry.id,
                "graph_name": entry.name,
                "subject_id": _subject_id_from_graph(entry),
                "group": _infer_group(entry.name, entry.source),
                "source": entry.source,
                "analysis_mode": str(analysis_mode),
                "min_conf": float(min_conf),
                "min_weight": float(min_weight),
                "seed": int(seed),
                "eff_sources_k": int(eff_sources_k),
                "compute_curvature": bool(compute_curvature),
                "curvature_sample_edges": int(curvature_sample_edges),
            }
            row.update(met)
            rows.append(row)
        except Exception as exc:
            # Export should be robust: keep failed rows with error description.
            rows.append(
                {
                    "graph_id": entry.id,
                    "graph_name": entry.name,
                    "subject_id": _subject_id_from_graph(entry),
                    "group": _infer_group(entry.name, entry.source),
                    "source": entry.source,
                    "analysis_mode": str(analysis_mode),
                    "min_conf": float(min_conf),
                    "min_weight": float(min_weight),
                    "seed": int(seed),
                    "eff_sources_k": int(eff_sources_k),
                    "compute_curvature": bool(compute_curvature),
                    "curvature_sample_edges": int(curvature_sample_edges),
                    "export_error": str(exc),
                }
            )

    if progress_cb is not None:
        progress_cb(total, total, "done")

    return pd.DataFrame(rows)


def build_mixfrac_subjects_table(
    experiments: list[ExperimentEntry],
    graphs: dict[str, GraphEntry],
) -> pd.DataFrame:
    """Build one-row-per-mixfrac-result table for mix_frac* group-level analyses."""
    rows: list[dict] = []

    for exp in experiments:
        params = exp.params or {}
        is_mixfrac = exp.attack_kind == "mix_frac_estimate" or params.get("attack_family") == "mixfrac"
        if not is_mixfrac:
            continue

        entry = graphs.get(exp.graph_id)
        graph_name = entry.name if entry is not None else exp.graph_id
        source = entry.source if entry is not None else ""

        rows.append(
            {
                "experiment_id": exp.id,
                "experiment_name": exp.name,
                "graph_id": exp.graph_id,
                "graph_name": graph_name,
                "subject_id": _subject_id_from_graph(entry) if entry is not None else exp.graph_id,
                "group": _infer_group(graph_name, source),
                "source": source,
                "created_at": exp.created_at,
                "mix_frac_star": params.get("mix_frac_star", np.nan),
                "ci_low": params.get("ci_low", np.nan),
                "ci_high": params.get("ci_high", np.nan),
                "distance_median": params.get("distance_median", np.nan),
                "replace_from": params.get("replace_from", ""),
                "healthy_n": params.get("healthy_n", np.nan),
                "match_mode": params.get("match_mode", ""),
                "used_metrics": ";".join([str(x) for x in params.get("used_metrics", [])]),
            }
        )

    return pd.DataFrame(rows)


def build_trajectories_long_table(
    experiments: list[ExperimentEntry],
    graphs: dict[str, GraphEntry],
) -> pd.DataFrame:
    """Build long-format trajectory table for repeated-measures and mixed models."""
    chunks: list[pd.DataFrame] = []

    for exp in experiments:
        hist = exp.history.copy()
        if hist is None or hist.empty:
            continue

        entry = graphs.get(exp.graph_id)
        graph_name = entry.name if entry is not None else exp.graph_id
        source = entry.source if entry is not None else ""
        params = exp.params or {}

        meta_cols = [c for c in DEFAULT_TRAJECTORY_META_COLS if c in hist.columns]
        numeric_cols = []
        for col in hist.columns:
            if col in meta_cols:
                continue
            s = pd.to_numeric(hist[col], errors="coerce")
            if np.isfinite(s.to_numpy(dtype=float)).any():
                numeric_cols.append(col)

        if not numeric_cols:
            continue

        base = hist.copy()
        if "mix_frac_effective" in base.columns:
            base["x_value"] = pd.to_numeric(base["mix_frac_effective"], errors="coerce")
            base["x_kind"] = "mix_frac_effective"
        elif "mix_frac" in base.columns:
            base["x_value"] = pd.to_numeric(base["mix_frac"], errors="coerce")
            base["x_kind"] = "mix_frac"
        elif "removed_frac" in base.columns:
            base["x_value"] = pd.to_numeric(base["removed_frac"], errors="coerce")
            base["x_kind"] = "removed_frac"
        else:
            base["x_value"] = pd.to_numeric(base.get("step", np.arange(len(base))), errors="coerce")
            base["x_kind"] = "step"

        id_vars = sorted(set(meta_cols + ["x_value", "x_kind"]))
        long_df = base.melt(
            id_vars=id_vars,
            value_vars=numeric_cols,
            var_name="metric",
            value_name="value",
        )
        long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce")
        long_df = long_df[np.isfinite(long_df["value"].to_numpy(dtype=float))]

        long_df.insert(0, "experiment_id", exp.id)
        long_df.insert(1, "experiment_name", exp.name)
        long_df.insert(2, "graph_id", exp.graph_id)
        long_df.insert(3, "graph_name", graph_name)
        long_df.insert(4, "subject_id", _subject_id_from_graph(entry) if entry is not None else exp.graph_id)
        long_df.insert(5, "group", _infer_group(graph_name, source))
        long_df.insert(6, "source", source)
        long_df.insert(7, "attack_kind", exp.attack_kind)
        long_df.insert(8, "attack_family", params.get("attack_family", ""))
        chunks.append(long_df)

    if not chunks:
        return pd.DataFrame(
            columns=[
                "experiment_id",
                "experiment_name",
                "graph_id",
                "graph_name",
                "subject_id",
                "group",
                "source",
                "attack_kind",
                "attack_family",
                "step",
                "x_kind",
                "x_value",
                "metric",
                "value",
            ]
        )
    return pd.concat(chunks, ignore_index=True)


def export_stats_zip_bytes(
    graphs: dict[str, GraphEntry],
    experiments: list[ExperimentEntry],
    *,
    min_conf: float,
    min_weight: float,
    analysis_mode: str,
    eff_sources_k: int = 32,
    seed: int = 42,
    compute_curvature: bool = True,
    curvature_sample_edges: int = 120,
    graph_ids: list[str] | None = None,
    progress_cb: Callable[[int, int, str], None] | None = None,
    lightweight: bool = False,
) -> bytes:
    """Export tidy statistical tables as zip with three CSV files."""
    df_subjects = build_subject_metrics_table(
        graphs,
        min_conf=min_conf,
        min_weight=min_weight,
        analysis_mode=analysis_mode,
        eff_sources_k=eff_sources_k,
        seed=seed,
        compute_curvature=compute_curvature,
        curvature_sample_edges=curvature_sample_edges,
        graph_ids=graph_ids,
        progress_cb=progress_cb,
        lightweight=lightweight,
    )
    df_mixfrac = build_mixfrac_subjects_table(experiments, graphs)
    df_traj = build_trajectories_long_table(experiments, graphs)

    buf = BytesIO()
    with ZipFile(buf, mode="w", compression=ZIP_DEFLATED) as zf:
        zf.writestr("subject_metrics.csv", df_subjects.to_csv(index=False))
        zf.writestr("mixfrac_subjects.csv", df_mixfrac.to_csv(index=False))
        zf.writestr("trajectories_long.csv", df_traj.to_csv(index=False))
    buf.seek(0)
    return buf.getvalue()


def export_stats_xlsx_bytes(
    graphs: dict[str, GraphEntry],
    experiments: list[ExperimentEntry],
    *,
    min_conf: float,
    min_weight: float,
    analysis_mode: str,
    eff_sources_k: int = 32,
    seed: int = 42,
    compute_curvature: bool = True,
    curvature_sample_edges: int = 120,
    graph_ids: list[str] | None = None,
    progress_cb: Callable[[int, int, str], None] | None = None,
    lightweight: bool = False,
) -> bytes:
    """Export tidy statistical tables as Excel workbook with three sheets."""
    df_subjects = build_subject_metrics_table(
        graphs,
        min_conf=min_conf,
        min_weight=min_weight,
        analysis_mode=analysis_mode,
        eff_sources_k=eff_sources_k,
        seed=seed,
        compute_curvature=compute_curvature,
        curvature_sample_edges=curvature_sample_edges,
        graph_ids=graph_ids,
        progress_cb=progress_cb,
        lightweight=lightweight,
    )
    df_mixfrac = build_mixfrac_subjects_table(experiments, graphs)
    df_traj = build_trajectories_long_table(experiments, graphs)

    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df_subjects.to_excel(writer, sheet_name="subject_metrics", index=False)
        df_mixfrac.to_excel(writer, sheet_name="mixfrac_subjects", index=False)
        df_traj.to_excel(writer, sheet_name="trajectories_long", index=False)
    buf.seek(0)
    return buf.getvalue()
