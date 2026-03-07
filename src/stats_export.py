from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Callable
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

EXPORT_SETTINGS_COLUMNS = ["key", "value"]
EXPORT_OVERVIEW_COLUMNS = ["table_name", "rows", "status", "note"]
EXPORT_ERRORS_COLUMNS = ["scope", "entity_id", "entity_name", "error_type", "message"]
EXPORT_MANIFEST_COLUMNS = [
    "graph_id",
    "graph_name",
    "subject_id",
    "group",
    "source",
    "created_at",
    "selected_for_export",
    "filtered_edge_rows",
    "experiment_count",
    "mixfrac_experiment_count",
    "trajectory_experiment_count",
]


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


def _ensure_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Return a frame containing at least the requested columns in stable order."""
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            out[col] = np.nan
    ordered = list(columns) + [c for c in out.columns if c not in columns]
    return out.loc[:, ordered]


def _clean_for_export(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize NaN/inf/object values for cleaner CSV/XLSX exports."""
    out = df.copy()
    if out.empty:
        return out
    out = out.replace([np.inf, -np.inf], np.nan)
    for col in out.columns:
        series = out[col]
        if pd.api.types.is_datetime64_any_dtype(series):
            continue
        if pd.api.types.is_float_dtype(series):
            continue
        if pd.api.types.is_integer_dtype(series):
            continue
        if pd.api.types.is_bool_dtype(series):
            continue
        out[col] = series.map(lambda x: "" if x is None else x)
    return out


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

            def _metric_progress(frac: float, _idx: int = idx, _name: str = entry.name) -> None:
                """Map inner metric progress to outer per-graph export progress."""
                if progress_cb is None:
                    return
                progress_cb(
                    (_idx - 1) + min(0.95, max(0.0, float(frac))),
                    total,
                    f"{_name} · Ricci {int(round(float(frac) * 100))}%",
                )

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
                progress_cb=_metric_progress if bool(compute_curvature and not lightweight) else None,
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
                "export_status": "ok",
                "export_error": "",
            }
            row.update(met)
            rows.append(row)
            if progress_cb is not None:
                progress_cb(idx, total, f"{entry.name} ✓")
        except Exception as exc:
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
                    "export_status": "error",
                    "export_error": str(exc),
                }
            )
            if progress_cb is not None:
                progress_cb(idx, total, f"{entry.name} ✗ {type(exc).__name__}")

    if progress_cb is not None:
        progress_cb(total, total, "done")

    return _clean_for_export(pd.DataFrame(rows))


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

    return _clean_for_export(pd.DataFrame(rows))


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
    return _clean_for_export(pd.concat(chunks, ignore_index=True))


def build_export_settings_table(*,
    min_conf: float,
    min_weight: float,
    analysis_mode: str,
    eff_sources_k: int,
    seed: int,
    compute_curvature: bool,
    curvature_sample_edges: int,
    graph_ids: list[str] | None,
    lightweight: bool,
) -> pd.DataFrame:
    """Build a compact settings table for traceability of exported stats."""
    selected = "ALL" if not graph_ids else ";".join([str(x) for x in graph_ids])
    rows = [
        {"key": "analysis_mode", "value": str(analysis_mode)},
        {"key": "min_conf", "value": float(min_conf)},
        {"key": "min_weight", "value": float(min_weight)},
        {"key": "eff_sources_k", "value": int(eff_sources_k)},
        {"key": "seed", "value": int(seed)},
        {"key": "compute_curvature", "value": bool(compute_curvature)},
        {"key": "curvature_sample_edges", "value": int(curvature_sample_edges)},
        {"key": "lightweight", "value": bool(lightweight)},
        {"key": "selected_graph_ids", "value": selected},
    ]
    return _ensure_columns(pd.DataFrame(rows), EXPORT_SETTINGS_COLUMNS)


def build_export_manifest_table(
    graphs: dict[str, GraphEntry],
    experiments: list[ExperimentEntry],
    *,
    min_conf: float,
    min_weight: float,
    graph_ids: list[str] | None,
) -> pd.DataFrame:
    """Build one-row-per-graph manifest for traceability and diagnostics."""
    selected = set(graph_ids or list(graphs.keys()))
    exp_by_graph: dict[str, list[ExperimentEntry]] = {}
    for exp in experiments:
        exp_by_graph.setdefault(exp.graph_id, []).append(exp)

    rows: list[dict] = []
    for gid, entry in graphs.items():
        exps = exp_by_graph.get(gid, [])
        mixfrac_n = sum(1 for exp in exps if exp.attack_kind == "mix_frac_estimate" or (exp.params or {}).get("attack_family") == "mixfrac")
        traj_n = sum(1 for exp in exps if exp.history is not None and not exp.history.empty)
        try:
            filtered_rows = int(len(entry.get_filtered_edges(float(min_conf), float(min_weight))))
        except Exception:
            filtered_rows = np.nan
        rows.append(
            {
                "graph_id": entry.id,
                "graph_name": entry.name,
                "subject_id": _subject_id_from_graph(entry),
                "group": _infer_group(entry.name, entry.source),
                "source": entry.source,
                "created_at": float(entry.created_at),
                "selected_for_export": entry.id in selected,
                "filtered_edge_rows": filtered_rows,
                "experiment_count": len(exps),
                "mixfrac_experiment_count": mixfrac_n,
                "trajectory_experiment_count": traj_n,
            }
        )
    return _clean_for_export(_ensure_columns(pd.DataFrame(rows), EXPORT_MANIFEST_COLUMNS))


def build_export_errors_table(
    df_subjects: pd.DataFrame,
    experiments: list[ExperimentEntry],
    graphs: dict[str, GraphEntry],
) -> pd.DataFrame:
    """Collect export-time diagnostics into a dedicated table."""
    rows: list[dict] = []

    if not df_subjects.empty and "export_error" in df_subjects.columns:
        bad = df_subjects[df_subjects["export_error"].fillna("").astype(str).str.len() > 0]
        for _, row in bad.iterrows():
            rows.append(
                {
                    "scope": "subject_metrics",
                    "entity_id": row.get("graph_id", ""),
                    "entity_name": row.get("graph_name", ""),
                    "error_type": "subject_export_error",
                    "message": row.get("export_error", ""),
                }
            )

    for exp in experiments:
        entry = graphs.get(exp.graph_id)
        if entry is None:
            rows.append(
                {
                    "scope": "experiments",
                    "entity_id": exp.id,
                    "entity_name": exp.name,
                    "error_type": "missing_graph_reference",
                    "message": f"graph_id={exp.graph_id} not found in graphs",
                }
            )
        if exp.history is None or exp.history.empty:
            rows.append(
                {
                    "scope": "experiments",
                    "entity_id": exp.id,
                    "entity_name": exp.name,
                    "error_type": "empty_history",
                    "message": "history is empty, so no trajectory rows were exported",
                }
            )
    return _clean_for_export(_ensure_columns(pd.DataFrame(rows), EXPORT_ERRORS_COLUMNS))


def build_export_overview_table(
    df_subjects: pd.DataFrame,
    df_mixfrac: pd.DataFrame,
    df_traj: pd.DataFrame,
    df_manifest: pd.DataFrame,
    df_errors: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize exported tables with row counts and human-readable status."""
    rows = []
    specs = [
        ("manifest", df_manifest, "traceability by graph"),
        ("subject_metrics", df_subjects, "subject-level graph metrics"),
        ("mixfrac_subjects", df_mixfrac, "mix_frac estimate results"),
        ("trajectories_long", df_traj, "long-format attack / trajectory metrics"),
        ("errors", df_errors, "diagnostics collected during export"),
    ]
    for name, df, note in specs:
        status = "ok" if len(df) > 0 else "empty"
        rows.append({"table_name": name, "rows": int(len(df)), "status": status, "note": note})
    return _ensure_columns(pd.DataFrame(rows), EXPORT_OVERVIEW_COLUMNS)


def _autosize_worksheet(ws) -> None:
    """Apply basic readability formatting for openpyxl worksheets."""
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions
    for column_cells in ws.columns:
        first = column_cells[0]
        letter = first.column_letter
        values = ["" if cell.value is None else str(cell.value) for cell in column_cells[:100]]
        max_len = max((len(v) for v in values), default=0)
        ws.column_dimensions[letter].width = min(max(max_len + 2, 10), 40)


def _build_export_bundle(
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
) -> dict[str, pd.DataFrame]:
    """Build all exported stats tables in one place to keep zip/xlsx consistent."""
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
    df_manifest = build_export_manifest_table(
        graphs,
        experiments,
        min_conf=min_conf,
        min_weight=min_weight,
        graph_ids=graph_ids,
    )
    df_errors = build_export_errors_table(df_subjects, experiments, graphs)
    df_overview = build_export_overview_table(df_subjects, df_mixfrac, df_traj, df_manifest, df_errors)
    df_settings = build_export_settings_table(
        min_conf=min_conf,
        min_weight=min_weight,
        analysis_mode=analysis_mode,
        eff_sources_k=eff_sources_k,
        seed=seed,
        compute_curvature=compute_curvature,
        curvature_sample_edges=curvature_sample_edges,
        graph_ids=graph_ids,
        lightweight=lightweight,
    )
    return {
        "overview": df_overview,
        "settings": df_settings,
        "manifest": df_manifest,
        "subject_metrics": df_subjects,
        "mixfrac_subjects": df_mixfrac,
        "trajectories_long": df_traj,
        "errors": df_errors,
    }


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
    """Export tidy statistical tables as ZIP with diagnostics and CSV files."""
    bundle = _build_export_bundle(
        graphs,
        experiments,
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

    buf = BytesIO()
    with ZipFile(buf, mode="w", compression=ZIP_DEFLATED) as zf:
        for name, df in bundle.items():
            zf.writestr(f"{name}.csv", _clean_for_export(df).to_csv(index=False))
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
    """Export tidy statistical tables as Excel workbook with diagnostics."""
    bundle = _build_export_bundle(
        graphs,
        experiments,
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

    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for name, df in bundle.items():
            clean_df = _clean_for_export(df)
            clean_df.to_excel(writer, sheet_name=name[:31], index=False)
        for ws in writer.book.worksheets:
            _autosize_worksheet(ws)
    buf.seek(0)
    return buf.getvalue()
