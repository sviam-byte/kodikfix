from __future__ import annotations

import gc
import hashlib
import json
import logging
import os
import traceback
from io import BytesIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st

# 1) Config & Logging
# TODO: move logger to separate module.
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("kodik")

try:
    _logdir = Path(__file__).resolve().parent / "logs"
    _logdir.mkdir(parents=True, exist_ok=True)
    _fh = logging.FileHandler(_logdir / "kodik.log", encoding="utf-8")
    _fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(_fh)
except Exception:
    pass

st.set_page_config(
    page_title="Kodik Lab",
    layout="wide",
    page_icon="🕸️",
    initial_sidebar_state="expanded",
)
st.title("Graph Lab")

from src.config import settings
from src.io_load import load_edges
from src.preprocess import coerce_fixed_format
from src.graph_build import build_graph
from src.core.graph_ops import calculate_metrics
from src.core.physics import simulate_energy_flow
from src.services.graph_service import GraphService
from src.stats_export import export_stats_xlsx_bytes, export_stats_zip_bytes
from src.exporters import payload_to_flat_row
from src.energy_export import (
    energy_run_summary_dict,
    frames_to_energy_edges_long,
    frames_to_energy_nodes_long,
    frames_to_energy_steps_summary,
)
from src.robustness import attack_trajectory_summary, graph_resistance_summary
from src.cli import _attack_payload_from_graph, _metrics_payload_from_graph
from src.state.session import ctx
from src.state_models import build_experiment_entry, build_graph_entry
from src.mat_packed import bundle_to_edge_frames
from src.ui_blocks import inject_custom_css

# session_io функциональность опциональна для UI.
# Если импорт упал (например, из-за отсутствующего xlsx-engine),
# приложение должно продолжить работу с остальными вкладками.
SESSION_IO_AVAILABLE = True
SESSION_IO_IMPORT_ERROR = None
try:
    from src.session_io import (
        export_experiments_json,
        export_experiments_xlsx,
        export_workspace_json,
        import_workspace_json,
    )
except Exception as e:  # pylint: disable=broad-except
    SESSION_IO_AVAILABLE = False
    SESSION_IO_IMPORT_ERROR = f"{type(e).__name__}: {e}"
    logger.exception("session_io import failed\n%s", traceback.format_exc())

    def _session_io_unavailable(*args, **kwargs):
        raise RuntimeError(
            "session_io недоступен. "
            f"Первичная ошибка импорта: {SESSION_IO_IMPORT_ERROR}"
        )

    export_experiments_json = _session_io_unavailable
    export_experiments_xlsx = _session_io_unavailable
    export_workspace_json = _session_io_unavailable
    import_workspace_json = _session_io_unavailable

from src.ui.tabs import attacks as tab_attacks
from src.ui.tabs import compare as tab_compare
from src.ui.tabs import dashboard as tab_dashboard
from src.ui.tabs import energy as tab_energy
from src.ui.tabs import structure as tab_structure
from src.batch_ops import build_ui_args, discover_batch_files, inspect_batch_file, make_run_dir, run_batch_plan, stage_batch_inputs, write_run_metadata

inject_custom_css()
ctx.ensure_initialized()


# --- Helpers ---

def new_id(prefix):
    import uuid

    return f"{prefix}_{uuid.uuid4().hex[:6]}"  # 6 символов — хватает


def _guess_cols(columns):
    cols = [str(c) for c in columns]
    low = [c.lower() for c in cols]

    def pick(cands):
        for cand in cands:
            if cand in low:
                return cols[low.index(cand)]
        return None

    src = pick(["src", "source", "from", "u", "a", "node_from"])
    dst = pick(["dst", "target", "to", "v", "b", "node_to"])
    w = pick(["weight", "w", "score", "value"])
    conf = pick(["confidence", "conf", "p", "prob", "support"])
    return src, dst, w, conf


def _norm_path_str(value: str) -> str:
    """Normalize path-like string from UI input."""
    return str(Path(str(value).strip()).expanduser()) if str(value).strip() else ""


def _default_batch_output_root() -> str:
    """Default root for UI-triggered batch run outputs."""
    return str((Path(__file__).resolve().parent / "batch_runs").resolve())


def _default_research_output_root() -> str:
    """Default root for persistent research-run outputs."""
    return str((Path(__file__).resolve().parent / "research_runs").resolve())


def _safe_run_label(value: str, fallback: str = "research") -> str:
    """Build a filesystem-safe short label for run files/directories."""
    raw = str(value or "").strip()
    if not raw:
        raw = str(fallback)
    cleaned = "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in raw)
    cleaned = cleaned.strip("._-")
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned[:96] or str(fallback)


EXCEL_MAX_ROWS = 1_048_576
EXCEL_MAX_COLS = 16_384


def _excel_sheet_fits(frame: pd.DataFrame) -> bool:
    """Return True when frame shape fits into a single XLSX worksheet."""
    frame = frame if isinstance(frame, pd.DataFrame) else pd.DataFrame(frame)
    return int(len(frame.index)) <= EXCEL_MAX_ROWS and int(len(frame.columns)) <= EXCEL_MAX_COLS


def _write_tables_xlsx_bytes(frames: dict[str, pd.DataFrame]) -> bytes:
    """Serialize dataframes to a workbook (sheet-per-frame).

    Oversized tables are replaced by a compact note sheet instead of crashing
    the whole research run. Full data should still be taken from CSV exports.
    """
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        wrote = False
        for sheet_name, df in (frames or {}).items():
            if df is None:
                continue
            frame = df if isinstance(df, pd.DataFrame) else pd.DataFrame(df)
            safe_sheet = _safe_run_label(sheet_name, "sheet")[:31]
            if _excel_sheet_fits(frame):
                frame.to_excel(writer, sheet_name=safe_sheet, index=False)
            else:
                logger.warning(
                    "Skipping oversized XLSX sheet %s with shape=%s; keep CSV as canonical export",
                    sheet_name,
                    tuple(frame.shape),
                )
                pd.DataFrame(
                    [
                        {
                            "sheet": str(sheet_name),
                            "status": "skipped_oversized_for_excel",
                            "rows": int(len(frame.index)),
                            "cols": int(len(frame.columns)),
                            "excel_max_rows": int(EXCEL_MAX_ROWS),
                            "excel_max_cols": int(EXCEL_MAX_COLS),
                            "note": "Table was too large for one Excel sheet. Use CSV export.",
                        }
                    ]
                ).to_excel(writer, sheet_name=safe_sheet, index=False)
            wrote = True
        if not wrote:
            pd.DataFrame([{"status": "empty"}]).to_excel(writer, sheet_name="summary", index=False)
    buf.seek(0)
    return buf.getvalue()


def _append_research_csv_rows(csv_path: Path, frame: pd.DataFrame | None) -> None:
    """Append one compact research table to CSV without rebuilding previous rows.

    Compatibility guard: if incoming columns differ from already persisted CSV
    headers, fall back to one safe read+rewrite for this table to avoid writing
    malformed rows with a shifted schema.
    """
    if frame is None or frame.empty:
        return
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists() or csv_path.stat().st_size == 0:
        frame.to_csv(csv_path, mode="a", header=True, index=False)
        return

    # Keep streaming append fast for the common stable-schema case.
    try:
        existing_cols = list(pd.read_csv(csv_path, nrows=0).columns)
    except Exception:
        logger.exception("Failed to inspect aggregate CSV header: %s", csv_path)
        existing_cols = []

    incoming_cols = list(frame.columns)
    if existing_cols and incoming_cols == existing_cols:
        frame.to_csv(csv_path, mode="a", header=False, index=False)
        return

    # Schema drift is rare, but when it happens we rewrite once to preserve data.
    try:
        prev = pd.read_csv(csv_path)
    except Exception:
        logger.exception("Failed to read aggregate CSV for rewrite: %s", csv_path)
        prev = pd.DataFrame(columns=existing_cols)

    merged = pd.concat([prev, frame], ignore_index=True, sort=False)
    merged.to_csv(csv_path, index=False)


def _load_research_aggregate_frames(run_dir: Path) -> dict[str, pd.DataFrame]:
    """Load aggregate compact research tables from canonical CSV files."""
    frames: dict[str, pd.DataFrame] = {}
    run_dir = Path(run_dir)
    for table_name in ["research_metrics", "research_resistance", "research_attacks", "research_energy_runs"]:
        csv_path = run_dir / f"{_safe_run_label(table_name, 'summary')}.csv"
        if not csv_path.exists():
            continue
        try:
            frames[table_name] = pd.read_csv(csv_path)
        except Exception:
            logger.exception("Failed to read research aggregate CSV: %s", csv_path)
    return frames


def _write_research_bundle_from_run_dir(run_dir: Path, frames: dict[str, pd.DataFrame]) -> None:
    """Build final ZIP directly from persisted run-dir artifacts."""
    run_dir = Path(run_dir)
    bundle_path = run_dir / "research_bundle.zip"
    with ZipFile(bundle_path, mode="w", compression=ZIP_DEFLATED) as zf:
        for name, df in (frames or {}).items():
            zf.writestr(f"{name}.csv", (df.copy() if df is not None else pd.DataFrame()).to_csv(index=False).encode("utf-8"))
        for folder_name in ["per_graph", "extras"]:
            base = run_dir / folder_name
            if not base.exists():
                continue
            for file_path in sorted(p for p in base.rglob("*") if p.is_file()):
                zf.write(file_path, arcname=str(file_path.relative_to(run_dir)).replace('\\', '/'))
        for manifest_name in ["manifest.csv", "manifest.xlsx", "run_meta.json", "research_summary.xlsx"]:
            file_path = run_dir / manifest_name
            if file_path.exists():
                zf.write(file_path, arcname=manifest_name)


def _finalize_research_run_outputs(run_dir: Path, *, manifest_rows: list[dict], aggregate_frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Write final manifest/summary artifacts once, at the end of the run."""
    run_dir = Path(run_dir).expanduser().resolve()
    manifest_df = pd.DataFrame(manifest_rows or [])
    manifest_df.to_csv(run_dir / "manifest.csv", index=False)
    with pd.ExcelWriter(run_dir / "manifest.xlsx", engine="openpyxl") as writer:
        manifest_df.to_excel(writer, sheet_name="manifest", index=False)
    for name, df in (aggregate_frames or {}).items():
        (df if df is not None else pd.DataFrame()).to_csv(run_dir / f"{_safe_run_label(name, 'summary')}.csv", index=False)
    summary_frames = aggregate_frames or _load_research_aggregate_frames(run_dir)
    (run_dir / "research_summary.xlsx").write_bytes(_write_tables_xlsx_bytes(summary_frames))
    _write_research_bundle_from_run_dir(run_dir, summary_frames)
    return summary_frames


def _persist_research_graph_result(
    run_dir: Path,
    *,
    graph_entry,
    per_graph_frames: dict[str, pd.DataFrame],
    per_graph_extras: dict[str, bytes],
    manifest_rows: list[dict],
    workbook_frames: dict[str, pd.DataFrame] | None = None,
    append_aggregate_rows: bool = True,
) -> None:
    """Persist only current graph outputs immediately; finalize summaries later."""
    run_dir = Path(run_dir).expanduser().resolve()
    per_graph_dir = run_dir / "per_graph"
    extras_dir = run_dir / "extras"
    per_graph_dir.mkdir(parents=True, exist_ok=True)
    extras_dir.mkdir(parents=True, exist_ok=True)
    stem = _safe_run_label(
        getattr(graph_entry, "name", None) or getattr(graph_entry, "id", "graph"),
        getattr(graph_entry, "id", "graph"),
    )

    workbook_payload: dict[str, pd.DataFrame] = {}
    workbook_payload.update(per_graph_frames or {})
    workbook_payload.update(workbook_frames or {})
    workbook_path = per_graph_dir / f"{stem}__full.xlsx"
    workbook_path.write_bytes(_write_tables_xlsx_bytes(workbook_payload))

    for rel_name, payload in (per_graph_extras or {}).items():
        rel_path = Path(rel_name)
        out_path = extras_dir / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(payload)

    manifest_df = pd.DataFrame(manifest_rows or [])
    manifest_df.to_csv(run_dir / "manifest.csv", index=False)

    if append_aggregate_rows:
        for table_name, frame in (per_graph_frames or {}).items():
            if not str(table_name).startswith("research_"):
                continue
            _append_research_csv_rows(run_dir / f"{_safe_run_label(table_name, 'summary')}.csv", frame)


def _research_graph_stem(graph_entry) -> str:
    """Build deterministic per-graph filename stem for research artifacts."""
    return _safe_run_label(
        getattr(graph_entry, "name", None) or getattr(graph_entry, "id", "graph"),
        getattr(graph_entry, "id", "graph"),
    )


def _research_manifest_path(run_dir: Path) -> Path:
    """Return canonical manifest.csv path for a research run directory."""
    return Path(run_dir) / "manifest.csv"


def _research_lock_path(run_dir: Path) -> Path:
    """Return lock-file path used to guard currently running research jobs."""
    return Path(run_dir) / "RUNNING.lock"


def _research_workbook_path(run_dir: Path, graph_entry) -> Path:
    """Return path to per-graph workbook generated in stream-save mode."""
    return Path(run_dir) / "per_graph" / f"{_research_graph_stem(graph_entry)}__full.xlsx"


def _research_table_csv_path(run_dir: Path, graph_entry, table_name: str) -> Path:
    """Return path to a per-graph CSV table inside stream-save run dir."""
    stem = _research_graph_stem(graph_entry)
    table_stem = _safe_run_label(table_name, "table")
    return Path(run_dir) / "per_graph" / f"{stem}__{table_stem}.csv"


def _research_workbook_sheet_name(table_name: str) -> str:
    """Return sanitized workbook sheet name for one research table."""
    return _safe_run_label(table_name, "sheet")[:31]


def _research_table_from_workbook(run_dir: Path, graph_entry, table_name: str) -> pd.DataFrame | None:
    """Read one per-graph research table from the unified workbook.

    New format: one workbook per graph with multiple sheets.
    Backward compatibility: if workbook/sheet is missing, fall back to legacy CSV.
    """
    workbook_path = _research_workbook_path(run_dir, graph_entry)
    sheet_name = _research_workbook_sheet_name(table_name)
    if workbook_path.exists():
        try:
            xls = pd.ExcelFile(workbook_path, engine="openpyxl")
            if sheet_name in xls.sheet_names:
                df = pd.read_excel(xls, sheet_name=sheet_name)
                return df if df is not None else pd.DataFrame()
        except Exception:
            logger.exception("Failed to read workbook sheet %s from %s", sheet_name, workbook_path)

    legacy_csv = _research_table_csv_path(run_dir, graph_entry, table_name)
    if legacy_csv.exists():
        try:
            return pd.read_csv(legacy_csv)
        except Exception:
            logger.exception("Failed to read legacy research table: %s", legacy_csv)
    return None


def _load_research_manifest_rows(run_dir: Path) -> list[dict]:
    """Load existing manifest rows; tolerate missing/invalid files safely."""
    path = _research_manifest_path(run_dir)
    if not path.exists():
        return []
    try:
        df = pd.read_csv(path)
    except Exception:
        logger.exception("Failed to read existing research manifest: %s", path)
        return []
    if df.empty:
        return []
    rows = df.replace({np.nan: None}).to_dict(orient="records")
    return [dict(row) for row in rows]


def _upsert_manifest_row(manifest_rows: list[dict], row: dict) -> list[dict]:
    """Insert or replace manifest row by graph_id while preserving order."""
    key = str(row.get("graph_id", "")).strip()
    if not key:
        manifest_rows.append(dict(row))
        return manifest_rows
    for idx, existing in enumerate(manifest_rows):
        if str(existing.get("graph_id", "")).strip() == key:
            manifest_rows[idx] = dict(row)
            return manifest_rows
    manifest_rows.append(dict(row))
    return manifest_rows


def _hydrate_existing_research_rows(run_dir: Path, graph_entry, results: dict[str, list[dict]]) -> bool:
    """Read previously persisted per-graph tables and merge into in-memory results.

    Preferred source is the per-graph workbook produced by stream-save mode.
    Legacy per-table CSV files are still supported for older runs.
    """
    found = False
    for table_name in ["research_metrics", "research_resistance", "research_attacks", "research_energy_runs"]:
        df = _research_table_from_workbook(run_dir, graph_entry, table_name)
        if df is None:
            continue
        if df.empty:
            continue
        results.setdefault(table_name, []).extend(df.replace({np.nan: None}).to_dict(orient="records"))
        found = True
    return found


def _is_research_graph_already_done(run_dir: Path, graph_entry) -> bool:
    """Check if graph has completed status and unified workbook in a previous run."""
    workbook_path = _research_workbook_path(run_dir, graph_entry)
    if not workbook_path.exists():
        return False
    manifest_rows = _load_research_manifest_rows(run_dir)
    gid = str(getattr(graph_entry, "id", "")).strip()
    for row in manifest_rows:
        if str(row.get("graph_id", "")).strip() == gid and str(row.get("status", "")).strip().lower() == "done":
            return True
    return False


def _find_existing_research_run(stream_root: Path, *, run_label: str, seed: int) -> Path | None:
    """Find latest compatible run dir with same label+seed and no active lock."""
    root = Path(stream_root).expanduser().resolve()
    if not root.exists():
        return None
    label = _safe_run_label(run_label, "research_run")
    prefix = f"{label}__seed_{int(seed)}__"
    candidates = [p for p in root.iterdir() if p.is_dir() and p.name.startswith(prefix)]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for candidate in candidates:
        lock_path = _research_lock_path(candidate)
        if lock_path.exists():
            logger.warning("Research resume skipped locked run: %s", candidate)
            continue
        return candidate
    return None


def add_graph_to_state(name, df, source, src, dst):
    gid = new_id("G")
    entry = build_graph_entry(
        name=name,
        source=source,
        edges=df,
        src_col=src,
        dst_col=dst,
        entry_id=gid,
    )
    ctx.add_graph_entry(entry, make_active=True)
    return gid


def save_experiment_to_state(name, gid, kind, params, df_hist):
    eid = new_id("EXP")
    exp = build_experiment_entry(
        name=name,
        graph_id=gid,
        attack_kind=kind,
        params=params,
        history=df_hist,
        entry_id=eid,
    )
    # Store directly in session state to keep the workflow lightweight.
    if hasattr(ctx, "add_experiment"):
        ctx.add_experiment(exp)
    else:
        ctx.experiments.append(exp)
    return eid


def _export_progress_ui(prefix: str = "stats"):
    """Create a compact progress UI bundle for long-running export jobs."""
    bar = st.progress(0.0)
    msg = st.empty()

    def _cb(done: int, total: int, label: str) -> None:
        total_ = max(1, int(total))
        frac = min(1.0, max(0.0, float(done) / float(total_)))
        bar.progress(frac)
        if label == "done":
            msg.caption("Экспорт завершён")
        else:
            msg.caption(f"{prefix}: {done}/{total_} · {label}")

    return bar, msg, _cb


def _stats_export_selection(export_scope: str, active_gid: str | None) -> list[str] | None:
    """Resolve export scope to graph id list; None means export all graphs."""
    if export_scope == "Active graph only" and active_gid:
        return [active_gid]
    return None


@st.cache_data(show_spinner=False)
def cached_load_packed_mat_graphs(file_bytes: bytes, filename: str) -> tuple[list[tuple[str, pd.DataFrame]], int]:
    """Load packed MAT connectomes into per-subject edge tables."""
    _ = filename
    return bundle_to_edge_frames(file_bytes, keep_zero_weight=False)


def _clear_pending_upload_state():
    """Очистить временное состояние загрузок/стейджинга для сайдбара."""
    for key in [
        "__pending_upload_df",
        "__pending_upload_name",
        "__pending_upload_error",
        "__mat_stage_name",
        "__mat_stage_graphs",
        "__mat_stage_subjects",
        "__mat_stage_n_nodes",
        "__mat_stage_source_filename",
    ]:
        st.session_state.pop(key, None)


def _import_staged_mat_graphs(selected_idx: list[int] | None = None):
    """Импортировать выбранные MAT-графы из staging в рабочий state."""
    graphs = st.session_state.get("__mat_stage_graphs")
    subjects = st.session_state.get("__mat_stage_subjects")
    source_filename = st.session_state.get("__mat_stage_source_filename", "mat-upload")
    if not graphs or not subjects:
        raise RuntimeError("Нет staged MAT-графов для импорта")

    if selected_idx is None:
        selected_idx = list(range(len(graphs)))

    added_ids = []
    base = Path(source_filename).stem
    for i in selected_idx:
        subj_name = subjects[i]
        df_edges = graphs[i]
        gid = add_graph_to_state(
            f"{base} :: {subj_name}",
            df_edges,
            "mat-upload",
            "src",
            "dst",
        )
        added_ids.append(gid)

    if added_ids:
        # Делаем первый импортированный граф активным через API менеджера,
        # чтобы синхронизировать session_state и UI epoch.
        ctx.set_active_graph(added_ids[0])
    return added_ids


@st.cache_data(show_spinner=False)
def cached_load_edges(file_bytes: bytes, filename: str, fixed: bool) -> tuple[pd.DataFrame, dict | None]:
    """Загрузить таблицу рёбер с кэшем (ускоряет переключение вкладок)."""
    df_any = load_edges(file_bytes, filename)
    if fixed:
        df_edges, meta = coerce_fixed_format(df_any)
        return df_edges, meta
    return df_any, None


def _hash_nx_graph_for_metrics(G: nx.Graph) -> str:
    """Return a stable cache key for metric calculation based on graph topology/weights."""
    if G is None:
        return "none"
    try:
        return nx.weisfeiler_lehman_graph_hash(G, edge_attr="weight")
    except Exception:  # pylint: disable=broad-except
        # Fallback keeps cache functional even if hashing fails for rare graph edge cases.
        return f"{G.number_of_nodes()}-{G.number_of_edges()}"


@st.cache_data(show_spinner=False, hash_funcs={nx.Graph: _hash_nx_graph_for_metrics})
def cached_calculate_metrics(
    G: nx.Graph,
    seed: int,
    curvature_sample_edges: int,
) -> dict:
    """Calculate base metrics for an already built graph and cache the result.

    Curvature is intentionally excluded here and computed separately in UI on demand.
    """
    if G is None:
        return {}
    if G.number_of_nodes() == 0:
        return {
            "N": 0,
            "E": 0,
            "C": 0,
            "density": 0.0,
            "avg_degree": 0.0,
        }

    # For very large graphs we trade some precision for responsiveness on rerenders.
    large_graph = G.number_of_nodes() > 300
    huge_graph = G.number_of_nodes() > 1200 or G.number_of_edges() > 8000
    return calculate_metrics(
        G,
        eff_sources_k=settings.APPROX_EFFICIENCY_K,
        seed=int(seed),
        compute_curvature=False,
        curvature_sample_edges=int(curvature_sample_edges),
        compute_heavy=not large_graph,
        skip_spectral=bool(huge_graph),
        diameter_samples=6 if large_graph else 16,
    )


def _normalize_ricci_payload(payload: dict | None) -> dict:
    """Convert rich Ricci payload into flat dashboard metric keys."""
    if not payload:
        return {}
    summary = dict(payload.get("summary", {}) or {})
    return {
        "kappa_mean": summary.get("kappa_mean"),
        "kappa_median": summary.get("kappa_median"),
        "kappa_frac_negative": summary.get("kappa_frac_negative"),
        "kappa_computed_edges": summary.get("computed_edges"),
        "kappa_skipped_edges": summary.get("skipped_edges"),
        "fragility_kappa": payload.get("fragility"),
    }




def _research_metric_catalog() -> list[tuple[str, str, str]]:
    """Available research-level calculations exposed in the dedicated tab."""
    return [
        ("basic", "Base graph metrics", "N, E, density, degree, components, beta, LCC"),
        ("efficiency", "Efficiency", "weighted global efficiency"),
        ("spectral", "Spectral", "lambda2, lmax, thresholds, modularity, tau"),
        ("clustering", "Clustering", "average clustering"),
        ("assortativity", "Assortativity", "degree assortativity"),
        ("entropy", "Entropy rates", "H_rw and H_evo"),
        ("curvature", "Ricci curvature", "kappa summary"),
        ("resistance", "Resistance summary", "connectivity/core/robustness summary"),
        ("energy", "Energy diffusion", "per-graph energy run + XLSX"),
        ("attack", "Attack experiment", "trajectory summary + history"),
    ]


def _research_selected_flags() -> dict[str, bool]:
    """Read research metric checkboxes from Streamlit session state."""
    flags = {}
    for key, _label, _help in _research_metric_catalog():
        flags[key] = bool(st.session_state.get(f"__research_flag_{key}", True))
    return flags


def _bundle_frames_to_xlsx_bytes(frames: dict[str, pd.DataFrame]) -> bytes:
    """Serialize research summary tables into a single XLSX payload."""
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        wrote = False
        for name, df in (frames or {}).items():
            frame = df.copy() if df is not None else pd.DataFrame()
            safe_sheet = _safe_run_label(name, "sheet")[:31]
            if _excel_sheet_fits(frame):
                frame.to_excel(writer, sheet_name=safe_sheet, index=False)
            else:
                logger.warning(
                    "Skipping oversized summary XLSX sheet %s with shape=%s; keep CSV as canonical export",
                    name,
                    tuple(frame.shape),
                )
                pd.DataFrame(
                    [
                        {
                            "sheet": str(name),
                            "status": "skipped_oversized_for_excel",
                            "rows": int(len(frame.index)),
                            "cols": int(len(frame.columns)),
                            "excel_max_rows": int(EXCEL_MAX_ROWS),
                            "excel_max_cols": int(EXCEL_MAX_COLS),
                            "note": "Table was too large for one Excel sheet. Use CSV export.",
                        }
                    ]
                ).to_excel(writer, sheet_name=safe_sheet, index=False)
            wrote = True
        if not wrote:
            pd.DataFrame([{"status": "empty"}]).to_excel(writer, sheet_name="summary", index=False)
    buf.seek(0)
    return buf.getvalue()


def _bundle_frames_to_zip_bytes(frames: dict[str, pd.DataFrame], extras: dict[str, bytes] | None = None) -> bytes:
    """Serialize research summary tables and optional artifacts into a ZIP payload."""
    buf = BytesIO()
    with ZipFile(buf, mode="w", compression=ZIP_DEFLATED) as zf:
        for name, df in frames.items():
            zf.writestr(f"{name}.csv", (df.copy() if df is not None else pd.DataFrame()).to_csv(index=False).encode("utf-8"))
        for name, payload in (extras or {}).items():
            zf.writestr(name, payload)
    buf.seek(0)
    return buf.getvalue()


def _run_research_workspace_plan(
    graph_ids: list[str],
    *,
    min_conf: float,
    min_weight: float,
    analysis_mode: str,
    seed_val: int,
    eff_k: int,
    curv_n: int,
    flags: dict[str, bool],
    attack_family: str,
    attack_kind: str,
    attack_frac: float,
    attack_steps: int,
    attack_heavy_every: int,
    attack_fast_mode: bool,
    energy_steps: int,
    energy_flow_mode: str,
    energy_damping: float,
    energy_phys_injection: float,
    energy_phys_leak: float,
    energy_phys_cap_mode: str,
    energy_rw_impulse: bool,
    stream_save: bool = False,
    stream_output_root: str = "",
    run_label: str = "",
    resume_existing_run: bool = True,
) -> tuple[dict[str, pd.DataFrame], dict[str, bytes], Path | None]:
    """Execute the full research batch plan over selected workspace graphs.

    Design notes:
    - Processing is intentionally sequential to keep Streamlit progress updates
      deterministic and easy to debug.
    - Partial failures should not abort the full run; every selected graph gets
      a chance to produce outputs.
    - Export artifacts are accumulated in-memory for a single download step.
    - Optional stream_save mode persists per-graph outputs immediately to disk.
    """
    results: dict[str, list[dict]] = {
        "research_metrics": [],
        "research_resistance": [],
        "research_attacks": [],
        "research_energy_runs": [],
    }
    keep_compact_results_in_memory = not bool(stream_save)
    keep_extras_in_memory = not bool(stream_save)
    extras: dict[str, bytes] = {}
    run_dir: Path | None = None
    manifest_rows: list[dict] = []
    safe_label = _safe_run_label(str(run_label).strip() or "research_run")
    if bool(stream_save):
        stream_root = Path(str(stream_output_root).strip() or _default_research_output_root()).expanduser().resolve()
        stream_root.mkdir(parents=True, exist_ok=True)
        if bool(resume_existing_run):
            run_dir = _find_existing_research_run(stream_root, run_label=safe_label, seed=int(seed_val))
            if run_dir is not None:
                manifest_rows = _load_research_manifest_rows(run_dir)
        if run_dir is None:
            run_dir = make_run_dir(
                stream_root,
                mode="research_stream",
                seed=int(seed_val),
                run_label=safe_label,
            )
            manifest_rows = []
        _research_lock_path(run_dir).write_text(str(os.getpid()), encoding="utf-8")
        write_run_metadata(
            run_dir,
            base_dir=stream_root,
            run_label=str(run_label).strip() or "research_run",
            seed=int(seed_val),
            mode="research_stream",
            status="running",
            extra={
                "analysis_mode": str(analysis_mode),
                "requested_graphs": len(list(graph_ids or [])),
                "min_conf": float(min_conf),
                "min_weight": float(min_weight),
                "resume_existing_run": bool(resume_existing_run),
            },
        )

    # Validate incoming IDs against current workspace state to guard against
    # stale selections after graph deletion, trim_memory(), rerun/switch/import.
    existing_graph_ids = list(ctx.graphs.keys()) if isinstance(ctx.graphs, dict) else []
    requested_graph_ids = [str(gid) for gid in graph_ids]
    missing_graph_ids = [gid for gid in requested_graph_ids if gid not in ctx.graphs]
    valid_graph_ids = [gid for gid in requested_graph_ids if gid in ctx.graphs]

    if missing_graph_ids:
        logger.warning("Research plan skipped missing graphs: %s", missing_graph_ids)
        results["research_metrics"].extend(
            {
                "graph_id": gid,
                "graph_name": "<missing>",
                "source": "<missing>",
                "analysis_mode": str(analysis_mode),
                "min_conf": float(min_conf),
                "min_weight": float(min_weight),
                "status": "missing_graph",
                "error_type": "KeyError",
                "error": f"Graph id not found in workspace: {gid}",
                "available_graph_ids": "|".join(existing_graph_ids),
            }
            for gid in missing_graph_ids
        )

    total = max(1, len(valid_graph_ids))
    bar = st.progress(0.0)
    msg = st.empty()

    if not valid_graph_ids:
        msg.caption("Нет доступных графов для расчёта")
        bar.progress(1.0)
        frames = {name: pd.DataFrame(rows) for name, rows in results.items()}
        if run_dir is not None:
            try:
                _research_lock_path(run_dir).unlink(missing_ok=True)
            except Exception:
                logger.exception("Failed to remove research run lock: %s", _research_lock_path(run_dir))
        return frames, extras, run_dir

    if run_dir is not None and manifest_rows and keep_compact_results_in_memory:
        for gid_existing in valid_graph_ids:
            entry_existing = ctx.graphs.get(gid_existing)
            if entry_existing is None:
                continue
            _hydrate_existing_research_rows(run_dir, entry_existing, results)

    for idx, gid in enumerate(valid_graph_ids, start=1):
        entry = ctx.graphs.get(gid)
        if entry is None:
            logger.warning("Research plan lost graph during run: %s", gid)
            if keep_compact_results_in_memory:
                results["research_metrics"].append({
                    "graph_id": gid,
                    "graph_name": "<missing>",
                    "source": "<missing>",
                    "analysis_mode": str(analysis_mode),
                    "min_conf": float(min_conf),
                    "min_weight": float(min_weight),
                    "status": "missing_graph",
                    "error_type": "KeyError",
                    "error": f"Graph disappeared during run: {gid}",
                })
            bar.progress(idx / total)
            continue
        msg.caption(f"[{idx}/{total}] {entry.name}")
        graph_manifest = {
            "idx": int(idx),
            "graph_id": getattr(entry, "id", gid),
            "graph_name": getattr(entry, "name", gid),
            "source": getattr(entry, "source", ""),
            "status": "running",
        }
        if run_dir is not None and _is_research_graph_already_done(run_dir, entry):
            if keep_compact_results_in_memory:
                _hydrate_existing_research_rows(run_dir, entry, results)
            graph_manifest["status"] = "skipped_existing"
            graph_manifest["tables"] = "existing"
            graph_manifest["extras_count"] = 0
            _upsert_manifest_row(manifest_rows, graph_manifest)
            write_run_metadata(
                run_dir,
                base_dir=run_dir.parent,
                run_label=str(run_label).strip() or "research_run",
                seed=int(seed_val),
                mode="research_stream",
                status="running",
                extra={
                    "processed_graphs": int(idx),
                    "total_graphs": int(total),
                    "last_graph": getattr(entry, "name", gid),
                    "last_action": "skip_existing",
                },
            )
            msg.caption(f"[{idx}/{total}] skip existing: {entry.name}")
            bar.progress(float(idx) / float(total))
            continue
        # Keep graph-local frames separate from global aggregations so that
        # stream-save writes one deterministic workbook per graph.
        per_graph_frames: dict[str, pd.DataFrame] = {}
        workbook_frames_local: dict[str, pd.DataFrame] = {}
        per_graph_extras: dict[str, bytes] = {}
        try:
            graph = cached_build_graph(
                entry.edges,
                entry.src_col,
                entry.dst_col,
                min_conf,
                min_weight,
                analysis_mode,
            )

            if any(flags.get(k, False) for k in ["basic", "efficiency", "spectral", "clustering", "assortativity", "entropy", "curvature"]):
                args_metrics = build_ui_args(
                    seed=int(seed_val),
                    eff_k=int(eff_k),
                    compute_curvature=bool(flags.get("curvature", False)),
                    curvature_sample_edges=int(curv_n),
                    compute_heavy=bool(flags.get("efficiency", False) or flags.get("entropy", False) or flags.get("clustering", False) or flags.get("assortativity", False) or flags.get("spectral", False)),
                    skip_spectral=not bool(flags.get("spectral", False)),
                    diameter_samples=16,
                    n_jobs=1,
                )
                payload = _metrics_payload_from_graph(args_metrics, graph, input_label=entry.name)
                row = payload_to_flat_row(payload)
                row.update({
                    "graph_id": entry.id,
                    "graph_name": entry.name,
                    "source": entry.source,
                    "analysis_mode": str(analysis_mode),
                    "min_conf": float(min_conf),
                    "min_weight": float(min_weight),
                })
                if not flags.get("clustering", False) and "clustering" in row:
                    row["clustering"] = np.nan
                if not flags.get("assortativity", False) and "assortativity" in row:
                    row["assortativity"] = np.nan
                if not flags.get("entropy", False):
                    for col in ["H_rw", "H_evo", "fragility_H", "fragility_evo"]:
                        if col in row:
                            row[col] = np.nan
                if not flags.get("curvature", False):
                    for col in ["kappa_mean", "kappa_median", "kappa_frac_negative", "kappa_computed_edges", "kappa_skipped_edges", "kappa_var", "kappa_skew", "kappa_entropy", "fragility_kappa"]:
                        if col in row:
                            row[col] = np.nan
                if keep_compact_results_in_memory:
                    results["research_metrics"].append(row)
                per_graph_frames["research_metrics"] = pd.DataFrame([row])

            if flags.get("resistance", False):
                row = graph_resistance_summary(graph)
                row.update({"graph_id": entry.id, "graph_name": entry.name, "source": entry.source})
                if keep_compact_results_in_memory:
                    results["research_resistance"].append(row)
                per_graph_frames["research_resistance"] = pd.DataFrame([row])

            if flags.get("attack", False):
                args_attack = build_ui_args(
                    family=str(attack_family),
                    kind=str(attack_kind),
                    frac=float(attack_frac),
                    steps=int(attack_steps),
                    seed=int(seed_val),
                    eff_k=int(eff_k),
                    heavy_every=int(attack_heavy_every),
                    fast_mode=bool(attack_fast_mode),
                    compute_curvature=bool(flags.get("curvature", False)),
                    curvature_sample_edges=int(curv_n),
                    n_jobs=1,
                )
                attack_payload, history = _attack_payload_from_graph(args_attack, graph, input_label=entry.name)
                attack_row = attack_trajectory_summary(history, attack_kind=str(attack_kind))
                attack_row.update({
                    "graph_id": entry.id,
                    "graph_name": entry.name,
                    "source": entry.source,
                    "family": attack_family,
                    "kind": attack_kind,
                    "frac": float(attack_frac),
                    "steps_requested": int(attack_steps),
                })
                if isinstance(attack_payload.get("final_row"), dict):
                    for k, v in attack_payload["final_row"].items():
                        attack_row[f"final__{k}"] = v
                if keep_compact_results_in_memory:
                    results["research_attacks"].append(attack_row)
                per_graph_frames["research_attacks"] = pd.DataFrame([attack_row])
                per_graph_extras[f"attack_histories/{entry.id}__{attack_family}__{attack_kind}.csv"] = history.to_csv(index=False).encode("utf-8")
                per_graph_extras[f"attack_payloads/{entry.id}__{attack_family}__{attack_kind}.json"] = json.dumps(attack_payload, ensure_ascii=False, indent=2, default=str).encode("utf-8")

            if flags.get("energy", False):
                node_frames, edge_frames = simulate_energy_flow(
                    graph,
                    steps=int(energy_steps),
                    flow_mode=str(energy_flow_mode),
                    damping=float(energy_damping),
                    phys_injection=float(energy_phys_injection),
                    phys_leak=float(energy_phys_leak),
                    phys_cap_mode=str(energy_phys_cap_mode),
                    rw_impulse=bool(energy_rw_impulse),
                )
                # Keep tab-level energy outputs aligned with batch/service exporters.
                energy_nodes_long = frames_to_energy_nodes_long(graph, node_frames, sources=None)
                energy_steps_summary = frames_to_energy_steps_summary(graph, node_frames, edge_frames, sources=None)
                energy_run_summary = energy_run_summary_dict(
                    graph,
                    node_frames,
                    edge_frames,
                    sources=None,
                    flow_mode=str(energy_flow_mode),
                )
                energy_edges_long = frames_to_energy_edges_long(graph, edge_frames, sources=None)
                energy_run_summary.update({
                    "graph_id": entry.id,
                    "graph_name": entry.name,
                    "source": entry.source,
                    "steps_requested": int(energy_steps),
                    "n_node_rows": int(len(energy_nodes_long)),
                    "n_edge_rows": int(len(energy_edges_long)),
                })
                if keep_compact_results_in_memory:
                    results["research_energy_runs"].append(energy_run_summary)
                per_graph_frames["research_energy_runs"] = pd.DataFrame([energy_run_summary])
                per_graph_extras[f"energy/{entry.id}__nodes_long.csv"] = energy_nodes_long.to_csv(index=False).encode("utf-8")
                per_graph_extras[f"energy/{entry.id}__edges_long.csv"] = energy_edges_long.to_csv(index=False).encode("utf-8")
                per_graph_extras[f"energy/{entry.id}__steps_summary.csv"] = energy_steps_summary.to_csv(index=False).encode("utf-8")
                per_graph_extras[f"energy/{entry.id}__run_summary.json"] = json.dumps(energy_run_summary, ensure_ascii=False, indent=2, default=str).encode("utf-8")
                workbook_frames_local["energy_nodes_long"] = energy_nodes_long
                workbook_frames_local["energy_edges_long"] = energy_edges_long
                workbook_frames_local["energy_steps_summary"] = energy_steps_summary
                workbook_frames_local["energy_run_summary"] = pd.DataFrame([energy_run_summary])
                # Free heavy intermediate objects to reduce peak memory in long runs.
                del node_frames, edge_frames, energy_nodes_long, energy_edges_long, energy_steps_summary
                gc.collect()

            graph_manifest["status"] = "done"
            graph_manifest["tables"] = ",".join(sorted(per_graph_frames.keys()))
            graph_manifest["extras_count"] = int(len(per_graph_extras))
            if keep_extras_in_memory:
                extras.update(per_graph_extras)
        except Exception:
            logger.exception("Research run failed for graph %s (%s)", gid, entry.name)
            err_text = traceback.format_exc()
            graph_manifest["status"] = "failed"
            graph_manifest["error"] = err_text[-4000:]
            msg.warning(f"⚠️ Ошибка при обработке {entry.name}: {err_text[-300:]}")
            per_graph_frames = {
                "research_errors": pd.DataFrame([
                    {
                        "graph_id": getattr(entry, "id", gid),
                        "graph_name": getattr(entry, "name", gid),
                        "source": getattr(entry, "source", ""),
                        "status": "failed",
                        "error": err_text,
                    }
                ])
            }
            workbook_frames_local = dict(per_graph_frames)
            per_graph_extras = {}
            graph_manifest["extras_count"] = 0

        _upsert_manifest_row(manifest_rows, graph_manifest)
        if run_dir is not None:
            _persist_research_graph_result(
                run_dir,
                graph_entry=entry,
                per_graph_frames=per_graph_frames,
                per_graph_extras=per_graph_extras,
                manifest_rows=manifest_rows,
                workbook_frames=workbook_frames_local,
                append_aggregate_rows=True,
            )
            write_run_metadata(
                run_dir,
                base_dir=run_dir.parent,
                run_label=str(run_label).strip() or "research_run",
                seed=int(seed_val),
                mode="research_stream",
                status="running",
                extra={
                    "processed_graphs": int(idx),
                    "total_graphs": int(total),
                    "last_graph": getattr(entry, "name", gid),
                },
            )

        del per_graph_frames, workbook_frames_local, per_graph_extras
        try:
            del graph
        except Exception:
            pass
        gc.collect()
        bar.progress(float(idx) / float(total))

    msg.caption("Research run complete")
    if run_dir is not None:
        frames = _load_research_aggregate_frames(run_dir)
        frames = _finalize_research_run_outputs(run_dir, manifest_rows=manifest_rows, aggregate_frames=frames)
        write_run_metadata(
            run_dir,
            base_dir=run_dir.parent,
            run_label=str(run_label).strip() or "research_run",
            seed=int(seed_val),
            mode="research_stream",
            status="finished",
            extra={
                "processed_graphs": int(len(valid_graph_ids)),
                "summary_tables": ",".join(sorted(frames.keys())),
            },
        )
        try:
            _research_lock_path(run_dir).unlink(missing_ok=True)
        except Exception:
            logger.exception("Failed to remove research run lock: %s", _research_lock_path(run_dir))
    else:
        frames = {name: pd.DataFrame(rows) for name, rows in results.items() if rows}
    return frames, extras, run_dir


def _render_research_tab(
    *,
    cur_gid: str,
    min_conf: float,
    min_weight: float,
    analysis_mode: str,
    seed_val: int,
    curv_n: int,
) -> None:
    """Render UI for the standalone research workspace batch calculations.

    State hygiene: this tab stores controls under the ``__research_*`` prefix to
    avoid collisions with controls from other tabs.
    """
    st.subheader("🧠 Research calc")
    st.caption("Отдельная вкладка для полного исследовательского расчёта по активному графу или по всему workspace.")

    stream_defaults = st.columns([1.2, 1.8])
    with stream_defaults[0]:
        research_stream_save = st.checkbox(
            "Потоково сохранять на диск",
            value=bool(st.session_state.get("__research_stream_save", True)),
            key="__research_stream_save",
            help="После каждого графа сразу пишет manifest, per-graph CSV/XLSX и итоговые summary-файлы в папку запуска.",
        )
        research_resume_existing = st.checkbox(
            "Проверять и продолжать предыдущий run",
            value=bool(st.session_state.get("__research_resume_existing", True)),
            key="__research_resume_existing",
            help="Ищет последний run с той же меткой и seed, подхватывает manifest и пропускает уже готовые графы.",
        )
        research_run_label = st.text_input(
            "Метка запуска",
            value=str(st.session_state.get("__research_run_label", "workspace_research")),
            key="__research_run_label",
        )
    with stream_defaults[1]:
        research_output_root = st.text_input(
            "Папка результатов research",
            value=str(st.session_state.get("__research_output_root", _default_research_output_root())),
            key="__research_output_root",
        )
        research_output_root_resolved = Path(str(research_output_root).strip() or _default_research_output_root()).expanduser().resolve()
        safe_research_label = _safe_run_label(str(research_run_label).strip() or "workspace_research")
        preview_run_dir = research_output_root_resolved / f"{safe_research_label}__seed_{int(seed_val)}__<timestamp>"
        st.caption(f"Research root: {research_output_root_resolved}")
        if bool(research_resume_existing):
            existing_run = _find_existing_research_run(research_output_root_resolved, run_label=safe_research_label, seed=int(seed_val))
            if existing_run is not None:
                st.caption(f"Будет проверен предыдущий run: {existing_run}")
            else:
                st.caption("Предыдущий совместимый run не найден — будет создан новый.")
        st.caption(f"Папка запуска будет вида: {preview_run_dir}")

    g1, g2 = st.columns([1.2, 1.8])
    with g1:
        st.markdown("**Что считать**")
        for key, label, help_text in _research_metric_catalog():
            st.checkbox(label, value=st.session_state.get(f"__research_flag_{key}", True), key=f"__research_flag_{key}", help=help_text)
        research_eff_k = int(st.number_input("Research eff_k", min_value=4, max_value=512, value=int(st.session_state.get("__research_eff_k", 32)), step=4, key="__research_eff_k"))
        scope = st.radio("Область расчёта", ["Активный граф", "Все графы workspace"], horizontal=False, key="__research_scope")

    with g2:
        st.markdown("**Параметры атак и энергии**")
        a1, a2, a3 = st.columns(3)
        with a1:
            attack_family = st.selectbox("Attack family", ["node", "edge", "mix"], key="__research_attack_family")
            if attack_family == "node":
                attack_kind = st.selectbox("Attack kind", ["random", "degree", "betweenness", "kcore", "richclub_top", "low_degree", "weak_strength"], key="__research_attack_kind_node")
            elif attack_family == "edge":
                attack_kind = st.selectbox("Attack kind", ["weak_edges_by_weight", "weak_edges_by_confidence", "strong_edges_by_weight", "strong_edges_by_confidence", "ricci_most_negative", "ricci_most_positive", "ricci_abs_max", "flux_high_rw", "flux_high_evo", "flux_high_rw_x_neg_ricci"], key="__research_attack_kind_edge")
            else:
                attack_kind = st.selectbox("Attack kind", ["hrish_mix", "mix_degree_preserving", "mix_weightconf_preserving"], key="__research_attack_kind_mix")
        with a2:
            attack_frac = float(st.number_input("Attack frac", min_value=0.0, max_value=1.0, value=float(st.session_state.get("__research_attack_frac", 0.5)), step=0.05, key="__research_attack_frac"))
            attack_steps = int(st.number_input("Attack steps", min_value=1, value=int(st.session_state.get("__research_attack_steps", 30)), step=1, key="__research_attack_steps"))
            attack_heavy_every = int(st.number_input("Attack heavy_every", min_value=1, value=int(st.session_state.get("__research_attack_heavy_every", 5)), step=1, key="__research_attack_heavy_every"))
        with a3:
            attack_fast_mode = st.checkbox("Attack fast mode", value=st.session_state.get("__research_attack_fast_mode", True), key="__research_attack_fast_mode")
            energy_steps = int(st.number_input("Energy steps", min_value=1, value=int(st.session_state.get("__research_energy_steps", 50)), step=1, key="__research_energy_steps"))
            energy_flow_mode = st.selectbox("Energy flow mode", ["rw", "evo", "phys"], key="__research_energy_flow_mode")

        e1, e2, e3 = st.columns(3)
        with e1:
            energy_damping = float(st.number_input("Energy damping", min_value=0.0, max_value=1.0, value=float(st.session_state.get("__research_energy_damping", 1.0)), step=0.05, key="__research_energy_damping"))
        with e2:
            energy_phys_injection = float(st.number_input("Phys injection", min_value=0.0, value=float(st.session_state.get("__research_energy_phys_injection", 0.15)), step=0.05, key="__research_energy_phys_injection"))
            energy_phys_leak = float(st.number_input("Phys leak", min_value=0.0, value=float(st.session_state.get("__research_energy_phys_leak", 0.02)), step=0.01, key="__research_energy_phys_leak"))
        with e3:
            energy_phys_cap_mode = st.selectbox("Phys cap mode", ["strength", "degree"], key="__research_energy_phys_cap_mode")
            energy_rw_impulse = st.checkbox("RW impulse", value=st.session_state.get("__research_energy_rw_impulse", True), key="__research_energy_rw_impulse")

    flags = _research_selected_flags()
    graph_ids = [cur_gid] if scope == "Активный граф" else list(ctx.graphs.keys())
    graph_ids = [gid for gid in graph_ids if gid in ctx.graphs]

    c_run1, c_run2 = st.columns(2)
    run_active = c_run1.button("Посчитать всё", type="primary", width="stretch")
    run_all = c_run2.button("Посчитать всё для всех графов", width="stretch")
    if run_active:
        graph_ids = [cur_gid]
    if run_all:
        graph_ids = list(ctx.graphs.keys())

    if run_active or run_all:
        # Защита от рассинхрона: после rerun/trim_memory/drop_graph набор id
        # может устареть, поэтому валидируем его прямо перед запуском.
        graph_ids = [gid for gid in graph_ids if gid in ctx.graphs]
        if not graph_ids:
            st.error(
                "В workspace нет доступных графов для расчёта. "
                "Возможно, список устарел после удаления графов или trim_memory()."
            )
            return
        frames, extras, run_dir = _run_research_workspace_plan(
            graph_ids,
            min_conf=float(min_conf),
            min_weight=float(min_weight),
            analysis_mode=str(analysis_mode),
            seed_val=int(seed_val),
            eff_k=int(research_eff_k),
            curv_n=int(curv_n),
            flags=flags,
            attack_family=str(attack_family),
            attack_kind=str(attack_kind),
            attack_frac=float(attack_frac),
            attack_steps=int(attack_steps),
            attack_heavy_every=int(attack_heavy_every),
            attack_fast_mode=bool(attack_fast_mode),
            energy_steps=int(energy_steps),
            energy_flow_mode=str(energy_flow_mode),
            energy_damping=float(energy_damping),
            energy_phys_injection=float(energy_phys_injection),
            energy_phys_leak=float(energy_phys_leak),
            energy_phys_cap_mode=str(energy_phys_cap_mode),
            energy_rw_impulse=bool(energy_rw_impulse),
            stream_save=bool(research_stream_save),
            stream_output_root=str(research_output_root),
            run_label=str(research_run_label),
            resume_existing_run=bool(research_resume_existing),
        )
        cache_key = (
            tuple(graph_ids),
            float(min_conf),
            float(min_weight),
            str(analysis_mode),
            int(seed_val),
            int(research_eff_k),
            int(curv_n),
            json.dumps(flags, sort_keys=True),
            str(attack_family),
            str(attack_kind),
            float(attack_frac),
            int(attack_steps),
            int(energy_steps),
            str(energy_flow_mode),
        )
        cached_payload = {
            "key": cache_key,
            "frames": frames,
            "run_dir": str(run_dir) if run_dir is not None else "",
        }
        if run_dir is None:
            cached_payload["extras"] = extras
            cached_payload["xlsx"] = _bundle_frames_to_xlsx_bytes(frames)
            cached_payload["zip"] = _bundle_frames_to_zip_bytes(frames, extras=extras)
        st.session_state["__research_results"] = cached_payload
        if run_dir is not None:
            st.session_state["__last_research_run_dir"] = str(run_dir)
            st.success(f"Готово: рассчитано {len(graph_ids)} граф(ов). Папка запуска: {run_dir}")
            st.code(str(run_dir), language=None)
        else:
            st.success(f"Готово: рассчитано {len(graph_ids)} граф(ов).")

    cached = st.session_state.get("__research_results")
    if cached:
        run_dir_cached = str(cached.get("run_dir", "")).strip()
        xlsx_data = cached.get("xlsx")
        zip_data = cached.get("zip")
        if run_dir_cached:
            st.caption(f"Последний research-run: {run_dir_cached}")
            run_dir_path = Path(run_dir_cached)
            xlsx_path = run_dir_path / "research_summary.xlsx"
            zip_path = run_dir_path / "research_bundle.zip"
            if xlsx_path.exists():
                xlsx_data = xlsx_path.read_bytes()
            if zip_path.exists():
                zip_data = zip_path.read_bytes()
        d1, d2 = st.columns(2)
        d1.download_button("Скачать research_summary.xlsx", data=xlsx_data, file_name="research_summary.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", width="stretch", disabled=xlsx_data is None)
        d2.download_button("Скачать research_bundle.zip", data=zip_data, file_name="research_bundle.zip", mime="application/zip", width="stretch", disabled=zip_data is None)
        for name, df in cached.get("frames", {}).items():
            st.markdown(f"**{name}**")
            st.dataframe(df, width="stretch", height=min(420, 44 + 36 * min(len(df), 8)))

def _run_article_plan(
    G_view: nx.Graph,
    *,
    cur_gid: str,
    analysis_mode: str,
    min_conf: float,
    min_weight: float,
    seed_val: int,
    curv_n: int,
    stats_eff_k: int,
    stats_do_curv: bool,
    stats_lightweight: bool,
    export_graph_ids: list[str] | None,
    export_key_base,
) -> None:
    """Run a single-click pipeline for article-ready metrics and exports.

    The function writes results into dedicated caches in ``st.session_state`` so
    tab reruns can reuse expensive intermediate results.
    """
    prog = st.progress(0.0)
    msg = st.empty()

    def _set(frac: float, text: str) -> None:
        prog.progress(min(1.0, max(0.0, float(frac))))
        msg.caption(text)

    metrics_key = (
        cur_gid,
        analysis_mode,
        float(min_conf),
        float(min_weight),
        int(seed_val),
    )
    ricci_key = (cur_gid, analysis_mode, float(min_conf), float(min_weight), int(seed_val), int(curv_n))

    _set(0.02, "План: базовые метрики")
    # Здесь считаем полный набор базовых метрик для текущего графа, а не UI-light версию.
    base = calculate_metrics(
        G_view,
        eff_sources_k=int(stats_eff_k),
        seed=int(seed_val),
        compute_curvature=False,
        curvature_sample_edges=int(curv_n),
        compute_heavy=True,
        skip_spectral=False,
        diameter_samples=16,
        skip_clustering=False,
        skip_assortativity=False,
    )
    st.session_state.setdefault("__base_metrics_cache", {})[metrics_key] = base

    _set(0.18, f"План: Ricci ({int(curv_n)} edges)")
    ricci = GraphService.compute_ricci_progress(
        G_view,
        sample_edges=curv_n,
        seed=seed_val,
        progress_cb=lambda frac: _set(0.18 + 0.32 * float(frac), f"Ricci: {int(round(100 * frac))}%"),
        status_cb=lambda text: _set(0.18 + 0.32 * 0.999, text),
    )
    st.session_state.setdefault("__ricci_cache", {})[ricci_key] = ricci

    _set(0.55, "План: готовлю ZIP для статьи")
    zip_payload = export_stats_zip_bytes(
        ctx.graphs,
        ctx.experiments,
        min_conf=float(min_conf),
        min_weight=float(min_weight),
        analysis_mode=str(analysis_mode),
        eff_sources_k=int(stats_eff_k),
        seed=int(seed_val),
        compute_curvature=bool(stats_do_curv),
        curvature_sample_edges=int(curv_n),
        graph_ids=export_graph_ids,
        progress_cb=lambda done, total, label: _set(0.55 + 0.20 * (float(done) / max(1.0, float(total))), f"ZIP: {label}"),
        lightweight=bool(stats_lightweight),
    )
    st.session_state.setdefault("__stats_export_cache", {})[("zip", export_key_base)] = zip_payload

    _set(0.77, "План: готовлю XLSX для статьи")
    xlsx_payload = export_stats_xlsx_bytes(
        ctx.graphs,
        ctx.experiments,
        min_conf=float(min_conf),
        min_weight=float(min_weight),
        analysis_mode=str(analysis_mode),
        eff_sources_k=int(stats_eff_k),
        seed=int(seed_val),
        compute_curvature=bool(stats_do_curv),
        curvature_sample_edges=int(curv_n),
        graph_ids=export_graph_ids,
        progress_cb=lambda done, total, label: _set(0.77 + 0.20 * (float(done) / max(1.0, float(total))), f"XLSX: {label}"),
        lightweight=bool(stats_lightweight),
    )
    st.session_state.setdefault("__stats_export_cache", {})[("xlsx", export_key_base)] = xlsx_payload
    _set(1.0, "План завершён")


@st.cache_resource(show_spinner=False)
def cached_build_graph(
    df_edges: pd.DataFrame,
    src_col: str,
    dst_col: str,
    min_conf: float,
    min_weight: float,
    analysis_mode: str,
) -> nx.Graph:
    """Собрать граф с кэшем (тяжёлый объект)."""
    return build_graph(
        df_edges,
        src_col=src_col,
        dst_col=dst_col,
        min_conf=min_conf,
        min_weight=min_weight,
        analysis_mode=analysis_mode,
    )


# ============================================================
# 4) SIDEBAR
# ============================================================
with st.sidebar:
    st.title("🎛️ Kodik Lab")
    page_mode = st.radio(
        "Страница",
        ["Граф и вкладки", "Batch-план"],
        index=0,
        key="__page_mode",
    )

    with st.expander("📥 Импорт / Экспорт", expanded=False):
        t1, t2 = st.tabs(["Workspace", "Exps"])

        with t1:
            if not SESSION_IO_AVAILABLE:
                st.warning("Workspace import/export временно недоступен.")
                with st.expander("Показать причину"):
                    st.code(SESSION_IO_IMPORT_ERROR or "unknown session_io error")
            else:
                if st.button("Export Workspace"):
                    b = export_workspace_json(ctx.graphs, ctx.experiments)
                    st.download_button("JSON", b, "workspace.json", "application/json")

                up_ws = st.file_uploader("Load Workspace", type=["json"], key="up_ws")
                if up_ws:
                    try:
                        gs, ex = import_workspace_json(up_ws.getvalue())
                        st.session_state["graphs"] = gs
                        st.session_state["experiments"] = ex
                        if gs:
                            ctx.set_active_graph(next(iter(gs.keys())))
                        st.rerun()
                    except Exception as e:  # pylint: disable=broad-except
                        st.error(f"Workspace import error: {type(e).__name__}: {e}")

        with t2:
            if not SESSION_IO_AVAILABLE:
                st.warning("Experiments export временно недоступен.")
            else:
                if st.button("Export Exps"):
                    b = export_experiments_json(ctx.experiments)
                    st.download_button("JSON", b, "experiments.json", "application/json")
                if st.button("Export Exps XLSX"):
                    b_xlsx = export_experiments_xlsx(ctx.experiments)
                    st.download_button(
                        "XLSX",
                        b_xlsx,
                        "experiments.xlsx",
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )

    st.markdown("---")
    st.markdown("---")
    st.subheader("📂 Данные")

    uploaded_file = st.file_uploader("CSV / Excel / MAT", type=["csv", "xlsx", "xls", "mat"], key="up_data")

    if uploaded_file:
        raw_bytes = uploaded_file.getvalue()
        file_hash = hashlib.md5(raw_bytes).hexdigest()

        if file_hash != st.session_state.get("last_upload_hash"):
            st.session_state["last_upload_hash"] = file_hash

            # На новом файле очищаем состояние прошлых попыток/staging.
            _clear_pending_upload_state()

            if uploaded_file.name.lower().endswith(".mat"):
                try:
                    packed_graphs, n_nodes = cached_load_packed_mat_graphs(raw_bytes, uploaded_file.name)
                    if not packed_graphs:
                        raise ValueError("MAT-файл прочитан, но графы не извлечены")

                    st.session_state["__mat_stage_name"] = uploaded_file.name
                    st.session_state["__mat_stage_source_filename"] = uploaded_file.name
                    st.session_state["__mat_stage_graphs"] = [df for _, df in packed_graphs]
                    st.session_state["__mat_stage_subjects"] = [name for name, _ in packed_graphs]
                    st.session_state["__mat_stage_n_nodes"] = int(n_nodes)
                    st.session_state["__upload_status"] = (
                        f"MAT распознан: {len(packed_graphs)} субъектов, {n_nodes} узлов. "
                        "Выбери ниже, что импортировать."
                    )
                    st.session_state.pop("__pending_upload_error", None)
                    st.rerun()
                except Exception as e:
                    st.session_state["__pending_upload_error"] = f"MAT import error: {e}"
                st.stop()

            try:
                df_raw, _ = cached_load_edges(raw_bytes, uploaded_file.name, fixed=False)
                st.session_state["__pending_upload_df"] = df_raw
                st.session_state["__pending_upload_name"] = uploaded_file.name

                # пытаемся авто-режимом
                try:
                    df_edges, meta = cached_load_edges(raw_bytes, uploaded_file.name, fixed=True)
                    add_graph_to_state(
                        uploaded_file.name,
                        df_edges,
                        "upload",
                        meta.get("src_col", "src"),
                        meta.get("dst_col", "dst"),
                    )
                    st.session_state.pop("__pending_upload_error", None)
                    st.session_state.pop("__pending_upload_df", None)
                    st.session_state.pop("__pending_upload_name", None)
                    st.rerun()
                except Exception as e:
                    st.session_state["__pending_upload_error"] = str(e)

            except Exception as e:
                st.session_state["__pending_upload_error"] = str(e)

    if st.session_state.get("__mat_stage_graphs"):
        st.markdown("---")
        st.subheader("🧠 MAT batch import")

        mat_name = st.session_state.get("__mat_stage_name", "unknown.mat")
        mat_subjects = st.session_state.get("__mat_stage_subjects", [])
        mat_n_nodes = st.session_state.get("__mat_stage_n_nodes", 0)
        mat_count = len(mat_subjects)

        st.caption(f"{mat_name}: {mat_count} subjects, {mat_n_nodes} nodes each")

        default_preview_n = min(10, mat_count)
        preview_n = st.number_input(
            "Сколько субъектов показать",
            min_value=1,
            max_value=max(1, mat_count),
            value=default_preview_n,
            step=1,
            key="__mat_preview_n",
        )
        st.dataframe(
            pd.DataFrame(
                {
                    "idx": list(range(min(preview_n, mat_count))),
                    "subject": mat_subjects[: int(preview_n)],
                }
            ),
            width="stretch",
            height=220,
        )

        selected_subjects = st.multiselect(
            "Import selected subjects",
            options=mat_subjects,
            default=mat_subjects[: min(3, mat_count)],
            key="__mat_selected_subjects",
        )

        b1, b2, b3 = st.columns(3)
        if b1.button("Import selected", width="stretch"):
            selected_set = set(selected_subjects)
            idx = [i for i, s in enumerate(mat_subjects) if s in selected_set]
            if not idx:
                st.warning("Ничего не выбрано для импорта.")
            else:
                added_ids = _import_staged_mat_graphs(idx)
                st.session_state["__upload_status"] = f"Импортировано {len(added_ids)} выбранных графов из {mat_name}."
                _clear_pending_upload_state()
                st.rerun()
        if b2.button("Import all", width="stretch"):
            added_ids = _import_staged_mat_graphs(None)
            st.session_state["__upload_status"] = f"Импортировано {len(added_ids)} графов из {mat_name}."
            _clear_pending_upload_state()
            st.rerun()
        if b3.button("Cancel", width="stretch"):
            _clear_pending_upload_state()
            st.info("MAT staging очищен.")
            st.rerun()

    # Column mapping UI (если авто-режим не взлетел)
    if st.session_state.get("__pending_upload_df") is not None:
        df_raw = st.session_state["__pending_upload_df"]
        err = st.session_state.get("__pending_upload_error")

        with st.expander("🧩 Сопоставление колонок", expanded=bool(err)):
            if err:
                st.warning("Авто-разбор не вышел. Нужны колонки руками.")
                st.caption(f"Причина: {err}")

            cols = list(df_raw.columns)
            if not cols:
                st.error("Файл пустой? колонок не вижу.")
            else:
                g_src, g_dst, g_w, g_c = _guess_cols(cols)

                src_col = st.selectbox("Source column", cols, index=cols.index(g_src) if g_src in cols else 0)
                dst_col = st.selectbox("Target column", cols, index=cols.index(g_dst) if g_dst in cols else min(1, len(cols)-1))

                w_col = st.selectbox(
                    "Weight column (optional)",
                    ["(нет)"] + cols,
                    index=(1 + cols.index(g_w)) if g_w in cols else 0,
                )
                c_col = st.selectbox(
                    "Confidence column (optional)",
                    ["(нет)"] + cols,
                    index=(1 + cols.index(g_c)) if g_c in cols else 0,
                )

                show_preview = st.checkbox("Показать первые строки", value=False)
                if show_preview:
                    st.dataframe(df_raw.head(30), width="stretch")

                if st.button("Загрузить с этим маппингом", type="primary"):
                    tmp_df = pd.DataFrame(
                        {
                            "src": df_raw[src_col].astype(str),
                            "dst": df_raw[dst_col].astype(str),
                        }
                    )

                    if w_col != "(нет)":
                        tmp_df["weight"] = pd.to_numeric(df_raw[w_col], errors="coerce")
                    else:
                        tmp_df["weight"] = 1.0

                    if c_col != "(нет)":
                        tmp_df["confidence"] = pd.to_numeric(df_raw[c_col], errors="coerce")
                    else:
                        tmp_df["confidence"] = 100.0

                    # NaN -> дефолты
                    tmp_df["weight"] = tmp_df["weight"].fillna(1.0)
                    tmp_df["confidence"] = tmp_df["confidence"].fillna(100.0)

                    name = st.session_state.get("__pending_upload_name", "upload")
                    add_graph_to_state(name, tmp_df, "upload", "src", "dst")

                    st.session_state.pop("__pending_upload_error", None)
                    st.session_state.pop("__pending_upload_df", None)
                    st.session_state.pop("__pending_upload_name", None)
                    st.rerun()

    with st.expander("🎲 Демо граф"):
        from src.null_models import make_er_gnm

        dt = st.selectbox("Тип", ["ER", "Barabasi", "Watts"], key="demo_t")
        if st.button("Создать"):
            import networkx as nx

            if dt == "ER":
                G0 = make_er_gnm(250, 800, 42)
            elif dt == "Barabasi":
                G0 = nx.barabasi_albert_graph(250, 3)
            else:
                G0 = nx.watts_strogatz_graph(250, 6, 0.1)

            edges = [[u, v, float(0.1 + 0.9 * np.random.rand()), 100.0] for u, v in G0.edges()]
            df_demo = pd.DataFrame(edges, columns=["src", "dst", "weight", "confidence"])
            add_graph_to_state(f"Demo {dt}", df_demo, "demo", "src", "dst")
            st.rerun()

    st.markdown("---")
    st.subheader("⚙️ Фильтры")
    min_conf = st.number_input("Min Confidence", 0, 100, 0)
    min_weight = st.number_input("Min Weight", 0.0, 1000.0, 0.0, step=0.1)

    st.markdown("---")
    st.subheader("📈 Вид")
    if "plot_height" not in st.session_state:
        st.session_state["plot_height"] = settings.PLOT_HEIGHT
    if "norm_mode" not in st.session_state:
        st.session_state["norm_mode"] = "none"

    st.session_state["plot_height"] = st.slider("Высота", 600, 1400, st.session_state["plot_height"], 50)
    st.session_state["norm_mode"] = st.selectbox(
        "Нормировка", ["none", "rel0", "delta0", "minmax", "zscore"], index=0
    )

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🧹 Clear cache", help="Сброс st.cache_* (иногда лечит странные подвисоны)"):
            try:
                st.cache_data.clear()
                st.cache_resource.clear()
            except Exception:
                pass
            # ещё и локальное
            st.session_state.pop("__ricci_cache", None)
            st.success("Cache cleared")
            st.rerun()

    with c2:
        if st.button("🧨 Trim memory", help="Обрезает лишние графы/экспы (чтобы вкладка не съедала 4ГБ)"):
            try:
                ctx.trim_memory()
            except Exception:
                pass
            st.rerun()

    if st.button("🗑️ Reset All", type="primary"):
        st.session_state.clear()
        st.rerun()



if page_mode == "Batch-план":
    st.header("Batch-план")
    st.caption("Отдельная страница для пакетного расчёта по папке, ZIP или набору файлов.")

    batch_source_mode = st.selectbox(
        "Источник данных",
        ["local_folder", "uploaded_files", "uploaded_zip"],
        index=0,
        format_func=lambda x: {
            "local_folder": "Локальная папка",
            "uploaded_files": "Загруженные файлы",
            "uploaded_zip": "ZIP-архив",
        }.get(x, x),
        key="__batch_source_mode_page",
    )

    batch_input_dir = ""
    batch_uploaded_files = []
    batch_uploaded_zip = None
    if batch_source_mode == "local_folder":
        batch_input_dir = st.text_input(
            "Папка с входными файлами",
            value=st.session_state.get("__batch_input_dir_page", os.getcwd()),
            key="__batch_input_dir_page",
        )
    elif batch_source_mode == "uploaded_files":
        batch_uploaded_files = st.file_uploader(
            "Загрузи набор файлов для batch",
            type=["mat", "csv", "tsv", "txt", "xlsx", "xls", "npy", "npz"],
            accept_multiple_files=True,
            key="__batch_uploaded_files_page",
        ) or []
    else:
        batch_uploaded_zip = st.file_uploader(
            "Загрузи ZIP с файлами для batch",
            type=["zip"],
            accept_multiple_files=False,
            key="__batch_uploaded_zip_page",
        )

    c1, c2 = st.columns(2)
    with c1:
        # По умолчанию берём все имена; затем фильтруем по поддерживаемым
        # расширениям в discover_batch_files/_iter_input_files.
        batch_pattern = st.text_input("Pattern", value=st.session_state.get("__batch_pattern_page", "*"), key="__batch_pattern_page")
        batch_recursive = st.checkbox("Recursive", value=st.session_state.get("__batch_recursive_page", True), key="__batch_recursive_page")
        batch_limit = st.number_input(
            "Limit (0 = all)",
            min_value=0,
            value=int(st.session_state.get("__batch_limit_page", 0)),
            step=1,
            key="__batch_limit_page",
        )
        p1, p2, p3, p4 = st.columns(4)
        with p1:
            batch_run_metrics = st.checkbox("Metrics", value=st.session_state.get("__batch_run_metrics_page", True), key="__batch_run_metrics_page")
        with p2:
            batch_run_attack = st.checkbox("Attack", value=st.session_state.get("__batch_run_attack_page", False), key="__batch_run_attack_page")
        with p3:
            batch_run_energy = st.checkbox("Energy", value=st.session_state.get("__batch_run_energy_page", False), key="__batch_run_energy_page")
        with p4:
            batch_run_resistance = st.checkbox("Resistance", value=st.session_state.get("__batch_run_resistance_page", False), key="__batch_run_resistance_page")
        batch_run_label = st.text_input(
            "Имя запуска (опционально)",
            value=st.session_state.get("__batch_run_label_page", "mat_batch"),
            key="__batch_run_label_page",
        )
        batch_output_root = st.text_input(
            "Корневая папка для результатов",
            value=st.session_state.get("__batch_output_root_page", _default_batch_output_root()),
            key="__batch_output_root_page",
        )
        batch_output_root_resolved = Path(str(batch_output_root).strip() or ".").expanduser().resolve()
        st.caption(f"Абсолютный путь root: {batch_output_root_resolved}")
    with c2:
        batch_input_kind = st.selectbox("Input kind", ["auto", "matrix", "edge"], index=1, key="__batch_input_kind_page")
        batch_mat_key = st.text_input("MAT key (optional)", value=st.session_state.get("__batch_mat_key_page", ""), key="__batch_mat_key_page")
        batch_sign_policy = st.selectbox("Sign policy", ["abs", "positive_only", "shift"], index=0, key="__batch_sign_policy_page")
        batch_threshold_mode = st.selectbox("Threshold mode", ["density", "absolute"], index=0, key="__batch_threshold_mode_page")
        batch_threshold_value = st.number_input(
            "Threshold value",
            value=float(st.session_state.get("__batch_threshold_value_page", 0.15)),
            min_value=0.0,
            step=0.01,
            key="__batch_threshold_value_page",
        )
        batch_shift = st.number_input(
            "Shift",
            value=float(st.session_state.get("__batch_shift_page", 0.0)),
            step=0.01,
            key="__batch_shift_page",
        )

    d1, d2, d3, d4 = st.columns(4)
    with d1:
        batch_seed = st.number_input("Seed", min_value=0, value=int(st.session_state.get("__batch_seed_page", int(settings.DEFAULT_SEED))), step=1, key="__batch_seed_page")
        batch_lcc = st.checkbox("LCC only", value=st.session_state.get("__batch_lcc_page", True), key="__batch_lcc_page")
    with d2:
        batch_compute_curv = st.checkbox("Compute curvature", value=st.session_state.get("__batch_compute_curv_page", False), key="__batch_compute_curv_page")
        batch_curv_n = st.number_input("Curvature sample edges", min_value=1, value=int(st.session_state.get("__batch_curv_n_page", 120)), step=1, key="__batch_curv_n_page")
    with d3:
        batch_eff_k = st.number_input("eff_k", min_value=1, value=int(st.session_state.get("__batch_eff_k_page", 32)), step=1, key="__batch_eff_k_page")
        batch_skip_spectral = st.checkbox("Skip spectral", value=st.session_state.get("__batch_skip_spectral_page", False), key="__batch_skip_spectral_page")
    with d4:
        batch_chunk_size = st.number_input("Chunk size", min_value=1, value=int(st.session_state.get("__batch_chunk_size_page", 10)), step=1, key="__batch_chunk_size_page")
        batch_write_full_bundle = st.checkbox("Full bundle zip", value=st.session_state.get("__batch_write_full_bundle_page", False), key="__batch_write_full_bundle_page")

    energy_box = st.container(border=True)
    with energy_box:
        st.caption("Параметры diffusion используются только если включён Energy")
        e1, e2, e3 = st.columns(3)
        with e1:
            batch_energy_steps = st.number_input("Energy steps", min_value=1, value=int(st.session_state.get("__batch_energy_steps_page", 50)), step=1, key="__batch_energy_steps_page")
            batch_energy_flow_mode = st.selectbox("Energy flow mode", ["rw", "evo", "phys"], index=0, key="__batch_energy_flow_mode_page")
        with e2:
            batch_energy_damping = st.number_input("Energy damping", min_value=0.0, max_value=1.0, value=float(st.session_state.get("__batch_energy_damping_page", 1.0)), step=0.05, key="__batch_energy_damping_page")
            batch_energy_rw_impulse = st.checkbox("Energy RW impulse", value=st.session_state.get("__batch_energy_rw_impulse_page", True), key="__batch_energy_rw_impulse_page")
        with e3:
            batch_energy_phys_injection = st.number_input("Energy phys injection", min_value=0.0, value=float(st.session_state.get("__batch_energy_phys_injection_page", 0.15)), step=0.05, key="__batch_energy_phys_injection_page")
            batch_energy_phys_leak = st.number_input("Energy phys leak", min_value=0.0, value=float(st.session_state.get("__batch_energy_phys_leak_page", 0.02)), step=0.01, key="__batch_energy_phys_leak_page")
            batch_energy_phys_cap_mode = st.selectbox("Energy cap mode", ["strength", "degree"], index=0, key="__batch_energy_phys_cap_mode_page")

    attack_box = st.container(border=True)
    with attack_box:
        st.caption("Параметры атаки используются только если включён Attack")
        a1, a2, a3, a4 = st.columns(4)
        with a1:
            batch_family = st.selectbox("Family", ["node", "edge", "mix"], index=0, key="__batch_family_page")
            batch_kind = st.text_input("Kind", value=st.session_state.get("__batch_kind_page", "degree"), key="__batch_kind_page")
        with a2:
            batch_frac = st.number_input("Frac", min_value=0.0, max_value=1.0, value=float(st.session_state.get("__batch_frac_page", 0.5)), step=0.05, key="__batch_frac_page")
            batch_steps = st.number_input("Steps", min_value=1, value=int(st.session_state.get("__batch_steps_page", 30)), step=1, key="__batch_steps_page")
        with a3:
            batch_heavy_every = st.number_input("Heavy every", min_value=1, value=int(st.session_state.get("__batch_heavy_every_page", 5)), step=1, key="__batch_heavy_every_page")
            batch_fast_mode = st.checkbox("Fast mode", value=st.session_state.get("__batch_fast_mode_page", False), key="__batch_fast_mode_page")
        with a4:
            batch_alpha_rewire = st.number_input("alpha_rewire", min_value=0.0, max_value=1.0, value=float(st.session_state.get("__batch_alpha_rewire_page", 0.6)), step=0.05, key="__batch_alpha_rewire_page")
            batch_beta_replace = st.number_input("beta_replace", min_value=0.0, max_value=1.0, value=float(st.session_state.get("__batch_beta_replace_page", 0.4)), step=0.05, key="__batch_beta_replace_page")
            batch_swaps_per_edge = st.number_input("swaps_per_edge", min_value=0.0, value=float(st.session_state.get("__batch_swaps_per_edge_page", 0.5)), step=0.1, key="__batch_swaps_per_edge_page")
            batch_replace_from = st.text_input("replace_from", value=st.session_state.get("__batch_replace_from_page", "CFG"), key="__batch_replace_from_page")

    preview_root = None
    preview_files = []
    cleanup_cb_preview = None
    try:
        preview_root, preview_files, cleanup_cb_preview = discover_batch_files(
            source_mode=batch_source_mode,
            input_dir=batch_input_dir,
            uploaded_files=batch_uploaded_files,
            uploaded_zip_name=batch_uploaded_zip.name if batch_uploaded_zip is not None else "",
            uploaded_zip_bytes=batch_uploaded_zip.getvalue() if batch_uploaded_zip is not None else None,
            pattern=batch_pattern,
            recursive=batch_recursive,
            limit=int(batch_limit),
        )
    except Exception as e:
        st.info(f"Пока нет списка файлов: {type(e).__name__}: {e}")

    selected_files_abs = []
    if preview_files:
        display_files = []
        preview_rows = []
        for p in preview_files:
            try:
                rel = str(p.relative_to(preview_root))
            except Exception:
                rel = str(p)
            display_files.append(rel)
            meta = inspect_batch_file(p)
            preview_rows.append(
                {
                    "file": rel,
                    "suffix": meta.get("suffix"),
                    "kind_guess": meta.get("kind_guess"),
                    "expanded_graphs": meta.get("expanded_graphs"),
                    "packed_mat": meta.get("packed_mat"),
                    "n_subjects": meta.get("n_subjects"),
                    "n_nodes": meta.get("n_nodes"),
                    "preview_error": meta.get("preview_error"),
                }
            )

        st.subheader("Что считать")
        pick_mode = st.radio(
            "Посчитать для",
            ["Всех найденных", "Только выбранных"],
            horizontal=True,
            key="__batch_pick_mode_page",
        )
        total_expanded = int(sum(int(row.get("expanded_graphs") or 1) for row in preview_rows))
        st.caption(f"Найдено файлов: {len(display_files)}; будет рассчитано графов: {total_expanded}. Промежуточный экспорт будет идти чанками по {int(batch_chunk_size)}.")
        st.dataframe(pd.DataFrame(preview_rows), width="stretch", height=260)
        if pick_mode == "Только выбранных":
            selected_display = st.multiselect(
                "Выбери файлы",
                options=display_files,
                default=display_files[: min(10, len(display_files))],
                key="__batch_selected_display_page",
            )
            selected_lookup = set(selected_display)
            selected_files_abs = [str(p) for p, rel in zip(preview_files, display_files) if rel in selected_lookup]
        else:
            selected_files_abs = [str(p) for p in preview_files]

        run_label = "всех" if pick_mode == "Всех найденных" else f"выбранных ({len(selected_files_abs)})"
        selected_modes = [name for name, flag in [("metrics", batch_run_metrics), ("attack", batch_run_attack), ("energy", batch_run_energy), ("resistance", batch_run_resistance)] if flag]
        mode_label = "_".join(selected_modes) if selected_modes else "batch"
        batch_output_root_resolved = Path(str(batch_output_root).strip() or ".").expanduser().resolve()
        preview_label = str(batch_run_label).strip() or f"batch_{mode_label}"
        preview_name = f"{preview_label}__seed_{int(batch_seed)}__<timestamp>"
        preview_run_dir = batch_output_root_resolved / preview_name
        st.info(
            "\n".join(
                [
                    f"Куда будет писать root: {batch_output_root_resolved}",
                    f"Папка запуска будет вида: {preview_run_dir}",
                    f"Файл-указатель последнего запуска: {batch_output_root_resolved / 'LAST_BATCH_RUN.txt'}",
                ]
            )
        )
        batch_status = st.empty()
        batch_prog = st.progress(0.0)
        run_batch_btn = st.button(f"Посчитать для {run_label}", type="primary", width="stretch")

        if run_batch_btn:
            cleanup_cb_run = None
            try:
                if not str(batch_output_root).strip():
                    raise ValueError("Не указана корневая папка для результатов")
                if not batch_run_metrics and not batch_run_attack and not batch_run_energy and not batch_run_resistance:
                    raise ValueError("Отметь хотя бы один расчёт")
                if not selected_files_abs:
                    raise ValueError("Нет выбранных файлов")

                staged_input_dir, _, cleanup_cb_run = stage_batch_inputs(
                    source_mode=batch_source_mode,
                    input_dir=batch_input_dir,
                    uploaded_files=batch_uploaded_files,
                    uploaded_zip_name=batch_uploaded_zip.name if batch_uploaded_zip is not None else "",
                    uploaded_zip_bytes=batch_uploaded_zip.getvalue() if batch_uploaded_zip is not None else None,
                )

                selected_rel = []
                for abs_path in selected_files_abs:
                    try:
                        selected_rel.append(str(Path(abs_path).resolve().relative_to(Path(preview_root).resolve())))
                    except Exception:
                        pass
                selected_run_files = [str((Path(staged_input_dir) / rel).resolve()) for rel in selected_rel] if selected_rel else None

                selected_modes = [name for name, flag in [("metrics", batch_run_metrics), ("attack", batch_run_attack), ("energy", batch_run_energy), ("resistance", batch_run_resistance)] if flag]
                mode_label = "_".join(selected_modes) if selected_modes else "batch"
                planned_dir = make_run_dir(
                    batch_output_root,
                    mode=f"batch_{mode_label}",
                    seed=int(batch_seed),
                    run_label=str(batch_run_label).strip(),
                )
                st.session_state["__last_batch_run_dir"] = str(planned_dir)
                write_run_metadata(
                    planned_dir,
                    base_dir=batch_output_root,
                    run_label=str(batch_run_label).strip() or f"batch_{mode_label}",
                    seed=int(batch_seed),
                    mode=f"batch_{mode_label}",
                    status="planned",
                    extra={
                        "source_mode": str(batch_source_mode),
                        "selected_modes": ",".join(selected_modes),
                        "selected_files_count": len(selected_files_abs),
                    },
                )
                batch_status.info(f"Создана папка запуска: {planned_dir}")

                args = build_ui_args(
                    input_dir=str(staged_input_dir),
                    out_dir=str(planned_dir),
                    pattern=str(batch_pattern),
                    recursive=bool(batch_recursive),
                    limit=int(batch_limit),
                    input_kind=str(batch_input_kind),
                    mat_key=str(batch_mat_key),
                    sign_policy=str(batch_sign_policy),
                    threshold_mode=str(batch_threshold_mode),
                    threshold_value=float(batch_threshold_value),
                    shift=float(batch_shift),
                    seed=int(batch_seed),
                    lcc=bool(batch_lcc),
                    eff_k=int(batch_eff_k),
                    compute_curvature=bool(batch_compute_curv),
                    curvature_sample_edges=int(batch_curv_n),
                    skip_spectral=bool(batch_skip_spectral),
                    family=str(batch_family),
                    kind=str(batch_kind),
                    frac=float(batch_frac),
                    alpha_rewire=float(batch_alpha_rewire),
                    beta_replace=float(batch_beta_replace),
                    swaps_per_edge=float(batch_swaps_per_edge),
                    replace_from=str(batch_replace_from),
                    steps=int(batch_steps),
                    heavy_every=int(batch_heavy_every),
                    fast_mode=bool(batch_fast_mode),
                    run_metrics=bool(batch_run_metrics),
                    run_attack=bool(batch_run_attack),
                    run_energy=bool(batch_run_energy),
                    run_resistance=bool(batch_run_resistance),
                    energy_steps=int(batch_energy_steps),
                    energy_flow_mode=str(batch_energy_flow_mode),
                    energy_damping=float(batch_energy_damping),
                    energy_phys_injection=float(batch_energy_phys_injection),
                    energy_phys_leak=float(batch_energy_phys_leak),
                    energy_phys_cap_mode=str(batch_energy_phys_cap_mode),
                    energy_rw_impulse=bool(batch_energy_rw_impulse),
                    source_mode=str(batch_source_mode),
                    selected_files=selected_run_files,
                    batch_chunk_size=int(batch_chunk_size),
                    write_full_bundle=bool(batch_write_full_bundle),
                )

                def _ui_progress(done: int, total: int, label: str):
                    frac = 1.0 if total <= 0 else min(1.0, max(0.0, float(done) / float(total)))
                    batch_prog.progress(frac)
                    batch_status.info(f"[{done:.2f}/{total}] {label}")

                run_dir, result_frames = run_batch_plan(args, progress_cb=_ui_progress)
                summary_parts = []
                for mode_name, df_batch in result_frames.items():
                    ok_n = int((df_batch.get("status") == "ok").sum()) if "status" in df_batch else len(df_batch)
                    summary_parts.append(f"{mode_name}: ok={ok_n}/{len(df_batch)}")
                st.session_state["__last_batch_run_dir"] = str(run_dir)
                batch_status.success(f"Готово: {'; '.join(summary_parts)}\nПапка запуска: {run_dir}")
                batch_prog.progress(1.0)
                st.code(str(run_dir), language=None)
                latest_ptr = Path(run_dir).parent / "LAST_BATCH_RUN.txt"
                if latest_ptr.exists():
                    st.caption(f"Указатель последнего запуска: {latest_ptr}")

                bundle_zip_path = Path(run_dir) / "batch_plan_bundle.zip"
                manifest_xlsx_path = Path(run_dir) / "batch_plan_manifest.xlsx"
                manifest_csv_path = Path(run_dir) / "batch_plan_manifest.csv"

                if manifest_xlsx_path.exists():
                    st.download_button(
                        "Скачать общий manifest (.xlsx)",
                        data=manifest_xlsx_path.read_bytes(),
                        file_name=manifest_xlsx_path.name,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        width="stretch",
                    )
                if manifest_csv_path.exists():
                    st.download_button(
                        "Скачать общий manifest (.csv)",
                        data=manifest_csv_path.read_bytes(),
                        file_name=manifest_csv_path.name,
                        mime="text/csv",
                        width="stretch",
                    )
                if bundle_zip_path.exists():
                    st.download_button(
                        "Скачать весь batch bundle (.zip)",
                        data=bundle_zip_path.read_bytes(),
                        file_name=bundle_zip_path.name,
                        mime="application/zip",
                        width="stretch",
                    )
            except Exception as e:
                batch_status.error(f"Batch run error: {type(e).__name__}: {e}")
            finally:
                if cleanup_cb_run is not None:
                    cleanup_cb_run()
    else:
        st.warning("Файлы пока не найдены. Проверь путь, pattern или загрузку.")

    last_batch_run_dir = str(st.session_state.get("__last_batch_run_dir", "")).strip()
    if last_batch_run_dir:
        st.caption(f"Последняя папка batch-запуска в этой сессии: {last_batch_run_dir}")

    if cleanup_cb_preview is not None:
        cleanup_cb_preview()
    st.stop()

# ============================================================
# 5) AАКТИВНЫЙ ГРАФЧИК
# ============================================================
if not ctx.graphs:
    st.warning("Workspace пуст. Загрузите файл или создайте демо-граф в сайдбаре.")
    st.stop()

# Защита от рассинхрона session_state во время rerun:
# даже если выше уже была проверка, здесь нельзя полагаться на то,
# что ctx.graphs точно словарь и точно не пустой.
cur_gids = list(ctx.graphs.keys()) if isinstance(ctx.graphs, dict) else []
cur_gid = ctx.active_graph_id

if not cur_gids:
    st.warning("Workspace пуст. Загрузите файл или создайте демо-граф в сайдбаре.")
    st.stop()

if cur_gid not in cur_gids:
    cur_gid = cur_gids[0]
    ctx.set_active_graph(cur_gid)

c1, c2 = st.columns([6, 1])
with c1:
    sel = st.selectbox(
        "Активный граф",
        cur_gids,
        index=cur_gids.index(cur_gid),
        format_func=lambda x: f"{ctx.graphs[x].name} ({ctx.graphs[x].source})",
        help="Выбери активный граф. Для MAT batch-формата здесь будут все субъекты.",
    )
    if sel != cur_gid:
        ctx.set_active_graph(sel)
        st.rerun()

active_entry = ctx.graphs[ctx.active_graph_id]
cur_gid = ctx.active_graph_id

with c2:
    if st.button("❌ Del"):
        ctx.drop_graph(cur_gid)
        st.rerun()


# ============================================================
# 6) CONTROLLER: DATA PREP
# ============================================================
with st.sidebar:
    st.markdown("---")
    st.markdown(f"**{active_entry.name}**")

    analysis_mode = st.radio("Режим", ["Global", "LCC"], horizontal=True)
    st.session_state["__analysis_mode"] = analysis_mode

    seed_val = int(st.number_input("Seed", value=settings.DEFAULT_SEED))

    curv_n = int(st.slider("Ricci edges", 20, 300, int(settings.RICCI_SAMPLE_EDGES)))
    do_ricci = st.button("Compute Ricci (slow)")

    st.markdown("---")
    st.subheader("📊 Export for statistics")
    st.caption("Tidy tables for p-value / regression / mixed models")

    stats_eff_k = int(st.number_input("Stats eff_k", min_value=4, max_value=512, value=32, step=4))
    stats_do_curv = st.checkbox("Include curvature in subject_metrics", value=False)
    stats_scope = st.radio("Что экспортировать", ["Active graph only", "All graphs"], horizontal=False, index=0)
    stats_lightweight = st.checkbox("Fast export (light metrics)", value=True)

    if "__stats_export_cache" not in st.session_state:
        st.session_state["__stats_export_cache"] = {}

    export_graph_ids = _stats_export_selection(stats_scope, cur_gid)
    export_key_base = (
        tuple(export_graph_ids) if export_graph_ids is not None else ("__all__",),
        str(cur_gid),
        int(curv_n),
        float(min_conf),
        float(min_weight),
        str(analysis_mode),
        int(stats_eff_k),
        int(seed_val),
        bool(stats_do_curv),
        int(curv_n),
        bool(stats_lightweight),
        len(ctx.experiments),
    )

    if st.button("🧠 Посчитать по плану", width="stretch"):
        _run_article_plan(
            # Собираем граф локально, чтобы не зависеть от инициализации G_view ниже по файлу.
            cached_build_graph(
                active_entry.edges,
                active_entry.src_col,
                active_entry.dst_col,
                min_conf,
                min_weight,
                analysis_mode,
            ),
            cur_gid=str(cur_gid),
            analysis_mode=str(analysis_mode),
            min_conf=float(min_conf),
            min_weight=float(min_weight),
            seed_val=int(seed_val),
            curv_n=int(curv_n),
            stats_eff_k=int(stats_eff_k),
            stats_do_curv=bool(stats_do_curv),
            stats_lightweight=bool(stats_lightweight),
            export_graph_ids=export_graph_ids,
            export_key_base=export_key_base,
        )
        st.success("План выполнен: base + Ricci + article exports.")

    st.caption("План считает полный набор метрик текущего графа, Ricci и готовит ZIP/XLSX для статьи. Для batch-атак/энергии используй отдельную вкладку Research calc.")

    b_zip, b_xlsx = st.columns(2)
    if b_zip.button("Prepare ZIP", width="stretch"):
        bar, msg, cb = _export_progress_ui("ZIP")
        try:
            payload = export_stats_zip_bytes(
                ctx.graphs,
                ctx.experiments,
                min_conf=float(min_conf),
                min_weight=float(min_weight),
                analysis_mode=str(analysis_mode),
                eff_sources_k=int(stats_eff_k),
                seed=int(seed_val),
                compute_curvature=bool(stats_do_curv),
                curvature_sample_edges=int(curv_n),
                graph_ids=export_graph_ids,
                progress_cb=cb,
                lightweight=bool(stats_lightweight),
            )
            st.session_state["__stats_export_cache"][("zip", export_key_base)] = payload
        finally:
            bar.empty()
            msg.empty()

    if b_xlsx.button("Prepare XLSX", width="stretch"):
        bar, msg, cb = _export_progress_ui("XLSX")
        try:
            payload = export_stats_xlsx_bytes(
                ctx.graphs,
                ctx.experiments,
                min_conf=float(min_conf),
                min_weight=float(min_weight),
                analysis_mode=str(analysis_mode),
                eff_sources_k=int(stats_eff_k),
                seed=int(seed_val),
                compute_curvature=bool(stats_do_curv),
                curvature_sample_edges=int(curv_n),
                graph_ids=export_graph_ids,
                progress_cb=cb,
                lightweight=bool(stats_lightweight),
            )
            st.session_state["__stats_export_cache"][("xlsx", export_key_base)] = payload
        finally:
            bar.empty()
            msg.empty()

    zip_payload = st.session_state["__stats_export_cache"].get(("zip", export_key_base))
    if zip_payload is not None:
        st.download_button(
            "Stats ZIP (CSV)",
            data=zip_payload,
            file_name="stats_tables.zip",
            mime="application/zip",
            width="stretch",
        )

    xlsx_payload = st.session_state["__stats_export_cache"].get(("xlsx", export_key_base))
    if xlsx_payload is not None:
        st.download_button(
            "Stats XLSX",
            data=xlsx_payload,
            file_name="stats_tables.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            width="stretch",
        )

    # DEBUG: если совсем странно
    # st.write(active_entry.edges.head(5))

G_view = cached_build_graph(
    active_entry.edges,
    active_entry.src_col,
    active_entry.dst_col,
    min_conf,
    min_weight,
    analysis_mode,
)

G_full = cached_build_graph(
    active_entry.edges,
    active_entry.src_col,
    active_entry.dst_col,
    min_conf,
    min_weight,
    "Global",
)

if st.session_state.get("__upload_status"):
    st.success(st.session_state["__upload_status"])
    st.session_state.pop("__upload_status", None)

if st.session_state.get("__pending_upload_error"):
    st.error(st.session_state["__pending_upload_error"])

# -----------------------------
# Base metrics: no auto-compute
# -----------------------------
metrics_key = (
    cur_gid,
    analysis_mode,
    float(min_conf),
    float(min_weight),
    int(seed_val),
)

if "__base_metrics_cache" not in st.session_state:
    st.session_state["__base_metrics_cache"] = {}

# Минимальный набор, чтобы дэшборд мог открыться мгновенно.
met = {
    "N": G_view.number_of_nodes() if G_view is not None else 0,
    "E": G_view.number_of_edges() if G_view is not None else 0,
}

with st.container(border=True):
    p1, p2, p3, p4, p5 = st.columns([1, 1, 1, 1, 1.2])
    p1.metric("Nodes", G_view.number_of_nodes())
    p2.metric("Edges", G_view.number_of_edges())
    p3.metric("Mode", analysis_mode)
    p4.metric("Ricci edges", curv_n)
    compute_base_now = p5.button("📊 Compute base metrics", key="btn_compute_base_metrics")

    if G_view.number_of_nodes() == 0:
        st.error(
            "После фильтров граф пустой. "
            "Уменьши Min Confidence / Min Weight или проверь входные данные."
        )
    else:
        if metrics_key in st.session_state["__base_metrics_cache"]:
            met = st.session_state["__base_metrics_cache"][metrics_key]
            st.success(
                f"Граф собран: {G_view.number_of_nodes()} узлов, "
                f"{G_view.number_of_edges()} рёбер. Базовые метрики уже в кэше."
            )
        else:
            st.warning(
                "Для ускорения старта базовые метрики больше не считаются автоматически. "
                "Нажми 'Compute base metrics', если они нужны."
            )

if compute_base_now and G_view.number_of_nodes() > 0:
    with st.spinner("Calculating base metrics..."):
        met = cached_calculate_metrics(
            G_view,
            int(seed_val),
            int(settings.RICCI_SAMPLE_EDGES),
        )
    st.session_state["__base_metrics_cache"][metrics_key] = met

if metrics_key in st.session_state["__base_metrics_cache"]:
    met = st.session_state["__base_metrics_cache"][metrics_key]

# Ricci отдельно, с прогрессом + свой кэш
ricci_key = (cur_gid, analysis_mode, float(min_conf), float(min_weight), int(seed_val), int(curv_n))
if "__ricci_cache" not in st.session_state:
    st.session_state["__ricci_cache"] = {}

if do_ricci:
    bar = st.progress(0.0)
    msg = st.empty()

    def _progress_cb(frac: float) -> None:
        frac_ = min(1.0, max(0.0, float(frac)))
        bar.progress(frac_)
        msg.caption(f"Ricci progress: {int(round(frac_ * 100))}%")

    def _status_cb(text: str) -> None:
        msg.caption(text)

    curv = GraphService.compute_ricci_progress(
        G_view,
        sample_edges=curv_n,
        seed=seed_val,
        progress_cb=_progress_cb,
        status_cb=_status_cb,
    )
    st.session_state["__ricci_cache"][ricci_key] = curv
    bar.progress(1.0)
    msg.caption("Ricci завершён")

if ricci_key in st.session_state["__ricci_cache"]:
    curv = st.session_state["__ricci_cache"][ricci_key]
    met.update(_normalize_ricci_payload(curv))


st.markdown("---")

# ============================================================
# 7) TABS ROUTER
# ============================================================
active_tab = st.radio(
    "Раздел",
    ["📊 Дэшборд", "🧠 Research", "⚡ Energy", "🕸️ 3D", "🧪 Null", "💥 Attack", "🆚 Compare"],
    horizontal=True,
    key="main_active_tab",
)

if active_tab == "📊 Дэшборд":
    tab_dashboard.render(G_view, met, active_entry)

elif active_tab == "🧠 Research":
    _render_research_tab(
        cur_gid=str(cur_gid),
        min_conf=float(min_conf),
        min_weight=float(min_weight),
        analysis_mode=str(analysis_mode),
        seed_val=int(seed_val),
        curv_n=int(curv_n),
    )

elif active_tab == "⚡ Energy":
    tab_energy.render(
        G_view,
        active_entry,
        seed_val,
        active_entry.src_col,
        active_entry.dst_col,
        min_conf,
        min_weight,
        analysis_mode,
    )

elif active_tab == "🕸️ 3D":
    tab_structure.render(
        G_view,
        active_entry,
        seed_val,
        active_entry.src_col,
        active_entry.dst_col,
        min_conf,
        min_weight,
        analysis_mode,
    )

elif active_tab == "🧪 Null":
    tab_attacks.render_null_models(
        G_view,
        G_full,
        met,
        active_entry,
        seed_val,
        add_graph_callback=add_graph_to_state,
    )

elif active_tab == "💥 Attack":
    tab_attacks.render_attack_lab(
        G_view,
        active_entry,
        seed_val,
        active_entry.src_col,
        active_entry.dst_col,
        min_conf,
        min_weight,
        analysis_mode,
        save_experiment_callback=save_experiment_to_state,
    )

elif active_tab == "🆚 Compare":
    tab_compare.render(
        G_view,
        active_entry,
        active_entry.src_col,
        active_entry.dst_col,
        min_conf,
        min_weight,
        analysis_mode,
    )

st.markdown("---")
st.caption("Kodik Лабчик")
