from __future__ import annotations

import hashlib
import io
import logging
import math
import os
import traceback
from pathlib import Path

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
from src.services.graph_service import GraphService
from src.stats_export import export_stats_xlsx_bytes, export_stats_zip_bytes
from src.state.session import ctx
from src.state_models import build_experiment_entry, build_graph_entry
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
from src.batch_ops import build_ui_args, make_run_dir, run_batch_attack, run_batch_metrics

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


def _packed_edge_count_to_n(m: int) -> int:
    """Восстановить число узлов n из packed-edge размера m=n*(n-1)/2."""
    if m <= 0:
        raise ValueError("Число packed-edge признаков должно быть > 0")

    disc = 1 + 8 * int(m)
    root = int(math.isqrt(disc))
    if root * root != disc:
        raise ValueError(
            f"Число столбцов {m} не похоже на верхний треугольник матрицы: "
            "m должно быть равно n*(n-1)/2"
        )

    n = (1 + root) // 2
    if n * (n - 1) // 2 != m:
        raise ValueError(f"Число столбцов {m} не раскладывается как n*(n-1)/2")
    return int(n)


def _mat_obj_to_subject_names(obj: np.ndarray, n_rows: int) -> list[str]:
    """Преобразовать MATLAB object-array `subj_id` в список строковых имён."""
    flat = np.asarray(obj, dtype=object).ravel().tolist()
    out: list[str] = []
    for x in flat:
        if isinstance(x, np.ndarray):
            if x.size == 1:
                out.append(str(x.item()))
            else:
                out.append(" ".join(map(str, x.ravel().tolist())))
        else:
            out.append(str(x))

    if len(out) < n_rows:
        out.extend([f"subject_{i:03d}" for i in range(len(out), n_rows)])
    return out[:n_rows]


@st.cache_data(show_spinner=False)
def cached_load_packed_mat_graphs(file_bytes: bytes, filename: str) -> tuple[list[tuple[str, pd.DataFrame]], int]:
    """
    Загрузить MAT-файл формата:
      - data: (subjects, packed_edges)
      - subj_id: optional subject ids

    где packed_edges = n*(n-1)/2 (верхний треугольник без диагонали).
    """
    _ = filename  # участвует в ключе cache_data и не используется в логике ниже.
    from scipy.io import loadmat

    mat = loadmat(io.BytesIO(file_bytes))
    if "data" not in mat:
        raise ValueError("В .mat не найден ключ 'data'")

    X = np.asarray(mat["data"], dtype=float)
    if X.ndim != 2:
        raise ValueError(f"'data' в .mat должна быть 2D, получено: ndim={X.ndim}")

    n_subjects, m = X.shape
    n_nodes = _packed_edge_count_to_n(int(m))
    iu, ju = np.triu_indices(n_nodes, k=1)

    if "subj_id" in mat:
        subj_names = _mat_obj_to_subject_names(mat["subj_id"], n_subjects)
    else:
        subj_names = [f"subject_{i:03d}" for i in range(n_subjects)]

    graphs: list[tuple[str, pd.DataFrame]] = []
    for i in range(n_subjects):
        row = np.asarray(X[i], dtype=float).ravel()
        keep = np.isfinite(row)
        df = pd.DataFrame(
            {
                "src": iu[keep].astype(int),
                "dst": ju[keep].astype(int),
                "weight": row[keep].astype(float),
                "confidence": np.full(int(np.sum(keep)), 100.0, dtype=float),
            }
        )
        graphs.append((subj_names[i], df))

    return graphs, int(n_nodes)


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


def _run_article_plan(
    G_view: nx.Graph,
    *,
    cur_gid: str,
    analysis_mode: str,
    min_conf: float,
    min_weight: float,
    seed_val: int,
    curv_n: int,
    stats_do_curv: bool,
    stats_lightweight: bool,
    export_graph_ids: list[str] | None,
    export_key_base,
) -> None:
    """Run a single-click pipeline for article-ready metrics and exports."""
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
    base = cached_calculate_metrics(G_view, int(seed_val), int(settings.RICCI_SAMPLE_EDGES))
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
        eff_sources_k=int(settings.APPROX_EFFICIENCY_K),
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
        eff_sources_k=int(settings.APPROX_EFFICIENCY_K),
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

    with st.expander("🗂 Batch runner", expanded=False):
        batch_input_dir = st.text_input(
            "Папка с входными файлами",
            value=st.session_state.get("__batch_input_dir", os.getcwd()),
            key="__batch_input_dir",
        )
        c1, c2 = st.columns(2)
        with c1:
            batch_pattern = st.text_input("Pattern", value=st.session_state.get("__batch_pattern", "*.mat"), key="__batch_pattern")
            batch_recursive = st.checkbox("Recursive", value=st.session_state.get("__batch_recursive", True), key="__batch_recursive")
            batch_limit = st.number_input(
                "Limit (0 = all)",
                min_value=0,
                value=int(st.session_state.get("__batch_limit", 0)),
                step=1,
                key="__batch_limit",
            )
            batch_mode = st.selectbox("Что считать", ["metrics", "attack"], index=0, key="__batch_mode")
            batch_run_label = st.text_input(
                "Имя запуска (опционально)",
                value=st.session_state.get("__batch_run_label", "mat_batch"),
                key="__batch_run_label",
            )
            batch_output_root = st.text_input(
                "Корневая папка для результатов",
                value=st.session_state.get("__batch_output_root", _default_batch_output_root()),
                key="__batch_output_root",
            )
        with c2:
            batch_input_kind = st.selectbox("Input kind", ["auto", "matrix", "edge"], index=1, key="__batch_input_kind")
            batch_mat_key = st.text_input("MAT key (optional)", value=st.session_state.get("__batch_mat_key", ""), key="__batch_mat_key")
            batch_sign_policy = st.selectbox("Sign policy", ["abs", "positive_only", "shift"], index=0, key="__batch_sign_policy")
            batch_threshold_mode = st.selectbox("Threshold mode", ["density", "absolute"], index=0, key="__batch_threshold_mode")
            batch_threshold_value = st.number_input(
                "Threshold value",
                value=float(st.session_state.get("__batch_threshold_value", 0.15)),
                min_value=0.0,
                step=0.01,
                key="__batch_threshold_value",
            )
            batch_shift = st.number_input(
                "Shift",
                value=float(st.session_state.get("__batch_shift", 0.0)),
                step=0.01,
                key="__batch_shift",
            )

        d1, d2, d3 = st.columns(3)
        with d1:
            batch_seed = st.number_input(
                "Seed",
                min_value=0,
                value=int(st.session_state.get("__batch_seed", int(settings.DEFAULT_SEED))),
                step=1,
                key="__batch_seed",
            )
            batch_lcc = st.checkbox("LCC only", value=st.session_state.get("__batch_lcc", True), key="__batch_lcc")
        with d2:
            batch_compute_curv = st.checkbox("Compute curvature", value=st.session_state.get("__batch_compute_curv", False), key="__batch_compute_curv")
            batch_curv_n = st.number_input(
                "Curvature sample edges",
                min_value=1,
                value=int(st.session_state.get("__batch_curv_n", 120)),
                step=1,
                key="__batch_curv_n",
            )
        with d3:
            batch_eff_k = st.number_input("eff_k", min_value=1, value=int(st.session_state.get("__batch_eff_k", 32)), step=1, key="__batch_eff_k")
            batch_skip_spectral = st.checkbox("Skip spectral", value=st.session_state.get("__batch_skip_spectral", False), key="__batch_skip_spectral")

        attack_box = st.container(border=True)
        with attack_box:
            st.caption("Параметры атаки используются только если выбран mode = attack")
            a1, a2, a3 = st.columns(3)
            with a1:
                batch_family = st.selectbox("Family", ["node", "edge", "mix"], index=0, key="__batch_family")
                batch_kind = st.text_input("Kind", value=st.session_state.get("__batch_kind", "degree"), key="__batch_kind")
            with a2:
                batch_frac = st.number_input(
                    "Frac",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(st.session_state.get("__batch_frac", 0.5)),
                    step=0.05,
                    key="__batch_frac",
                )
                batch_steps = st.number_input("Steps", min_value=1, value=int(st.session_state.get("__batch_steps", 30)), step=1, key="__batch_steps")
            with a3:
                batch_heavy_every = st.number_input(
                    "Heavy every",
                    min_value=1,
                    value=int(st.session_state.get("__batch_heavy_every", 5)),
                    step=1,
                    key="__batch_heavy_every",
                )
                batch_fast_mode = st.checkbox("Fast mode", value=st.session_state.get("__batch_fast_mode", False), key="__batch_fast_mode")

        in_dir_path = Path(_norm_path_str(batch_input_dir)) if str(batch_input_dir).strip() else None
        out_root_path = Path(_norm_path_str(batch_output_root)) if str(batch_output_root).strip() else None
        if in_dir_path and in_dir_path.exists():
            try:
                candidates = sorted([
                    p
                    for p in (in_dir_path.rglob(batch_pattern) if batch_recursive else in_dir_path.glob(batch_pattern))
                    if p.is_file() and p.suffix.lower() in {".mat", ".csv", ".tsv", ".txt", ".xlsx", ".xls", ".npy", ".npz"}
                ])
                shown = candidates[:10]
                st.caption(f"Найдено файлов: {len(candidates)}")
                if shown:
                    st.dataframe(
                        pd.DataFrame({"file": [str(p.relative_to(in_dir_path)) for p in shown]}),
                        use_container_width=True,
                        height=180,
                    )
            except Exception as e:  # pylint: disable=broad-except
                st.warning(f"Не удалось просканировать папку: {type(e).__name__}: {e}")
        else:
            st.caption("Укажи существующую папку с файлами.")

        batch_status = st.empty()
        batch_prog = st.progress(0.0)
        run_batch_btn = st.button("Run batch", type="primary", use_container_width=True)

        if run_batch_btn:
            try:
                if not in_dir_path or not in_dir_path.exists():
                    raise FileNotFoundError("Папка с входными файлами не найдена")
                if not out_root_path:
                    raise ValueError("Не указана корневая папка для результатов")

                planned_dir = make_run_dir(
                    out_root_path,
                    mode=f"batch_{batch_mode}",
                    seed=int(batch_seed),
                    run_label=str(batch_run_label).strip(),
                )

                args = build_ui_args(
                    input_dir=str(in_dir_path),
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
                    steps=int(batch_steps),
                    heavy_every=int(batch_heavy_every),
                    fast_mode=bool(batch_fast_mode),
                )

                def _ui_progress(done: int, total: int, label: str):
                    frac = 1.0 if total <= 0 else min(1.0, max(0.0, float(done) / float(total)))
                    batch_prog.progress(frac)
                    batch_status.info(f"[{done}/{total}] {label}")

                if batch_mode == "metrics":
                    run_dir, df_batch = run_batch_metrics(args, progress_cb=_ui_progress)
                    ok_n = int((df_batch.get("status") == "ok").sum()) if "status" in df_batch else len(df_batch)
                    batch_status.success(f"Готово: metrics batch, ok={ok_n}/{len(df_batch)}\n{run_dir}")
                else:
                    run_dir, df_batch = run_batch_attack(args, progress_cb=_ui_progress)
                    ok_n = int((df_batch.get("status") == "ok").sum()) if "status" in df_batch else len(df_batch)
                    batch_status.success(f"Готово: attack batch, ok={ok_n}/{len(df_batch)}\n{run_dir}")
                batch_prog.progress(1.0)
            except Exception as e:  # pylint: disable=broad-except
                batch_status.error(f"Batch run error: {type(e).__name__}: {e}")

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
            use_container_width=True,
            height=220,
        )

        selected_subjects = st.multiselect(
            "Import selected subjects",
            options=mat_subjects,
            default=mat_subjects[: min(3, mat_count)],
            key="__mat_selected_subjects",
        )

        b1, b2, b3 = st.columns(3)
        if b1.button("Import selected", use_container_width=True):
            selected_set = set(selected_subjects)
            idx = [i for i, s in enumerate(mat_subjects) if s in selected_set]
            if not idx:
                st.warning("Ничего не выбрано для импорта.")
            else:
                added_ids = _import_staged_mat_graphs(idx)
                st.session_state["__upload_status"] = f"Импортировано {len(added_ids)} выбранных графов из {mat_name}."
                _clear_pending_upload_state()
                st.rerun()
        if b2.button("Import all", use_container_width=True):
            added_ids = _import_staged_mat_graphs(None)
            st.session_state["__upload_status"] = f"Импортировано {len(added_ids)} графов из {mat_name}."
            _clear_pending_upload_state()
            st.rerun()
        if b3.button("Cancel", use_container_width=True):
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
                    st.dataframe(df_raw.head(30), use_container_width=True)

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


# ============================================================
# 5) AАКТИВНЫЙ ГРАФЧИК
# ============================================================
if not ctx.graphs:
    st.warning("Workspace пуст. Загрузите файл или создайте демо-граф в сайдбаре.")
    st.stop()

cur_gids = list(ctx.graphs.keys())
cur_gid = ctx.active_graph_id
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

    if st.button("🧠 Посчитать по плану", use_container_width=True):
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
            stats_do_curv=bool(stats_do_curv),
            stats_lightweight=bool(stats_lightweight),
            export_graph_ids=export_graph_ids,
            export_key_base=export_key_base,
        )
        st.success("План выполнен: base + Ricci + article exports.")

    st.caption("План считает базовые метрики текущего графа, Ricci и готовит ZIP/XLSX для статьи.")

    b_zip, b_xlsx = st.columns(2)
    if b_zip.button("Prepare ZIP", use_container_width=True):
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

    if b_xlsx.button("Prepare XLSX", use_container_width=True):
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
            use_container_width=True,
        )

    xlsx_payload = st.session_state["__stats_export_cache"].get(("xlsx", export_key_base))
    if xlsx_payload is not None:
        st.download_button(
            "Stats XLSX",
            data=xlsx_payload,
            file_name="stats_tables.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
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
    ["📊 Дэшборд", "⚡ Energy", "🕸️ 3D", "🧪 Null", "💥 Attack", "🆚 Compare"],
    horizontal=True,
    key="main_active_tab",
)

if active_tab == "📊 Дэшборд":
    tab_dashboard.render(G_view, met, active_entry)

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
