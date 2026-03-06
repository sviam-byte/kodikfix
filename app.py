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
    ctx.set_graph_entry(entry)
    ctx.active_graph_id = gid
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
        ctx.active_graph_id = added_ids[0]
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
                            ctx.active_graph_id = next(iter(gs.keys()))
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
    ctx.active_graph_id = cur_gid

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
        ctx.active_graph_id = sel
        st.rerun()

active_entry = ctx.graphs[cur_gid]

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
    stats_do_curv = st.checkbox("Include curvature in subject_metrics", value=True)

    stats_zip = export_stats_zip_bytes(
        ctx.graphs,
        ctx.experiments,
        min_conf=float(min_conf),
        min_weight=float(min_weight),
        analysis_mode=str(analysis_mode),
        eff_sources_k=int(stats_eff_k),
        seed=int(seed_val),
        compute_curvature=bool(stats_do_curv),
        curvature_sample_edges=int(curv_n),
    )
    st.download_button(
        "Stats ZIP (CSV)",
        data=stats_zip,
        file_name="stats_tables.zip",
        mime="application/zip",
        use_container_width=True,
    )

    stats_xlsx = export_stats_xlsx_bytes(
        ctx.graphs,
        ctx.experiments,
        min_conf=float(min_conf),
        min_weight=float(min_weight),
        analysis_mode=str(analysis_mode),
        eff_sources_k=int(stats_eff_k),
        seed=int(seed_val),
        compute_curvature=bool(stats_do_curv),
        curvature_sample_edges=int(curv_n),
    )
    st.download_button(
        "Stats XLSX",
        data=stats_xlsx,
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
        bar.progress(min(1.0, max(0.0, frac)))

    def _status_cb(text: str) -> None:
        msg.caption(text)

    curv = GraphService.compute_ricci_progress(
        G_view,
        sample_edges=curv_n,
        seed=seed_val,
        progress_cb=_progress_cb,
        status_cb=_status_cb,
    )
    bar.empty()
    msg.empty()
    st.session_state["__ricci_cache"][ricci_key] = curv

if ricci_key in st.session_state["__ricci_cache"]:
    curv = st.session_state["__ricci_cache"][ricci_key]
    met.update(
        {
            "kappa_mean": curv.get("kappa_mean"),
            "kappa_median": curv.get("kappa_median"),
            "kappa_frac_negative": curv.get("kappa_frac_negative"),
            "fragility_kappa": curv.get("fragility_kappa"),
        }
    )


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
