from __future__ import annotations

import hashlib
import logging
import os
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
from src.session_io import export_experiments_json, export_workspace_json, import_workspace_json
from src.state.session import ctx
from src.state_models import build_experiment_entry, build_graph_entry
from src.ui_blocks import inject_custom_css

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


@st.cache_data(show_spinner=False)
def cached_load_edges(file_bytes: bytes, filename: str, fixed: bool) -> tuple[pd.DataFrame, dict | None]:
    """Загрузить таблицу рёбер с кэшем (ускоряет переключение вкладок)."""
    df_any = load_edges(file_bytes, filename)
    if fixed:
        df_edges, meta = coerce_fixed_format(df_any)
        return df_edges, meta
    return df_any, None


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
            if st.button("Export Workspace"):
                b = export_workspace_json(ctx.graphs, ctx.experiments)
                st.download_button("JSON", b, "workspace.json", "application/json")

            up_ws = st.file_uploader("Load Workspace", type=["json"], key="up_ws")
            if up_ws:
                gs, ex = import_workspace_json(up_ws.getvalue())
                st.session_state["graphs"] = gs
                st.session_state["experiments"] = ex
                if gs:
                    ctx.active_graph_id = next(iter(gs.keys()))
                st.rerun()

        with t2:
            if st.button("Export Exps"):
                b = export_experiments_json(ctx.experiments)
                st.download_button("JSON", b, "experiments.json", "application/json")

    st.markdown("---")
    st.subheader("📂 Данные")

    uploaded_file = st.file_uploader("CSV / Excel", type=["csv", "xlsx", "xls"], key="up_data")

    if uploaded_file:
        raw_bytes = uploaded_file.getvalue()
        file_hash = hashlib.md5(raw_bytes).hexdigest()

        if file_hash != st.session_state.get("last_upload_hash"):
            st.session_state["last_upload_hash"] = file_hash

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

c1, c2, c3 = st.columns([3, 1, 1])
with c1:
    sel = st.selectbox(
        "Активный граф",
        cur_gids,
        index=cur_gids.index(cur_gid),
        format_func=lambda x: f"{ctx.graphs[x].name} ({ctx.graphs[x].source})",
    )
    if sel != cur_gid:
        ctx.active_graph_id = sel
        st.rerun()

active_entry = ctx.graphs[cur_gid]

with c3:
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

with st.spinner("Calculating metrics..."):
    met = GraphService.compute_metrics(
        active_entry.edges,
        active_entry.src_col,
        active_entry.dst_col,
        min_conf,
        min_weight,
        analysis_mode,
        seed_val,
        False,  # curvature отдельно
        int(settings.RICCI_SAMPLE_EDGES),
    )

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


# ============================================================
# 7) TABS ROUTER
# ============================================================
tab_names = ["📊 Дэшборд", "⚡ Energy", "🕸️ 3D", "🧪 Null", "💥 Attack", "🆚 Compare"]
current_tab = st.radio("Разделы", tab_names, horizontal=True, label_visibility="collapsed")

st.markdown("---")

if current_tab == tab_names[0]:
    tab_dashboard.render(G_view, met, active_entry)

elif current_tab == tab_names[1]:
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

elif current_tab == tab_names[2]:
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

elif current_tab == tab_names[3]:
    tab_attacks.render_null_models(
        G_view,
        G_full,
        met,
        active_entry,
        seed_val,
        add_graph_callback=add_graph_to_state,
    )

elif current_tab == tab_names[4]:
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

elif current_tab == tab_names[5]:
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
