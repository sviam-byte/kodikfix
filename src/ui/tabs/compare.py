from __future__ import annotations

"""UI tab for comparing graph metrics and experiment trajectories."""

import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st

import plotly.express as px
from src.metrics import calculate_metrics
from src.preprocess import filter_edges
from src.graph_build import build_graph_from_edges, lcc_subgraph
from src.state_models import GraphEntry
from src.ui.plots.charts import (
    AUC_TRAP,
    apply_plot_defaults as _apply_plot_defaults,
    auto_y_range as _auto_y_range,
    forward_fill_heavy as _forward_fill_heavy,
)
from src.plotting import fig_compare_attacks


def _hash_graph(G: nx.Graph) -> str:
    """Build a stable hash for graph metric cache keys."""
    if G is None:
        return "none"
    try:
        return nx.weisfeiler_lehman_graph_hash(G, edge_attr="weight")
    except Exception:  # pylint: disable=broad-except
        return f"{G.number_of_nodes()}-{G.number_of_edges()}"


@st.cache_data(show_spinner=False, hash_funcs={nx.Graph: _hash_graph})
def _cached_scalar_metrics(G: nx.Graph, eff_sources_k: int = 16, seed: int = 42) -> dict:
    """Cache scalar metric calculations used by compare-tab graph benchmarking."""
    # Compare-tab is often rerendered; keep heavy metrics off for larger graphs.
    large_graph = G.number_of_nodes() > 300
    huge_graph = G.number_of_nodes() > 1200 or G.number_of_edges() > 8000
    return calculate_metrics(
        G,
        eff_sources_k=int(eff_sources_k),
        seed=int(seed),
        compute_curvature=False,
        compute_heavy=not large_graph,
        skip_spectral=bool(huge_graph),
        diameter_samples=6 if large_graph else 16,
    )


def render(
    G_view,
    active_entry: GraphEntry,
    src_col: str,
    dst_col: str,
    min_conf: float,
    min_weight: float,
    analysis_mode: str,
) -> None:
    """Render the compare tab for scalar and trajectory comparisons."""
    if G_view is None:
        return

    st.header("🆚 Сравнение")

    mode_cmp = st.radio("Что сравниваем?", ["Графы (скаляры)", "Эксперименты (траектории)"], horizontal=True)

    graphs = st.session_state["graphs"]
    all_gids = list(graphs.keys())

    if mode_cmp.startswith("Графы"):
        st.subheader("Сравнение скаляров по графам")
        selected_gids = st.multiselect(
            "Выберите графы",
            all_gids,
            default=[active_entry.id] if active_entry.id in all_gids else [],
            format_func=lambda gid: f"{graphs[gid].name} ({graphs[gid].source})",
        )

        scalar_metric = st.selectbox(
            "Метрика",
            ["density", "l2_lcc", "mod", "eff_w", "avg_degree", "clustering", "assortativity", "lcc_frac"],
            index=1
        )

        if selected_gids:
            rows = []
            for gid in selected_gids:
                entry = graphs[gid]
                _df = filter_edges(
                    entry.edges,
                    entry.src_col,
                    entry.dst_col,
                    min_conf, min_weight
                )
                _G = build_graph_from_edges(_df, entry.src_col, entry.dst_col)
                if analysis_mode.startswith("LCC"):
                    _G = lcc_subgraph(_G)

                # Compute scalar metrics for each graph under current filters.
                _m = _cached_scalar_metrics(_G, eff_sources_k=16, seed=42)
                rows.append({"Name": entry.name, scalar_metric: _m.get(scalar_metric, np.nan)})

            df_cmp = pd.DataFrame(rows)
            fig_bar = px.bar(df_cmp, x="Name", y=scalar_metric, title=f"Comparison: {scalar_metric}", color="Name")
            fig_bar.update_layout(template="plotly_dark", height=780)
            st.plotly_chart(fig_bar, width="stretch", key="plot_compare_bar")
            st.dataframe(df_cmp, width="stretch")
        else:
            st.info("Выбери графы.")

    else:
        st.subheader("Сравнение экспериментов (кривые)")
        exps = st.session_state["experiments"]
        if not exps:
            st.warning("Нет сохраненных экспериментов.")
        else:
            exp_opts = {e.id: e.name for e in exps}
            sel_exps = st.multiselect("Выберите эксперименты", list(exp_opts.keys()), format_func=lambda x: exp_opts[x])

            y_axis = st.selectbox("Y Axis", ["lcc_frac", "eff_w", "mod", "l2_lcc"], index=0)
            if sel_exps:
                curves = []
                x_candidates = []
                effective_mix_ok = True
                for eid in sel_exps:
                    e = next(x for x in exps if x.id == eid)
                    df_hist = _forward_fill_heavy(e.history)
                    curves.append((e.name, df_hist))
                    if "mix_frac" in df_hist.columns:
                        x_candidates.append("mix_frac")
                        if "mix_frac_effective" not in df_hist.columns:
                            effective_mix_ok = False
                    else:
                        x_candidates.append("removed_frac")
                        effective_mix_ok = False

                all_mix = bool(x_candidates) and all(x == "mix_frac" for x in x_candidates)
                if all_mix:
                    x_axis_options = ["mix_frac"]
                    if effective_mix_ok:
                        x_axis_options.append("mix_frac_effective")
                    x_col = st.selectbox("X Axis", x_axis_options, index=0, key="cmp_mix_x_axis")
                else:
                    x_col = "removed_frac"

                fig_lines = fig_compare_attacks(
                    curves,
                    x_col,
                    y_axis,
                    f"Comparison: {y_axis}",
                    normalize_mode=st.session_state["norm_mode"],
                    height=st.session_state["plot_height"],
                )
                fig_lines.update_layout(template="plotly_dark")
                all_y = pd.concat([pd.to_numeric(df[y_axis], errors="coerce") for _, df in curves if y_axis in df.columns], ignore_index=True)
                fig_lines = _apply_plot_defaults(fig_lines, height=st.session_state["plot_height"], y_range=_auto_y_range(all_y))
                st.plotly_chart(fig_lines, width="stretch", key="plot_compare_lines")

                st.markdown("#### Robustness (AUC)")
                auc_rows = []
                for name, df in curves:
                    if y_axis in df.columns and x_col in df.columns:
                        xs = pd.to_numeric(df[x_col], errors="coerce")
                        ys = pd.to_numeric(df[y_axis], errors="coerce")
                        mask = xs.notna() & ys.notna()
                        if mask.sum() >= 2:
                            auc = float(AUC_TRAP(ys[mask].to_numpy(), xs[mask].to_numpy()))
                            auc_rows.append({"Experiment": name, "AUC": auc})

                if auc_rows:
                    st.dataframe(pd.DataFrame(auc_rows).sort_values("AUC", ascending=False), width="stretch")
            else:
                st.info("Выбери эксперименты.")
