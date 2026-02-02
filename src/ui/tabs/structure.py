from __future__ import annotations

import networkx as nx
import streamlit as st

import plotly.express as px
import plotly.graph_objects as go

from src.services.graph_service import GraphService
from src.state_models import GraphEntry
from src.ui.plots.scene3d import make_3d_traces
from src.utils import as_simple_undirected

_layout_cached = GraphService.compute_layout3d


def render(G_view: nx.Graph | None, active_entry: GraphEntry, seed_val: int, src_col: str, dst_col: str, min_conf: float, min_weight: float, analysis_mode: str) -> None:
    """Render the Structure & 3D tab."""
    if G_view is None:
        return

    if G_view.number_of_nodes() > 1500:
        st.warning("⚠️ Граф большой. Тяжелые метрики (Ricci, Efficiency) считаются в фоновом режиме.")
    col_vis_ctrl, col_vis_main = st.columns([1, 4])

    with col_vis_ctrl:
        st.subheader("Настройки 3D")
        show_labels = st.checkbox("Показать ID узлов", False)
        node_size = st.slider("Размер узлов", 1, 20, 4)
        max_nodes_viz = st.slider("Макс. узлов (виз)", 500, 20000, 6000, step=500)
        # 20k edges перегружают браузер; 2.5k держат FPS комфортным.
        max_edges_viz = st.slider("Макс. рёбер (виз)", 500, 10000, 2500, step=500)
        layout_mode = st.selectbox("Layout", ["Fixed (по исходному графу)", "Recompute (по текущему виду)"], index=0)

        st.info("3D-визуализация: фиксированный layout лучше для сравнения по шагам (не прыгает).")

        if st.button("🔄 Обновить layout seed (анти-кэш)"):
            st.session_state["layout_seed_bump"] = int(st.session_state.get("layout_seed_bump", 0)) + 1

        # Edge overlay options for 3D (coloring by edge-specific metrics).
        edge_overlay_ui = st.selectbox(
            "Разметка рёбер",
            [
                "Energy flux (RW)",
                "Energy flux (Demetrius)",
                "Weight (log10)",
                "Confidence",
                "Ricci sign (κ<0/κ>0)",
                "None",
            ],
            # Ricci per-edge слишком медленный для интерактива на больших графах;
            # дефолт — вес/потоки.
            index=2,
        )

    with col_vis_main:
        if G_view.number_of_nodes() > 2000:
            st.warning(f"Граф большой ({G_view.number_of_nodes()} узлов). 3D может тормозить.")

        # Небольшой прогрессбар: Streamlit кэш скрывает длительные вычисления, но UX без индикации — боль.
        pb = st.progress(0.0)
        pb_msg = st.empty()

        # Seed учитывает "анти-кэш" и делает layout детерминированным между перерисовками.
        base_seed = int(seed_val) + int(st.session_state.get("layout_seed_bump", 0))

        # 1) Получаем pos3d (режимы остаются детерминированными через seed).
        pb_msg.caption("3D layout...")
        if layout_mode.startswith("Fixed"):
            pos3d = _layout_cached(
                active_entry.edges,
                src_col,
                dst_col,
                float(min_conf),
                float(min_weight),
                analysis_mode,
                base_seed,
            )
        else:
            pos3d = _layout_cached(
                active_entry.edges,
                src_col,
                dst_col,
                float(min_conf),
                float(min_weight),
                analysis_mode,
                base_seed,
            )

        pb.progress(0.55)

        edge_overlay = "ricci"
        flow_mode = "rw"
        if edge_overlay_ui.startswith("Energy flux"):
            edge_overlay = "flux"
            flow_mode = "evo" if "Demetrius" in edge_overlay_ui else "rw"
        elif edge_overlay_ui.startswith("Weight"):
            edge_overlay = "weight"
        elif edge_overlay_ui.startswith("Confidence"):
            edge_overlay = "confidence"
        elif edge_overlay_ui.startswith("None"):
            edge_overlay = "none"

        # 2) Всегда строим трэйсы, чтобы 3D работал и для Fixed, и для Recompute.
        pb_msg.caption("Building Plotly traces...")
        edge_traces, node_trace = make_3d_traces(
            G_view,
            pos3d,
            show_scale=True,
            edge_overlay=edge_overlay,
            flow_mode=flow_mode,
            max_nodes_viz=int(max_nodes_viz),
            max_edges_viz=int(max_edges_viz),
            edge_subset_mode="top_abs",
            coord_round=4,
        )

        pb.progress(1.0)
        pb.empty(); pb_msg.empty()

        # 3) Рисуем внутри col_vis_main, чтобы не ломать сетку.
        if node_trace is not None:
            node_trace.marker.size = node_size
            if show_labels:
                node_trace.mode = "markers+text"

            fig_3d = go.Figure(data=[*edge_traces, node_trace])
            fig_3d.update_layout(
                title=f"3D Structure: {active_entry.name}",
                template="plotly_dark",
                showlegend=False,
                height=820,
                margin=dict(l=0, r=0, t=30, b=0),
                uirevision=f"{active_entry.id}_{base_seed}",
                scene=dict(
                    xaxis=dict(showbackground=False, showticklabels=False, title=""),
                    yaxis=dict(showbackground=False, showticklabels=False, title=""),
                    zaxis=dict(showbackground=False, showticklabels=False, title=""),
                ),
            )
            st.plotly_chart(fig_3d, use_container_width=True, key="plot_struct_3d")
        else:
            st.write("Граф пуст.")

    st.markdown("---")
    st.subheader("Матрица смежности (heatmap)")
    if G_view.number_of_nodes() < 1000 and G_view.number_of_nodes() > 0:
        adj = nx.adjacency_matrix(as_simple_undirected(G_view), weight="weight").todense()
        fig_hm = px.imshow(adj, title="Adjacency Heatmap", color_continuous_scale="Viridis")
        fig_hm.update_layout(template="plotly_dark", height=760, width=760)
        st.plotly_chart(fig_hm, use_container_width=False, key="plot_adj_heatmap")
    else:
        st.info("Матрица слишком большая для отображения (N >= 1000) или граф пуст.")
