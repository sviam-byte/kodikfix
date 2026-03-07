from __future__ import annotations

import networkx as nx
import pandas as pd
import streamlit as st

from src.config import settings
from src.exporters import export_energy_tables_csv_zip, export_energy_tables_xlsx
from src.services.graph_service import GraphService
from src.state_models import GraphEntry
from src.ui.plots.scene3d import make_energy_flow_figure_3d

_layout_cached = GraphService.compute_layout3d
_energy_frames_cached = GraphService.compute_energy_frames
_energy_tables_cached = GraphService.compute_energy_export_tables


def render(G_view: nx.Graph | None, active_entry: GraphEntry, seed_val: int, src_col: str, dst_col: str, min_conf: float, min_weight: float, analysis_mode: str) -> None:
    """Render the Energy & Dynamics tab."""
    st.header("⚡ Динамика и распространение (Energy Flow)")

    if G_view is None:
        st.info("Сначала загрузите граф в сайдбаре (Load graph).")
        return

    ui_epoch = int(st.session_state.get("__active_graph_ui_epoch", 0))

    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("1. Физика процесса")
        flow_mode_ui = st.selectbox(
            "Тип распространения",
            ["phys", "rw", "evo"],
            help="Phys: давление/поток (как вода). RW: диффузия (как газ).",
            key=f"energy_flow_mode_{active_entry.id}_{ui_epoch}",
        )
        rw_impulse = st.toggle("Импульсный режим (всплеск)", value=True)

        if "energy_sources" not in st.session_state:
            st.session_state["energy_sources"] = []

        sources_ui = st.multiselect(
            "Источники (откуда течет)",
            options=list(G_view.nodes()),
            default=[x for x in st.session_state.get("energy_sources", []) if x in G_view.nodes()],
            key="src_select",
        )
        st.session_state["energy_sources"] = sources_ui

        final_sources = list(sources_ui)
        if not final_sources:
            deg = dict(G_view.degree(weight="weight"))
            auto_src = max(deg, key=deg.get)
            st.info(f"🤖 Авто-выбор источника: узел **{auto_src}** (max strength)")

    with c2:
        st.subheader("2. Параметры потока")
        if flow_mode_ui == "phys":
            phys_inj = st.slider("Сила впрыска (Injection)", 0.1, 5.0, settings.DEFAULT_INJECTION, 0.1)
            phys_leak = st.slider("Утечка (Leak)", 0.0, 0.1, settings.DEFAULT_LEAK, 0.001)
            phys_cap = st.selectbox("Емкость узлов", ["strength", "degree"])
            st.session_state["__phys_injection"] = phys_inj
            st.session_state["__phys_leak"] = phys_leak
            st.session_state["__phys_cap"] = phys_cap
        else:
            st.info("Для RW/Evo параметров меньше.")

        flow_steps = st.slider("Длительность (шаги)", 10, 200, 50)

    st.markdown("---")
    st.subheader("🎨 Настройка Вида (Сделай красиво)")

    vc1, vc2, vc3 = st.columns(3)
    with vc1:
        anim_duration = st.slider("Скорость анимации (мс/кадр)", 50, 1000, settings.ANIMATION_DURATION_MS, 50)
        vis_contrast = st.slider("Яркость (Gamma)", 1.0, 10.0, 4.5)
    with vc2:
        node_size_energy = st.slider("Размер узлов", 2, 20, 7)
        vis_clip = st.slider("Срез пиков (Clip)", 0.0, 0.5, 0.05)
    with vc3:
        edge_subset_mode = st.selectbox("Отрисовка связей", ["top_flux", "top_weight", "all"], index=0)
        max_edges_viz = st.slider("Макс. кол-во ребер", 100, 5000, 1500)
        max_nodes_viz = st.slider("Макс. узлов", 500, 20000, 6000, step=500)

    if st.button("🔥 ЗАПУСТИТЬ СИМУЛЯЦИЮ", type="primary", use_container_width=True):
        bar = st.progress(0.0)
        stage = st.empty()
        with st.spinner("Моделирование физики..."):
            stage.caption("3D layout...")
            base_seed = int(seed_val) + int(st.session_state.get("layout_seed_bump", 0))
            pos3d_local = _layout_cached(
                active_entry.edges,
                src_col,
                dst_col,
                float(min_conf),
                float(min_weight),
                analysis_mode,
                base_seed,
            )
            bar.progress(0.25)

            stage.caption("Simulating energy flow...")
            src_key = tuple(final_sources) if final_sources else tuple()
            inj_val = float(st.session_state.get("__phys_injection", settings.DEFAULT_INJECTION))
            leak_val = float(st.session_state.get("__phys_leak", settings.DEFAULT_LEAK))
            cap_val = str(st.session_state.get("__phys_cap", "strength"))

            node_frames, edge_frames = _energy_frames_cached(
                active_entry.edges,
                src_col,
                dst_col,
                float(min_conf),
                float(min_weight),
                analysis_mode,
                steps=int(flow_steps),
                flow_mode=str(flow_mode_ui),
                damping=settings.DEFAULT_DAMPING,
                sources=src_key,
                phys_injection=inj_val,
                phys_leak=leak_val,
                phys_cap_mode=cap_val,
                rw_impulse=bool(rw_impulse),
            )
            energy_nodes_long, energy_steps_summary, energy_run_summary = _energy_tables_cached(
                active_entry.edges,
                src_col,
                dst_col,
                float(min_conf),
                float(min_weight),
                analysis_mode,
                steps=int(flow_steps),
                flow_mode=str(flow_mode_ui),
                damping=settings.DEFAULT_DAMPING,
                sources=src_key,
                phys_injection=inj_val,
                phys_leak=leak_val,
                phys_cap_mode=cap_val,
                rw_impulse=bool(rw_impulse),
            )
            bar.progress(0.72)

            stage.caption("Rendering Plotly frames...")
            try:
                fig_flow = make_energy_flow_figure_3d(
                    G_view,
                    pos3d_local,
                    steps=int(flow_steps),
                    node_frames=node_frames,
                    edge_frames=edge_frames,
                    node_size=int(node_size_energy),
                    vis_contrast=float(vis_contrast),
                    vis_clip=float(vis_clip),
                    anim_duration=int(anim_duration),
                    max_edges_viz=int(max_edges_viz),
                    max_nodes_viz=int(max_nodes_viz),
                    edge_subset_mode=str(edge_subset_mode),
                    height=860,
                )
            except TypeError:
                fig_flow = make_energy_flow_figure_3d(
                    G_view,
                    pos3d_local,
                    steps=int(flow_steps),
                    node_frames=node_frames,
                    edge_frames=edge_frames,
                    node_size=int(node_size_energy),
                    vis_contrast=float(vis_contrast),
                    vis_clip=float(vis_clip),
                    anim_duration=int(anim_duration),
                    max_edges_viz=int(max_edges_viz),
                    max_nodes_viz=int(max_nodes_viz),
                    edge_subset_mode=str(edge_subset_mode),
                )
                fig_flow.update_layout(height=860)
            except Exception as e:
                st.error(f"Energy 3D render failed: {type(e).__name__}: {e}")
                st.exception(e)
                bar.empty(); stage.empty()
                return

        st.session_state[f"energy_nodes_long__{active_entry.id}"] = energy_nodes_long
        st.session_state[f"energy_steps_summary__{active_entry.id}"] = energy_steps_summary
        st.session_state[f"energy_run_summary__{active_entry.id}"] = energy_run_summary
        bar.progress(1.0)
        bar.empty(); stage.empty()
        st.plotly_chart(fig_flow, use_container_width=True, key=f"plot_energy_flow_{active_entry.id}_{ui_epoch}")

    energy_nodes_long = st.session_state.get(f"energy_nodes_long__{active_entry.id}")
    energy_steps_summary = st.session_state.get(f"energy_steps_summary__{active_entry.id}")
    energy_run_summary = st.session_state.get(f"energy_run_summary__{active_entry.id}")
    if energy_run_summary:
        st.markdown("---")
        st.subheader("📊 Численный слой diffusion")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Final active frac", f"{float(energy_run_summary.get('final_active_frac', 0.0)):.3f}")
        m2.metric("t50", f"{float(energy_run_summary.get('time_to_50pct_nodes', float('nan'))):.1f}" if pd.notna(energy_run_summary.get("time_to_50pct_nodes")) else "—")
        m3.metric("Final entropy", f"{float(energy_run_summary.get('final_entropy', 0.0)):.3f}")
        m4.metric("Final gini", f"{float(energy_run_summary.get('final_gini', 0.0)):.3f}")
        m5, m6, m7, m8 = st.columns(4)
        m5.metric("AUC active", f"{float(energy_run_summary.get('auc_active_nodes', 0.0)):.3f}")
        m6.metric("AUC energy", f"{float(energy_run_summary.get('auc_total_energy', 0.0)):.3f}")
        m7.metric("Diffusion radius", f"{float(energy_run_summary.get('final_diffusion_radius', 0.0)):.3f}")
        m8.metric("Peak step", f"{int(energy_run_summary.get('peak_step', 0))}")

        ex1, ex2, ex3 = st.columns(3)
        ex1.download_button(
            "Скачать energy summary (.csv)",
            pd.DataFrame([energy_run_summary]).to_csv(index=False).encode("utf-8"),
            file_name=f"{active_entry.name}_energy_run_summary.csv",
            mime="text/csv",
            use_container_width=True,
        )
        ex2.download_button(
            "Скачать energy nodes long (.csv)",
            (energy_nodes_long if energy_nodes_long is not None else pd.DataFrame()).to_csv(index=False).encode("utf-8"),
            file_name=f"{active_entry.name}_energy_nodes_long.csv",
            mime="text/csv",
            use_container_width=True,
        )
        ex3.download_button(
            "Скачать energy export (.xlsx)",
            export_energy_tables_xlsx(energy_nodes_long, energy_steps_summary, energy_run_summary),
            file_name=f"{active_entry.name}_energy_export.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
        st.download_button(
            "Скачать все energy tables (.zip)",
            export_energy_tables_csv_zip(energy_nodes_long, energy_steps_summary, energy_run_summary),
            file_name=f"{active_entry.name}_energy_export.zip",
            mime="application/zip",
            use_container_width=True,
        )

        if st.checkbox("Показать численные таблицы", key=f"show_energy_tables_{active_entry.id}"):
            st.markdown("#### Energy steps summary")
            st.dataframe(energy_steps_summary, use_container_width=True, height=260)
            st.markdown("#### Energy nodes long")
            st.dataframe(energy_nodes_long, use_container_width=True, height=320)
