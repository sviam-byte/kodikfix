from __future__ import annotations

import networkx as nx
import streamlit as st

from src.config import settings
from src.services.graph_service import GraphService
from src.state_models import GraphEntry
from src.ui.plots.scene3d import make_energy_flow_figure_3d

# keep legacy helper names used in the tab body
_layout_cached = GraphService.compute_layout3d
_energy_frames_cached = GraphService.compute_energy_frames


def render(G_view: nx.Graph | None, active_entry: GraphEntry, seed_val: int, src_col: str, dst_col: str, min_conf: float, min_weight: float, analysis_mode: str) -> None:
    """Render the Energy & Dynamics tab."""
    st.header("⚡ Динамика и распространение (Energy Flow)")

    if G_view is None:
        st.info("Сначала загрузите граф в сайдбаре (Load graph).")
        return

    # --- БЛОК 1: МОДЕЛЬ И ИСТОЧНИКИ ---
    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("1. Физика процесса")
        flow_mode_ui = st.selectbox(
            "Тип распространения",
            ["phys", "rw", "evo"],
            help="Phys: давление/поток (как вода). RW: диффузия (как газ).",
        )
        rw_impulse = st.toggle("Импульсный режим (всплеск)", value=True)

        # Логика источников с пояснением.
        if "energy_sources" not in st.session_state:
            st.session_state["energy_sources"] = []

        sources_ui = st.multiselect(
            "Источники (откуда течет)",
            options=list(G_view.nodes()),
            default=st.session_state.get("energy_sources", []),
            key="src_select",
        )
        st.session_state["energy_sources"] = sources_ui

        # Вычисляем и показываем авто-источник, если список пуст.
        final_sources = list(sources_ui)
        if not final_sources:
            # Быстрый расчет "сильного" узла для UI.
            deg = dict(G_view.degree(weight="weight"))
            auto_src = max(deg, key=deg.get)
            st.info(f"🤖 Авто-выбор источника: узел **{auto_src}** (max strength)")

    with c2:
        st.subheader("2. Параметры потока")
        if flow_mode_ui == "phys":
            phys_inj = st.slider(
                "Сила впрыска (Injection)",
                0.1,
                5.0,
                settings.DEFAULT_INJECTION,
                0.1,
            )
            phys_leak = st.slider("Утечка (Leak)", 0.0, 0.1, settings.DEFAULT_LEAK, 0.001)
            phys_cap = st.selectbox("Емкость узлов", ["strength", "degree"])
            st.session_state["__phys_injection"] = phys_inj
            st.session_state["__phys_leak"] = phys_leak
            st.session_state["__phys_cap"] = phys_cap
        else:
            st.info("Для RW/Evo параметров меньше.")

        flow_steps = st.slider("Длительность (шаги)", 10, 200, 50)

    st.markdown("---")

    # --- БЛОК 2: ВИЗУАЛИЗАЦИЯ ---
    st.subheader("🎨 Настройка Вида (Сделай красиво)")

    vc1, vc2, vc3 = st.columns(3)
    with vc1:
        # Важный слайдер для "замедления".
        anim_duration = st.slider(
            "Скорость анимации (мс/кадр)",
            50,
            1000,
            settings.ANIMATION_DURATION_MS,
            50,
            help="Больше = медленнее. Позволяет вращать граф во время полета.",
        )
        vis_contrast = st.slider("Яркость (Gamma)", 1.0, 10.0, 4.5)
    with vc2:
        node_size_energy = st.slider("Размер узлов", 2, 20, 7)
        vis_clip = st.slider("Срез пиков (Clip)", 0.0, 0.5, 0.05)
    with vc3:
        edge_subset_mode = st.selectbox("Отрисовка связей", ["top_flux", "top_weight", "all"], index=0)
        max_edges_viz = st.slider("Макс. кол-во ребер", 100, 5000, 1500)
        max_nodes_viz = st.slider("Макс. узлов", 500, 20000, 6000, step=500)

    # КНОПКА ЗАПУСКА
    if st.button("🔥 ЗАПУСТИТЬ СИМУЛЯЦИЮ", type="primary", use_container_width=True):
        bar = st.progress(0.0)
        stage = st.empty()
        with st.spinner("Моделирование физики..."):
            # Layout.
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
            bar.progress(0.35)

            # Simulation.
            stage.caption("Simulating energy flow...")
            src_key = tuple(final_sources) if final_sources else tuple()

            # Параметры физики берем из стейта или дефолтов.
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
                damping=settings.DEFAULT_DAMPING,  # Дефолт.
                sources=src_key,
                phys_injection=inj_val,
                phys_leak=leak_val,
                phys_cap_mode=cap_val,
                rw_impulse=bool(rw_impulse),
            )
            bar.progress(0.75)

            # Rendering.
            stage.caption("Rendering Plotly frames...")
            try:
                fig_flow = make_energy_flow_figure_3d(
                    G_view,
                    pos3d_local,
                    steps=int(flow_steps),
                    node_frames=node_frames,
                    edge_frames=edge_frames,
                    # Передаем параметры визуализации (часть из них игнорируется внутри plotter).
                    node_size=int(node_size_energy),
                    vis_contrast=float(vis_contrast),
                    vis_clip=float(vis_clip),
                    anim_duration=int(anim_duration),
                    max_edges_viz=int(max_edges_viz),
                    max_nodes_viz=int(max_nodes_viz),
                    edge_subset_mode=str(edge_subset_mode),
                )
            except Exception as e:
                # Streamlit Cloud иногда редактирует текст ошибки. Покажем тип/сообщение явно.
                st.error(f"Energy 3D render failed: {type(e).__name__}: {e}")
                st.exception(e)
                bar.empty(); stage.empty()
                return

        bar.progress(1.0)
        bar.empty(); stage.empty()
        st.plotly_chart(fig_flow, use_container_width=True, key="plot_energy_flow")
