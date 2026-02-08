from __future__ import annotations

import textwrap
import time

import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st

import plotly.express as px
import plotly.graph_objects as go

from src.config import settings
from src.null_models import make_er_gnm, make_configuration_model, rewire_mix
from src.attacks import run_attack, run_edge_attack
from src.attacks_mix import run_mix_attack
from src.core_math import classify_phase_transition
from src.config_loader import load_metrics_info
from src.plotting import fig_metrics_over_steps, fig_compare_attacks
from src.services.graph_service import GraphService
from src.state_models import GraphEntry
from src.ui.plots.charts import (
    AUC_TRAP,
    apply_plot_defaults as _apply_plot_defaults,
    auto_y_range as _auto_y_range,
    forward_fill_heavy as _forward_fill_heavy,
)
from src.ui.plots.scene3d import make_3d_traces
from src.ui_blocks import help_icon
from src.utils import as_simple_undirected, get_node_strength

_layout_cached = GraphService.compute_layout3d


def _hash_graph(G: nx.Graph) -> str:
    """Stable hash for caching graph-derived metrics."""
    if G is None:
        return "none"
    try:
        return nx.weisfeiler_lehman_graph_hash(G, edge_attr="weight")
    except Exception:
        return f"{G.number_of_nodes()}-{G.number_of_edges()}"


@st.cache_data(show_spinner=False, hash_funcs={nx.Graph: _hash_graph})
def _cached_betweenness(G: nx.Graph) -> dict:
    """Cached betweenness centrality (costly on large graphs)."""
    return nx.betweenness_centrality(G, normalized=True)

# Загружаем справку по метрикам один раз на модуль.
_info = load_metrics_info()
METRIC_HELP = _info.get("metric_help", {})

# presets moved out of app.py
ATTACK_PRESETS_NODE = {
    "Random": {"kind": "random"},
    "Degree": {"kind": "degree"},
    "Strength": {"kind": "strength"},
    "Betweenness": {"kind": "betweenness"},
    "Closeness": {"kind": "closeness"},
    "Eigenvector": {"kind": "eigenvector"},
    "PageRank": {"kind": "pagerank"},
    "Katz": {"kind": "katz"},
    "k-core": {"kind": "kcore"},
    "Community bridge": {"kind": "community_bridge"},
}
ATTACK_PRESETS_EDGE = {
    "Random": {"kind": "edge_random"},
    "Weight": {"kind": "edge_weight"},
    "Betweenness": {"kind": "edge_betweenness"},
    "Rici (Ollivier)": {"kind": "edge_ricci"},
}

def _extract_removed_order(aux):
    if isinstance(aux, dict):
        for k in ["removed_nodes", "removed_order", "order", "removal_order", "removed"]:
            v = aux.get(k)
            if isinstance(v, (list, tuple)) and v:
                return list(v)
    if isinstance(aux, (list, tuple)) and aux:
        if not isinstance(aux[0], (pd.DataFrame, np.ndarray, dict, list, tuple)):
            return list(aux)
    return None

def _fallback_removal_order(G: nx.Graph, kind: str, seed: int):
    """
    Fallback для 3D-декомпозиции, если src.attacks не вернул порядок удаления.
    ВАЖНО: это не адаптивная атака, только визуальный fallback.
    """
    if G.number_of_nodes() == 0:
        return []

    rng = np.random.default_rng(int(seed))
    H = as_simple_undirected(G)
    nodes = list(H.nodes())

    if kind in ("random",):
        rng.shuffle(nodes)
        return nodes

    if kind in ("degree",):
        nodes.sort(key=lambda n: H.degree(n), reverse=True)
        return nodes

    if kind in ("low_degree",):  
        nodes.sort(key=lambda n: H.degree(n))
        return nodes

    if kind in ("weak_strength",): 
        nodes.sort(key=lambda n: get_node_strength(H, n))
        return nodes

    if kind in ("betweenness",):
        if H.number_of_nodes() > 5000:
            nodes.sort(key=lambda n: H.degree(n), reverse=True)
            return nodes
        b = _cached_betweenness(H)
        nodes.sort(key=lambda n: b.get(n, 0.0), reverse=True)
        return nodes

    if kind in ("kcore",):
        core = nx.core_number(H)
        nodes.sort(key=lambda n: core.get(n, 0), reverse=True)
        return nodes

    if kind in ("richclub_top",):
        nodes.sort(key=lambda n: get_node_strength(H, n), reverse=True)
        return nodes

    rng.shuffle(nodes)
    return nodes

def render_null_models(G_view: nx.Graph | None, G_full: nx.Graph | None, met: dict, active_entry: GraphEntry, seed_val: int, add_graph_callback) -> None:
    """Render the null models tab."""
    if G_view is None:
        return

    st.header("🧪 Нулевые модели и синтетика")

    nm_col1, nm_col2 = st.columns([1, 2])

    with nm_col1:
        st.subheader("Параметры")
        null_kind = st.selectbox("Тип модели", ["ER G(n,m)", "Configuration Model", "Mix/Rewire (p)"])

        mix_p = 0.0
        if null_kind == "Mix/Rewire (p)":
            mix_p = st.slider("p (rewiring probability)", 0.0, 1.0, 0.2, 0.05, help=help_icon("Mix/Rewire"))

        nm_seed = st.number_input("Seed генерации", value=int(seed_val), step=1)
        new_name_suffix = st.text_input("Суффикс имени", value="_null")

        if st.button("⚙️ Создать и добавить", type="primary"):
            with st.spinner("Генерация..."):
                if null_kind == "ER G(n,m)":
                    G_new = make_er_gnm(G_full.number_of_nodes(), G_full.number_of_edges(), seed=int(nm_seed))
                    src_tag = "ER"
                elif null_kind == "Configuration Model":
                    G_new = make_configuration_model(G_full, seed=int(nm_seed))
                    src_tag = "CFG"
                else:
                    G_new = rewire_mix(G_full, p=float(mix_p), seed=int(nm_seed))
                    src_tag = f"MIX(p={mix_p})"

                edges = [[u, v, 1.0, 1.0] for u, v in as_simple_undirected(G_new).edges()]
                df_new = pd.DataFrame(edges, columns=["src", "dst", "weight", "confidence"])

                add_graph_callback(
                    f"{active_entry.name}{new_name_suffix}",
                    df_new,
                    f"null:{src_tag}",
                    "src",
                    "dst",
                )
                st.success("Граф создан. Переключаюсь на него...")
                st.rerun()

    with nm_col2:
        st.info("Быстрая проверка против ER-ожиданий (очень грубо):")
        N = G_view.number_of_nodes()
        M = G_view.number_of_edges()
        er_density = 2 * M / (N * (N - 1)) if N > 1 else 0.0
        er_clustering = er_density

        met_light = met
        cmp_df = pd.DataFrame({
            "Metric": ["Avg Degree", "Density", "Clustering (C)", "Modularity (примерно)"],
            "Active Graph": [met_light.get("avg_degree", np.nan), met_light.get("density", np.nan), met_light.get("clustering", np.nan), met_light.get("mod", np.nan)],
            "ER Expected": [met_light.get("avg_degree", np.nan), er_density, er_clustering, "~0.0"],
        })
        st.dataframe(cmp_df, use_container_width=True)

def render_attack_lab(G_view: nx.Graph | None, active_entry: GraphEntry, seed_val: int, src_col: str, dst_col: str, min_conf: float, min_weight: float, analysis_mode: str, save_experiment_callback) -> None:
    """Render the Attack Lab tab."""
    if G_view is None:
        return

    # Совместимость по сигнатуре колбэка сохранения эксперимента.
    # В app.py сейчас: save_experiment_to_state(name, gid, kind, params, df_hist)
    # В старых/других местах мог быть вариант с keyword graph_id=...
    def _save_experiment(*, name: str, graph_id: str, kind: str, params: dict, df_hist):
        try:
            return save_experiment_callback(
                name=name,
                graph_id=graph_id,
                kind=kind,
                params=params,
                df_hist=df_hist,
            )
        except TypeError:
            # fallback на (name, gid, ...)
            return save_experiment_callback(name, graph_id, kind, params, df_hist)

    st.header("💥 Attack Lab (node + edge + weak)")

    # --------------------------
    # SINGLE RUN
    # --------------------------
    st.subheader("Single run")
    family = st.radio(
        "Тип атаки",
        ["Node (узлы)", "Edge (рёбра: слабые/сильные)", "Mix/Entropy (Hrish)"],
        horizontal=True,
    )

    col_setup, _ = st.columns([1, 2])

    with col_setup:
        with st.container(border=True):
            st.markdown("### Параметры")

            frac = st.slider("Доля удаления", 0.05, 0.95, 0.5, 0.05)
            steps = st.slider("Шаги", 5, 150, 30)
            seed_run = st.number_input("Seed", value=int(seed_val), step=1)

            with st.expander("Дополнительно"):
                eff_k = st.slider("Efficiency samples (k)", 8, 256, 32)
                heavy_freq = st.slider("Тяжёлые метрики каждые N шагов", 1, 10, 2)
                fast_mode = st.checkbox("⚡ Fast Mode (approx)", value=True, help="Сильно ускоряет расчет за счет снижения точности на промежуточных шагах.")

                tag = st.text_input("Тег", "")

            if family.startswith("Node"):
                attack_ui = st.selectbox(
                    "Стратегия (узлы)",
                    [
                        "random",
                        "degree (Hubs)",
                        "betweenness (Bridges)",
                        "kcore (Deep Core)",
                        "richclub_top (Top Strength)",
                        "low_degree (Weak nodes)",
                        "weak_strength (Weak strength)",
                    ],
                )
                kind_map = {
                    "random": "random",
                    "degree (Hubs)": "degree",
                    "betweenness (Bridges)": "betweenness",
                    "kcore (Deep Core)": "kcore",
                    "richclub_top (Top Strength)": "richclub_top",
                    "low_degree (Weak nodes)": "low_degree",
                    "weak_strength (Weak strength)": "weak_strength",
                }
                kind = kind_map.get(attack_ui, "random")

            elif family.startswith("Edge"):
                attack_ui = st.selectbox(
                    "Стратегия (рёбра)",
                    [
                        "weak_edges_by_weight",
                        "weak_edges_by_confidence",
                        "strong_edges_by_weight",
                        "strong_edges_by_confidence",
                        "ricci_most_negative (κ min)",
                        "ricci_most_positive (κ max)",
                        "ricci_abs_max (|κ| max)",
                        "flux_high_rw",
                        "flux_high_evo",
                        "flux_high_rw_x_neg_ricci",
                    ],
                    help=help_icon("Weak edges")
                )
                kind = str(attack_ui).split(" ")[0]

            else:
                kind = st.selectbox(
                    "Режим Hrish",
                    [
                        "hrish_mix",
                        "mix_degree_preserving",
                        "mix_weightconf_preserving",
                    ],
                    help="hrish_mix = rewire (degree-preserving) + replace из нулевой модели.",
                )
                replace_from = st.selectbox("Replace source", ["ER", "CFG"], index=0)
                alpha_rewire = st.slider("alpha (rewire)", 0.0, 1.0, 0.6, 0.05)
                beta_replace = st.slider("beta (replace)", 0.0, 1.0, 0.4, 0.05)
                swaps_per_edge = st.slider("swaps_per_edge", 0.0, 3.0, 0.5, 0.1)
                st.caption("Ось X здесь: mix_frac (0..1), а не removed_frac.")

            if st.button("🚀 RUN", type="primary", use_container_width=True):
                if family.startswith("Mix/Entropy"):
                    with st.spinner(f"Mix attack: {kind}"):
                        bar = st.progress(0.0)
                        msg = st.empty()

                        def _cb(i, total, x=None):
                            frac_ = 0.0 if total <= 0 else min(1.0, max(0.0, i / total))
                            bar.progress(frac_)
                            if x is not None:
                                msg.caption(f"mix: {i}/{total}  mix_frac={x:.3f}")

                        df_hist, aux = run_mix_attack(
                            G_view,
                            kind=str(kind),
                            steps=int(steps),
                            seed=int(seed_run),
                            eff_sources_k=int(eff_k),
                            heavy_every=int(heavy_freq),
                            alpha_rewire=float(alpha_rewire),
                            beta_replace=float(beta_replace),
                            swaps_per_edge=float(swaps_per_edge),
                            replace_from=str(replace_from),
                            progress_cb=_cb,
                            fast_mode=fast_mode,
                        )
                        bar.empty(); msg.empty()
                        df_hist = _forward_fill_heavy(df_hist)
                        phase_info = classify_phase_transition(
                            df_hist.rename(columns={"mix_frac": "removed_frac"})
                        )

                        label = f"{active_entry.name} | mix:{kind} | seed={seed_run}"
                        if tag:
                            label += f" [{tag}]"

                        _save_experiment(
                            name=label,
                            graph_id=active_entry.id,
                            kind=str(kind),
                            params={
                                "attack_family": "mix",
                                "steps": int(steps),
                                "seed": int(seed_run),
                                "phase": phase_info,
                                "eff_k": int(eff_k),
                                "heavy_every": int(heavy_freq),
                                **aux,
                            },
                            df_hist=df_hist,
                        )
                    st.success("Готово.")
                    st.rerun()

                if family.startswith("Node"):
                    with st.spinner(f"Node attack: {kind}"):
                        # TODO: stabilize the progress indicator updates in Streamlit.
                        bar = st.progress(0.0)
                        msg = st.empty()

                        def _cb(i, total, k=None):
                            frac_ = 0.0 if total <= 0 else min(1.0, max(0.0, i / total))
                            bar.progress(frac_)
                            if k is not None:
                                msg.caption(f"node attack: {i}/{total}  target_k={k}")

                        df_hist, aux = run_attack(
                            G_view, kind, float(frac), int(steps), int(seed_run), int(eff_k),
                            rc_frac=0.1, compute_heavy_every=int(heavy_freq),
                            progress_cb=_cb,
                        )
                        bar.empty(); msg.empty()
                        df_hist = _forward_fill_heavy(df_hist)
                        removed_order = _extract_removed_order(aux) or _fallback_removal_order(G_view, kind, int(seed_run))
                        phase_info = classify_phase_transition(df_hist)

                        label = f"{active_entry.name} | node:{kind} | seed={seed_run}"
                        if tag:
                            label += f" [{tag}]"

                        _save_experiment(
                            name=label,
                            graph_id=active_entry.id,
                            kind=kind,
                            params={
                                "attack_family": "node",
                                "frac": float(frac),
                                "steps": int(steps),
                                "seed": int(seed_run),
                                "phase": phase_info,
                                "compute_heavy_every": int(heavy_freq),
                                "eff_k": int(eff_k),
                                "removed_order": removed_order,
                                "mode": "src_run_attack_or_fallback",
                            },
                            df_hist=df_hist
                        )
                    st.success("Готово.")
                    st.rerun()

                else:
                    with st.spinner(f"Edge attack: {kind}"):
                        bar = st.progress(0.0)
                        msg = st.empty()

                        def _cb(i, total, k=None):
                            # i=0..total; на больших графах это прям спасает психику
                            frac_ = 0.0 if total <= 0 else min(1.0, max(0.0, i / total))
                            bar.progress(frac_)
                            if k is not None:
                                msg.caption(f"edge attack: {i}/{total}  target_edges={k}")

                        df_hist, aux = run_edge_attack(
                            G_view, kind, float(frac), int(steps), int(seed_run), int(eff_k),
                            compute_heavy_every=int(heavy_freq),
                            compute_curvature=bool(st.session_state.get("__compute_curvature", False)),
                            curvature_sample_edges=int(st.session_state.get("__curvature_sample_edges", 80)),
                            progress_cb=_cb,
                        )
                        bar.empty(); msg.empty()
                        df_hist = _forward_fill_heavy(df_hist)
                        phase_info = classify_phase_transition(df_hist)

                        label = f"{active_entry.name} | edge:{kind} | seed={seed_run}"
                        if tag:
                            label += f" [{tag}]"

                        _save_experiment(
                            name=label,
                            graph_id=active_entry.id,
                            kind=kind,
                            params={
                                "attack_family": "edge",
                                "frac": float(frac),
                                "steps": int(steps),
                                "seed": int(seed_run),
                                "phase": phase_info,
                                "compute_heavy_every": int(heavy_freq),
                                "eff_k": int(eff_k),
                                "removed_edges_order": aux.get("removed_edges_order", []),
                                "total_edges": aux.get("total_edges", None),
                            },
                            df_hist=df_hist
                        )
                    st.success("Готово.")
                    st.rerun()

    st.markdown("---")
    st.markdown("## Последний результат (для текущего графа)")

    exps_here = [e for e in st.session_state["experiments"] if e.graph_id == active_entry.id]
    if not exps_here:
        st.info("Нет экспериментов. Запусти сверху.")
    else:
        exps_here.sort(key=lambda x: x.created_at, reverse=True)
        last_exp = exps_here[0]
        df_res = _forward_fill_heavy(last_exp.history.copy())
        params = last_exp.params or {}
        fam = params.get("attack_family", "node")
        xcol = "mix_frac" if fam == "mix" and "mix_frac" in df_res.columns else "removed_frac"

        ph = last_exp.params.get("phase", {}) if last_exp.params else {}
        if ph:
            st.caption(
                f"Phase: {'🔥 Abrupt' if ph.get('is_abrupt') else '🌊 Continuous'}"
                f" | critical_x ≈ {float(ph.get('critical_x', 0.0)):.3f}"
            )

        attack_tabs = ["📉 Curves", "🌀 Phase views", "🧊 3D step-by-step"]
        # Stateful selector avoids tab resets when animation uses st.rerun().
        selected_attack_tab = st.radio(
            "Просмотр результатов",
            attack_tabs,
            horizontal=True,
            key="attack_results_tab",
        )

        if selected_attack_tab == attack_tabs[0]:
            with st.expander("❔ Что означают метрики на графиках", expanded=False):
                st.markdown(
                    "- **lcc_frac**: доля узлов в гигантской компоненте (порядковый параметр перколяции)\n"
                    "- **eff_w**: глобальная эффективность (в среднем насколько короткие пути; выше = сеть “связнее”)\n"
                    "- **l2_lcc**: λ₂ (алгебраическая связность) для LCC; близко к 0 = “на грани распада”\n"
                    "- **mod**: модульность сообществ; рост часто означает фрагментацию на кластеры\n"
                    "- **H_***: энтропии распределений (рост “случайности” структуры)\n"
                )
            fig = fig_metrics_over_steps(
                df_res,
                title="Метрики по шагам",
                normalize_mode=st.session_state["norm_mode"],
                height=st.session_state["plot_height"],
            )
            fig.update_layout(template="plotly_dark")
            fig.update_traces(mode="lines+markers")
            fig.update_traces(line_width=3)
            fig = _apply_plot_defaults(fig, height=st.session_state["plot_height"])
            st.plotly_chart(fig, use_container_width=True, key="plot_attack_metrics")

            st.markdown("#### AUC (robustness) по выбранной метрике")
            y_axis = st.selectbox(
                "Метрика для AUC",
                [c for c in ["lcc_frac", "eff_w", "l2_lcc", "mod", "H_deg", "H_w", "H_conf", "H_tri"] if c in df_res.columns],
                index=0,
                key="auc_y_single",
            )
            st.caption(METRIC_HELP.get(y_axis, ""))

            if y_axis in df_res.columns and xcol in df_res.columns:
                xs = pd.to_numeric(df_res[xcol], errors="coerce")
                ys = pd.to_numeric(df_res[y_axis], errors="coerce")
                mask = xs.notna() & ys.notna()
                if mask.sum() >= 2:
                    auc_val = float(AUC_TRAP(ys[mask].to_numpy(), xs[mask].to_numpy()))
                    st.metric("AUC", f"{auc_val:.6f}")
                else:
                    st.info("Недостаточно точек для AUC.")

            with st.expander("❓ Что на этих графиках", expanded=False):
                txt = """
                Ось X:
                  - removed_frac: доля удалённых узлов/рёбер (атаки).
                  - mix_frac: уровень энтропизации (Hrish mix), 0..1.

                Ось Y:
                  - lcc_frac: доля LCC (перколяция).
                  - eff_w: эффективность (качество глобальной связности путей).
                  - l2_lcc: λ₂ (спектральная связность LCC).
                  - mod: модульность (структура сообществ).
                  - H_*: энтропии распределений (рост “случайности”).
                """
                st.text(textwrap.dedent(txt).strip())

        elif selected_attack_tab == attack_tabs[1]:
            if xcol in df_res.columns and "lcc_frac" in df_res.columns:
                fig_lcc = px.line(df_res, x=xcol, y="lcc_frac", title="Order parameter: LCC fraction vs removed fraction")
                fig_lcc.update_layout(template="plotly_dark")
                fig_lcc = _apply_plot_defaults(fig_lcc, height=780, y_range=_auto_y_range(df_res["lcc_frac"]))
                st.plotly_chart(fig_lcc, use_container_width=True, key="plot_phase_lcc")

            if xcol in df_res.columns and "lcc_frac" in df_res.columns:
                dfp = df_res.sort_values(xcol).copy()
                dx = pd.to_numeric(dfp[xcol], errors="coerce").diff()
                dy = pd.to_numeric(dfp["lcc_frac"], errors="coerce").diff()
                dfp["suscep"] = (dy / dx).replace([np.inf, -np.inf], np.nan)
                fig_s = px.line(dfp, x=xcol, y="suscep", title="Susceptibility proxy: d(LCC)/dx")
                fig_s.update_layout(template="plotly_dark")
                fig_s = _apply_plot_defaults(fig_s, height=780, y_range=_auto_y_range(dfp["suscep"]))
                st.plotly_chart(fig_s, use_container_width=True, key="plot_phase_suscep")

            if "mod" in df_res.columns and "l2_lcc" in df_res.columns:
                dfp2 = df_res.copy()
                dfp2["mod"] = pd.to_numeric(dfp2["mod"], errors="coerce")
                dfp2["l2_lcc"] = pd.to_numeric(dfp2["l2_lcc"], errors="coerce")
                dfp2 = dfp2.dropna(subset=["mod", "l2_lcc"])
                if not dfp2.empty:
                    fig_phase = px.line(dfp2, x="l2_lcc", y="mod", title="Phase portrait (trajectory): Q vs λ₂")
                    fig_phase.update_layout(template="plotly_dark")
                    fig_phase = _apply_plot_defaults(fig_phase, height=780)
                    st.plotly_chart(fig_phase, use_container_width=True, key="plot_phase_portrait")

        elif selected_attack_tab == attack_tabs[2]:
            edge_overlay_ui = st.selectbox(
                "Разметка рёбер (3D step-by-step)",
                [
                    "Ricci sign (κ<0/κ>0)",
                    "Energy flux (RW)",
                    "Energy flux (Demetrius)",
                    "Weight (log10)",
                    "Confidence",
                    "None",
                ],
                index=0,
                key="edge_overlay_tabc",
            )
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

            base_seed = int(seed_val) + int(st.session_state.get("layout_seed_bump", 0))
            pos_base = _layout_cached(
                active_entry.edges,
                src_col,
                dst_col,
                float(min_conf),
                float(min_weight),
                analysis_mode,
                base_seed,
            )

            if fam == "mix":
                st.info("Для Mix/Entropy 3D-декомпозиция не поддерживается (нет порядка удаления).")
            elif fam == "node":
                removed_order = params.get("removed_order") or []
                if not removed_order:
                    st.warning("Нет removed_order для 3D. (src.run_attack не дал, а fallback не сохранился.)")
                else:
                    max_steps = max(1, len(df_res) - 1)
                    step_val = st.slider(
                        "Шаг (3D)",
                        0,
                        max_steps,
                        int(st.session_state.get("__decomp_step", 0)),
                        key="__decomp_step_slider",
                    )
                    st.session_state["__decomp_step"] = int(step_val)

                    play = st.toggle("▶ Play", value=False, key="play3d")
                    fps = st.slider("FPS", 1, 10, 3, key="fps3d")

                    frac_here = float(df_res.iloc[int(step_val)]["removed_frac"]) if "removed_frac" in df_res.columns else (step_val / max_steps)
                    k_remove = int(round(frac_here * G_view.number_of_nodes()))
                    k_remove = max(0, min(k_remove, len(removed_order)))

                    removed_set = set(removed_order[:k_remove])
                    H = as_simple_undirected(G_view).copy()
                    H.remove_nodes_from([n for n in removed_set if H.has_node(n)])

                    pos_k = {n: pos_base[n] for n in H.nodes() if n in pos_base}
                    edge_traces, node_trace = make_3d_traces(
                        H,
                        pos_k,
                        show_scale=True,
                        edge_overlay=edge_overlay,
                        flow_mode=flow_mode,
                    )

                    if node_trace is not None:
                        fig = go.Figure(data=[*edge_traces, node_trace])
                        fig.update_layout(template="plotly_dark", height=860, showlegend=False)
                        fig.update_layout(title=f"Node removal | step={step_val}/{max_steps} | removed~{k_remove} | frac={frac_here:.3f}")
                        st.plotly_chart(fig, use_container_width=True, key="plot_attack_3d_node_step")
                    else:
                        st.info("На этом шаге граф пуст.")

                    if play:
                        time.sleep(1.0 / float(fps))
                        nxt = int(step_val) + 1
                        if nxt > max_steps:
                            nxt = 0
                        st.session_state["__decomp_step"] = nxt
                        st.rerun()

            else:
                removed_edges_order = params.get("removed_edges_order") or []
                total_edges = params.get("total_edges") or len(as_simple_undirected(G_view).edges())
                if not removed_edges_order:
                    st.warning("Нет removed_edges_order для 3D.")
                else:
                    max_steps = max(1, len(df_res) - 1)
                    step_val = st.slider(
                        "Шаг (3D)",
                        0,
                        max_steps,
                        int(st.session_state.get("__decomp_step", 0)),
                        key="__decomp_step_slider_edge",
                    )
                    st.session_state["__decomp_step"] = int(step_val)

                    play = st.toggle("▶ Play", value=False, key="play3d_edge")
                    fps = st.slider("FPS", 1, 10, 3, key="fps3d_edge")

                    frac_here = float(df_res.iloc[int(step_val)]["removed_frac"]) if "removed_frac" in df_res.columns else (step_val / max_steps)
                    k_remove = int(round(frac_here * float(total_edges)))
                    k_remove = max(0, min(k_remove, len(removed_edges_order)))

                    H = as_simple_undirected(G_view).copy()
                    for (u, v) in removed_edges_order[:k_remove]:
                        if H.has_edge(u, v):
                            H.remove_edge(u, v)

                    pos_k = {n: pos_base[n] for n in H.nodes() if n in pos_base}
                    edge_traces, node_trace = make_3d_traces(
                        H,
                        pos_k,
                        show_scale=True,
                        edge_overlay=edge_overlay,
                        flow_mode=flow_mode,
                    )

                    if node_trace is not None:
                        fig = go.Figure(data=[*edge_traces, node_trace])
                        fig.update_layout(template="plotly_dark", height=860, showlegend=False)
                        fig.update_layout(title=f"Edge removal | step={step_val}/{max_steps} | removed~{k_remove} edges | frac={frac_here:.3f}")
                        st.plotly_chart(fig, use_container_width=True, key="plot_attack_3d_edge_step")
                    else:
                        st.info("На этом шаге граф пуст.")

                    if play:
                        time.sleep(1.0 / float(fps))
                        nxt = int(step_val) + 1
                        if nxt > max_steps:
                            nxt = 0
                        st.session_state["__decomp_step"] = nxt
                        st.rerun()

    st.markdown("---")

    # --------------------------
    # PRESET BATCH (same graph)
    # --------------------------
    st.subheader("Preset batch (на одном графе)")
    bcol1, bcol2 = st.columns([1, 2])

    with bcol1:
        batch_family = st.radio("Batch тип", ["Node presets", "Edge presets"], horizontal=True, key="batch_family")

        if batch_family.startswith("Node"):
            preset_name = st.selectbox("Preset", list(ATTACK_PRESETS_NODE.keys()), key="preset_node")
            preset = ATTACK_PRESETS_NODE[preset_name]
        else:
            preset_name = st.selectbox("Preset", list(ATTACK_PRESETS_EDGE.keys()), key="preset_edge")
            preset = ATTACK_PRESETS_EDGE[preset_name]

        frac_b = st.slider("Доля удаления (batch)", 0.05, 0.95, 0.5, 0.05, key="batch_frac")
        steps_b = st.slider("Шаги (batch)", 5, 150, 30, key="batch_steps")
        seed_b = st.number_input("Base seed (batch)", value=123, step=1, key="batch_seed")

        with st.expander("Batch advanced"):
            eff_k_b = st.slider("Efficiency k", 8, 256, 32, key="batch_effk")
            heavy_b = st.slider("Heavy every N", 1, 10, 2, key="batch_heavy")
            tag_b = st.text_input("Тег batch", "", key="batch_tag")

        if st.button("🚀 RUN PRESET SUITE", type="primary", use_container_width=True, key="run_suite"):
            with st.spinner(f"Running preset: {preset_name}"):
                if batch_family.startswith("Node"):
                    curves = run_node_attack_suite(
                        G_view, active_entry, preset,
                        frac=float(frac_b), steps=int(steps_b), base_seed=int(seed_b),
                        eff_k=int(eff_k_b), heavy_freq=int(heavy_b),
                        rc_frac=0.1, tag=tag_b
                    )
                else:
                    curves = run_edge_attack_suite(
                        G_view, active_entry, preset,
                        frac=float(frac_b), steps=int(steps_b), base_seed=int(seed_b),
                        eff_k=int(eff_k_b), heavy_freq=int(heavy_b),
                        tag=tag_b
                    )

            st.session_state["last_suite_curves"] = curves
            st.success(f"Готово: {len(curves)} прогонов сохранено.")
            st.rerun()

    with bcol2:
        curves = st.session_state.get("last_suite_curves")
        if curves:
            st.markdown("### Сравнение suite")
            y_axis = st.selectbox("Y", ["lcc_frac", "eff_w", "l2_lcc", "mod"], index=0, key="suite_y")
            fig = fig_compare_attacks(
                curves,
                "removed_frac",
                y_axis,
                f"Suite compare: {y_axis}",
                normalize_mode=st.session_state["norm_mode"],
                height=st.session_state["plot_height"],
            )
            fig.update_layout(template="plotly_dark")
            all_y = pd.concat([pd.to_numeric(df[y_axis], errors="coerce") for _, df in curves if y_axis in df.columns], ignore_index=True)
            fig = _apply_plot_defaults(fig, height=st.session_state["plot_height"], y_range=_auto_y_range(all_y))
            st.plotly_chart(fig, use_container_width=True, key="plot_suite_compare")

            st.markdown("#### AUC ranking")
            rows = []
            for name, df in curves:
                if "removed_frac" in df.columns and y_axis in df.columns:
                    xs = pd.to_numeric(df["removed_frac"], errors="coerce")
                    ys = pd.to_numeric(df[y_axis], errors="coerce")
                    mask = xs.notna() & ys.notna()
                    if mask.sum() >= 2:
                        rows.append({"run": name, "AUC": float(AUC_TRAP(ys[mask].to_numpy(), xs[mask].to_numpy()))})
            if rows:
                df_auc = pd.DataFrame(rows).sort_values("AUC", ascending=False)
                st.dataframe(df_auc, use_container_width=True)
        else:
            st.info("Запусти suite слева, чтобы увидеть сравнение.")

    st.markdown("---")

    # --------------------------
    # MULTI-GRAPH BATCH
    # --------------------------
    st.subheader("Multi-graph batch (на нескольких графах)")
    graphs = st.session_state["graphs"]
    gid_list = list(graphs.keys())

    mg_col1, mg_col2 = st.columns([1, 2])

    with mg_col1:
        mg_family = st.radio("Multi тип", ["Node presets", "Edge presets"], horizontal=True, key="mg_family")

        sel_gids = st.selectbox(
            "Графы (multi) — выбери несколько в списке ниже",
            options=["(выбрать ниже)"],
            index=0,
            help="Основной выбор — в multiselect ниже"
        )

        sel_gids = st.multiselect(
            "Выбери графы",
            gid_list,
            default=[st.session_state["active_graph_id"]] if st.session_state["active_graph_id"] else [],
            format_func=lambda gid: f"{graphs[gid].name} ({graphs[gid].source})",
            key="mg_gids"
        )

        if mg_family.startswith("Node"):
            preset_name_mg = st.selectbox("Preset (multi)", list(ATTACK_PRESETS_NODE.keys()), key="mg_preset_node")
            preset_mg = ATTACK_PRESETS_NODE[preset_name_mg]
        else:
            preset_name_mg = st.selectbox("Preset (multi)", list(ATTACK_PRESETS_EDGE.keys()), key="mg_preset_edge")
            preset_mg = ATTACK_PRESETS_EDGE[preset_name_mg]

        mg_frac = st.slider("Доля удаления", 0.05, 0.95, 0.5, 0.05, key="mg_frac")
        mg_steps = st.slider("Шаги", 5, 150, 30, key="mg_steps")
        mg_seed = st.number_input("Base seed", value=321, step=1, key="mg_seed")

        with st.expander("Multi advanced"):
            mg_effk = st.slider("Efficiency k", 8, 256, 32, key="mg_effk")
            mg_heavy = st.slider("Heavy every N", 1, 10, 2, key="mg_heavy")
            mg_tag = st.text_input("Тег multi", "", key="mg_tag")

        if st.button("🚀 RUN MULTI-GRAPH SUITE", type="primary", use_container_width=True, key="run_mg"):
            if not sel_gids:
                st.error("Выбери хотя бы один граф.")
            else:
                all_curves = []
                with st.spinner("Running multi-graph suite..."):
                    for gid in sel_gids:
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

                        if mg_family.startswith("Node"):
                            curves = run_node_attack_suite(
                                _G, entry, preset_mg,
                                frac=float(mg_frac), steps=int(mg_steps),
                                base_seed=int(mg_seed), eff_k=int(mg_effk),
                                heavy_freq=int(mg_heavy),
                                rc_frac=0.1,
                                tag=f"MG:{mg_tag}"
                            )
                        else:
                            curves = run_edge_attack_suite(
                                _G, entry, preset_mg,
                                frac=float(mg_frac), steps=int(mg_steps),
                                base_seed=int(mg_seed), eff_k=int(mg_effk),
                                heavy_freq=int(mg_heavy),
                                tag=f"MG:{mg_tag}"
                            )

                        all_curves.extend(curves)

                st.session_state["last_multi_curves"] = all_curves
                st.success(f"Готово: {len(all_curves)} прогонов.")
                st.rerun()

    with mg_col2:
        multi_curves = st.session_state.get("last_multi_curves")
        if multi_curves:
            st.markdown("### Multi сравнение")
            y = st.selectbox("Y (multi)", ["lcc_frac", "eff_w", "l2_lcc", "mod"], index=0, key="mg_y")
            fig = fig_compare_attacks(
                multi_curves,
                "removed_frac",
                y,
                f"Multi compare: {y}",
                normalize_mode=st.session_state["norm_mode"],
                height=st.session_state["plot_height"],
            )
            fig.update_layout(template="plotly_dark")
            all_y = pd.concat([pd.to_numeric(df[y], errors="coerce") for _, df in multi_curves if y in df.columns], ignore_index=True)
            fig = _apply_plot_defaults(fig, height=st.session_state["plot_height"], y_range=_auto_y_range(all_y))
            st.plotly_chart(fig, use_container_width=True, key="plot_multi_compare")
        else:
            st.info("Запусти multi suite слева, чтобы увидеть сравнение.")
