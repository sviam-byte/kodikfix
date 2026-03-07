from __future__ import annotations

import numpy as np
import streamlit as st
import plotly.express as px

from .config_loader import load_css, load_metrics_info


def help_icon(key: str) -> str:
    info = load_metrics_info()
    return info.get("help_text", {}).get(key, "")


def inject_custom_css() -> None:
    css = load_css()
    if not css:
        return
    st.markdown(f"<style>\n{css}\n</style>", unsafe_allow_html=True)


def fmt(val, precision=4, is_percent=False):
    if val is None:
        return "—"
    try:
        f = float(val)
    except (TypeError, ValueError):
        return str(val)

    if not np.isfinite(f):
        return "—"

    if is_percent:
        return f"{f * 100:.1f}%"

    # Handle negative zero.
    if f == 0.0:
        f = 0.0

    return f"{f:.{precision}f}"


def render_dashboard_metrics(G_view, met: dict) -> None:
    with st.container(border=True):
        st.markdown("#### 📐 Основные параметры")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("N (Nodes)", met.get("N", G_view.number_of_nodes()), help=help_icon("N"))
        k2.metric("E (Edges)", met.get("E", G_view.number_of_edges()), help=help_icon("E"))
        k3.metric("Density", fmt(met.get("density"), 6), help=help_icon("Density"))
        k4.metric("Avg Degree", fmt(met.get("avg_degree"), 2))

    with st.container(border=True):
        st.markdown("#### 🔗 Связность и пути")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Components", met.get("C", "N/A"))
        c2.metric(
            "LCC Size",
            fmt(met.get("lcc_size"), 0),
            fmt(met.get("lcc_frac"), is_percent=True),
            help=help_icon("LCC frac"),
        )
        c3.metric("Diameter (approx)", fmt(met.get("diameter_approx"), 0))
        c4.metric("Efficiency", fmt(met.get("eff_w")), help=help_icon("Efficiency"))

    with st.container(border=True):
        st.markdown("#### 🕸️ Топология и Спектр")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Modularity Q", fmt(met.get("mod")), help=help_icon("Modularity Q"))
        m2.metric("Lambda2 (LCC)", fmt(met.get("l2_lcc"), 6), help=help_icon("Lambda2"))
        m3.metric("Assortativity", fmt(met.get("assortativity")), help=help_icon("Assortativity"))
        m4.metric("Clustering", fmt(met.get("clustering")), help=help_icon("Clustering"))

    with st.container(border=True):
        st.markdown("#### 🎲 Энтропия и Устойчивость")
        e1, e2, e3 = st.columns(3)
        e1.metric("H_deg", fmt(met.get("H_deg")), help=help_icon("H_deg"))
        e2.metric("H_w", fmt(met.get("H_w")), help=help_icon("H_w"))
        e3.metric("H_conf", fmt(met.get("H_conf")), help=help_icon("H_conf"))

        with st.expander("❔", expanded=False):
            st.markdown(
                "- **H_deg**: насколько разнообразны роли узлов (иерархия vs распределённость)\n"
                "- **H_w**: насколько «тонко» настроены силы связей (разнообразие весов)\n"
                "- **H_conf**: неоднородность/надёжность структуры (по confidence)\n"
            )

        st.divider()
        a1, a2, a3 = st.columns(3)
        a1.metric("τ (Relaxation)", fmt(met.get("tau_relax")), help=help_icon("tau_relax"))
        a2.metric("β (Redundancy)", fmt(met.get("beta_red")), help=help_icon("beta_red"))
        a3.metric("1/λ_max (Epi thr)", fmt(met.get("epi_thr")), help=help_icon("epi_thr"))

    st.subheader("🧭 Геометрия / робастность")

    g1, g2, g3, g4 = st.columns(4)
    g1.metric("H_rw (entropy rate)", fmt(met.get("H_rw")), help=help_icon("H_rw"))
    g2.metric("H_evo (Demetrius)", fmt(met.get("H_evo")), help=help_icon("H_evo"))
    g3.metric("κ̄ (mean Ricci)", fmt(met.get("kappa_mean")), help=help_icon("kappa_mean"))
    g4.metric(
        "% κ<0",
        fmt(met.get("kappa_frac_negative"), is_percent=True),
        help=help_icon("kappa_frac_negative"),
    )

    h1, h2, h3, h4 = st.columns(4)
    h1.metric("Frag(H_rw)", fmt(met.get("fragility_H")), help=help_icon("fragility_H"))
    h2.metric(
        "Frag(H_evo)", fmt(met.get("fragility_evo")), help=help_icon("fragility_evo")
    )
    h3.metric(
        "Frag(κ̄)", fmt(met.get("fragility_kappa")), help=help_icon("fragility_kappa")
    )
    h4.metric(
        "κ edges (ok/skip)",
        f"{int(met.get('kappa_computed_edges', 0))}/{int(met.get('kappa_skipped_edges', 0))}",
        help="Сколько рёбер реально посчитали κ (остальные пропущены из-за ограничения support).",
    )

    with st.expander("❔", expanded=False):
        st.markdown(
            "- **τ ~ 1/λ₂**: если τ больше, сеть медленнее «расслабляется» после возмущения\n"
            "- **β**: сколько альтернативных путей есть (сколько «циклов» сверх остова)\n"
            "- **1/λ_max**: насколько легко распространяется возбуждение по сети (порог)\n"
        )


def render_dashboard_charts(G_view, apply_plot_defaults) -> None:
    st.markdown("### 📈 Распределения")
    d1, d2 = st.columns(2)

    with d1:
        degrees = [d for _, d in G_view.degree()]
        if degrees:
            fig_deg = px.histogram(
                x=degrees,
                nbins=30,
                title="Degree Distribution",
                labels={"x": "Degree", "y": "Count"},
            )
            fig_deg.update_layout(template="plotly_dark")
            apply_plot_defaults(fig_deg, height=620)
            st.plotly_chart(fig_deg, width="stretch", key="plot_deg_hist")
        else:
            st.info("Граф пуст: degree distribution не построить.")

    with d2:
        weights = [float(d.get("weight", 1.0)) for _, _, d in G_view.edges(data=True)]
        weights = [w for w in weights if np.isfinite(w)]
        if weights:
            fig_w = px.histogram(
                x=weights,
                nbins=30,
                title="Weight Distribution",
                labels={"x": "Weight", "y": "Count"},
            )
            fig_w.update_layout(template="plotly_dark")
            apply_plot_defaults(fig_w, height=620)
            st.plotly_chart(fig_w, width="stretch", key="plot_w_hist")
        else:
            st.info("Нет валидных весов для histogram.")
