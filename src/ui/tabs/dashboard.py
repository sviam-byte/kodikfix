from __future__ import annotations

import io

import networkx as nx
import pandas as pd
import streamlit as st

from src.state_models import GraphEntry
from src.ui.plots.charts import apply_plot_defaults
from src.ui_blocks import render_dashboard_metrics, render_dashboard_charts


def _node_metrics_df(G: nx.Graph) -> pd.DataFrame:
    # TODO: нормально оформить метрики узлов (центральности и т.п.). Сейчас нужно «чтобы выгружалось».
    nodes = list(G.nodes())

    deg = dict(G.degree())
    strength = {}
    for n in nodes:
        s = 0.0
        for _, _, d in G.edges(n, data=True):
            s += float(d.get("weight", 1.0))
        strength[n] = s

    # clustering в nx медленный на огромных графах — поэтому ограничение. Да, костыль.
    if G.number_of_nodes() <= 3000:
        clust = nx.clustering(G)
    else:
        clust = {n: float("nan") for n in nodes}

    df = pd.DataFrame(
        {
            "node": nodes,
            "degree": [deg.get(n, 0) for n in nodes],
            "strength": [strength.get(n, 0.0) for n in nodes],
            "clustering": [clust.get(n, float("nan")) for n in nodes],
        }
    )
    return df


def _graph_export_bytes(G: nx.Graph, fmt: str) -> bytes:
    # fmt: graphml | gexf
    fmt = fmt.lower().strip()
    sio = io.StringIO()
    if fmt == "graphml":
        nx.write_graphml(G, sio)
    elif fmt == "gexf":
        nx.write_gexf(G, sio)
    else:
        raise ValueError(f"unknown fmt: {fmt}")
    return sio.getvalue().encode("utf-8")


def render(G_view: nx.Graph | None, met: dict, active_entry: GraphEntry) -> None:
    """Дэшборд: быстрый обзор + простые выгрузки."""
    if G_view is None:
        st.info("Граф не загружен.")
        return

    st.header(f"Обзор: {active_entry.name}")

    if G_view.number_of_nodes() > 1500:
        st.warning("⚠️ Граф большой. Тяжелые метрики считаются отдельно и могут быть медленными.")

    # (в процессе отладки тут часто сидело st.write(met), оставлю как напоминание)
    # st.write(met)

    render_dashboard_metrics(G_view, met)

    st.markdown("---")

    render_dashboard_charts(G_view, apply_plot_defaults)

    st.markdown("---")
    st.subheader("Экспорт")

    c1, c2 = st.columns(2)

    with c1:
        if st.button("📄 Собрать node-metrics CSV", key="exp_nodes_csv"):
            df_nodes = _node_metrics_df(G_view)
            csv = df_nodes.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Скачать nodes.csv",
                data=csv,
                file_name=f"{active_entry.name}_nodes.csv",
                mime="text/csv",
                key="dl_nodes_csv",
            )

    with c2:
        st.caption("Граф для Gephi/других тулов")
        try:
            b1 = _graph_export_bytes(G_view, "graphml")
            st.download_button(
                "⬇️ GraphML",
                data=b1,
                file_name=f"{active_entry.name}.graphml",
                mime="application/xml",
                key="dl_graphml",
            )
        except Exception as e:
            st.warning(f"GraphML export failed: {e}")

        try:
            b2 = _graph_export_bytes(G_view, "gexf")
            st.download_button(
                "⬇️ GEXF",
                data=b2,
                file_name=f"{active_entry.name}.gexf",
                mime="application/xml",
                key="dl_gexf",
            )
        except Exception as e:
            st.warning(f"GEXF export failed: {e}")
