import numpy as np
import networkx as nx
import pandas as pd

"""
табличка -> нетворк граф
+ повторно проверяем (на случай, если оно пришло не из препроцесс, а из случайного графа)
ещё здесь выделяется LCC
"""


def build_graph_from_edges(df_edges: pd.DataFrame, src_col: str, dst_col: str) -> nx.Graph:
    G = nx.from_pandas_edgelist(
        df_edges,
        source=src_col,
        target=dst_col,
        edge_attr=["weight", "confidence"],
        create_using=nx.Graph(),
    )
    for _, _, d in G.edges(data=True):
        w_raw = d.get("weight", 1.0)
        c_raw = d.get("confidence", 0.0)
        try:
            w = float(w_raw)
        except (TypeError, ValueError):
            w = 1.0
        try:
            c = float(c_raw)
        except (TypeError, ValueError):
            c = 0.0

        if not np.isfinite(w) or w <= 0:
            raise ValueError(f"edge weight must be finite and >0, got {w!r}")
        if not np.isfinite(c):
            raise ValueError(f"edge confidence must be finite, got {c!r}")

        d["weight"] = w
        d["confidence"] = c
    return G


def lcc_subgraph(G: nx.Graph) -> nx.Graph:
    if G.number_of_nodes() == 0:
        return G.copy()
    comp = max(nx.connected_components(G), key=len)
    return G.subgraph(comp).copy()


def graph_to_edge_df(G: nx.Graph) -> pd.DataFrame:
    rows = []
    for u, v, d in G.edges(data=True):
        rows.append(
            {
                "src": int(u) if isinstance(u, (int, np.integer)) else u,
                "dst": int(v) if isinstance(v, (int, np.integer)) else v,
                "weight": float(d.get("weight", 1.0)),
                "confidence": float(d.get("confidence", 1.0)),
            }
        )
    return pd.DataFrame(rows)


def graph_summary(G: nx.Graph) -> str:
    N = G.number_of_nodes()
    E = G.number_of_edges()
    C = nx.number_connected_components(G) if N > 0 else 0
    dens = nx.density(G) if N > 1 else 0.0
    return (
        f"N={N}\n"
        f"E={E}\n"
        f"Components={C}\n"
        f"Density={dens:.6g}\n"
        f"Selfloops={nx.number_of_selfloops(G)}\n"
    )
