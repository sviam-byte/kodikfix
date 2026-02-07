import networkx as nx
import numpy as np
import pandas as pd

from .config import settings
from .weights import policy_from_settings, apply_weight_policy_scalar
from .preprocess import filter_edges

"""
табличка -> нетворк граф
+ повторно проверяем (на случай, если оно пришло не из препроцесс, а из случайного графа)
ещё здесь выделяется LCC
"""


def build_graph_from_edges(
    df_edges: pd.DataFrame,
    src_col: str,
    dst_col: str,
    *,
    strict: bool = True,
) -> nx.Graph:
    G = nx.from_pandas_edgelist(
        df_edges,
        source=src_col,
        target=dst_col,
        edge_attr=["weight", "confidence"],
        create_using=nx.Graph(),
    )
    pol = policy_from_settings(settings.WEIGHT_POLICY, settings.WEIGHT_EPS, settings.WEIGHT_SHIFT)
    to_drop = []

    for u, v, d in G.edges(data=True):
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

        w2 = apply_weight_policy_scalar(w, pol)
        if w2 is None:
            if strict:
                raise ValueError(f"edge weight must be finite and >0 under policy={pol.mode!r}, got {w!r}")
            to_drop.append((u, v))
            continue
        w = float(w2)
        if not np.isfinite(c):
            raise ValueError(f"edge confidence must be finite, got {c!r}")

        d["weight"] = float(w)
        d["confidence"] = c
    if to_drop:
        G.remove_edges_from(to_drop)
    if G.number_of_edges() == 0:
        raise ValueError("Graph has zero edges after weight sanitization.")
    return G


def build_graph(
    df_edges: pd.DataFrame,
    *,
    src_col: str = "src",
    dst_col: str = "dst",
    min_conf: float = 0.0,
    min_weight: float = 0.0,
    analysis_mode: str = "Global",
    strict: bool = True,
) -> nx.Graph:
    """Полный пайплайн сборки: фильтры -> граф -> (опционально) LCC."""
    df_filtered = filter_edges(df_edges, src_col, dst_col, float(min_conf), float(min_weight))
    G = build_graph_from_edges(df_filtered, src_col, dst_col, strict=strict)
    if str(analysis_mode).startswith("LCC"):
        G = lcc_subgraph(G)
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
