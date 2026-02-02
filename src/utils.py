import networkx as nx
import numpy as np


def as_simple_undirected(G: nx.Graph) -> nx.Graph:
    """Привести граф к простому неориентированному виду.

    Предполагается, что weight/confidence уже числовые (это обязан гарантировать preprocess).
    """
    H = G
    if hasattr(H, "is_directed") and H.is_directed():
        H = H.to_undirected(as_view=False)

    if isinstance(H, (nx.MultiGraph, nx.MultiDiGraph)):
        # Склеиваем мульти-рёбра: суммируем веса и сохраняем первый набор атрибутов.
        # Это сохраняет поведение "aggregate weight", но избегает has_edge для каждого ребра.
        simple = nx.Graph()
        simple.add_nodes_from(H.nodes(data=True))
        edge_weights = {}
        edge_attrs = {}
        edge_nodes = {}
        for u, v, d in H.edges(data=True):
            w = d.get("weight", 1.0)
            key = frozenset((u, v))
            edge_weights[key] = float(edge_weights.get(key, 0.0)) + float(w)
            # Сохраняем атрибуты первого ребра как базовые.
            if key not in edge_attrs:
                edge_attrs[key] = dict(d)
                edge_nodes[key] = (u, v)

        for key, w in edge_weights.items():
            u, v = edge_nodes[key]
            attrs = edge_attrs[key]
            attrs["weight"] = w
            simple.add_edge(u, v, **attrs)
        return simple

    return nx.Graph(H)


def get_node_strength(G: nx.Graph, n) -> float:
    """Сумма весов всех инцидентных рёбер узла."""
    strength = 0.0
    for _, _, d in G.edges(n, data=True):
        w = float(d.get("weight", 1.0))
        if not np.isfinite(w):
            raise ValueError(f"non-finite edge weight for node={n}: {w!r}")
        strength += w
    return strength
