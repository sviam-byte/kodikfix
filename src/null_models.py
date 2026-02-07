import networkx as nx


def make_er_gnm(n: int, m: int, seed: int) -> nx.Graph:
    return nx.gnm_random_graph(int(n), int(m), seed=int(seed))

def make_configuration_model(G_base: nx.Graph, seed: int) -> nx.Graph:
    """
    всё ещё мультиграфы не поддерживаются :(
    Мы приводим к простому графу.
    """
    degs = [d for _, d in G_base.degree()]
    M = nx.configuration_model(degs, seed=int(seed))
    H = nx.Graph(M)  
    H.remove_edges_from(nx.selfloop_edges(H))
    return H


def rewire_mix(G_base: nx.Graph, p: float, seed: int) -> nx.Graph:
    """
    Постепенная хаотизация через double_edge_swap.
    p=0 -> оригинал
    p=1 -> сильная рандомизация (но сохраняем степени)
    """
    p = float(max(0.0, min(1.0, p)))
    H = G_base.copy()
    if H.number_of_edges() < 2 or H.number_of_nodes() < 4 or p <= 0:
        return H

    swaps = int(p * H.number_of_edges() * 5)  
    swaps = max(1, swaps)
    tries = swaps * 10

    nx.double_edge_swap(H, nswap=swaps, max_tries=tries, seed=seed)

    H.remove_edges_from(nx.selfloop_edges(H))
    return H
