import hashlib
import networkx as nx

def graph_to_edge_payload(G: nx.Graph, decimals: int = 8) -> tuple:
    edges = []
    for u, v, d in G.edges(data=True):
        if u > v:
            u, v = v, u
        w = float(d.get("weight", 1.0))
        edges.append((int(u), int(v), round(w, decimals)))
    edges.sort()
    return tuple(edges)

def fingerprint_edge_payload(edge_payload: tuple) -> str:
    payload = repr(edge_payload).encode("utf-8")
    return hashlib.blake2b(payload, digest_size=16).hexdigest()

def payload_to_graph(edge_payload: tuple) -> nx.Graph:
    G = nx.Graph()
    for u, v, w in edge_payload:
        G.add_edge(int(u), int(v), weight=float(w), confidence=0.0)
    return G
