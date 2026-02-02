from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import networkx as nx

from ..utils import as_simple_undirected


def validate_edge_weights(G: nx.Graph) -> None:
    for u, v, d in G.edges(data=True):
        w = float(d.get("weight", 1.0))
        if not np.isfinite(w) or w <= 0:
            raise ValueError(f"bad edge weight on ({u!r}, {v!r}): {w!r}")


def _rw_transition_matrix(G: nx.Graph, nodes: List) -> np.ndarray:
    n = len(nodes)
    idx = {nodes[i]: i for i in range(n)}
    P = np.zeros((n, n), dtype=float)
    for u in nodes:
        i = idx[u]
        nbrs = list(G.neighbors(u))
        if not nbrs:
            P[i, i] = 1.0
            continue
        js = []
        ws = []
        for v in nbrs:
            w = float(G[u][v].get("weight", 1.0))
            js.append(idx[v])
            ws.append(w)
        s = float(np.sum(ws))
        if s <= 0:
            P[i, i] = 1.0
        else:
            for w, j in zip(ws, js):
                P[i, j] = w / s
    return P


def _pf_markov(
    G: nx.Graph,
    nodes: List,
    iters: int = 2000,
    tol: float = 1e-10,
) -> Tuple[np.ndarray, np.ndarray]:
    n = len(nodes)
    idx = {nodes[i]: i for i in range(n)}
    A = np.zeros((n, n), dtype=float)
    for u, v, d in G.edges(data=True):
        i = idx[u]
        j = idx[v]
        w = float(d.get("weight", 1.0))
        A[i, j] += w
        A[j, i] += w

    x = np.ones(n, dtype=float) / max(1, n)
    lam_old = 0.0
    for _ in range(int(max(10, iters))):
        y = A @ x
        norm = float(np.linalg.norm(y))
        if norm == 0:
            break
        x = y / norm
        lam = float((x @ (A @ x)) / max(1e-12, (x @ x)))
        if abs(lam - lam_old) < tol:
            break
        lam_old = lam

    v = np.abs(x) + 1e-15
    lam = float((v @ (A @ v)) / max(1e-12, (v @ v)))
    if not np.isfinite(lam) or lam <= 0:
        P = _rw_transition_matrix(G, nodes)
        pi = np.ones(n, dtype=float) / max(1, n)
        return P, pi

    P = np.zeros((n, n), dtype=float)
    for i in range(n):
        denom = lam * v[i]
        if denom <= 0:
            P[i, i] = 1.0
            continue
        row = (A[i, :] * v) / denom
        rs = float(row.sum())
        if rs <= 0:
            P[i, i] = 1.0
        else:
            P[i, :] = row / rs

    pi_raw = v * v
    pi = pi_raw / max(1e-12, float(pi_raw.sum()))
    return P, pi


def compute_energy_flow(
    G: nx.Graph,
    steps: int = 20,
    flow_mode: str = "rw",
    damping: float = 1.0,
    sources: Optional[List] = None,
) -> Tuple[Dict, Dict[Tuple, float]]:
    H = as_simple_undirected(G)
    validate_edge_weights(H)
    nodes = list(H.nodes())
    if not nodes:
        return {}, {}

    if sources:
        srcs = [s for s in sources if s in H]
    else:
        srcs = []
    if not srcs:
        strengths = dict(H.degree(weight="weight"))
        srcs = [max(strengths, key=strengths.get)]

    steps = int(max(0, steps))

    if str(flow_mode).lower().startswith("evo"):
        P, _pi = _pf_markov(H, nodes)
    else:
        P = _rw_transition_matrix(H, nodes)

    n = len(nodes)
    idx = {nodes[i]: i for i in range(n)}
    e = np.zeros(n, dtype=float)
    for s in srcs:
        e[idx[s]] += 1.0 / len(srcs)

    damp = float(damping)
    if not np.isfinite(damp):
        damp = 1.0
    damp = max(0.0, min(1.0, damp))

    for _ in range(steps):
        e = e @ P
        if damp != 1.0:
            e *= damp

    node_energy = {nodes[i]: float(e[i]) for i in range(n)}

    edge_flux: Dict[Tuple, float] = {}
    for u, v in H.edges():
        iu = idx[u]
        iv = idx[v]
        f_uv = float(e[iu] * P[iu, iv])
        f_vu = float(e[iv] * P[iv, iu])
        edge_flux[(u, v)] = max(f_uv, f_vu)

    return node_energy, edge_flux


def _simulate_energy_physical(
    G: nx.Graph,
    steps: int,
    damping: float,
    sources: Optional[List],
    cap_mode: str = "strength",
    injection: float = 0.15,
    leak: float = 0.02,
) -> Tuple[List[Dict], List[Dict[Tuple, float]]]:
    H = as_simple_undirected(G)
    validate_edge_weights(H)
    nodes = list(H.nodes())
    if not nodes:
        return [], []

    if cap_mode == "degree":
        cap = {n: float(H.degree(n)) for n in nodes}
    else:
        cap = {n: float(H.degree(n, weight="weight")) for n in nodes}

    for n in cap:
        if cap[n] <= 0:
            cap[n] = 1.0

    srcs = []
    if sources:
        srcs = [s for s in sources if s in H]
    if not srcs:
        srcs = [max(cap, key=cap.get)] if cap else [nodes[0]]

    E = {n: 0.0 for n in nodes}
    for s in srcs:
        E[s] = 10.0

    max_w = max([d.get("weight", 1.0) for _, _, d in H.edges(data=True)] or [1.0])
    dt = 0.1 / max(1.0, float(max_w))

    node_frames = []
    edge_frames = []

    for t in range(steps + 1):
        node_frames.append(E.copy())

        P = {n: E[n] / cap[n] for n in nodes}

        dE = {n: 0.0 for n in nodes}
        edge_flux = {}

        for u, v, d in H.edges(data=True):
            w = d.get("weight", 1.0)
            flux = w * (P[u] - P[v]) * dt

            dE[u] -= flux
            dE[v] += flux
            edge_flux[(u, v)] = abs(flux)

        edge_frames.append(edge_flux)

        if t == steps:
            break

        for n in nodes:
            E[n] += dE[n]

            if n in srcs:
                E[n] += float(injection) * dt * 10.0

            E[n] *= damping
            E[n] -= float(leak) * dt
            if E[n] < 0:
                E[n] = 0.0

    return node_frames, edge_frames


def simulate_energy_flow(
    G: nx.Graph,
    steps: int = 25,
    flow_mode: str = "rw",
    damping: float = 1.0,
    sources: Optional[List] = None,
    phys_injection: float = 0.15,
    phys_leak: float = 0.02,
    phys_cap_mode: str = "strength",
    rw_impulse: bool = True,
) -> Tuple[List[Dict], List[Dict[Tuple, float]]]:
    fm = str(flow_mode).lower().strip()
    if fm in ("phys", "pressure", "flow"):
        return _simulate_energy_physical(
            G,
            steps=int(steps),
            damping=float(damping),
            sources=sources,
            cap_mode=str(phys_cap_mode),
            injection=float(phys_injection),
            leak=float(phys_leak),
        )

    H = as_simple_undirected(G)
    validate_edge_weights(H)
    nodes = list(H.nodes())
    if not nodes:
        return [], []

    srcs: List = []
    if sources:
        srcs = [s for s in sources if s in H]
    if not srcs:
        strengths = dict(H.degree(weight="weight"))
        srcs = [max(strengths, key=strengths.get)] if strengths else [nodes[0]]

    steps = int(max(0, steps))

    if str(flow_mode).lower().startswith("evo"):
        P, _pi = _pf_markov(H, nodes)
    else:
        P = _rw_transition_matrix(H, nodes)

    n = len(nodes)
    idx = {nodes[i]: i for i in range(n)}

    e = np.zeros(n, dtype=float)
    for s in srcs:
        e[idx[s]] += 1.0 / float(len(srcs))

    inj = float(phys_injection)
    if not np.isfinite(inj):
        inj = 0.0
    inj = max(0.0, min(1.0, inj))

    if bool(rw_impulse) and inj > 0.0:
        add = inj / float(len(srcs))
        for s in srcs:
            e[idx[s]] += add
        inj = 0.0

    damp = float(damping)
    if not np.isfinite(damp):
        damp = 1.0
    damp = max(0.0, min(1.0, damp))

    node_frames: List[Dict] = []
    edge_frames: List[Dict[Tuple, float]] = []

    def _snapshot(evec: np.ndarray) -> Tuple[Dict, Dict[Tuple, float]]:
        node_energy = {nodes[i]: float(evec[i]) for i in range(n)}
        edge_flux: Dict[Tuple, float] = {}
        for u, v in H.edges():
            iu = idx[u]
            iv = idx[v]
            f_uv = float(evec[iu] * P[iu, iv])
            f_vu = float(evec[iv] * P[iv, iu])
            edge_flux[(u, v)] = max(f_uv, f_vu)
        return node_energy, edge_flux

    ne0, ef0 = _snapshot(e)
    node_frames.append(ne0)
    edge_frames.append(ef0)

    for _ in range(steps):
        e = e @ P
        if damp != 1.0:
            e = e * damp
        if inj > 0.0:
            add = inj / float(len(srcs))
            for s in srcs:
                e[idx[s]] += add
        ne, ef = _snapshot(e)
        node_frames.append(ne)
        edge_frames.append(ef)

    return node_frames, edge_frames
