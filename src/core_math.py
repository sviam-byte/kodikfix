"""Core numerical routines (entropy, phase, robust geometry).

Сюда свалена вычислительная математика (без Streamlit/UI).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import multiprocessing
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import networkx as nx
from joblib import Parallel, delayed
from scipy.stats import entropy as scipy_entropy

from .utils import as_simple_undirected
from .profiling import timeit

# -----------------------------
# Entropy
# -----------------------------
def entropy_histogram(x, bins="fd") -> float:
    """Estimate entropy from a histogram using SciPy (default: Freedman–Diaconis bins)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return float("nan")
    hist, _ = np.histogram(x, bins=bins)
    # SciPy handles normalization internally; we keep the raw counts for clarity.
    # Guard against tiny negative values due to floating error.
    return abs(float(scipy_entropy(hist)))


def entropy_degree(G: nx.Graph) -> float:
    """Entropy of the degree distribution."""
    degrees = np.fromiter((d for _, d in G.degree()), dtype=float)
    if degrees.size == 0:
        return float("nan")
    _, counts = np.unique(degrees, return_counts=True)
    # Guard against tiny negative values due to floating error.
    return abs(float(scipy_entropy(counts)))


def entropy_weights(G: nx.Graph) -> float:
    """Entropy of edge weight distribution."""
    vals: list[float] = []
    for _, _, d in G.edges(data=True):
        w = float(d.get("weight", 1.0))
        if np.isfinite(w) and w > 0:
            vals.append(w)
    return entropy_histogram(vals, bins="fd")


def entropy_confidence(G: nx.Graph) -> float:
    """Entropy of edge confidence distribution."""
    vals: list[float] = []
    for _, _, d in G.edges(data=True):
        c = float(d.get("confidence", 1.0))
        if np.isfinite(c) and c > 0:
            vals.append(c)
    return entropy_histogram(vals, bins="fd")


def triangle_support_edge(G: nx.Graph):
    """Count how many triangles each edge participates in (heavy for large graphs)."""
    tri = nx.triangles(G)
    out = []
    for u, v in G.edges():
        out.append(min(tri.get(u, 0), tri.get(v, 0)))
    return out


def entropy_triangle_support(G: nx.Graph) -> float:
    """Энтропия распределения triangle-support по рёбрам (число треугольников на ребро)."""
    ts = triangle_support_edge(G)
    return entropy_histogram(ts, bins="fd")

# -----------------------------
# Phase transition heuristics
# -----------------------------
def classify_phase_transition(
    df: pd.DataFrame,
    x_col: str = "removed_frac",
    y_col: str = "lcc_frac",
) -> dict:
    """
    Эвристика "взрывного" распада:
    - смотрим на дискретные скачки y между соседними точками
    - если самый большой отрицательный скачок занимает большую долю амплитуды
      и происходит в узком окне x -> считаем abrupt
    """
    if df is None or df.empty or x_col not in df.columns or y_col not in df.columns:
        return {
            "is_abrupt": False,
            "critical_x": float("nan"),
            "jump": 0.0,
            "jump_fraction": 0.0,
        }

    x = np.asarray(df[x_col], dtype=float)
    y = np.asarray(df[y_col], dtype=float)

    if len(x) < 3:
        return {
            "is_abrupt": False,
            "critical_x": float(x[-1]) if len(x) else float("nan"),
            "jump": 0.0,
            "jump_fraction": 0.0,
        }

    dy = np.diff(y)
    # largest negative drop
    idx = int(np.argmin(dy))
    jump = float(-dy[idx])  # positive magnitude of drop
    y_span = float(max(1e-12, np.nanmax(y) - np.nanmin(y)))
    jump_fraction = float(jump / y_span)

    critical_x = float(x[idx + 1]) if idx + 1 < len(x) else float(x[-1])

    # "first-order like" heuristic thresholds
    # jump_fraction >= 0.35 means ~ huge discontinuity relative to range
    is_abrupt = bool(jump_fraction >= 0.35)

    return {
        "is_abrupt": is_abrupt,
        "critical_x": critical_x,
        "jump": jump,
        "jump_fraction": jump_fraction,
    }

# -----------------------------
# Robust geometry / curvature
# -----------------------------
def add_dist_attr(G: nx.Graph) -> nx.Graph:
    """Копия графа + атрибут dist=1/weight для путей.

    Некорректные веса заменяются малым положительным значением, чтобы не падать.
    """
    H = G.copy()
    for _, _, d in H.edges(data=True):
        w = float(d.get("weight", 1.0))
        if not np.isfinite(w) or w <= 0:
            raise ValueError(f"edge weight must be finite and >0, got {w!r}")
        d["dist"] = 1.0 / w
    return H


def _normalize_edge_weights(G: nx.Graph) -> nx.Graph:
    """Нормализовать веса рёбер: finite и >=0 (best-effort)."""
    H = G.copy()
    for _, _, d in H.edges(data=True):
        w_raw = d.get("weight", 1.0)
        try:
            w = float(w_raw)
        except (TypeError, ValueError):
            w = 1.0
        if not np.isfinite(w) or w <= 0:
            w = 1.0
        d["weight"] = w
    return H

# -----------------------------
# 1) Entropy rate of random walk
# -----------------------------
def network_entropy_rate(G: nx.Graph, base: float = math.e) -> float:
    """
    Entropy rate:
        H_rw = - Σ_i π_i Σ_j P_ij log P_ij
    where P_ij = w_ij / strength(i), π_i = strength(i)/Σ strength.
    """
    # Assume G is already simplified and weights are numeric (preprocess validates inputs).
    if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
        return 0.0
    A = nx.adjacency_matrix(G, weight="weight").astype(float).tocsr()
    if A.nnz == 0:
        return 0.0
    d = np.asarray(A.sum(axis=1)).flatten()
    total_s = float(d.sum())
    if total_s <= 0:
        return 0.0

    # Normalize rows: P_ij = w_ij / d_i (zero-degree rows stay zero).
    inv_d = np.reciprocal(d, out=np.zeros_like(d), where=d > 0)
    P = A.multiply(inv_d[:, None])

    # Считаем энтропию напрямую на разреженных данных, без toarray():
    # H(row) = -sum(p * log(p)). Учитываем только ненулевые элементы.
    data_log = np.log(P.data + 1e-20) / np.log(base)
    ent_data = -(P.data * data_log)

    # Перекладываем энтропийные веса в копию разреженной матрицы и суммируем по строкам.
    P_ent = P.copy()
    P_ent.data = ent_data
    row_ents = np.asarray(P_ent.sum(axis=1)).flatten()
    return float(np.sum((d / total_s) * row_ents))


# -----------------------------
# 2) Ollivier–Ricci curvature (transport W1)
# -----------------------------
def _one_step_measure(H: nx.Graph, x) -> Dict:
    neigh = list(H.neighbors(x))
    if not neigh:
        return {}
    ws = []
    for y in neigh:
        d = H[x][y]
        w = float(d.get("weight", 1.0))
        ws.append(max(0.0, w))
    s = float(sum(ws))
    if s <= 0:
        p = 1.0 / len(neigh)
        return {y: p for y in neigh}
    return {y: w / s for y, w in zip(neigh, ws)}


def _quantize_probs(probs: Dict, scale: int) -> Tuple[List, List[int]]:
    items = [(k, float(v)) for k, v in probs.items() if float(v) > 0]
    if not items:
        return [], []
    nodes = [k for k, _ in items]
    ps = np.array([p for _, p in items], dtype=float)
    s = float(ps.sum())
    if s <= 0:
        return [], []
    ps = ps / s

    masses = np.floor(ps * scale).astype(int)
    rem = int(scale - masses.sum())

    if rem > 0:
        frac = ps * scale - np.floor(ps * scale)
        order = np.argsort(-frac)
        for idx in order[:rem]:
            masses[int(idx)] += 1
    elif rem < 0:
        order = np.argsort(-masses)
        k = 0
        while rem < 0 and k < len(order):
            idx = int(order[k])
            if masses[idx] > 0:
                masses[idx] -= 1
                rem += 1
            else:
                k += 1

    if int(masses.sum()) != int(scale):
        masses[-1] += int(scale - masses.sum())

    return nodes, masses.tolist()


def _emd_w1_transport(
    supply: Dict,
    demand: Dict,
    dist: Dict[Tuple, float],
    *,
    scale: int,
    missing_cost: float,
) -> float:
    S_nodes, S_mass = _quantize_probs(supply, scale)
    D_nodes, D_mass = _quantize_probs(demand, scale)
    if not S_nodes or not D_nodes:
        return 0.0

    Gf = nx.DiGraph()
    for u, m in zip(S_nodes, S_mass):
        Gf.add_node(("S", u), demand=-int(m))
    for v, m in zip(D_nodes, D_mass):
        Gf.add_node(("D", v), demand=+int(m))

    for u in S_nodes:
        for v in D_nodes:
            c = dist.get((u, v), dist.get((v, u), None))
            if c is None or not np.isfinite(c):
                c = float(missing_cost)
            Gf.add_edge(("S", u), ("D", v), weight=float(c), capacity=int(scale))

    cost, _ = nx.network_simplex(Gf)
    return float(cost) / float(scale)


def ollivier_ricci_edge(
    G: nx.Graph,
    x,
    y,
    *,
    max_support: int = 60,
    cutoff: float = 8.0,
    scale: int = 120_000,
    missing_cost: float = 1e6,
) -> Optional[float]:
    """
    κ(x,y) = 1 - W1(µ_x, µ_y)/d(x,y)
    Distances use dist=1/weight.
    """
    H = _normalize_edge_weights(as_simple_undirected(G))
    if not H.has_edge(x, y):
        return None

    Hw = add_dist_attr(H)
    mu_x = _one_step_measure(H, x)
    mu_y = _one_step_measure(H, y)
    if not mu_x or not mu_y:
        return None

    sx = list(mu_x.keys())
    sy = list(mu_y.keys())
    if (len(sx) + len(sy)) > int(max_support):
        return None

    dxy = float(Hw[x][y].get("dist", 1.0))
    if not np.isfinite(dxy) or dxy <= 0:
        return None

    dist = {}
    for u in sx:
        dists = nx.single_source_dijkstra_path_length(Hw, u, cutoff=float(cutoff), weight="dist")
        for v in sy:
            if v in dists:
                dist[(u, v)] = float(dists[v])

    W1 = _emd_w1_transport(mu_x, mu_y, dist, scale=int(scale), missing_cost=float(missing_cost))
    return float(1.0 - (W1 / dxy))


@dataclass
class CurvatureSummary:
    kappa_mean: float
    kappa_median: float
    kappa_frac_negative: float
    computed_edges: int
    skipped_edges: int


@timeit('ollivier_ricci_summary')
def ollivier_ricci_summary(
    G: nx.Graph,
    sample_edges: int = 150,
    seed: int = 42,
    max_support: int = 60,
    cutoff: float = 8.0,
    scale: int = 120_000,
    progress_cb=None,
    force_sequential: bool = False,
    **_ignored,
) -> CurvatureSummary:
    # NOTE: progress_cb нужен только для UI (streamlit). В core он остаётся "немым" колбэком.
    H = _normalize_edge_weights(as_simple_undirected(G))
    if H.number_of_edges() == 0:
        return CurvatureSummary(0.0, 0.0, 0.0, 0, 0)

    edges = list(H.edges())
    rng = random.Random(int(seed))
    if len(edges) > int(sample_edges):
        edges = rng.sample(edges, int(sample_edges))

    # Если UI попросил прогресс — считаем последовательно (иначе joblib прогресс не отдаёт).
    if (progress_cb is not None) or bool(force_sequential):
        results = []
        total = max(1, len(edges))
        for i, (x, y) in enumerate(edges, start=1):
            results.append(
                ollivier_ricci_edge(H, x, y, max_support=max_support, cutoff=cutoff, scale=scale)
            )
            if progress_cb is not None:
                try:
                    progress_cb(i, total, x, y)
                except TypeError:
                    progress_cb(i, total)
    else:
        # Parallelize per-edge curvature for a big speedup on multi-core machines.
        num_cores = max(1, min(multiprocessing.cpu_count(), len(edges)))
        results = Parallel(n_jobs=num_cores)(
            delayed(ollivier_ricci_edge)(
                H, x, y, max_support=max_support, cutoff=cutoff, scale=scale
            )
            for x, y in edges
        )

    kappas = [float(k) for k in results if k is not None and np.isfinite(k)]
    skipped = len(edges) - len(kappas)

    if len(kappas) == 0:
        return CurvatureSummary(float('nan'), float('nan'), float('nan'), 0, int(skipped))

    arr = np.array(kappas, dtype=float)
    return CurvatureSummary(
        kappa_mean=float(arr.mean()),
        kappa_median=float(np.median(arr)),
        kappa_frac_negative=float((arr < 0).mean()),
        computed_edges=int(arr.size),
        skipped_edges=int(skipped),
    )



# -----------------------------
# 3) Fragility proxies
# -----------------------------
def fragility_from_entropy(h: float, eps: float = 1e-9) -> float:
    if not np.isfinite(h):
        return float("nan")
    return float(1.0 / max(eps, float(h)))


def fragility_from_curvature(kappa_mean: float, eps: float = 1e-9) -> float:
    if not np.isfinite(kappa_mean):
        return float("nan")
    return float(1.0 / max(eps, 1.0 + float(kappa_mean)))


# -----------------------------
# 4) Demetrius evolutionary entropy (PF-Markov)
# -----------------------------
def _pf_eigs_sparse(A):
    import scipy.sparse.linalg as spla
    vals_r, vecs_r = spla.eigs(A, k=1, which="LR")
    lam = float(np.real(vals_r[0]))
    u = np.real(vecs_r[:, 0])
    vals_l, vecs_l = spla.eigs(A.T, k=1, which="LR")
    v = np.real(vecs_l[:, 0])
    return lam, u, v


def evolutionary_entropy_demetrius(G: nx.Graph, base: float = math.e) -> float:
    """
    Build PF-Markov chain from adjacency A and compute entropy rate:
      P_ij = a_ij * u_j / (lam * u_i),  π_i ∝ u_i v_i
      H_evo = -Σ_i π_i Σ_j P_ij log P_ij
    """
    H = _normalize_edge_weights(as_simple_undirected(G))
    if H.number_of_nodes() < 2 or H.number_of_edges() == 0:
        return float("nan")

    # For undirected: treat as directed both ways for A
    A = nx.adjacency_matrix(H.to_directed(), weight="weight").astype(float).tocsr()
    if A.nnz == 0:
        return float("nan")
    if A.data.size:
        A.data = np.maximum(A.data, 0.0)

    n = A.shape[0]
    if n <= 600:
        Ad = A.toarray()
        vals, vecs = np.linalg.eig(Ad)
        idx = int(np.argmax(np.real(vals)))
        lam = float(np.real(vals[idx]))
        u = np.real(vecs[:, idx])
        # Для симметричной матрицы (неориентированный граф) u == v.
        v = u
    else:
        # Для undirected графа достаточно правого собственного вектора.
        import scipy.sparse.linalg as spla

        vals_r, vecs_r = spla.eigs(A, k=1, which="LR")
        lam = float(np.real(vals_r[0]))
        u = np.real(vecs_r[:, 0])
        v = u

    if not np.isfinite(lam) or lam <= 0:
        return float("nan")

    u = np.abs(u) + 1e-15
    v = np.abs(v) + 1e-15

    P = np.zeros((n, n), dtype=float)
    A_coo = A.tocoo()
    for i, j, aij in zip(A_coo.row, A_coo.col, A_coo.data):
        if aij <= 0:
            continue
        P[i, j] += float(aij) * float(u[j]) / (float(lam) * float(u[i]))

    row_sums = P.sum(axis=1)
    for i in range(n):
        s = float(row_sums[i])
        if s > 0:
            P[i, :] /= s
        else:
            P[i, :] = 1.0 / n

    pi = u * v
    Z = float(pi.sum())
    if not np.isfinite(Z) or Z <= 0:
        return float("nan")
    pi = pi / Z

    log_base = math.log(base)
    H_evo = 0.0
    for i in range(n):
        row = P[i, :]
        mask = row > 0
        if not np.any(mask):
            continue
        h_i = -float(np.sum(row[mask] * (np.log(row[mask]) / log_base)))
        H_evo += float(pi[i]) * h_i

    return float(H_evo)
