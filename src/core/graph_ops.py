from __future__ import annotations

import math
import random
from typing import Any, Callable

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse.linalg as spla
from networkx.algorithms.community import louvain_communities, modularity

from ..config import EPS_W, settings
from ..core_math import (
    add_dist_attr,
    evolutionary_entropy_demetrius,
    fragility_from_curvature,
    fragility_from_entropy,
    network_entropy_rate,
    ollivier_ricci_summary,
)
from ..profiling import timeit

GraphMetrics = dict[str, int | float | str | Any]


def spectral_radius_weighted_adjacency(G: nx.Graph) -> float:
    if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
        return 0.0
    A = nx.adjacency_matrix(G, weight="weight").astype(float)
    try:
        if A.shape[0] < 10:
            vals = np.linalg.eigvals(A.toarray())
            return float(np.max(np.abs(vals)))
        vals = spla.eigs(A, k=1, which="LR", return_eigenvectors=False)
        return float(np.abs(vals[0]))
    except (spla.ArpackNoConvergence, spla.ArpackError, ValueError):
        return 0.0


def lambda2_on_lcc(G: nx.Graph) -> float:
    if G.number_of_nodes() < 2 or G.number_of_edges() == 0:
        return 0.0
    H = G.to_undirected(as_view=False) if G.is_directed() else G

    comps = list(nx.connected_components(H))
    if not comps:
        return 0.0
    lcc = max(comps, key=len)
    if len(lcc) < 2:
        return 0.0

    Hs = H.subgraph(lcc).copy()
    L = nx.normalized_laplacian_matrix(Hs, weight="weight").astype(float)
    if L.shape[0] < 10:
        vals = np.sort(np.real(np.linalg.eigvals(L.toarray())))
        return float(max(0.0, vals[1])) if vals.size >= 2 else 0.0
    try:
        vals = spla.eigs(L, k=2, which="SM", sigma=1e-5, return_eigenvectors=False)
        vals = np.sort(np.real(vals))
        if vals.size >= 2:
            return float(max(0.0, vals[1]))
        return 0.0
    except (spla.ArpackNoConvergence, spla.ArpackError, ValueError):
        if L.shape[0] < 2000:
            vals = np.sort(np.real(np.linalg.eigvals(L.toarray())))
            return float(max(0.0, vals[1])) if vals.size >= 2 else 0.0
        return 0.0


def lcc_fraction(G: nx.Graph, N0: int) -> float:
    if N0 <= 0 or G.number_of_nodes() == 0:
        return 0.0
    H_u = G.to_undirected(as_view=False) if G.is_directed() else G
    lcc = max(nx.connected_components(H_u), key=len)
    return float(len(lcc) / float(N0))


def approx_weighted_efficiency(
    G: nx.Graph,
    sources_k: int = settings.APPROX_EFFICIENCY_K,
    seed: int = 0,
) -> float:
    N = G.number_of_nodes()
    if N < 2 or G.number_of_edges() == 0:
        return 0.0

    H = add_dist_attr(G)
    rng = random.Random(int(seed))
    nodes = list(H.nodes())

    # Защита от зависания на больших графах
    req_k = int(sources_k)
    if N > 1500:
        req_k = min(req_k, 10)
    elif N > 500:
        req_k = min(req_k, 25)

    k = min(req_k, len(nodes))
    if k <= 0:
        return 0.0
    sources = rng.sample(nodes, k)

    inv_sum = 0.0
    cnt = 0
    for s in sources:
        dists = nx.single_source_dijkstra_path_length(H, s, weight="dist")
        for t, d in dists.items():
            if s == t:
                continue
            if d and np.isfinite(d) and d > 0:
                inv_sum += 1.0 / float(d)
                cnt += 1
    return float(inv_sum / cnt) if cnt > 0 else 0.0


def compute_modularity_louvain(G: nx.Graph, seed: int = 0) -> float:
    if G.number_of_nodes() < 2 or G.number_of_edges() == 0:
        return 0.0
    H = G.to_undirected(as_view=False) if G.is_directed() else G
    parts = louvain_communities(H, weight="weight", seed=seed)
    return float(modularity(H, parts, weight="weight"))


def degree_entropy(G: nx.Graph) -> float:
    N = G.number_of_nodes()
    if N == 0:
        return 0.0
    degs = np.array([d for _, d in G.degree()], dtype=float)
    if degs.size == 0:
        return 0.0
    _, counts = np.unique(degs, return_counts=True)
    p = counts.astype(float) / float(counts.sum())
    p = p[p > 0]
    return float(-np.sum(p * np.log(p))) if p.size > 0 else 0.0


def approx_diameter_lcc(G: nx.Graph, seed: int = 0, samples: int = 16):
    if G.number_of_nodes() < 2 or G.number_of_edges() == 0:
        return None
    H = G.to_undirected(as_view=False) if G.is_directed() else G

    comps = list(nx.connected_components(H))
    if not comps:
        return None
    lcc = max(comps, key=len)
    if len(lcc) < 2:
        return 0
    S = H.subgraph(lcc).copy()

    rng = random.Random(int(seed))
    nodes = list(S.nodes())
    if not nodes:
        return None
    k = min(int(samples), len(nodes))
    starts = rng.sample(nodes, k)

    best = 0
    for s in starts:
        d1 = nx.single_source_shortest_path_length(S, s)
        if not d1:
            continue
        u = max(d1, key=d1.get)
        d2 = nx.single_source_shortest_path_length(S, u)
        if not d2:
            continue
        diam = max(d2.values())
        best = max(best, diam)
    return int(best)


def _shannon_entropy_from_counts(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=float)
    s = float(np.sum(counts))
    if s <= 0:
        return float("nan")
    p = counts / s
    p = p[(p > 1e-15) & np.isfinite(p)]
    if p.size == 0:
        return float("nan")
    return abs(float(-np.sum(p * np.log(p))))


def _shannon_entropy_from_values(values, bins: int = 32) -> float:
    xs = np.asarray(values, dtype=float)
    xs = xs[np.isfinite(xs)]
    if xs.size == 0:
        return float("nan")
    hist, _ = np.histogram(xs, bins=int(bins))
    return _shannon_entropy_from_counts(hist)


@timeit("calculate_metrics")
def calculate_metrics(
    G: nx.Graph,
    eff_sources_k: int,
    seed: int,
    compute_curvature: bool = True,
    curvature_sample_edges: int = 150,
    curvature_max_support: int = settings.RICCI_MAX_SUPPORT,
    curvature_cutoff: float = settings.RICCI_CUTOFF,
    progress_cb: Callable[[float], None] | None = None,
    skip_spectral: bool = False,
    **kwargs,
) -> GraphMetrics:
    N = G.number_of_nodes()
    E = G.number_of_edges()
    # Получаем флаг тяжелых вычислений (по умолчанию True, если не передан)
    is_heavy = kwargs.get("compute_heavy", True)
    if not is_heavy:
        compute_curvature = False
    if N > 0:
        C = (
            nx.number_connected_components(G)
            if not G.is_directed()
            else nx.number_weakly_connected_components(G)
        )
    else:
        C = 0

    dens = nx.density(G) if N > 1 else 0.0
    avg_deg = (2 * E / N) if N > 0 else 0.0

    H_u = G.to_undirected(as_view=False) if G.is_directed() else G
    lcc_size = len(max(nx.connected_components(H_u), key=len)) if N > 0 else 0
    lcc_frac = lcc_size / N if N > 0 else 0.0

    if not skip_spectral:
        lmax = spectral_radius_weighted_adjacency(G)
        thresh = (1.0 / lmax) if lmax > 0 else 0.0

        l2 = lambda2_on_lcc(G)
        tau = (1.0 / l2) if l2 > 0 else float("inf")

        Q = compute_modularity_louvain(G, seed=seed)
    else:
        lmax = 0.0
        thresh = 0.0
        l2 = 0.0
        tau = float("inf")
        Q = float("nan")

    # Efficiency — самая дорогая операция (Dijkstra).
    # Если шаг "легкий", мы вообще её не считаем (либо считаем на микро-выборке для UI)
    if is_heavy or N < 300:
        eff_w = approx_weighted_efficiency(G, sources_k=eff_sources_k, seed=seed)
    else:
        eff_w = float("nan")

    ent = degree_entropy(G)
    assort = nx.degree_assortativity_coefficient(G) if N > 2 and E > 0 else 0.0
    clust = nx.average_clustering(H_u) if N > 2 and E > 0 else 0.0
    diam = approx_diameter_lcc(G, seed=seed, samples=16)

    beta = int(E - N + C) if N > 0 else 0

    degs = np.array([d for _, d in G.degree()], dtype=float)
    if degs.size > 0:
        _, deg_counts = np.unique(degs, return_counts=True)
        H_deg = _shannon_entropy_from_counts(deg_counts)
    else:
        H_deg = float("nan")

    if G.number_of_edges() > 0:
        edf = pd.DataFrame([d for _, _, d in G.edges(data=True)])
        ws = (
            pd.to_numeric(edf["weight"], errors="coerce").dropna().to_numpy()
            if "weight" in edf
            else np.array([])
        )
        cs = (
            pd.to_numeric(edf["confidence"], errors="coerce").dropna().to_numpy()
            if "confidence" in edf
            else np.array([])
        )
    else:
        ws = np.array([])
        cs = np.array([])

    H_w = _shannon_entropy_from_values(ws, bins=32) if ws.size else float("nan")
    H_conf = _shannon_entropy_from_values(cs, bins=32) if cs.size else float("nan")

    beta_red = (E - (N - C)) / float(E) if E > 0 else float("nan")

    tau_relax = (1.0 / l2) if (np.isfinite(l2) and l2 > EPS_W) else float("nan")
    epi_thr = (1.0 / lmax) if (np.isfinite(lmax) and lmax > EPS_W) else float("nan")

    H_rw = float(network_entropy_rate(G, base=math.e))
    H_evo = float(evolutionary_entropy_demetrius(G, base=math.e))

    if compute_curvature and G.number_of_edges() > 0:
        def _progress_cb(i: int, total: int, *_args) -> None:
            if progress_cb is None:
                return
            frac = float(i) / float(max(1, total))
            progress_cb(min(1.0, max(0.0, frac)))

        curv = ollivier_ricci_summary(
            G,
            sample_edges=curvature_sample_edges,
            seed=seed,
            max_support=curvature_max_support,
            cutoff=curvature_cutoff,
            progress_cb=_progress_cb if progress_cb is not None else None,
        )
        kappa_mean = float(curv.kappa_mean)
        kappa_median = float(curv.kappa_median)
        kappa_frac_negative = float(curv.kappa_frac_negative)
        kappa_computed_edges = int(curv.computed_edges)
        kappa_skipped_edges = int(curv.skipped_edges)
    else:
        kappa_mean = float("nan")
        kappa_median = float("nan")
        kappa_frac_negative = float("nan")
        kappa_computed_edges = 0
        kappa_skipped_edges = 0

    frag_H = float(fragility_from_entropy(H_rw)) if np.isfinite(H_rw) else float("nan")
    frag_evo = float(fragility_from_entropy(H_evo)) if np.isfinite(H_evo) else float("nan")
    frag_k = float(fragility_from_curvature(kappa_mean)) if np.isfinite(kappa_mean) else float("nan")

    return {
        "N": N,
        "E": E,
        "C": C,
        "density": dens,
        "avg_degree": avg_deg,
        "beta": beta,
        "beta_red": beta_red,
        "lcc_size": lcc_size,
        "lcc_frac": lcc_frac,
        "eff_w": eff_w,
        "l2_lcc": l2,
        "tau_lcc": tau,
        "tau_relax": tau_relax,
        "lmax": lmax,
        "thresh": thresh,
        "epi_thr": epi_thr,
        "mod": Q,
        "entropy_deg": ent,
        "H_deg": H_deg,
        "H_w": H_w,
        "H_conf": H_conf,
        "assortativity": assort,
        "clustering": clust,
        "diameter_approx": diam,
        "H_rw": H_rw,
        "H_evo": H_evo,
        "kappa_mean": kappa_mean,
        "kappa_median": kappa_median,
        "kappa_frac_negative": kappa_frac_negative,
        "kappa_computed_edges": kappa_computed_edges,
        "kappa_skipped_edges": kappa_skipped_edges,
        "fragility_H": frag_H,
        "fragility_evo": frag_evo,
        "fragility_kappa": frag_k,
    }


def compute_3d_layout(G: nx.Graph, seed: int) -> dict:

    N = G.number_of_nodes()
    if N == 0:
        return {}

    seed = int(seed)

    FAST_N = 500

    if N <= FAST_N:
        return nx.spring_layout(G, dim=3, weight="weight", seed=seed, iterations=40, threshold=1e-4)

    # Fast path: считаем 2D (это намного быстрее сходится),
    # а Z-координату синтезируем из centrality
    # Improves convergence for dense layouts without overworking the solver.
    pos2 = nx.spring_layout(G, dim=2, weight="weight", seed=seed, iterations=15, threshold=1e-3)

   
    rng = np.random.default_rng(seed)

    d_dict = dict(G.degree())
    d_vals = np.array([d_dict[n] for n in pos2.keys()], dtype=float)

    if d_vals.size:
        dmin, dmax = d_vals.min(), d_vals.max()
        denom = (dmax - dmin) if (dmax - dmin) > 1e-9 else 1.0
        dz = 2.0 * ((d_vals - dmin) / denom) - 1.0
    else:
        dz = np.zeros(len(pos2), dtype=float)

    dz += rng.normal(0.0, 0.15, size=dz.shape)

    out = {}
    nodes = list(pos2.keys())
    for i, n in enumerate(nodes):
        x, y = pos2[n]
        out[n] = (float(x), float(y), float(dz[i]))
    return out
