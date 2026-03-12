"""
Сюда свалена вычислительная математика.
"""

from __future__ import annotations

import math
import multiprocessing
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import entropy as scipy_entropy

from .config import (
    EPS_LOG,
    EPS_W,
    RICCI_CUTOFF,
    RICCI_MASS_SCALE,
    RICCI_MAX_SUPPORT,
    RICCI_SAMPLE_EDGES,
)
from .null import compute_null_threshold
from .profiling import timeit
from .utils import as_simple_undirected


# -----------------------------
# Entropy
# -----------------------------
def entropy_histogram(x, bins="fd") -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return float("nan")
    hist, _ = np.histogram(x, bins=bins)
    return abs(float(scipy_entropy(hist)))


def entropy_degree(G: nx.Graph) -> float:
    degrees = np.fromiter((d for _, d in G.degree()), dtype=float)
    if degrees.size == 0:
        return float("nan")
    _, counts = np.unique(degrees, return_counts=True)
    return abs(float(scipy_entropy(counts)))


def entropy_weights(G: nx.Graph) -> float:
    vals: list[float] = []
    for _, _, d in G.edges(data=True):
        w = float(d.get("weight", 1.0))
        if np.isfinite(w) and w > 0:
            vals.append(w)
    return entropy_histogram(vals, bins="fd")


def entropy_confidence(G: nx.Graph) -> float:
    vals: list[float] = []
    for _, _, d in G.edges(data=True):
        c = float(d.get("confidence", 1.0))
        if np.isfinite(c) and c > 0:
            vals.append(c)
    return entropy_histogram(vals, bins="fd")


def triangle_support_edge(G: nx.Graph):
    tri = nx.triangles(G)
    out = []
    for u, v in G.edges():
        out.append(min(tri.get(u, 0), tri.get(v, 0)))
    return out


def entropy_triangle_support(G: nx.Graph) -> float:
    ts = triangle_support_edge(G)
    return entropy_histogram(ts, bins="fd")

# -----------------------------
# Phase transition heuristics
# -----------------------------
def classify_phase_transition(
    df: pd.DataFrame,
    x_col: str = "removed_frac",
    y_col: str = "lcc_frac",
    null_jump_samples: Optional[List[float]] = None,
) -> dict:
    """
    Эвристика "взрывного" распада:
    - смотрим на дискретные скачки y между соседними точками
    - если самый большой отрицательный скачок занимает большую долю амплитуды
      и происходит в узком окне x -> считаем abrupt
      и получаем типа критическую точку и фазовый переход
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
    idx = int(np.argmin(dy))
    jump = float(-dy[idx]) 
    y_span = float(max(EPS_W, np.nanmax(y) - np.nanmin(y)))
    jump_fraction = float(jump / y_span)

    critical_x = float(x[idx + 1]) if idx + 1 < len(x) else float(x[-1])

    threshold = compute_null_threshold(null_jump_samples or [])
    is_abrupt = bool(jump_fraction >= threshold)

    return {
        "is_abrupt": is_abrupt,
        "critical_x": critical_x,
        "jump": jump,
        "jump_fraction": jump_fraction,
        "threshold": threshold,
    }

# -----------------------------
# Robust geometry / curvature
# -----------------------------
def add_dist_attr(G: nx.Graph) -> nx.Graph:

    H = G.copy()
    for _, _, d in H.edges(data=True):
        w = float(d.get("weight", 1.0))
        if not np.isfinite(w) or w <= 0:
            raise ValueError(f"edge weight must be finite and >0, got {w!r}")
        d["dist"] = 1.0 / max(w, EPS_W)
    return H


def _normalize_edge_weights(G: nx.Graph) -> nx.Graph:
    """Normalize operational edge weights used by positive-weight algorithms.

    The project uses a dual representation in ``signed_split`` mode:
    ``raw_weight`` (signed) and ``weight`` (non-negative magnitude).  This
    helper always produces a positive operational ``weight`` and keeps signed
    fields synchronized when present.
    """
    H = G.copy()
    for _, _, d in H.edges(data=True):
        raw_signed = d.get("raw_weight", d.get("weight_signed", d.get("weight", 1.0)))
        try:
            signed = float(raw_signed)
        except (TypeError, ValueError):
            signed = 1.0
        if not np.isfinite(signed):
            signed = 1.0
        w = abs(signed)
        if w <= 0:
            w = 1.0
        d["weight"] = float(w)
        if "raw_weight" in d or "weight_signed" in d:
            sign = 1.0 if signed > 0 else (-1.0 if signed < 0 else 1.0)
            d["sign"] = float(sign)
            d["raw_weight"] = float(signed if signed != 0 else sign * w)
            d["weight_signed"] = float(d["raw_weight"])
            d["weight_abs"] = float(w)
    return H


# -----------------------------
# Signed Laplacian spectrum
# -----------------------------
def _build_signed_adjacency(G: nx.Graph) -> np.ndarray:
    """Build a dense signed adjacency matrix from edge attributes.

    Preference order is ``raw_weight`` -> ``weight_signed`` -> ``weight`` so
    unsigned graphs degrade gracefully while signed graphs preserve sign.
    """
    nodes = sorted(G.nodes())
    n = len(nodes)
    idx = {node: i for i, node in enumerate(nodes)}
    A = np.zeros((n, n), dtype=float)
    for u, v, d in G.edges(data=True):
        raw = d.get("raw_weight", d.get("weight_signed", d.get("weight", 1.0)))
        try:
            w = float(raw)
        except (TypeError, ValueError):
            w = 0.0
        if not np.isfinite(w):
            w = 0.0
        i, j = idx[u], idx[v]
        A[i, j] = w
        A[j, i] = w
    return A


def signed_laplacian_spectrum(G: nx.Graph, k: int = 3) -> dict[str, float]:
    """Compute signed normalized Laplacian summary metrics.

    Definition::
        D = diag(sum_j |A_ij|),
        L_s = I - D^{-1/2} A D^{-1/2}.

    Returns the smallest eigenvalue (`signed_lambda_min`), second smallest
    (`signed_lambda2`), and `frustration_index` (alias of lambda_min).
    Parameter ``k`` is accepted for API compatibility and currently ignored.
    """
    n = G.number_of_nodes()
    if n < 2 or G.number_of_edges() == 0:
        nan = float("nan")
        return {"signed_lambda_min": nan, "signed_lambda2": nan, "frustration_index": nan}

    A = _build_signed_adjacency(G)
    d_abs = np.abs(A).sum(axis=1)
    mask = d_abs > 1e-15
    if int(np.sum(mask)) < 2:
        nan = float("nan")
        return {"signed_lambda_min": nan, "signed_lambda2": nan, "frustration_index": nan}

    inv_sqrt_d = np.zeros_like(d_abs)
    inv_sqrt_d[mask] = 1.0 / np.sqrt(d_abs[mask])
    D_inv_sqrt = np.diag(inv_sqrt_d)
    L_norm = np.eye(n, dtype=float) - D_inv_sqrt @ A @ D_inv_sqrt
    L_norm = 0.5 * (L_norm + L_norm.T)

    eigvals = np.sort(np.real(np.linalg.eigvalsh(L_norm)))
    lmin = float(max(0.0, eigvals[0]))
    l2 = float(max(0.0, eigvals[1])) if eigvals.size >= 2 else float("nan")
    return {
        "signed_lambda_min": lmin,
        "signed_lambda2": l2,
        "frustration_index": lmin,
    }

# -----------------------------
# 1) Entropy rate of random walk
# -----------------------------
def network_entropy_rate(G: nx.Graph, base: float = math.e) -> float:

    if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
        return 0.0
    A = nx.adjacency_matrix(G, weight="weight").astype(float).tocsr()
    if A.nnz == 0:
        return 0.0
    d = np.asarray(A.sum(axis=1)).flatten()
    total_s = float(d.sum())
    if total_s <= 0:
        return 0.0

    inv_d = np.reciprocal(d, out=np.zeros_like(d), where=d > 0)
    P = A.multiply(inv_d[:, None])

    data_log = np.log(P.data + EPS_LOG) / np.log(base)
    ent_data = -(P.data * data_log)

    P_ent = P.copy()
    P_ent.data = ent_data
    row_ents = np.asarray(P_ent.sum(axis=1)).flatten()
    return float(np.sum((d / total_s) * row_ents))


# -----------------------------
# 2) Ollivier–Ricci curvature (transport W1)
# вычислительно тяжёоая штука, запускается только по кнопке
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
    return {y: w / s for y, w in zip(neigh, ws, strict=False)}


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
    for u, m in zip(S_nodes, S_mass, strict=False):
        Gf.add_node(("S", u), demand=-int(m))
    for v, m in zip(D_nodes, D_mass, strict=False):
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
    max_support: int = RICCI_MAX_SUPPORT,
    cutoff: float = RICCI_CUTOFF,
    scale: int = RICCI_MASS_SCALE,
    missing_cost: float = 1e6,
    _precomputed_Hw: Optional[nx.Graph] = None,
    _sp_cache: Optional[Dict] = None,
) -> Optional[float]:
    """
    κ(x,y) = 1 - W1(µ_x, µ_y)/d(x,y)
    """
    # В summary-режиме сюда может приходить уже нормализованный граф + готовые dist.
    if _precomputed_Hw is not None:
        H = G
        Hw = _precomputed_Hw
    else:
        H = _normalize_edge_weights(as_simple_undirected(G))
        Hw = add_dist_attr(H)
    if not H.has_edge(x, y):
        return None

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
    sp_cache = _sp_cache if _sp_cache is not None else {}
    for u in sx:
        dists = sp_cache.get(u)
        if dists is None:
            dists = nx.single_source_dijkstra_path_length(Hw, u, cutoff=float(cutoff), weight="dist")
            sp_cache[u] = dists
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
    # Расширенные статистики распределения кривизны (для batch/research анализа).
    kappa_var: float = float("nan")
    kappa_skew: float = float("nan")
    kappa_entropy: float = float("nan")
    # Полный вектор kappa; tuple сохраняет hashability dataclass-объекта.
    kappa_values: tuple = ()
    # Доля edge-сэмплов, для которых curvature не удалось вычислить.
    skipped_frac: float = float("nan")
    # Размер подвыборки ребер, фактически использованной в summary.
    sampled_edges: int = 0


@timeit('ollivier_ricci_summary')
def ollivier_ricci_summary(
    G: nx.Graph,
    sample_edges: int = RICCI_SAMPLE_EDGES,
    seed: int = 42,
    max_support: int = RICCI_MAX_SUPPORT,
    cutoff: float = RICCI_CUTOFF,
    scale: int = RICCI_MASS_SCALE,
    progress_cb=None,
    force_sequential: bool = False,
    n_jobs: Optional[int] = None,
    **_ignored,
) -> CurvatureSummary:
    # ЗАМЕТКА: progress_cb нужен только для UI.
    H = _normalize_edge_weights(as_simple_undirected(G))
    Hw = add_dist_attr(H)
    if H.number_of_edges() == 0:
        return CurvatureSummary(
            0.0, 0.0, 0.0, 0, 0, sampled_edges=0, skipped_frac=0.0
        )

    edges = list(H.edges())
    rng = random.Random(int(seed))
    if len(edges) > int(sample_edges):
        edges = rng.sample(edges, int(sample_edges))

    if (progress_cb is not None) or bool(force_sequential):
        results = []
        total = max(1, len(edges))
        sp_cache: Dict = {}
        for i, (x, y) in enumerate(edges, start=1):
            results.append(
                ollivier_ricci_edge(
                    H,
                    x,
                    y,
                    max_support=max_support,
                    cutoff=cutoff,
                    scale=scale,
                    _precomputed_Hw=Hw,
                    _sp_cache=sp_cache,
                )
            )
            if progress_cb is not None:
                try:
                    progress_cb(i, total, x, y)
                except TypeError:
                    progress_cb(i, total)
    else:
        # Управляем параллелизмом снаружи (например, чтобы избежать oversubscription в batch).
        if n_jobs is None:
            num_cores = max(1, min(multiprocessing.cpu_count(), len(edges)))
        else:
            num_cores = max(1, min(int(n_jobs), len(edges)))
        results = Parallel(n_jobs=num_cores)(
            delayed(ollivier_ricci_edge)(
                H,
                x,
                y,
                max_support=max_support,
                cutoff=cutoff,
                scale=scale,
                _precomputed_Hw=Hw,
            )
            for x, y in edges
        )

    kappas = [float(k) for k in results if k is not None and np.isfinite(k)]
    skipped = len(edges) - len(kappas)

    if len(kappas) == 0:
        sampled = int(len(edges))
        return CurvatureSummary(
            float('nan'),
            float('nan'),
            float('nan'),
            0,
            int(skipped),
            float('nan'),
            float('nan'),
            float('nan'),
            (),
            skipped_frac=float(skipped / max(1, sampled)),
            sampled_edges=sampled,
        )

    arr = np.array(kappas, dtype=float)

    # Исследовательские признаки формы распределения kappa.
    # variance: определена от 2 наблюдений.
    k_var = float(arr.var()) if arr.size >= 2 else float("nan")
    # skewness: стабильна только от 3 наблюдений.
    if arr.size >= 3:
        m3 = float(((arr - arr.mean()) ** 3).mean())
        s3 = float(arr.std() ** 3)
        k_skew = m3 / s3 if s3 > 1e-15 else float("nan")
    else:
        k_skew = float("nan")
    # Энтропия гистограммы kappa — признак «размазанности» распределения.
    k_ent = entropy_histogram(arr, bins="fd") if arr.size >= 5 else float("nan")
    sampled = int(len(edges))

    return CurvatureSummary(
        kappa_mean=float(arr.mean()),
        kappa_median=float(np.median(arr)),
        kappa_frac_negative=float((arr < 0).mean()),
        computed_edges=int(arr.size),
        skipped_edges=int(skipped),
        kappa_var=k_var,
        kappa_skew=k_skew,
        kappa_entropy=k_ent,
        kappa_values=tuple(arr.tolist()),
        skipped_frac=float(skipped / max(1, sampled)),
        sampled_edges=sampled,
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
    """Demetrius evolutionary entropy (Perron-Frobenius Markov chain).

    We build the PF-induced Markov chain:
        P_ij = a_ij * u_j / (lam * u_i),  pi_i proportional to u_i * v_i
        H_evo = -sum_i pi_i sum_j P_ij log_base(P_ij)

    Implementation notes:
    - Works with non-negative edge weights. Non-finite / non-positive weights are sanitized to 1.0.
    - Entropy is computed in O(nnz) time without materializing a dense P matrix.
    """
    H = _normalize_edge_weights(as_simple_undirected(G))
    if H.number_of_nodes() < 2 or H.number_of_edges() == 0:
        return float("nan")

    A = nx.adjacency_matrix(H.to_directed(), weight="weight").astype(float).tocsr()
    if A.nnz == 0:
        return float("nan")
    if A.data.size:
        A.data = np.maximum(A.data, 0.0)

    n = int(A.shape[0])

    # Right PF eigenpair (A u = lam u) and left PF eigenvector (A^T v = lam v).
    if n <= 600:
        Ad = A.toarray()
        vals_r, vecs_r = np.linalg.eig(Ad)
        idx_r = int(np.argmax(np.real(vals_r)))
        lam = float(np.real(vals_r[idx_r]))
        u = np.real(vecs_r[:, idx_r])
        vals_l, vecs_l = np.linalg.eig(Ad.T)
        idx_l = int(np.argmax(np.real(vals_l)))
        v = np.real(vecs_l[:, idx_l])
    else:
        lam, u, v = _pf_eigs_sparse(A)

    if not np.isfinite(lam) or lam <= 0:
        return float("nan")

    # Ensure strictly positive vectors (PF theorem gives non-negative; we regularize).
    u = np.abs(u) + 1e-15
    v = np.abs(v) + 1e-15

    pi = u * v
    Z = float(pi.sum())
    if not np.isfinite(Z) or Z <= 0:
        return float("nan")
    pi = pi / Z

    log_base = math.log(base)
    # For row i: P_ij = a_ij * u_j / (lam * u_i). The normalization over j cancels lam*u_i:
    #   sum_j P_ij = (sum_j a_ij u_j) / (lam u_i)
    # so normalized probabilities per row are:
    #   p_ij = (a_ij u_j) / sum_j(a_ij u_j)
    H_evo = 0.0
    indptr = A.indptr
    indices = A.indices
    data = A.data
    for i in range(n):
        start, end = int(indptr[i]), int(indptr[i + 1])
        if start == end:
            continue
        js = indices[start:end]
        aijs = data[start:end]
        weights = aijs * u[js]
        s = float(weights.sum())
        if not np.isfinite(s) or s <= 0:
            continue
        p_row = weights / s
        mask = p_row > 0
        if not mask.any():
            continue
        h_i = -float((p_row[mask] * (np.log(p_row[mask]) / log_base)).sum())
        H_evo += float(pi[i]) * h_i

    return float(H_evo)
