import math
from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd

from .config import settings
from .core_math import ollivier_ricci_edge
from .metrics import add_dist_attr, calculate_metrics, compute_energy_flow
from .utils import as_simple_undirected, get_node_strength


def _pick_nodes_adaptive(
    H: nx.Graph,
    attack_kind: str,
    k: int,
    rng: np.random.Generator,
) -> Optional[list]:

    if k <= 0 or H.number_of_nodes() == 0:
        return []

    nodes = list(H.nodes())

    if attack_kind == "random":
        rng.shuffle(nodes)
        return nodes[:k]

    if attack_kind == "low_degree":
        nodes.sort(key=lambda n: H.degree(n))
        return nodes[:k]

    if attack_kind == "weak_strength":
        nodes.sort(key=lambda n: get_node_strength(H, n))
        return nodes[:k]

    return None

# =========================
# Rich-Club helpers
# =========================
def strength_ranking(G: nx.Graph) -> list:
    strength = dict(G.degree(weight="weight"))
    nodes = list(G.nodes())
    nodes_sorted = sorted(nodes, key=lambda n: strength.get(n, 0.0), reverse=True)
    return nodes_sorted


def richclub_top_fraction(G: nx.Graph, rc_frac: float) -> list:
    nodes_sorted = strength_ranking(G)
    if not nodes_sorted:
        return []
    k = max(1, int(len(nodes_sorted) * float(rc_frac)))
    return nodes_sorted[:k]


def richclub_by_density_threshold(G: nx.Graph, min_density: float, max_frac: float) -> list:
    nodes_sorted = strength_ranking(G)
    n = len(nodes_sorted)
    if n == 0:
        return []
    if n < 3:
        return nodes_sorted

    maxK = max(3, int(n * float(max_frac)))
    maxK = min(maxK, n)

    best = nodes_sorted[:3]
    for K in range(3, maxK + 1):
        club = nodes_sorted[:K]
        H = G.subgraph(club)
        dens = nx.density(H)
        if dens >= float(min_density):
            best = club
    return best


def pick_targets_for_attack(
    G: nx.Graph,
    attack_kind: str,
    step_size: int,
    seed: int,
    rc_frac: float,
    rc_min_density: float,
    rc_max_frac: float,
    fast_mode: bool = False,
) -> list:
    nodes = list(G.nodes())
    if not nodes:
        return []

    rng = np.random.default_rng(int(seed))

    if attack_kind == "random":
        k = min(len(nodes), step_size)
        if k <= 0:
            return []
        # Generator.choice returns ndarray; convert to plain python list.
        return rng.choice(nodes, size=k, replace=False).tolist()

    if attack_kind == "degree":
        strength = dict(G.degree(weight="weight"))
        return sorted(nodes, key=lambda n: strength.get(n, 0.0), reverse=True)[:step_size]

    if attack_kind == "betweenness":
        H = add_dist_attr(G)
        n = H.number_of_nodes()
        # ОПТИМИЗАЦИЯ: контролируем k_samples.
        # В fast_mode делаем очень грубую оценку (15 сэмплов достаточно для топ-узлов)
        if fast_mode:
            k_samples = min(15, n)
        else:
            k_samples = min(int(math.sqrt(n)) + 20, 100, n)
        bc = nx.betweenness_centrality(H, k=k_samples, weight="dist", normalized=True, seed=int(seed))
        return sorted(nodes, key=lambda n: bc.get(n, 0.0), reverse=True)[:step_size]

    if attack_kind == "kcore":
        core = nx.core_number(G)
        return sorted(nodes, key=lambda n: core.get(n, 0), reverse=True)[:step_size]

    if attack_kind == "richclub_top":
        club = richclub_top_fraction(G, rc_frac=rc_frac)
        if not club:
            return []
        return club[:min(step_size, len(club))]

    if attack_kind == "richclub_density":
        club = richclub_by_density_threshold(G, min_density=rc_min_density, max_frac=rc_max_frac)
        if not club:
            return []
        return club[:min(step_size, len(club))]

    return []


def lcc_fraction(G: nx.Graph, N0: int) -> float:
    if G.number_of_nodes() == 0 or N0 <= 0:
        return 0.0
    lcc = len(max(nx.connected_components(G), key=len))
    return float(lcc) / float(N0)

# =========================
# Attack simulation
# =========================
def run_attack(
    G_in: nx.Graph,
    attack_kind: str,
    remove_frac: float,
    steps: int,
    seed: int,
    eff_sources_k: int,
    rc_frac: float = 0.10,
    rc_min_density: float = 0.30,
    rc_max_frac: float = 0.30,
    compute_heavy_every: int = 1,
    keep_states: bool = False,
    fast_mode: bool = False,
    progress_cb=None,
):
    G = as_simple_undirected(G_in).copy()
    N0 = G.number_of_nodes()

    if N0 < 2:
        return pd.DataFrame(), {"removed_nodes": [], "states": []}

    total_to_remove = int(N0 * float(remove_frac))
    is_adaptive = attack_kind in ("low_degree", "weak_strength", "random")

    static_targets = []
    if not is_adaptive:
        static_targets = pick_targets_for_attack(
            G, attack_kind, total_to_remove, int(seed), rc_frac, rc_min_density, rc_max_frac,
            fast_mode=fast_mode
        )

    ks = np.linspace(0, total_to_remove, int(steps) + 1).round().astype(int).tolist()
    np_rng = np.random.default_rng(int(seed))
    history, states, removed_log = [], [], []

    for i, target_k in enumerate(ks):
        # прогрессбар для UI
        if progress_cb is not None:
            try:
                progress_cb(i, len(ks) - 1, target_k)
            except TypeError:
                progress_cb(i, len(ks) - 1)
        if G.number_of_nodes() == 0 and i > 0:
            break

        if keep_states:
            states.append(G.copy())
        # В fast_mode считаем тяжелые метрики только на первых/последних шагах
        is_first_last = (i == 0) or (i == len(ks) - 1)
        if fast_mode:
            heavy = is_first_last or (i % max(1, int(compute_heavy_every) * 2) == 0)
        else:
            heavy = (i % max(1, int(compute_heavy_every)) == 0)

        skip_spectral = not heavy

        met = calculate_metrics(
            G,
            int(eff_sources_k),
            int(seed),
            False,
            skip_spectral=skip_spectral,
            compute_heavy=heavy,
        )
        met.update(
            {
                "step": i,
                "removed_frac": len(removed_log) / N0,
                "lcc_frac": lcc_fraction(G, N0),
                "N": met.get("N", G.number_of_nodes()),
                "E": met.get("E", G.number_of_edges()),
                "eff_w": met.get("eff_w", np.nan),
                "mod": met.get("mod", np.nan),
                "l2_lcc": met.get("l2_lcc", np.nan),
                "clustering": met.get("clustering", np.nan),
            }
        )
        history.append(met)

        if i < len(ks) - 1:
            num_to_del = ks[i + 1] - len(removed_log)
            if num_to_del > 0:
                if is_adaptive:
                    targets = _pick_nodes_adaptive(G, attack_kind, num_to_del, np_rng) or []
                else:
                    targets = static_targets[len(removed_log) : ks[i + 1]]

                G.remove_nodes_from(targets)
                removed_log.extend(targets)

    return pd.DataFrame(history), {"removed_nodes": removed_log, "states": states}


def run_edge_attack(
    G: nx.Graph,
    kind: str,
    frac: float,
    steps: int,
    seed: int,
    eff_k: int,
    compute_heavy_every: int = 2,
    compute_curvature: bool = False,
    curvature_sample_edges: int = 80,
    progress_cb=None,
):

    if G.number_of_edges() == 0:
        df = pd.DataFrame(
            [
                {
                    "step": 0,
                    "removed_frac": 0.0,
                    "N": G.number_of_nodes(),
                    "E": 0,
                    "lcc_frac": 0.0,
                }
            ]
        )
        return df, {"removed_edges_order": [], "total_edges": 0, "kind": kind}

    H0 = as_simple_undirected(G)
    edges = list(H0.edges(data=True))
    kind = str(kind)

    # --------------------------
    # --------------------------
    def _safe_float(value, default: float = 0.0) -> float:
        """Convert to float with finite fallback for edge attributes."""
        try:
            v = float(value)
        except (TypeError, ValueError):
            return float(default)
        return v if np.isfinite(v) else float(default)

    if kind in (
        "weak_edges_by_weight",
        "weak_edges_by_confidence",
        "strong_edges_by_weight",
        "strong_edges_by_confidence",
    ):
        if "confidence" in kind:
            key = lambda e: float(e[2].get("confidence", 1.0))
        else:
            key = lambda e: _safe_float(e[2].get("weight", 1.0), 1.0)
        edges.sort(key=key, reverse=kind.startswith("strong_"))

    else:
        # --------------------------
        # Ricci / Flux
        # --------------------------
        rng = np.random.default_rng(int(seed))
        max_eval = 600 
        edge_list = [(u, v) for (u, v, _d) in edges]
        if len(edge_list) > max_eval:
            sample_idx = rng.choice(len(edge_list), size=max_eval, replace=False)
            sampled = [edge_list[i] for i in sample_idx]
        else:
            sampled = edge_list

        kappa: dict[tuple, float] = {}
        flux: dict[tuple, float] = {}

        if kind in ("flux_high_rw", "flux_high_evo", "flux_high_rw_x_neg_ricci"):
            flow_mode = "evo" if kind.endswith("_evo") else "rw"
            _ne, ef = compute_energy_flow(H0, steps=20, flow_mode=flow_mode, damping=1.0)
            flux = dict(ef)

        if kind.startswith("ricci_") or kind == "flux_high_rw_x_neg_ricci":
            for (u, v) in sampled:
                try:
                    val = ollivier_ricci_edge(
                        H0,
                        u,
                        v,
                        max_support=settings.RICCI_MAX_SUPPORT,
                        cutoff=settings.RICCI_CUTOFF,
                    )
                except (ValueError, RuntimeError):
                    val = None
                if val is None:
                    continue
                val = _safe_float(val, default=float("nan"))
                if not np.isfinite(val):
                    continue
                kappa[(u, v)] = val

        def flux_uv(u, v) -> float:
            if (u, v) in flux:
                return float(flux[(u, v)])
            if (v, u) in flux:
                return float(flux[(v, u)])
            return 0.0

        def kappa_uv(u, v) -> float:
            if (u, v) in kappa:
                return float(kappa[(u, v)])
            if (v, u) in kappa:
                return float(kappa[(v, u)])
            return 0.0

        def score(u, v, d) -> float:
            if kind == "flux_high_rw":
                return flux_uv(u, v)
            if kind == "flux_high_evo":
                return flux_uv(u, v)
            if kind == "ricci_most_negative":
                return -kappa_uv(u, v)
            if kind == "ricci_most_positive":
                return kappa_uv(u, v)
            if kind == "ricci_abs_max":
                return abs(kappa_uv(u, v))
            if kind == "flux_high_rw_x_neg_ricci":
                return flux_uv(u, v) * max(0.0, -kappa_uv(u, v))
            return _safe_float(d.get("weight", 1.0), 1.0)

        edges.sort(key=lambda e: score(e[0], e[1], e[2]), reverse=True)

    total_e = len(edges)
    remove_total = int(round(float(frac) * total_e))
    remove_total = max(0, min(remove_total, total_e))

    steps = max(1, int(steps))
    ks = np.linspace(0, remove_total, steps + 1).round().astype(int).tolist()

    removed_order = [(u, v) for (u, v, _) in edges[:remove_total]]
    H = H0.copy()

    rows = []
    for i, k in enumerate(ks):

        if progress_cb is not None:
            try:
                progress_cb(i, len(ks) - 1, k)
            except TypeError:
                progress_cb(i, len(ks) - 1)

        if i > 0:
            prev = ks[i - 1]
            for (u, v) in removed_order[prev:k]:
                if H.has_edge(u, v):
                    H.remove_edge(u, v)

        removed_frac = (k / total_e) if total_e else 0.0
        heavy = (i % int(max(1, compute_heavy_every)) == 0) or (i == steps)

        is_really_heavy_step = (i == 0) or (i == len(ks) - 1)
        skip_spectral = (H.number_of_nodes() > 500) and not is_really_heavy_step

        metrics = calculate_metrics(
            H,
            eff_sources_k=int(eff_k),
            seed=int(seed),
            compute_curvature=bool(compute_curvature and heavy),
            curvature_sample_edges=int(curvature_sample_edges),
            skip_spectral=skip_spectral,
        )

        row = {
            "step": i,
            "removed_frac": float(removed_frac),
            "removed_k": int(k),
            "N": int(metrics.get("N", H.number_of_nodes())),
            "E": int(metrics.get("E", H.number_of_edges())),
            "C": int(metrics.get("C", np.nan)) if "C" in metrics else np.nan,
            "lcc_size": int(metrics.get("lcc_size", np.nan)) if "lcc_size" in metrics else np.nan,
            "lcc_frac": float(metrics.get("lcc_frac", np.nan)) if "lcc_frac" in metrics else np.nan,
            "density": float(metrics.get("density", np.nan)) if "density" in metrics else np.nan,
            "avg_degree": float(metrics.get("avg_degree", np.nan)) if "avg_degree" in metrics else np.nan,
            "clustering": float(metrics.get("clustering", np.nan)) if "clustering" in metrics else np.nan,
            "assortativity": float(metrics.get("assortativity", np.nan)) if "assortativity" in metrics else np.nan,
            "eff_w": float(metrics.get("eff_w", np.nan)) if "eff_w" in metrics else np.nan,
            "mod": float(metrics.get("mod", np.nan)) if heavy else np.nan,
            "l2_lcc": float(metrics.get("l2_lcc", np.nan)) if heavy else np.nan,
        }
        rows.append(row)

    df_hist = pd.DataFrame(rows)
    aux = {
        "removed_edges_order": removed_order,
        "total_edges": total_e,
        "kind": kind,
    }
    return df_hist, aux
