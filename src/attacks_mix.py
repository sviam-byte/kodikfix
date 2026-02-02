"""Mix/Hrish attacks that rewire or replace edges while preserving distributions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import networkx as nx

from .metrics import calculate_metrics
from .null_models import make_er_gnm, make_configuration_model
from .core_math import (
    entropy_degree,
    entropy_weights,
    entropy_confidence,
    entropy_triangle_support,
)
from .utils import as_simple_undirected


def _safe_attr_float(x, default: float = 1.0) -> float:
    """Привести к float и вернуть default при нечисловых/бесконечных значениях."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float(default)
    return float(v) if np.isfinite(v) else float(default)


def _sample_edge_attrs_from_empirical(
    rng: np.random.Generator,
    attrs_pool: list[dict],
) -> dict:
    """Sample edge attrs (weight/confidence) from empirical pool."""
    if not attrs_pool:
        return {"weight": 1.0, "confidence": 1.0}
    d = attrs_pool[int(rng.integers(0, len(attrs_pool)))]
    out = {}
    if "weight" in d:
        out["weight"] = _safe_attr_float(d.get("weight", 1.0))
    if "confidence" in d:
        out["confidence"] = _safe_attr_float(d.get("confidence", 1.0))
    return out


def _edge_swap_degree_preserving(H: nx.Graph, n_swaps: int, seed: int) -> int:
    """Degree-preserving double-edge swap (best-effort)."""
    if H.number_of_edges() < 2 or H.number_of_nodes() < 4 or n_swaps <= 0:
        return 0
    try:
        nx.double_edge_swap(
            H,
            nswap=int(n_swaps),
            max_tries=int(n_swaps) * 20,
            seed=int(seed),
        )
        return int(n_swaps)
    except nx.NetworkXError:
        # Не удалось найти валидные свопы (граф слишком мал или жесткая топология).
        return 0


def _replace_edges_from_source(
    H: nx.Graph,
    source_edges: list[tuple],
    k_replace: int,
    rng: np.random.Generator,
    attrs_pool: list[dict],
) -> int:
    """
    Remove k edges at random, then add k edges sampled from a source null graph.
    Edge attrs are sampled from the empirical pool.
    """
    if k_replace <= 0 or H.number_of_edges() == 0:
        return 0
    edges_H = list(H.edges())
    rng.shuffle(edges_H)
    to_remove = edges_H[: min(k_replace, len(edges_H))]
    H.remove_edges_from(to_remove)

    added = 0
    tries = 0
    max_tries = max(1000, 20 * k_replace)
    while added < k_replace and tries < max_tries:
        tries += 1
        u, v = source_edges[int(rng.integers(0, len(source_edges)))]
        if u == v or H.has_edge(u, v):
            continue
        attrs = _sample_edge_attrs_from_empirical(rng, attrs_pool)
        H.add_edge(u, v, **attrs)
        added += 1

    return int(added)


def run_mix_attack(
    G: nx.Graph,
    kind: str,
    steps: int,
    seed: int,
    eff_sources_k: int,
    heavy_every: int = 2,
    alpha_rewire: float = 0.6,
    beta_replace: float = 0.4,
    swaps_per_edge: float = 0.5,
    replace_from: str = "ER",
    progress_cb=None,
):
    """
    Run mix-style attacks with a progress axis in mix_frac [0,1].

    kind:
      - "mix_degree_preserving": rewire only via double-edge-swap
      - "mix_weightconf_preserving": replace edges from null model, sample attrs
      - "hrish_mix": both channels using alpha/beta
    """
    rng = np.random.default_rng(int(seed))
    H0 = as_simple_undirected(G)
    H = H0.copy()

    N = H.number_of_nodes()
    M = H.number_of_edges()
    if N == 0:
        return pd.DataFrame([{"step": 0, "mix_frac": 0.0, "N": 0, "E": 0}]), {"kind": kind}

    steps = max(1, int(steps))
    xs = np.linspace(0.0, 1.0, steps + 1).tolist()

    attrs_pool = []
    for _, _, d in H0.edges(data=True):
        attrs_pool.append(
            {
                "weight": _safe_attr_float(d.get("weight", 1.0), 1.0),
                "confidence": _safe_attr_float(d.get("confidence", 1.0), 1.0),
            }
        )

    if replace_from.upper() == "CFG":
        Gsrc = make_configuration_model(H0, seed=int(seed) + 999)
    else:
        Gsrc = make_er_gnm(N, M, seed=int(seed) + 999)

    source_edges = list(as_simple_undirected(Gsrc).edges())

    rows = []
    total_swaps_done = 0
    total_replaced_done = 0

    for i, x in enumerate(xs):
        # UI-progress: пригодится на больших графах, иначе просто None
        if progress_cb is not None:
            try:
                progress_cb(i, len(xs) - 1, x)
            except TypeError:
                progress_cb(i, len(xs) - 1)

        heavy = (i % int(max(1, heavy_every)) == 0) or (i == steps)

        if i > 0:
            dx = float(x) - float(xs[i - 1])
            dx = max(0.0, dx)
            target_ops = max(1, int(round(dx * float(M))))

            if kind == "mix_degree_preserving":
                n_swaps = int(round(target_ops * float(swaps_per_edge)))
                total_swaps_done += _edge_swap_degree_preserving(H, n_swaps=n_swaps, seed=int(seed) + i)
            elif kind == "mix_weightconf_preserving":
                k_rep = int(round(target_ops))
                total_replaced_done += _replace_edges_from_source(
                    H, source_edges, k_replace=k_rep, rng=rng, attrs_pool=attrs_pool
                )
            else:
                a = float(alpha_rewire)
                b = float(beta_replace)
                s = max(1e-9, a + b)
                a /= s
                b /= s

                n_swaps = int(round(a * target_ops * float(swaps_per_edge)))
                k_rep = int(round(b * target_ops))
                total_swaps_done += _edge_swap_degree_preserving(H, n_swaps=n_swaps, seed=int(seed) + i)
                total_replaced_done += _replace_edges_from_source(
                    H, source_edges, k_replace=k_rep, rng=rng, attrs_pool=attrs_pool
                )

        m = calculate_metrics(
            H,
            eff_sources_k=int(eff_sources_k),
            seed=int(seed),
            compute_curvature=False,
        )

        row = {
            "step": int(i),
            "mix_frac": float(x),
            "N": int(m.get("N", H.number_of_nodes())),
            "E": int(m.get("E", H.number_of_edges())),
            "C": int(m.get("C", np.nan)) if "C" in m else np.nan,
            "lcc_size": int(m.get("lcc_size", np.nan)) if "lcc_size" in m else np.nan,
            "lcc_frac": float(m.get("lcc_frac", np.nan)) if "lcc_frac" in m else np.nan,
            "density": float(m.get("density", np.nan)) if "density" in m else np.nan,
            "avg_degree": float(m.get("avg_degree", np.nan)) if "avg_degree" in m else np.nan,
            "clustering": float(m.get("clustering", np.nan)) if "clustering" in m else np.nan,
            "assortativity": float(m.get("assortativity", np.nan)) if "assortativity" in m else np.nan,
            "eff_w": float(m.get("eff_w", np.nan)) if "eff_w" in m else np.nan,
            "H_deg": entropy_degree(H),
            "H_w": entropy_weights(H),
            "H_conf": entropy_confidence(H),
        }

        if heavy:
            row["H_tri"] = entropy_triangle_support(H)
            row["mod"] = float(m.get("mod", np.nan)) if "mod" in m else np.nan
            row["l2_lcc"] = float(m.get("l2_lcc", np.nan)) if "l2_lcc" in m else np.nan
        else:
            row["H_tri"] = np.nan
            row["mod"] = np.nan
            row["l2_lcc"] = np.nan

        rows.append(row)

    df = pd.DataFrame(rows)
    for col in ["H_tri", "mod", "l2_lcc"]:
        if col in df.columns:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan).ffill()

    aux = {
        "kind": kind,
        "total_swaps_done": int(total_swaps_done),
        "total_replaced_done": int(total_replaced_done),
        "replace_from": replace_from,
        "alpha_rewire": float(alpha_rewire),
        "beta_replace": float(beta_replace),
        "swaps_per_edge": float(swaps_per_edge),
    }
    return df, aux
