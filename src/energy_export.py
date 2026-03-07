from __future__ import annotations

from typing import Iterable

import networkx as nx
import numpy as np
import pandas as pd


def _safe_float(x, default: float = 0.0) -> float:
    try:
        v = float(x)
    except Exception:
        return float(default)
    if not np.isfinite(v):
        return float(default)
    return v


def _gini(x: np.ndarray) -> float:
    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    arr = np.clip(arr, 0.0, None)
    s = float(arr.sum())
    if s <= 0:
        return 0.0
    arr = np.sort(arr)
    n = arr.size
    idx = np.arange(1, n + 1, dtype=float)
    return float((2.0 * np.sum(idx * arr) / (n * s)) - (n + 1.0) / n)


def _entropy(x: np.ndarray) -> float:
    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    arr = np.clip(arr, 0.0, None)
    s = float(arr.sum())
    if s <= 0:
        return 0.0
    p = arr / s
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    return float(-(p * np.log2(p)).sum())


def _top_share(arr: np.ndarray, k: int) -> float:
    vals = np.asarray(arr, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0
    vals = np.clip(vals, 0.0, None)
    s = float(vals.sum())
    if s <= 0:
        return 0.0
    vals.sort()
    return float(vals[-max(1, int(k)):].sum() / s)


def _trapz_compat(y, x=None, dx: float = 1.0) -> float:
    """Compat trapezoidal integration across NumPy versions.

    Newer NumPy provides ``np.trapezoid`` and older releases expose ``np.trapz``.
    This wrapper keeps behavior stable and includes a small local fallback for
    rare environments where neither helper is available.
    """
    if hasattr(np, "trapezoid"):
        if x is None:
            return float(np.trapezoid(y, dx=dx))
        return float(np.trapezoid(y, x=x))
    if hasattr(np, "trapz"):
        if x is None:
            return float(np.trapz(y, dx=dx))
        return float(np.trapz(y, x=x))

    y = np.asarray(y, dtype=float)
    if y.size < 2:
        return float(y.sum())
    if x is None:
        return float(dx * (0.5 * y[0] + y[1:-1].sum() + 0.5 * y[-1]))

    x = np.asarray(x, dtype=float)
    if x.size != y.size:
        raise ValueError("x and y must have the same length")
    return float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) * 0.5))


def _time_to_frac(active_counts: np.ndarray, target: float, n_nodes: int) -> float:
    if n_nodes <= 0:
        return float("nan")
    threshold = float(target) * float(n_nodes)
    hits = np.where(active_counts >= threshold)[0]
    return float(hits[0]) if hits.size else float("nan")


def _diffusion_radius(G: nx.Graph, energies: dict, sources: Iterable | None) -> float:
    if G.number_of_nodes() <= 1:
        return 0.0
    srcs = [s for s in (sources or []) if s in G]
    if not srcs:
        if not energies:
            return 0.0
        srcs = [max(energies, key=lambda n: _safe_float(energies.get(n, 0.0)))]
    pos_energy = {n: max(0.0, _safe_float(v)) for n, v in energies.items()}
    total = float(sum(pos_energy.values()))
    if total <= 0:
        return 0.0
    dists = []
    for s in srcs:
        try:
            dists.append(nx.single_source_shortest_path_length(G, s))
        except Exception:
            pass
    if not dists:
        return 0.0
    out = 0.0
    for n, e in pos_energy.items():
        if e <= 0:
            continue
        best = min((dm.get(n, np.nan) for dm in dists), default=np.nan)
        if np.isfinite(best):
            out += float(e) * float(best)
    return float(out / total) if total > 0 else 0.0


def frames_to_energy_nodes_long(
    G: nx.Graph,
    node_frames: list[dict],
    *,
    sources: list | None = None,
) -> pd.DataFrame:
    nodes = list(G.nodes())
    seed_set = set([s for s in (sources or []) if s in G])
    strength = dict(G.degree(weight="weight")) if G is not None else {}
    degree = dict(G.degree()) if G is not None else {}
    rows: list[dict] = []
    for step, frame in enumerate(node_frames or []):
        vals = np.array([_safe_float(frame.get(n, 0.0)) for n in nodes], dtype=float)
        total = float(vals.sum())
        order = np.argsort(-vals) if vals.size else np.array([], dtype=int)
        rank_map = {nodes[idx]: int(rank + 1) for rank, idx in enumerate(order)}
        for n in nodes:
            e = _safe_float(frame.get(n, 0.0))
            rows.append(
                {
                    "step": int(step),
                    "node": str(n),
                    "energy": float(e),
                    "energy_norm": float(e / total) if total > 0 else 0.0,
                    "rank_energy": int(rank_map.get(n, len(nodes) + 1)),
                    "is_seed": bool(n in seed_set),
                    "degree": float(degree.get(n, 0.0)),
                    "strength": float(strength.get(n, 0.0)),
                }
            )
    return pd.DataFrame(rows)


def frames_to_energy_steps_summary(
    G: nx.Graph,
    node_frames: list[dict],
    edge_frames: list[dict],
    *,
    sources: list | None = None,
) -> pd.DataFrame:
    nodes = list(G.nodes())
    n_nodes = len(nodes)
    rows: list[dict] = []
    for step, frame in enumerate(node_frames or []):
        vals = np.array([_safe_float(frame.get(n, 0.0)) for n in nodes], dtype=float)
        active = int(np.sum(vals > 1e-12))
        flux_vals = np.array(list((edge_frames[step] or {}).values()), dtype=float) if step < len(edge_frames or []) else np.array([], dtype=float)
        rows.append(
            {
                "step": int(step),
                "total_energy": float(vals.sum()) if vals.size else 0.0,
                "active_nodes": active,
                "active_frac": float(active / n_nodes) if n_nodes > 0 else 0.0,
                "max_energy": float(vals.max()) if vals.size else 0.0,
                "mean_energy": float(vals.mean()) if vals.size else 0.0,
                "std_energy": float(vals.std()) if vals.size else 0.0,
                "entropy": _entropy(vals),
                "gini": _gini(vals),
                "top1_share": _top_share(vals, 1),
                "top5_share": _top_share(vals, 5),
                "flux_total": float(flux_vals.sum()) if flux_vals.size else 0.0,
                "flux_mean": float(flux_vals.mean()) if flux_vals.size else 0.0,
                "flux_max": float(flux_vals.max()) if flux_vals.size else 0.0,
            }
        )
    return pd.DataFrame(rows)


def energy_run_summary_dict(
    G: nx.Graph,
    node_frames: list[dict],
    edge_frames: list[dict],
    *,
    sources: list | None = None,
    flow_mode: str = "rw",
) -> dict:
    steps_df = frames_to_energy_steps_summary(G, node_frames, edge_frames, sources=sources)
    if steps_df.empty:
        return {
            "flow_mode": str(flow_mode),
            "n_steps": 0,
            "n_nodes": int(G.number_of_nodes()),
            "n_edges": int(G.number_of_edges()),
            "seed_count": int(len([s for s in (sources or []) if s in G])),
        }
    total_energy = pd.to_numeric(steps_df["total_energy"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    active_nodes = pd.to_numeric(steps_df["active_nodes"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    entropy = pd.to_numeric(steps_df["entropy"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    gini = pd.to_numeric(steps_df["gini"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    final_frame = (node_frames or [{}])[-1] if node_frames else {}
    final_vals = np.array([_safe_float(final_frame.get(n, 0.0)) for n in G.nodes()], dtype=float)
    peak_step = int(np.nanargmax(total_energy)) if total_energy.size else 0
    return {
        "flow_mode": str(flow_mode),
        "n_steps": int(max(0, len(node_frames or []) - 1)),
        "n_nodes": int(G.number_of_nodes()),
        "n_edges": int(G.number_of_edges()),
        "seed_count": int(len([s for s in (sources or []) if s in G])),
        "initial_total_energy": float(total_energy[0]) if total_energy.size else 0.0,
        "final_total_energy": float(total_energy[-1]) if total_energy.size else 0.0,
        "peak_total_energy": float(np.nanmax(total_energy)) if total_energy.size else 0.0,
        "peak_step": int(peak_step),
        "final_active_nodes": int(active_nodes[-1]) if active_nodes.size else 0,
        "final_active_frac": float(active_nodes[-1] / max(1, G.number_of_nodes())) if active_nodes.size else 0.0,
        "time_to_50pct_nodes": _time_to_frac(active_nodes, 0.5, G.number_of_nodes()),
        "time_to_90pct_nodes": _time_to_frac(active_nodes, 0.9, G.number_of_nodes()),
        "final_entropy": _entropy(final_vals),
        "final_gini": _gini(final_vals),
        "final_top1_share": _top_share(final_vals, 1),
        "final_top5_share": _top_share(final_vals, 5),
        "auc_active_nodes": _trapz_compat(active_nodes, dx=1.0) if active_nodes.size >= 2 else float(active_nodes.sum()),
        "auc_total_energy": _trapz_compat(total_energy, dx=1.0) if total_energy.size >= 2 else float(total_energy.sum()),
        "auc_entropy": _trapz_compat(entropy, dx=1.0) if entropy.size >= 2 else float(entropy.sum()),
        "auc_gini": _trapz_compat(gini, dx=1.0) if gini.size >= 2 else float(gini.sum()),
        "final_diffusion_radius": _diffusion_radius(G, final_frame, sources),
    }
