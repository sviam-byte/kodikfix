from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd


def _trapz_compat(y, x=None, dx: float = 1.0) -> float:
    """Compat trapezoidal integration across NumPy versions."""
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


def _safe_float(x, default: float = float("nan")) -> float:
    try:
        v = float(x)
    except Exception:
        return float(default)
    if not np.isfinite(v):
        return float(default)
    return v


def _largest_component_frac(G: nx.Graph) -> float:
    n = G.number_of_nodes()
    if n <= 0:
        return 0.0
    if G.is_directed():
        comps = list(nx.weakly_connected_components(G))
    else:
        comps = list(nx.connected_components(G))
    if not comps:
        return 0.0
    return float(max(len(c) for c in comps) / n)


def graph_resistance_summary(G: nx.Graph) -> dict:
    H = G.to_undirected() if getattr(G, "is_directed", lambda: False)() else G
    n = int(H.number_of_nodes())
    e = int(H.number_of_edges())
    out = {
        "n_nodes": n,
        "n_edges": e,
        "density": float(nx.density(H)) if n > 1 else 0.0,
        "n_components": int(nx.number_connected_components(H)) if n > 0 else 0,
        "giant_component_frac": _largest_component_frac(H),
        "avg_clustering": float(nx.average_clustering(H, weight="weight")) if n > 0 else 0.0,
        "max_core": float("nan"),
        "mean_core": float("nan"),
        "algebraic_connectivity": float("nan"),
        "node_connectivity": float("nan"),
        "edge_connectivity": float("nan"),
        "assortativity": float("nan"),
    }
    try:
        core = nx.core_number(H) if n > 0 and e > 0 else {}
        if core:
            vals = np.array(list(core.values()), dtype=float)
            out["max_core"] = float(vals.max())
            out["mean_core"] = float(vals.mean())
    except Exception:
        pass
    try:
        out["assortativity"] = float(nx.degree_assortativity_coefficient(H))
    except Exception:
        pass
    try:
        if nx.is_connected(H) and n <= 400:
            out["algebraic_connectivity"] = float(nx.algebraic_connectivity(H, weight="weight"))
    except Exception:
        pass
    try:
        if n <= 250 and e > 0:
            out["node_connectivity"] = float(nx.node_connectivity(H))
            out["edge_connectivity"] = float(nx.edge_connectivity(H))
    except Exception:
        pass
    return out


def attack_trajectory_summary(df_hist: pd.DataFrame, *, attack_kind: str = "") -> dict:
    if df_hist is None or df_hist.empty:
        return {"attack_kind": str(attack_kind), "steps": 0}
    df = df_hist.copy()
    xcol = "mix_frac" if "mix_frac" in df.columns else "removed_frac" if "removed_frac" in df.columns else "step"
    xs = pd.to_numeric(df[xcol], errors="coerce").to_numpy(dtype=float)
    out = {
        "attack_kind": str(attack_kind),
        "x_col": xcol,
        "steps": int(len(df)),
    }
    if "lcc_frac" in df.columns:
        ys = pd.to_numeric(df["lcc_frac"], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(xs) & np.isfinite(ys)
        if mask.sum() >= 1:
            yv = ys[mask]
            xv = xs[mask]
            out["final_lcc_frac"] = float(yv[-1])
            out["auc_lcc_frac"] = _trapz_compat(yv, x=xv) if yv.size >= 2 else float(yv.sum())
            hits50 = np.where(yv <= 0.5)[0]
            hits10 = np.where(yv <= 0.1)[0]
            out["collapse_step_50"] = float(xv[hits50[0]]) if hits50.size else float("nan")
            out["collapse_step_10"] = float(xv[hits10[0]]) if hits10.size else float("nan")
    if "eff_w" in df.columns:
        ys = pd.to_numeric(df["eff_w"], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(xs) & np.isfinite(ys)
        if mask.sum() >= 1:
            yv = ys[mask]
            xv = xs[mask]
            out["final_eff_w"] = float(yv[-1])
            out["auc_eff_w"] = _trapz_compat(yv, x=xv) if yv.size >= 2 else float(yv.sum())
    return out
