from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd
from networkx.algorithms.community import louvain_communities

from .attacks import run_edge_attack
from .attacks_mix import run_mix_attack
from .metrics import calculate_metrics
from .utils import as_simple_undirected


@dataclass
class ModuleInfo:
    membership: dict
    communities: list[set]


def _safe_float(x, default: float = 0.0) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float(default)
    return float(v) if np.isfinite(v) else float(default)


def _copy_graph_with_same_nodes(G: nx.Graph) -> nx.Graph:
    H = nx.Graph()
    H.add_nodes_from(G.nodes(data=True))
    return H


def _baseline_density(G: nx.Graph) -> float:
    if G.number_of_nodes() <= 1:
        return 0.0
    return float(nx.density(G))


def _target_edge_count_for_density(n_nodes: int, density: float) -> int:
    if n_nodes <= 1:
        return 0
    max_edges = n_nodes * (n_nodes - 1) // 2
    k = int(round(float(density) * float(max_edges)))
    return max(0, min(max_edges, k))


def prepare_module_info(
    G: nx.Graph,
    *,
    seed: int = 42,
    resolution: float = 1.0,
) -> ModuleInfo:
    H = as_simple_undirected(G)
    if H.number_of_nodes() == 0:
        return ModuleInfo(membership={}, communities=[])

    communities = louvain_communities(
        H,
        weight="weight",
        seed=int(seed),
        resolution=float(resolution),
    )
    membership = {}
    for idx, comm in enumerate(communities):
        for node in comm:
            membership[node] = int(idx)
    return ModuleInfo(
        membership=membership,
        communities=[set(c) for c in communities],
    )


def _classify_edges_by_modules(
    G: nx.Graph,
    membership: dict,
) -> tuple[list[tuple], list[tuple]]:
    inter_edges: list[tuple] = []
    intra_edges: list[tuple] = []
    for u, v, d in G.edges(data=True):
        mu = membership.get(u, -1)
        mv = membership.get(v, -1)
        item = (u, v, d)
        if mu != mv:
            inter_edges.append(item)
        else:
            intra_edges.append(item)
    return inter_edges, intra_edges


def _sort_edges_for_removal(
    edges: list[tuple],
    *,
    mode: str,
    rng: np.random.Generator,
) -> list[tuple]:
    out = list(edges)

    if mode == "random":
        rng.shuffle(out)
        return out

    if mode == "weak_weight":
        out.sort(key=lambda e: _safe_float(e[2].get("weight", 1.0), 1.0))
        return out

    if mode == "strong_weight":
        out.sort(key=lambda e: _safe_float(e[2].get("weight", 1.0), 1.0), reverse=True)
        return out

    raise ValueError(f"Unsupported edge sort mode: {mode}")


def _rank_edges_with_noise(
    G: nx.Graph,
    *,
    sigma: float,
    seed: int,
) -> list[tuple]:
    H = as_simple_undirected(G)
    rng = np.random.default_rng(int(seed))

    weights = []
    for _, _, d in H.edges(data=True):
        weights.append(_safe_float(d.get("weight", 1.0), 1.0))
    ws = np.asarray(weights, dtype=float)
    scale = float(np.nanstd(ws, ddof=1)) if ws.size >= 2 else 1.0
    if (not np.isfinite(scale)) or scale <= 1e-12:
        scale = 1.0

    ranked = []
    for u, v, d in H.edges(data=True):
        w = _safe_float(d.get("weight", 1.0), 1.0)
        eps = float(rng.normal(loc=0.0, scale=float(sigma) * scale))
        noisy = max(0.0, w + eps)
        ranked.append((u, v, d, noisy))

    ranked.sort(key=lambda x: x[3], reverse=True)
    return ranked


def _rebuild_graph_from_ranked_edges(
    G: nx.Graph,
    ranked_edges: Sequence[tuple],
    *,
    keep_k: int,
) -> nx.Graph:
    H = _copy_graph_with_same_nodes(G)
    if keep_k <= 0:
        return H

    for item in ranked_edges[: int(keep_k)]:
        u, v, d = item[0], item[1], item[2]
        H.add_edge(u, v, **dict(d))
    return H


def _metrics_row(
    G: nx.Graph,
    *,
    step: int,
    damage_frac: float,
    eff_sources_k: int,
    seed: int,
    heavy: bool,
    compute_curvature: bool = False,
    curvature_sample_edges: int = 80,
    metric_names: Sequence[str] | None = None,
) -> dict:
    skip_light = not bool(heavy)
    m = calculate_metrics(
        G,
        eff_sources_k=int(eff_sources_k),
        seed=int(seed),
        compute_curvature=bool(compute_curvature and heavy),
        curvature_sample_edges=int(curvature_sample_edges),
        compute_heavy=bool(heavy),
        skip_spectral=skip_light,
        skip_clustering=skip_light,
        skip_assortativity=skip_light,
        diameter_samples=16 if heavy else 6,
    )

    row = {
        "step": int(step),
        "damage_frac": float(damage_frac),
        "N": int(m.get("N", G.number_of_nodes())),
        "E": int(m.get("E", G.number_of_edges())),
        "C": int(m.get("C", np.nan)) if "C" in m else np.nan,
        "lcc_size": int(m.get("lcc_size", np.nan)) if "lcc_size" in m else np.nan,
        "lcc_frac": float(m.get("lcc_frac", np.nan)) if "lcc_frac" in m else np.nan,
        "density": float(m.get("density", np.nan)) if "density" in m else np.nan,
        "avg_degree": float(m.get("avg_degree", np.nan)) if "avg_degree" in m else np.nan,
        "clustering": float(m.get("clustering", np.nan)) if "clustering" in m else np.nan,
        "assortativity": float(m.get("assortativity", np.nan)) if "assortativity" in m else np.nan,
        "eff_w": float(m.get("eff_w", np.nan)) if "eff_w" in m else np.nan,
        "mod": float(m.get("mod", np.nan)) if heavy else np.nan,
        "l2_lcc": float(m.get("l2_lcc", np.nan)) if heavy else np.nan,
        "H_rw": float(m.get("H_rw", np.nan)) if "H_rw" in m else np.nan,
        "fragility_H": float(m.get("fragility_H", np.nan)) if "fragility_H" in m else np.nan,
        "kappa_mean": float(m.get("kappa_mean", np.nan)) if "kappa_mean" in m else np.nan,
        "kappa_frac_negative": float(m.get("kappa_frac_negative", np.nan))
        if "kappa_frac_negative" in m
        else np.nan,
    }

    if metric_names:
        keep = {
            "step",
            "damage_frac",
            "N",
            "E",
            "C",
            "lcc_size",
            "lcc_frac",
        }.union(set(metric_names))
        row = {k: v for k, v in row.items() if k in keep}

    return row


def _forward_fill_heavy_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    heavy_cols = [
        "clustering",
        "assortativity",
        "eff_w",
        "mod",
        "l2_lcc",
        "H_rw",
        "fragility_H",
        "kappa_mean",
        "kappa_frac_negative",
    ]
    for col in heavy_cols:
        if col in out.columns:
            out[col] = out[col].replace([np.inf, -np.inf], np.nan).ffill()
    return out


def _run_noise_trajectory(
    G: nx.Graph,
    *,
    steps: int,
    frac: float,
    seed: int,
    eff_sources_k: int,
    sigma_max: float,
    keep_density_from_baseline: bool,
    compute_heavy_every: int,
    compute_curvature: bool,
    curvature_sample_edges: int,
    metric_names: Sequence[str] | None,
    row_cb=None,
    progress_cb=None,
) -> tuple[pd.DataFrame, dict]:
    H0 = as_simple_undirected(G)
    n0 = H0.number_of_nodes()
    e0 = H0.number_of_edges()

    if e0 == 0:
        row = _metrics_row(
            H0,
            step=0,
            damage_frac=0.0,
            eff_sources_k=eff_sources_k,
            seed=seed,
            heavy=True,
            compute_curvature=compute_curvature,
            curvature_sample_edges=curvature_sample_edges,
            metric_names=metric_names,
        )
        return pd.DataFrame([row]), {
            "kind": "weight_noise",
            "baseline_edges": 0,
            "baseline_density": 0.0,
        }

    base_density = _baseline_density(H0)
    total_remove = int(round(float(frac) * float(e0)))
    total_remove = max(0, min(e0, total_remove))

    xs = np.linspace(0, 1, int(max(1, steps)) + 1)
    rows = []

    for i, x in enumerate(xs):
        if progress_cb is not None:
            try:
                progress_cb(i, len(xs) - 1, x)
            except TypeError:
                progress_cb(i, len(xs) - 1)

        if keep_density_from_baseline:
            remove_k = int(round(float(x) * total_remove))
            keep_k = max(0, e0 - remove_k)
        else:
            keep_k = _target_edge_count_for_density(n0, base_density)

        sigma = float(x) * float(sigma_max)
        ranked = _rank_edges_with_noise(H0, sigma=sigma, seed=int(seed) + i)
        H = _rebuild_graph_from_ranked_edges(H0, ranked, keep_k=keep_k)

        heavy = (i % int(max(1, compute_heavy_every)) == 0) or (i == len(xs) - 1)
        row = _metrics_row(
            H,
            step=i,
            damage_frac=(e0 - H.number_of_edges()) / float(max(1, e0)),
            eff_sources_k=eff_sources_k,
            seed=int(seed) + i,
            heavy=heavy,
            compute_curvature=compute_curvature,
            curvature_sample_edges=curvature_sample_edges,
            metric_names=metric_names,
        )
        row["noise_sigma"] = float(sigma)
        row["kept_edges"] = int(H.number_of_edges())
        rows.append(row)

        if row_cb is not None:
            row_cb(dict(row), i, len(xs) - 1)

    df = pd.DataFrame(rows)
    df = _forward_fill_heavy_columns(df)
    aux = {
        "kind": "weight_noise",
        "baseline_edges": int(e0),
        "baseline_density": float(base_density),
        "sigma_max": float(sigma_max),
        "keep_density_from_baseline": bool(keep_density_from_baseline),
    }
    return df, aux


def _run_module_selective_edge_removal(
    G: nx.Graph,
    *,
    kind: str,
    steps: int,
    frac: float,
    seed: int,
    eff_sources_k: int,
    module_info: ModuleInfo | None,
    recompute_modules: bool,
    removal_mode: str,
    compute_heavy_every: int,
    compute_curvature: bool,
    curvature_sample_edges: int,
    metric_names: Sequence[str] | None,
    row_cb=None,
    progress_cb=None,
) -> tuple[pd.DataFrame, dict]:
    H0 = as_simple_undirected(G).copy()
    rng = np.random.default_rng(int(seed))

    if module_info is None:
        module_info = prepare_module_info(H0, seed=int(seed))

    if H0.number_of_edges() == 0:
        row = _metrics_row(
            H0,
            step=0,
            damage_frac=0.0,
            eff_sources_k=eff_sources_k,
            seed=seed,
            heavy=True,
            compute_curvature=compute_curvature,
            curvature_sample_edges=curvature_sample_edges,
            metric_names=metric_names,
        )
        return pd.DataFrame([row]), {
            "kind": kind,
            "removed_edges_order": [],
            "candidate_edges_total": 0,
        }

    current_info = prepare_module_info(H0, seed=int(seed)) if recompute_modules else module_info

    inter_edges, intra_edges = _classify_edges_by_modules(H0, current_info.membership)
    candidate_edges = inter_edges if kind == "inter_module_removal" else intra_edges
    ranked = _sort_edges_for_removal(candidate_edges, mode=removal_mode, rng=rng)

    total_candidates = len(ranked)
    remove_total = int(round(float(frac) * float(total_candidates)))
    remove_total = max(0, min(remove_total, total_candidates))
    ks = np.linspace(0, remove_total, int(max(1, steps)) + 1).round().astype(int).tolist()

    removed_order = [(u, v) for (u, v, _d) in ranked[:remove_total]]
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

        heavy = (i % int(max(1, compute_heavy_every)) == 0) or (i == len(ks) - 1)
        row = _metrics_row(
            H,
            step=i,
            damage_frac=(k / float(max(1, total_candidates))),
            eff_sources_k=eff_sources_k,
            seed=int(seed) + i,
            heavy=heavy,
            compute_curvature=compute_curvature,
            curvature_sample_edges=curvature_sample_edges,
            metric_names=metric_names,
        )
        row["removed_k"] = int(k)
        row["candidate_edges_total"] = int(total_candidates)
        row["removal_mode"] = str(removal_mode)
        rows.append(row)

        if row_cb is not None:
            row_cb(dict(row), i, len(ks) - 1)

    df = pd.DataFrame(rows)
    df = _forward_fill_heavy_columns(df)
    aux = {
        "kind": kind,
        "removed_edges_order": removed_order,
        "candidate_edges_total": int(total_candidates),
        "recompute_modules": bool(recompute_modules),
        "removal_mode": str(removal_mode),
        "n_modules": int(len(module_info.communities)),
    }
    return df, aux


def _normalize_legacy_trajectory(df: pd.DataFrame, *, source_kind: str) -> pd.DataFrame:
    out = df.copy()
    if "damage_frac" not in out.columns:
        if "removed_frac" in out.columns:
            out["damage_frac"] = pd.to_numeric(out["removed_frac"], errors="coerce")
        elif "mix_frac_effective" in out.columns:
            out["damage_frac"] = pd.to_numeric(out["mix_frac_effective"], errors="coerce")
        elif "mix_frac" in out.columns:
            out["damage_frac"] = pd.to_numeric(out["mix_frac"], errors="coerce")
        else:
            out["damage_frac"] = np.nan
    out["attack_kind"] = str(source_kind)
    return out


def run_degradation_trajectory(
    G: nx.Graph,
    *,
    kind: str,
    steps: int = 12,
    frac: float = 0.5,
    seed: int = 42,
    eff_sources_k: int = 16,
    compute_heavy_every: int = 2,
    compute_curvature: bool = False,
    curvature_sample_edges: int = 80,
    metric_names: Sequence[str] | None = None,
    noise_sigma_max: float = 0.5,
    keep_density_from_baseline: bool = True,
    module_info: ModuleInfo | None = None,
    recompute_modules: bool = False,
    module_resolution: float = 1.0,
    removal_mode: str = "random",
    fast_mode: bool = False,
    progress_cb=None,
    row_cb=None,
) -> tuple[pd.DataFrame, dict]:
    kind = str(kind)
    H = as_simple_undirected(G)

    if kind in {"weak_edges_by_weight", "strong_edges_by_weight"}:
        df, aux = run_edge_attack(
            H,
            kind=kind,
            frac=float(frac),
            steps=int(steps),
            seed=int(seed),
            eff_k=int(eff_sources_k),
            compute_heavy_every=int(compute_heavy_every),
            compute_curvature=bool(compute_curvature),
            curvature_sample_edges=int(curvature_sample_edges),
            fast_mode=bool(fast_mode),
            progress_cb=progress_cb,
            row_cb=row_cb,
        )
        return _normalize_legacy_trajectory(df, source_kind=kind), aux

    if kind in {"mix_default", "mix_degree_preserving"}:
        mix_kind = "mix_default" if kind == "mix_default" else "mix_degree_preserving"
        actual_kind = "mix_weightconf_preserving" if kind == "mix_default" else kind
        df, aux = run_mix_attack(
            H,
            kind=actual_kind,
            steps=int(steps),
            seed=int(seed),
            eff_sources_k=int(eff_sources_k),
            heavy_every=int(compute_heavy_every),
            fast_mode=bool(fast_mode),
            progress_cb=progress_cb,
            row_cb=row_cb,
        )
        return _normalize_legacy_trajectory(df, source_kind=mix_kind), aux

    if kind == "weight_noise":
        return _run_noise_trajectory(
            H,
            steps=int(steps),
            frac=float(frac),
            seed=int(seed),
            eff_sources_k=int(eff_sources_k),
            sigma_max=float(noise_sigma_max),
            keep_density_from_baseline=bool(keep_density_from_baseline),
            compute_heavy_every=int(compute_heavy_every),
            compute_curvature=bool(compute_curvature),
            curvature_sample_edges=int(curvature_sample_edges),
            metric_names=metric_names,
            row_cb=row_cb,
            progress_cb=progress_cb,
        )

    if kind in {"inter_module_removal", "intra_module_removal"}:
        mi = module_info
        if mi is None and not recompute_modules:
            mi = prepare_module_info(
                H,
                seed=int(seed),
                resolution=float(module_resolution),
            )
        return _run_module_selective_edge_removal(
            H,
            kind=kind,
            steps=int(steps),
            frac=float(frac),
            seed=int(seed),
            eff_sources_k=int(eff_sources_k),
            module_info=mi,
            recompute_modules=bool(recompute_modules),
            removal_mode=str(removal_mode),
            compute_heavy_every=int(compute_heavy_every),
            compute_curvature=bool(compute_curvature),
            curvature_sample_edges=int(curvature_sample_edges),
            metric_names=metric_names,
            row_cb=row_cb,
            progress_cb=progress_cb,
        )

    raise ValueError(f"Unsupported degradation kind: {kind}")
