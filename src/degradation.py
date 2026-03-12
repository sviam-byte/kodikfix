from __future__ import annotations

import logging
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

_logger = logging.getLogger(__name__)

WEIGHT_ATTACK_KINDS = {"weight_noise", "weight_noise_pure"}
TOPOLOGY_ATTACK_KINDS = {
    "weak_edges_by_weight",
    "strong_edges_by_weight",
    "weak_positive_edges",
    "strong_negative_edges",
    "negative_edges_only",
    "negative_edges_by_magnitude",
    "random_edges",
    "inter_module_removal",
    "intra_module_removal",
    "mix_default",
    "mix_degree_preserving",
}

# Signed-aware edge attacks dispatched through run_edge_attack.
SIGNED_EDGE_ATTACK_KINDS = {
    "weak_positive_edges",
    "strong_negative_edges",
    "negative_edges_only",
    "negative_edges_by_magnitude",
}


def classify_attack_family(kind: str) -> str:
    """Classify an attack kind into weight/topology buckets for reporting."""
    k = str(kind)
    if k in WEIGHT_ATTACK_KINDS:
        return "weight"
    if k in TOPOLOGY_ATTACK_KINDS:
        return "topology"
    return "unknown"


def validate_graph_for_regime(G: nx.Graph, *, graph_regime: str) -> dict:
    """Audit graph compatibility with the selected processing regime."""
    H = as_simple_undirected(G)
    weights = [float(d.get("raw_weight", d.get("weight", 1.0))) for _u, _v, d in H.edges(data=True)]
    has_negative = any(w < 0 for w in weights)
    has_zero = any(abs(w) <= 1e-12 for w in weights)
    return {
        "graph_regime": str(graph_regime),
        "n_nodes": int(H.number_of_nodes()),
        "n_edges": int(H.number_of_edges()),
        "has_negative_weights": bool(has_negative),
        "has_zero_weights": bool(has_zero),
    }


def _trajectory_guardrails(df: pd.DataFrame) -> pd.DataFrame:
    """Annotate invalid trajectory rows to protect downstream winner selection."""
    out = df.copy()
    out["is_invalid_state"] = False
    out["invalid_reason"] = ""
    if "E" in out.columns:
        e = pd.to_numeric(out["E"], errors="coerce")
        bad_e = ~np.isfinite(e) | (e < 0)
        out.loc[bad_e, "is_invalid_state"] = True
        out.loc[bad_e, "invalid_reason"] = "invalid_E"
    if "distance_to_target" in out.columns:
        d = pd.to_numeric(out["distance_to_target"], errors="coerce")
        bad_d = ~np.isfinite(d)
        out.loc[bad_d, "is_invalid_state"] = True
        out.loc[bad_d, "invalid_reason"] = out.loc[bad_d, "invalid_reason"].replace("", "invalid_distance")
    return out


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
    """Rank edges by noisy perturbation of *signed* raw weights.

    Returns tuples ``(u, v, data, noisy_abs, noisy_signed)`` sorted by
    ``noisy_abs`` descending. The signed value is preserved for callers that
    need to update sign-aware attributes in pure-noise mode.
    """
    H = as_simple_undirected(G)
    rng = np.random.default_rng(int(seed))

    raw_weights = []
    for _, _, d in H.edges(data=True):
        rw = d.get("raw_weight", d.get("weight_signed", d.get("weight", 1.0)))
        raw_weights.append(_safe_float(rw, 0.0))
    ws = np.asarray(raw_weights, dtype=float)
    scale = float(np.nanstd(ws, ddof=1)) if ws.size >= 2 else 1.0
    if (not np.isfinite(scale)) or scale <= 1e-12:
        scale = 1.0

    ranked = []
    for u, v, d in H.edges(data=True):
        raw = _safe_float(d.get("raw_weight", d.get("weight_signed", d.get("weight", 1.0))), 0.0)
        eps = float(rng.normal(loc=0.0, scale=float(sigma) * scale))
        noisy_signed = float(raw + eps)
        if abs(noisy_signed) < 1e-12:
            noisy_signed = 1e-12 if noisy_signed >= 0 else -1e-12
        noisy_abs = abs(noisy_signed)
        ranked.append((u, v, d, noisy_abs, noisy_signed))

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

    total_weight = float(sum(_safe_float(d.get("weight", 1.0), 1.0) for _u, _v, d in G.edges(data=True)))
    total_signed_weight = float(sum(_safe_float(d.get("raw_weight", d.get("weight_signed", d.get("weight", 1.0))), 0.0) for _u, _v, d in G.edges(data=True)))

    row = {
        "step": int(step),
        "damage_frac": float(damage_frac),
        "N": int(m.get("N", G.number_of_nodes())),
        "E": int(m.get("E", G.number_of_edges())),
        "total_weight": total_weight,
        "total_signed_weight": total_signed_weight,
    }
    # Preserve all metrics returned by calculate_metrics() so new fields propagate
    # automatically to trajectories (e.g. algebraic_connectivity).
    for k, v in m.items():
        if k in {"step", "damage_frac", "total_weight"}:
            continue
        row[k] = _safe_float(v, np.nan)

    if metric_names:
        keep = {"step", "damage_frac", "N", "E", "total_weight", "total_signed_weight"}.union(set(metric_names))
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
    keep_all_edges: bool,
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

        if keep_all_edges:
            keep_k = e0
        elif keep_density_from_baseline:
            remove_k = int(round(float(x) * total_remove))
            keep_k = max(0, e0 - remove_k)
        else:
            keep_k = _target_edge_count_for_density(n0, base_density)

        sigma = float(x) * float(sigma_max)
        ranked = _rank_edges_with_noise(H0, sigma=sigma, seed=int(seed) + i)
        H = _rebuild_graph_from_ranked_edges(H0, ranked, keep_k=keep_k)
        if keep_all_edges:
            # In pure-noise mode we keep topology fixed and perturb signed weights.
            # This allows weak edges to flip sign under noise while keeping
            # operational ``weight`` strictly positive as |raw_weight|.
            for u, v, _d, noisy_abs, noisy_signed in ranked:
                if not H.has_edge(u, v):
                    continue
                if abs(noisy_signed) <= 1e-12:
                    noisy_signed = 1e-12 if noisy_signed >= 0 else -1e-12
                sign = 1.0 if noisy_signed > 0 else -1.0
                H[u][v]["weight"] = float(noisy_abs)
                H[u][v]["weight_abs"] = float(noisy_abs)
                H[u][v]["sign"] = float(sign)
                H[u][v]["raw_weight"] = float(noisy_signed)
                H[u][v]["weight_signed"] = float(noisy_signed)

        heavy = (i % int(max(1, compute_heavy_every)) == 0) or (i == len(xs) - 1)
        row = _metrics_row(
            H,
            step=i,
            damage_frac=(float(x) if keep_all_edges else (e0 - H.number_of_edges()) / float(max(1, e0))),
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
        "kind": "weight_noise_pure" if keep_all_edges else "weight_noise",
        "baseline_edges": int(e0),
        "baseline_density": float(base_density),
        "sigma_max": float(sigma_max),
        "keep_density_from_baseline": bool(keep_density_from_baseline),
        "keep_all_edges": bool(keep_all_edges),
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
    module_resolution: float,
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

    if recompute_modules:
        base_candidates = _classify_edges_by_modules(
            H0,
            prepare_module_info(H0, seed=int(seed), resolution=float(module_resolution)).membership,
        )
        total_candidates = len(base_candidates[0] if kind == "inter_module_removal" else base_candidates[1])
    else:
        inter_edges, intra_edges = _classify_edges_by_modules(H0, module_info.membership)
        candidate_edges = inter_edges if kind == "inter_module_removal" else intra_edges
        ranked = _sort_edges_for_removal(candidate_edges, mode=removal_mode, rng=rng)
        total_candidates = len(ranked)
    remove_total = int(round(float(frac) * float(total_candidates)))
    remove_total = max(0, min(remove_total, total_candidates))
    ks = np.linspace(0, remove_total, int(max(1, steps)) + 1).round().astype(int).tolist()

    removed_order: list[tuple[int, int]] = []
    if not recompute_modules:
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
            if recompute_modules:
                for local_seed in range(prev, k):
                    current_info = prepare_module_info(
                        H,
                        seed=int(seed) + local_seed,
                        resolution=float(module_resolution),
                    )
                    inter_edges, intra_edges = _classify_edges_by_modules(H, current_info.membership)
                    candidate_edges = inter_edges if kind == "inter_module_removal" else intra_edges
                    ranked_step = _sort_edges_for_removal(candidate_edges, mode=removal_mode, rng=rng)
                    if not ranked_step:
                        break
                    u, v, _d = ranked_step[0]
                    if H.has_edge(u, v):
                        H.remove_edge(u, v)
                        removed_order.append((u, v))
            else:
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
        "module_resolution": float(module_resolution),
        "n_modules": int(len(module_info.communities)) if module_info is not None else np.nan,
    }
    return df, aux


def _run_random_edge_removal(
    G: nx.Graph,
    *,
    steps: int,
    frac: float,
    seed: int,
    eff_sources_k: int,
    compute_heavy_every: int,
    compute_curvature: bool,
    curvature_sample_edges: int,
    metric_names: Sequence[str] | None,
    row_cb=None,
    progress_cb=None,
) -> tuple[pd.DataFrame, dict]:
    H0 = as_simple_undirected(G).copy()
    edges = list(H0.edges(data=True))
    total_e = len(edges)
    if total_e == 0:
        row = _metrics_row(H0, step=0, damage_frac=0.0, eff_sources_k=eff_sources_k, seed=seed, heavy=True, compute_curvature=compute_curvature, curvature_sample_edges=curvature_sample_edges, metric_names=metric_names)
        return pd.DataFrame([row]), {"kind": "random_edges", "total_edges": 0, "removed_edges_order": []}

    rng = np.random.default_rng(int(seed))
    rng.shuffle(edges)
    remove_total = int(round(float(frac) * total_e))
    remove_total = max(0, min(remove_total, total_e))
    ks = np.linspace(0, remove_total, int(max(1, steps)) + 1).round().astype(int).tolist()
    removed_order = [(u, v) for (u, v, _d) in edges[:remove_total]]

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
        row = _metrics_row(H, step=i, damage_frac=(k / float(max(1, total_e))), eff_sources_k=eff_sources_k, seed=int(seed) + i, heavy=heavy, compute_curvature=compute_curvature, curvature_sample_edges=curvature_sample_edges, metric_names=metric_names)
        row["removed_k"] = int(k)
        row["candidate_edges_total"] = int(total_e)
        rows.append(row)
        if row_cb is not None:
            row_cb(dict(row), i, len(ks) - 1)

    df = pd.DataFrame(rows)
    df = _forward_fill_heavy_columns(df)
    df["attack_kind"] = "random_edges"
    return df, {"kind": "random_edges", "total_edges": int(total_e), "removed_edges_order": removed_order}


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
    graph_regime: str = "full_weighted_unsigned",
) -> tuple[pd.DataFrame, dict]:
    kind = str(kind)
    H = as_simple_undirected(G)
    graph_audit = validate_graph_for_regime(H, graph_regime=str(graph_regime))

    if kind in {"weak_edges_by_weight", "strong_edges_by_weight"} | SIGNED_EDGE_ATTACK_KINDS:
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
        out = _trajectory_guardrails(_normalize_legacy_trajectory(df, source_kind=kind))
        return out, {**dict(aux or {}), "attack_family": classify_attack_family(kind), "graph_audit": graph_audit}

    if kind in {"mix_default", "mix_degree_preserving"}:
        # Guard: mix attacks are ineffective on near-complete graphs.
        # double_edge_swap cannot find valid swaps when density > 0.80;
        # replacement from donor edges hits a collision wall.
        _N_guard = H.number_of_nodes()
        _E_guard = H.number_of_edges()
        _max_e_guard = _N_guard * (_N_guard - 1) // 2 if _N_guard > 1 else 1
        _dens_guard = float(_E_guard) / float(max(1, _max_e_guard))
        if _dens_guard > 0.80:
            _logger.warning(
                "Skipping mix attack '%s': graph density %.3f > 0.80; "
                "edge swap/replacement is ineffective on near-complete graphs.",
                kind,
                _dens_guard,
            )
            row = _metrics_row(
                H,
                step=0,
                damage_frac=0.0,
                eff_sources_k=int(eff_sources_k),
                seed=int(seed),
                heavy=True,
                compute_curvature=bool(compute_curvature),
                curvature_sample_edges=int(curvature_sample_edges),
                metric_names=metric_names,
            )
            row["attack_kind"] = str(kind)
            row["skipped_reason"] = "density_too_high"
            df = pd.DataFrame([row])
            df["damage_frac"] = 0.0
            if row_cb is not None:
                row_cb(dict(row), 0, 0)
            out = _trajectory_guardrails(df)
            return out, {"kind": kind, "skipped": True, "density": float(_dens_guard), "attack_family": classify_attack_family(kind), "graph_audit": graph_audit}

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
        out = _trajectory_guardrails(_normalize_legacy_trajectory(df, source_kind=mix_kind))
        return out, {**dict(aux or {}), "attack_family": classify_attack_family(kind), "graph_audit": graph_audit}

    if kind == "random_edges":
        df, aux = _run_random_edge_removal(
            H,
            steps=int(steps),
            frac=float(frac),
            seed=int(seed),
            eff_sources_k=int(eff_sources_k),
            compute_heavy_every=int(compute_heavy_every),
            compute_curvature=bool(compute_curvature),
            curvature_sample_edges=int(curvature_sample_edges),
            metric_names=metric_names,
            row_cb=row_cb,
            progress_cb=progress_cb,
        )
        return _trajectory_guardrails(df), {**dict(aux or {}), "attack_family": classify_attack_family(kind), "graph_audit": graph_audit}

    if kind in {"weight_noise", "weight_noise_pure"}:
        df, aux = _run_noise_trajectory(
            H,
            steps=int(steps),
            frac=float(frac),
            seed=int(seed),
            eff_sources_k=int(eff_sources_k),
            sigma_max=float(noise_sigma_max),
            keep_density_from_baseline=bool(keep_density_from_baseline),
            keep_all_edges=bool(kind == "weight_noise_pure"),
            compute_heavy_every=int(compute_heavy_every),
            compute_curvature=bool(compute_curvature),
            curvature_sample_edges=int(curvature_sample_edges),
            metric_names=metric_names,
            row_cb=row_cb,
            progress_cb=progress_cb,
        )
        return _trajectory_guardrails(df), {**dict(aux or {}), "attack_family": classify_attack_family(kind), "graph_audit": graph_audit}

    if kind in {"inter_module_removal", "intra_module_removal"}:
        mi = module_info
        if mi is None and not recompute_modules:
            mi = prepare_module_info(
                H,
                seed=int(seed),
                resolution=float(module_resolution),
            )
        df, aux = _run_module_selective_edge_removal(
            H,
            kind=kind,
            steps=int(steps),
            frac=float(frac),
            seed=int(seed),
            eff_sources_k=int(eff_sources_k),
            module_info=mi,
            recompute_modules=bool(recompute_modules),
            module_resolution=float(module_resolution),
            removal_mode=str(removal_mode),
            compute_heavy_every=int(compute_heavy_every),
            compute_curvature=bool(compute_curvature),
            curvature_sample_edges=int(curvature_sample_edges),
            metric_names=metric_names,
            row_cb=row_cb,
            progress_cb=progress_cb,
        )
        return _trajectory_guardrails(df), {**dict(aux or {}), "attack_family": classify_attack_family(kind), "graph_audit": graph_audit}

    raise ValueError(f"Unsupported degradation kind: {kind}")
