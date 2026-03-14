from __future__ import annotations

import textwrap
import logging
import time
import json
import re
import gc
import traceback
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st

import plotly.express as px
import plotly.graph_objects as go

from src.config import settings
from src.null_models import make_er_gnm, make_configuration_model, rewire_mix
from src.attacks import run_attack, run_edge_attack
from src.attacks_mix import run_mix_attack
from src.graph_build import build_graph_from_edges, lcc_subgraph
from src.core_math import classify_phase_transition
from src.config_loader import load_metrics_info
from src.metrics import calculate_metrics
from src.mix_frac_estimator import estimate_mix_frac_star
from src.metric_registry import get_default_metrics_for_regime, describe_metrics_for_regime
from src.degradation import run_degradation_trajectory, prepare_module_info
from src.phenotype_matching import (
    compare_degradation_models,
    build_group_target_vector,
    resolve_metric_scales,
    normalize_metric_families,
    compute_profile_distance,
)
from src.phenotype_reporting import build_paper_ready_summary, export_phenotype_match_excel
from src.phenotype_scalar import build_scalar_subject_results, build_scalar_summary, build_scalar_winners
from src.plotting import fig_metrics_over_steps, fig_compare_attacks
from src.services.graph_service import GraphService
from src.robustness import attack_trajectory_summary, graph_resistance_summary
from src.state_models import GraphEntry
from src.batch_ops import (
    _infer_metadata_group_col,
    _infer_metadata_id_col,
    _norm_meta_token,
)
from src.ui.plots.charts import (
    AUC_TRAP,
    apply_plot_defaults as _apply_plot_defaults,
    auto_y_range as _auto_y_range,
    forward_fill_heavy as _forward_fill_heavy,
)
from src.ui.plots.scene3d import make_3d_traces
from src.ui_blocks import help_icon
from src.utils import as_simple_undirected, get_node_strength
from src.timeout_worker import run_with_timeout as _run_with_timeout_safe
from src.timeout_worker import build_graph_safe, compute_metrics_safe

_layout_cached = GraphService.compute_layout3d
logger = logging.getLogger(__name__)


def _hash_graph(G: nx.Graph) -> str:
    """Stable hash for caching graph-derived metrics."""
    if G is None:
        return "none"
    try:
        return nx.weisfeiler_lehman_graph_hash(G, edge_attr="weight")
    except Exception:
        return f"{G.number_of_nodes()}-{G.number_of_edges()}"


@st.cache_data(show_spinner=False, hash_funcs={nx.Graph: _hash_graph})
def _cached_betweenness(G: nx.Graph) -> dict:
    """Cached betweenness centrality with approximation on large graphs."""
    n = G.number_of_nodes()
    if n <= 250:
        return nx.betweenness_centrality(G, normalized=True)

    # Для больших графов используем сэмплирование source-узлов (k),
    # чтобы удержать время расчета в интерактивных пределах.
    if n <= 1000:
        k = min(64, n)
    else:
        k = min(32, n)

    return nx.betweenness_centrality(
        G,
        normalized=True,
        k=k,
        seed=42,
    )

# Загружаем справку по метрикам один раз на модуль.
_info = load_metrics_info()
METRIC_HELP = _info.get("metric_help", {})

# presets moved out of app.py
ATTACK_PRESETS_NODE = {
    "Random": {"kind": "random"},
    "Degree": {"kind": "degree"},
    "Strength": {"kind": "strength"},
    "Betweenness": {"kind": "betweenness"},
    "Closeness": {"kind": "closeness"},
    "Eigenvector": {"kind": "eigenvector"},
    "PageRank": {"kind": "pagerank"},
    "Katz": {"kind": "katz"},
    "k-core": {"kind": "kcore"},
    "Community bridge": {"kind": "community_bridge"},
}
ATTACK_PRESETS_EDGE = {
    "Random": {"kind": "edge_random"},
    "Weight": {"kind": "edge_weight"},
    "Betweenness": {"kind": "edge_betweenness"},
    "Rici (Ollivier)": {"kind": "edge_ricci"},
}

# Метрики для UI блока mix_frac*.
# Список оставлен явным, чтобы пользователь видел «безопасный» набор полей,
# которые гарантированно поддерживаются calculate_metrics / trajectory-кривыми.
MIX_FRAC_METRIC_OPTIONS = [
    "kappa_mean",
    "kappa_frac_negative",
    "kappa_median",
    "kappa_var",
    "kappa_skew",
    "kappa_entropy",
    "clustering",
    "mod",
    "avg_degree",
    "density",
    "eff_w",
    "l2_lcc",
    "lcc_frac",
    "H_rw",
    "H_evo",
    "fragility_kappa",
]


DEGRADATION_METRIC_OPTIONS = [
    "l2_lcc",
    "H_rw",
    "fragility_H",
    "mod",
    # signed-hybrid core
    "frac_negative_weight",
    "signed_balance_weight",
    "signed_std_weight",
    "frustration_index",
    "signed_lambda_min",

    # signed-hybrid secondary
    "signed_mean_weight",
    "signed_median_weight",
    "neg_abs_mean_weight",
    "pos_mean_weight",
    "signed_entropy_weight",
    "signed_lambda2",
    "strength_pos_mean",
    "strength_neg_mean",
    "strength_pos_std",
    "strength_neg_std",

    # extra weighted/spectral
    "H_w",
    "eff_w",
    "algebraic_connectivity",
    "tau_relax",

    # older / optional / discouraged
    "density",
    "clustering",
    "lcc_frac",
    "kappa_mean",
    "kappa_frac_negative",
]

SZ_ML_METRICS = [
    "l2_lcc",
    "H_rw",
    "fragility_H",
    "mod",
    "frac_negative_weight",
    "signed_balance_weight",
    "signed_std_weight",
    "frustration_index",
    "signed_lambda_min",
]

DEGRADATION_KIND_OPTIONS = [
    "weight_noise",
    "inter_module_removal",
    "intra_module_removal",
    "weak_edges_by_weight",
    "strong_edges_by_weight",
    "weak_positive_edges",
    "strong_negative_edges",
    "negative_edges_only",
    "negative_edges_by_magnitude",
    "mix_default",
    "mix_degree_preserving",
]


def _read_uploaded_metrics_table(uploaded_file) -> pd.DataFrame:
    """Read uploaded CSV/XLSX metrics table."""
    if uploaded_file is None:
        raise ValueError("Файл не загружен")
    name = str(uploaded_file.name).lower()
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded_file)
    if name.endswith(".tsv"):
        return pd.read_csv(uploaded_file, sep="	")
    return pd.read_csv(uploaded_file)


def _read_uploaded_metadata_table(uploaded_file) -> pd.DataFrame:
    """Read uploaded metadata CSV/XLSX."""
    return _read_uploaded_metrics_table(uploaded_file)


def _prepare_uploaded_metadata(
    uploaded_file,
    *,
    id_col: str = "",
    group_col: str = "",
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Normalize uploaded metadata and infer ID/group columns.

    The helper keeps normalized subject/group keys used for robust matching.
    """
    meta = _read_uploaded_metadata_table(uploaded_file).copy()
    if meta.empty:
        raise ValueError("Metadata file is empty")

    meta.columns = [str(col) for col in meta.columns]

    id_col_resolved = str(id_col or "").strip() or _infer_metadata_id_col(meta)
    if not id_col_resolved or id_col_resolved not in meta.columns:
        raise ValueError("Не удалось определить колонку subject id в metadata.")

    group_col_resolved = str(group_col or "").strip() or _infer_metadata_group_col(meta, exclude=str(id_col_resolved))
    if not group_col_resolved or group_col_resolved not in meta.columns:
        raise ValueError("Не удалось определить колонку группы в metadata.")

    meta["__meta_subject_key"] = meta[id_col_resolved].map(_norm_meta_token)
    meta["__meta_group_key"] = meta[group_col_resolved].map(_norm_meta_token)
    meta = meta[meta["__meta_subject_key"] != ""].copy()
    meta = meta.drop_duplicates(subset=["__meta_subject_key"], keep="first")

    return meta, {
        "id_col": str(id_col_resolved),
        "group_col": str(group_col_resolved),
    }


def _split_tokens(raw: str) -> list[str]:
    """Split comma-separated text into a list of trimmed tokens."""
    return [str(x).strip() for x in str(raw or "").split(",") if str(x).strip()]


def _token_set(raw: str) -> set[str]:
    """Build a normalized token set from comma-separated values."""
    return {_norm_meta_token(x) for x in _split_tokens(raw) if _norm_meta_token(x)}


def _parse_token_list(raw: str) -> set[str]:
    """Parse comma/newline/space separated token list into unique values."""
    if not str(raw or "").strip():
        return set()
    toks = re.split(r"[,;\n\r\t ]+", str(raw))
    return {str(x).strip() for x in toks if str(x).strip()}


def _metadata_match_candidates(entry: GraphEntry | None, gid: str) -> list[str]:
    """Build tolerant metadata keys for one workspace graph.

    MAT imports often store subject ids inside labels like
    ``cobre_resolution_325 :: szxxx0040000`` while workspace ids look like
    ``G_ab12cd``. We therefore try both whole labels and extracted suffixes.
    """
    raw_candidates: list[str] = [str(gid)]
    if entry is not None:
        raw_candidates.extend([
            str(getattr(entry, "name", "") or ""),
            str(getattr(entry, "source", "") or ""),
            Path(str(getattr(entry, "source", "") or "")).stem,
            Path(str(getattr(entry, "source", "") or "")).name,
        ])

    out: list[str] = []
    seen: set[str] = set()

    def _push(value: str) -> None:
        key = _norm_meta_token(value)
        if key and key not in seen:
            seen.add(key)
            out.append(key)

    for raw in raw_candidates:
        txt = str(raw or "").strip()
        if not txt:
            continue
        _push(txt)

        parts = [seg.strip() for seg in txt.split("::") if str(seg).strip()]
        for seg in parts:
            _push(seg)
        if parts:
            _push(parts[-1])

        p = Path(txt)
        _push(p.stem)
        _push(p.name)

        # Loose token extraction helps with labels like
        # "cobre_resolution_325 :: szxxx0040000".
        for token in re.findall(r"[A-Za-z]+[A-Za-z0-9_-]*\d+[A-Za-z0-9_-]*", txt):
            _push(token)

    return out


def _pm_safe_stem(name: str, fallback: str = "phenotype_match") -> str:
    """Normalize arbitrary names to filesystem-safe stems."""
    s = str(name or "").strip()
    s = re.sub(r"[^\w\-.]+", "_", s, flags=re.UNICODE)
    s = re.sub(r"_+", "_", s).strip("_.")
    return s or fallback


def _pm_repo_root() -> Path:
    """Best-effort repository root for local writes."""
    try:
        return Path(__file__).resolve().parents[3]
    except Exception:
        return Path.cwd().resolve()


def _pm_resolve_out_dir(raw_value: str | Path | None) -> Path:
    """Resolve output directory; relative paths are anchored to repo root."""
    raw = str(raw_value or "").strip()
    if not raw:
        return (_pm_repo_root() / "phenotype_runs").resolve()
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = _pm_repo_root() / p
    return p.resolve()


def _pm_write_progress_snapshot(run_dir: Path, payload: dict) -> None:
    """Persist current run progress for debugging/resume visibility."""
    _pm_write_json(run_dir / "progress.json", payload)


def _pm_append_event_log(run_dir: Path, payload: dict) -> None:
    """Append one structured event line to run_dir/events.log."""
    run_dir.mkdir(parents=True, exist_ok=True)
    row = dict(payload or {})
    row.setdefault("ts", time.strftime("%Y-%m-%d %H:%M:%S"))
    with open(run_dir / "events.log", "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _pm_emit_progress(run_dir: Path, payload: dict, *, write_event: bool = False) -> None:
    """Write progress snapshot with heartbeat and optionally mirror to event log."""
    row = dict(payload or {})
    row["heartbeat_ts"] = time.strftime("%Y-%m-%d %H:%M:%S")
    row["heartbeat_unix"] = time.time()
    _pm_write_progress_snapshot(run_dir, row)
    if write_event:
        _pm_append_event_log(run_dir, row)


def _pm_has_curvature_metrics(metrics: list[str]) -> bool:
    """Whether selected metric set implies curvature/Ricci calculations."""
    toks = {str(x).strip() for x in (metrics or [])}
    return any(
        t.startswith("kappa_") or t in {"fragility_kappa"}
        for t in toks
    )


def _pm_has_curvature_attacks(attack_kinds: list[str]) -> bool:
    """Whether selected attacks imply Ricci/curvature usage."""
    toks = {str(x).strip() for x in (attack_kinds or [])}
    ricci_like = {
        "ricci_most_negative",
        "ricci_most_positive",
        "ricci_abs_max",
        "flux_high_rw_x_neg_ricci",
        "edge_ricci",
    }
    return any(t in ricci_like or "ricci" in t.lower() for t in toks)


def _pm_heavy_metrics(metrics: list[str]) -> list[str]:
    """Return selected metrics that are relatively expensive."""
    heavy = {"mod", "eff_w", "H_rw", "H_evo", "fragility_H", "fragility_kappa"}
    return [str(x) for x in (metrics or []) if str(x) in heavy]


def _pm_heavy_attacks(attack_kinds: list[str]) -> list[str]:
    """Return selected degradation models that are relatively expensive."""
    heavy = {
        "inter_module_removal",
        "intra_module_removal",
        "mix_default",
        "mix_degree_preserving",
    }
    out = []
    for x in (attack_kinds or []):
        s = str(x)
        if s in heavy or "ricci" in s.lower():
            out.append(s)
    return out


def _pm_cost_level(
    *,
    hc_n: int,
    sz_n: int,
    attacks_n: int,
    steps_n: int,
    heavy_metrics_n: int,
    heavy_attacks_n: int,
    curvature_baseline: bool,
    curvature_trajectory: bool,
    export_subject_xlsx: bool,
    export_run_xlsx: bool,
) -> str:
    """Very rough qualitative estimate of run heaviness."""
    score = 0
    score += max(0, hc_n + sz_n) // 20
    score += max(0, hc_n * max(1, attacks_n) * max(1, steps_n)) // 800
    score += int(heavy_metrics_n >= 2)
    score += int(heavy_metrics_n >= 4)
    score += int(heavy_attacks_n >= 1)
    score += int(heavy_attacks_n >= 2)
    score += 3 if curvature_baseline else 0
    score += 3 if curvature_trajectory else 0
    score += 1 if export_subject_xlsx else 0
    score += 1 if export_run_xlsx else 0

    if score <= 2:
        return "cheap"
    if score <= 5:
        return "medium"
    return "heavy"


def _pm_est_time_per_subject_sec(
    *,
    attacks_n: int,
    steps: int,
    heavy_every: int,
    density_estimate: float,
    n_edges_approx: int,
    curvature_trajectory: bool,
) -> float:
    """Rough estimate of wall-clock seconds per HC subject.

    The dominant cost is ``calculate_metrics`` on heavy steps. Coefficients are
    empirical and intentionally conservative for dense NetworkX workloads.
    """
    heavy_every = max(1, int(heavy_every))
    n_steps = int(steps) + 1
    heavy_steps_per_attack = sum(
        1 for i in range(n_steps) if (i % heavy_every == 0) or (i == n_steps - 1)
    )
    light_steps_per_attack = max(0, n_steps - heavy_steps_per_attack)

    # Base cost per heavy step (Dijkstra + Louvain + clustering + spectral + entropy)
    if density_estimate > 0.5:
        cost_heavy = 5.0
    elif density_estimate > 0.2:
        cost_heavy = 3.0
    else:
        cost_heavy = 1.5

    if n_edges_approx > 30000:
        cost_heavy *= 1.4

    if curvature_trajectory:
        cost_heavy *= 3.0

    cost_light = 0.08
    per_attack = heavy_steps_per_attack * cost_heavy + light_steps_per_attack * cost_light
    return float(int(attacks_n) * per_attack + 5.0)


def _pm_build_preflight_summary(
    *,
    hc_n: int,
    sz_n: int,
    attack_kinds: list[str],
    metrics: list[str],
    steps: int,
    compute_curv: bool,
    export_subject_xlsx: bool,
    export_run_xlsx: bool,
    density_estimate: float = 0.0,
    n_edges_approx: int = 0,
    n_nodes_approx: int = 0,
    heavy_every: int = 2,
) -> dict:
    """Prepare a transparent summary of what exactly will be computed."""
    attacks_n = len(list(attack_kinds or []))
    metrics_n = len(list(metrics or []))
    traj_steps_total = int(hc_n) * int(attacks_n) * int(steps)
    baseline_graph_builds = int(hc_n) + int(sz_n)
    baseline_metric_passes = int(hc_n) + int(sz_n)

    curvature_from_metrics = _pm_has_curvature_metrics(metrics)
    curvature_from_attacks = _pm_has_curvature_attacks(attack_kinds)
    curvature_baseline = bool(compute_curv) and bool(curvature_from_metrics)
    curvature_trajectory = bool(compute_curv) and bool(curvature_from_attacks)

    heavy_metrics = _pm_heavy_metrics(metrics)
    heavy_attacks = _pm_heavy_attacks(attack_kinds)

    total_cost = _pm_cost_level(
        hc_n=int(hc_n),
        sz_n=int(sz_n),
        attacks_n=int(attacks_n),
        steps_n=int(steps),
        heavy_metrics_n=len(heavy_metrics),
        heavy_attacks_n=len(heavy_attacks),
        curvature_baseline=curvature_baseline,
        curvature_trajectory=curvature_trajectory,
        export_subject_xlsx=bool(export_subject_xlsx),
        export_run_xlsx=bool(export_run_xlsx),
    )

    # Density-aware runtime estimation for transparent preflight expectations.
    est_sec = _pm_est_time_per_subject_sec(
        attacks_n=int(attacks_n),
        steps=int(steps),
        heavy_every=int(heavy_every),
        density_estimate=float(density_estimate),
        n_edges_approx=int(n_edges_approx),
        curvature_trajectory=bool(curvature_trajectory),
    )
    est_baseline_sec = float(int(hc_n) + int(sz_n)) * max(1.0, est_sec / max(1, attacks_n))
    has_mix = any(str(x) in {"mix_default", "mix_degree_preserving"} for x in (attack_kinds or []))

    return {
        "hc_n": int(hc_n),
        "sz_n": int(sz_n),
        "attacks_n": int(attacks_n),
        "metrics_n": int(metrics_n),
        "steps": int(steps),
        "baseline_graph_builds": int(baseline_graph_builds),
        "baseline_metric_passes": int(baseline_metric_passes),
        "trajectory_steps_total": int(traj_steps_total),
        "curvature_from_metrics": bool(curvature_from_metrics),
        "curvature_from_attacks": bool(curvature_from_attacks),
        "curvature_baseline": bool(curvature_baseline),
        "curvature_trajectory": bool(curvature_trajectory),
        "heavy_metrics": list(heavy_metrics),
        "heavy_attacks": list(heavy_attacks),
        "export_subject_xlsx": bool(export_subject_xlsx),
        "export_run_xlsx": bool(export_run_xlsx),
        "estimated_cost": str(total_cost),
        "density_estimate": float(density_estimate),
        "n_edges_approx": int(n_edges_approx),
        "n_nodes_approx": int(n_nodes_approx),
        "est_time_per_subject_sec": float(est_sec),
        "est_baseline_sec": float(est_baseline_sec),
        "est_total_sec": float(est_sec * int(hc_n) + est_baseline_sec),
        "mix_attacks_selected": bool(has_mix),
        "mix_on_dense_graph": bool(has_mix and float(density_estimate) > 0.50),
    }



def _pm_append_csv(path: Path, rows: pd.DataFrame | list[dict] | dict) -> None:
    """Append rows to CSV, accepting dict/list/DataFrame inputs.

    The implementation is intentionally tolerant to concurrent writers.
    We avoid a separate ``path.exists()`` check because it opens a race
    between header detection and append mode in Streamlit reruns.
    """
    if isinstance(rows, dict):
        df = pd.DataFrame([rows])
    elif isinstance(rows, list):
        df = pd.DataFrame(rows)
    else:
        df = rows.copy()
    if df.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if path.stat().st_size > 0:
            df.to_csv(path, mode="a", header=False, index=False)
            return
    except (FileNotFoundError, OSError):
        pass
    df.to_csv(path, mode="w", header=True, index=False)


def _pm_read_csv_if_exists(path: Path) -> pd.DataFrame:
    """Read CSV if present and non-empty; otherwise return empty frame."""
    if path.exists() and path.stat().st_size > 0:
        return pd.read_csv(path)
    return pd.DataFrame()


def _pm_subject_dir(run_dir: Path, subject_id: str) -> Path:
    """Return the canonical per-subject folder path for a run."""
    return run_dir / "per_subject" / _pm_safe_stem(subject_id, "subject")


def _pm_subject_file(run_dir: Path, subject_id: str, filename: str, *, ensure_dir: bool = False) -> Path:
    """Return a per-subject file path and optionally create its parent directory."""
    sdir = _pm_subject_dir(run_dir, subject_id)
    if ensure_dir:
        sdir.mkdir(parents=True, exist_ok=True)
    return sdir / filename


def _pm_subject_done(run_dir: Path, subject_id: str) -> bool:
    """Check resumable marker file for a processed subject."""
    sdir = _pm_subject_dir(run_dir, subject_id)
    return (sdir / "done.json").exists()


def _pm_write_json(path: Path, payload: dict) -> None:
    """Write UTF-8 JSON with deterministic formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _pm_load_run_result(run_dir: Path, *, include_trajectory: bool = True) -> dict:
    """Load aggregate run artifacts into phenotype-matching result structure."""
    agg = run_dir / "aggregate"
    subject_df = _pm_read_csv_if_exists(agg / "subject_results.csv")
    winners_df = _pm_read_csv_if_exists(agg / "winner_results.csv")
    traj_df = _pm_read_csv_if_exists(agg / "trajectory_results.csv") if include_trajectory else pd.DataFrame()
    scalar_subject_df = _pm_read_csv_if_exists(agg / "scalar_subject_results.csv")
    scalar_winners_df = _pm_read_csv_if_exists(agg / "scalar_winners.csv")
    scalar_summary_df = _pm_read_csv_if_exists(agg / "scalar_summary.csv")

    cfg = {}
    cfg_path = run_dir / "config.json"
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception:
            cfg = {}

    return {
        "target_vector": dict(cfg.get("target_vector", {}) or {}),
        "scales": dict(cfg.get("scales", {}) or {}),
        "subject_results": subject_df,
        "winner_results": winners_df,
        "trajectory_results": traj_df,
        "scalar_subject_results": scalar_subject_df,
        "scalar_winners": scalar_winners_df,
        "scalar_summary": scalar_summary_df,
        "metrics_used": list(cfg.get("metrics", []) or []),
        "metric_families": dict(cfg.get("metric_families", {}) or {}),
        "distance_mode": str(cfg.get("distance_mode", "raw")),
    }


def _pm_finalize_scalar_aggregates(run_dir: Path, *, metrics: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recompute run-level scalar winners/summary from aggregate scalar_subject_results."""
    agg = run_dir / "aggregate"
    scalar_subject_df = build_scalar_subject_results(
        _pm_read_csv_if_exists(agg / "scalar_subject_results.csv"),
        metrics=metrics,
    )
    scalar_winners_df = build_scalar_winners(scalar_subject_df)
    scalar_summary_df = build_scalar_summary(scalar_subject_df, scalar_winners_df)

    for path, df in [
        (agg / "scalar_winners.csv", scalar_winners_df),
        (agg / "scalar_summary.csv", scalar_summary_df),
    ]:
        if path.exists():
            path.unlink()
        if isinstance(df, pd.DataFrame) and not df.empty:
            df.to_csv(path, index=False)

    return scalar_winners_df, scalar_summary_df


def _pm_write_run_location_note(run_dir: Path) -> None:
    """Write a tiny human-readable note with absolute save path."""
    note = textwrap.dedent(
        f"""\
        Phenotype matching run
        run_dir: {run_dir}
        aggregate_dir: {run_dir / 'aggregate'}
        per_subject_dir: {run_dir / 'per_subject'}
        written_at: {time.strftime('%Y-%m-%d %H:%M:%S')}
        """
    ).strip() + "\n"
    (run_dir / "SAVE_LOCATION.txt").write_text(note, encoding="utf-8")


def _pm_save_run_inputs(
    run_dir: Path,
    *,
    config: dict,
    metadata_upload=None,
    sz_upload=None,
    hc_baseline_upload=None,
) -> None:
    """Persist run config and raw uploaded files for reproducibility."""
    raw_dir = run_dir / "raw_inputs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    _pm_write_json(run_dir / "config.json", config)

    for upload, fallback in [
        (metadata_upload, "metadata"),
        (sz_upload, "sz_metrics"),
        (hc_baseline_upload, "hc_baseline"),
    ]:
        if upload is None:
            continue
        data = None
        try:
            data = upload.getvalue()
        except Exception:
            try:
                data = upload.read()
            except Exception:
                data = None
        if data:
            name = getattr(upload, "name", "") or fallback
            (raw_dir / _pm_safe_stem(name, fallback)).write_bytes(data)


def _pm_stream_subject(
    *,
    run_dir: Path,
    G: nx.Graph,
    subject_id: str,
    subject_idx: int,
    attack_kinds: list[str],
    metric_list: list[str],
    target_vector: dict[str, float],
    scales: dict[str, float],
    normalized_families: dict[str, list[str]],
    steps: int,
    frac: float,
    seed: int,
    eff_sources_k: int,
    compute_heavy_every: int,
    compute_curvature: bool,
    curvature_sample_edges: int,
    noise_sigma_max: float,
    keep_density_from_baseline: bool,
    recompute_modules: bool,
    module_resolution: float,
    removal_mode: str,
    fast_mode: bool,
    subject_meta: dict | None = None,
    timeout_seconds: float = 0.0,
    attack_timeout_seconds: float = 0.0,
    progress_cb=None,
    export_subject_excel: bool = False,
    include_subject_trajectory_in_return: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Process one subject in streaming mode and persist intermediate outputs."""
    sdir = _pm_subject_dir(run_dir, subject_id)
    sdir.mkdir(parents=True, exist_ok=True)

    traj_path = sdir / "trajectory.csv"
    scalar_path = sdir / "scalar_subject_results.csv"
    best_path = sdir / "subject_results.csv"
    winner_path = sdir / "winner_results.csv"

    for path in [traj_path, scalar_path, best_path, winner_path]:
        if path.exists():
            path.unlink()

    # При resume subject может быть пересчитан заново (done.json ещё не создан),
    # поэтому заранее удаляем старые строки этого subject из aggregate CSV,
    # чтобы не накапливать дубли в результирующих таблицах.
    for agg_file in [
        "trajectory_results.csv",
        "subject_results.csv",
        "scalar_subject_results.csv",
        "winner_results.csv",
    ]:
        agg_path = run_dir / "aggregate" / agg_file
        if not agg_path.exists():
            continue
        try:
            agg_df = pd.read_csv(agg_path)
            if "subject_id" in agg_df.columns:
                agg_df = agg_df[agg_df["subject_id"].astype(str) != str(subject_id)]
                agg_df.to_csv(agg_path, index=False)
        except Exception:
            # Не блокируем run из-за повреждённого/частично записанного CSV.
            pass

    subject_meta = dict(subject_meta or {})
    subject_meta.setdefault("subject_id", str(subject_id))
    t0 = time.perf_counter()

    subject_rows: list[dict] = []
    scalar_rows: list[dict] = []
    winner_rows: list[dict] = []

    module_info = None
    if any(k in {"inter_module_removal", "intra_module_removal"} for k in attack_kinds):
        module_info = prepare_module_info(
            G,
            seed=int(seed) + int(subject_idx),
            resolution=float(module_resolution),
        )

    for attack_pos, kind in enumerate(attack_kinds, start=1):
        attack_t0 = time.perf_counter()
        best_distance = float("inf")
        best_row = None
        metric_best: dict[str, dict] = {}
        last_step = max(1, int(steps))
        baseline_holder = {}
        min_required_metrics = max(1, int(np.ceil(0.75 * len(metric_list))))

        def _to_float(value, *, default=np.nan) -> float:
            """Fast scalar coercion for hot loops (avoids tiny pandas Series)."""
            try:
                out = float(value)
            except (TypeError, ValueError):
                return float(default)
            return out if np.isfinite(out) else float(default)

        def _check_timeout() -> None:
            if timeout_seconds and (time.perf_counter() - t0) > float(timeout_seconds):
                raise TimeoutError(f"subject timeout > {float(timeout_seconds):.1f}s")
            if attack_timeout_seconds and (time.perf_counter() - attack_t0) > float(attack_timeout_seconds):
                raise TimeoutError(f"attack timeout [{kind}] > {float(attack_timeout_seconds):.1f}s")

        def _row_cb(row: dict, i: int, total_steps: int) -> None:
            nonlocal best_distance, best_row, last_step
            _check_timeout()
            last_step = max(1, int(total_steps))

            row2 = dict(row)
            row2["subject_idx"] = int(subject_idx)
            row2["subject_id"] = str(subject_id)
            row2["attack_kind"] = str(kind)
            for k, v in subject_meta.items():
                row2.setdefault(k, v)

            info = compute_profile_distance(
                row2,
                target_vector=target_vector,
                metrics=metric_list,
                scales=scales,
                distance_mode="raw",
                metric_families=normalized_families,
            )
            row2["distance_to_target"] = float(info.get("distance", np.nan))
            row2["n_used_metrics"] = int(info.get("n_used_metrics", 0))
            row2["distance_mode"] = str(info.get("distance_mode", "raw"))
            row2["used_metrics"] = ",".join(info.get("used_metrics", []) or [])
            if row2["n_used_metrics"] < min_required_metrics:
                # Защита от winner bias: шаги с большим числом NaN-метрик не
                # должны выигрывать только потому, что в L2 сумме меньше слагаемых.
                row2["distance_to_target"] = float("inf")
                row2["distance_excluded_reason"] = "too_few_metrics"

            fam_d = info.get("family_distances", {}) or {}
            for fam in normalized_families:
                row2[f"family_dist__{fam}"] = float(fam_d.get(fam, np.nan))

            if not baseline_holder:
                baseline_holder.update(row2)
            baseline = baseline_holder

            for col in ["density", "eff_w", "lcc_frac", "E", "total_weight"]:
                base_val = _to_float(baseline.get(col, np.nan))
                cur_val = _to_float(row2.get(col, np.nan))
                row2[f"delta_{col}"] = (
                    float(cur_val - base_val)
                    if pd.notna(base_val) and pd.notna(cur_val)
                    else np.nan
                )

            base_e = _to_float(baseline.get("E", np.nan))
            cur_e = _to_float(row2.get("E", np.nan))
            row2["removed_edge_fraction_from_baseline"] = (
                float(1.0 - (cur_e / base_e))
                if pd.notna(base_e) and base_e > 0 and pd.notna(cur_e)
                else np.nan
            )

            _pm_append_csv(traj_path, row2)
            _pm_append_csv(run_dir / "aggregate" / "trajectory_results.csv", row2)

            dist = float(row2.get("distance_to_target", np.nan))
            if np.isfinite(dist) and dist < best_distance:
                best_distance = dist
                best_row = dict(row2)

            for m in metric_list:
                val = _to_float(row2.get(m, np.nan))
                tgt = _to_float(target_vector.get(m, np.nan))
                sc = _to_float(scales.get(m, 1.0), default=1.0)

                if pd.isna(val) or pd.isna(tgt):
                    err = np.nan
                else:
                    if pd.isna(sc) or float(sc) <= 1e-12:
                        sc = 1.0
                    err = abs((float(val) - float(tgt)) / float(sc))

                prev = metric_best.get(m)
                prev_err = np.nan if prev is None else float(prev.get("best_scalar_error", np.nan))
                if prev is None or (np.isfinite(err) and (not np.isfinite(prev_err) or err < prev_err)):
                    metric_best[m] = {
                        "subject_idx": int(subject_idx),
                        "subject_id": str(subject_id),
                        "attack_kind": str(kind),
                        "metric": str(m),
                        "metric_family": next(
                            (fam for fam, fam_metrics in normalized_families.items() if m in fam_metrics),
                            "singleton",
                        ),
                        "best_step": row2.get("step", None),
                        "best_damage_frac": float(row2.get("damage_frac", np.nan)),
                        "best_scalar_error": float(err) if np.isfinite(err) else np.nan,
                        "best_value": float(val) if pd.notna(val) else np.nan,
                    }

        def _progress_cb(i: int, total_steps: int, current_value=None) -> None:
            _check_timeout()
            if callable(progress_cb):
                progress_cb(
                    attack_pos - 1,
                    len(attack_kinds),
                    i,
                    total_steps,
                    str(kind),
                    current_value,
                )

        run_degradation_trajectory(
            G,
            kind=str(kind),
            steps=int(steps),
            frac=float(frac),
            seed=int(seed) + int(subject_idx),
            eff_sources_k=int(eff_sources_k),
            compute_heavy_every=int(compute_heavy_every),
            compute_curvature=bool(compute_curvature),
            curvature_sample_edges=int(curvature_sample_edges),
            metric_names=metric_list,
            noise_sigma_max=float(noise_sigma_max),
            keep_density_from_baseline=bool(keep_density_from_baseline),
            module_info=module_info,
            recompute_modules=bool(recompute_modules),
            module_resolution=float(module_resolution),
            removal_mode=str(removal_mode),
            fast_mode=bool(fast_mode),
            progress_cb=_progress_cb,
            row_cb=_row_cb,
        )

        subject_row = {
            "subject_idx": int(subject_idx),
            "subject_id": str(subject_id),
            "attack_kind": str(kind),
            "best_step": None if best_row is None else best_row.get("step", None),
            "best_damage_frac": np.nan if best_row is None else float(best_row.get("damage_frac", np.nan)),
            "best_distance": np.nan if best_row is None else float(best_row.get("distance_to_target", np.nan)),
            "n_used_metrics": 0 if best_row is None else int(best_row.get("n_used_metrics", 0)),
            "distance_mode": "raw" if best_row is None else str(best_row.get("distance_mode", "raw")),
            "best_delta_density": np.nan if best_row is None else float(best_row.get("delta_density", np.nan)),
            "best_delta_total_weight": np.nan if best_row is None else float(best_row.get("delta_total_weight", best_row.get("delta_E", np.nan))),
            "best_delta_E": np.nan if best_row is None else float(best_row.get("delta_E", np.nan)),
            "best_removed_edge_fraction_from_baseline": np.nan if best_row is None else float(best_row.get("removed_edge_fraction_from_baseline", np.nan)),
        }
        subject_row.update(subject_meta)
        subject_rows.append(subject_row)

        _pm_append_csv(best_path, subject_row)
        _pm_append_csv(run_dir / "aggregate" / "subject_results.csv", subject_row)

        scalar_block = list(metric_best.values())
        if scalar_block:
            scalar_rows.extend(scalar_block)
            _pm_append_csv(scalar_path, scalar_block)
            _pm_append_csv(run_dir / "aggregate" / "scalar_subject_results.csv", scalar_block)

        if callable(progress_cb):
            progress_cb(attack_pos, len(attack_kinds), last_step, last_step, str(kind), None)

    subject_df = pd.DataFrame(subject_rows)

    if not subject_df.empty:
        valid = subject_df[np.isfinite(pd.to_numeric(subject_df["best_distance"], errors="coerce"))].copy()
        if not valid.empty:
            idx = int(valid["best_distance"].astype(float).idxmin())
            winner_row = valid.loc[idx].to_dict()
            winner_row["is_subject_winner"] = True
            winner_rows.append(winner_row)
            _pm_append_csv(winner_path, winner_row)
            _pm_append_csv(run_dir / "aggregate" / "winner_results.csv", winner_row)

    scalar_subject_df = build_scalar_subject_results(pd.DataFrame(scalar_rows), metrics=metric_list)
    if bool(export_subject_excel):
        scalar_winners_df = build_scalar_winners(scalar_subject_df)
        scalar_summary_df = build_scalar_summary(scalar_subject_df, scalar_winners_df)
        result = {
            "target_vector": target_vector,
            "scales": scales,
            "subject_results": subject_df,
            "winner_results": pd.DataFrame(winner_rows),
            "trajectory_results": _pm_read_csv_if_exists(traj_path),
            "scalar_subject_results": scalar_subject_df,
            "scalar_winners": scalar_winners_df,
            "scalar_summary": scalar_summary_df,
            "metrics_used": metric_list,
            "metric_families": normalized_families,
            "distance_mode": "raw",
        }
        export_phenotype_match_excel(result, sdir / "subject_bundle.xlsx")

    _pm_write_json(
        sdir / "done.json",
        {
            "subject_id": str(subject_id),
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_sec": float(time.perf_counter() - t0),
            "n_attacks": int(len(attack_kinds)),
        },
    )

    return (
        subject_df,
        pd.DataFrame(winner_rows),
        scalar_subject_df,
        _pm_read_csv_if_exists(traj_path) if bool(include_subject_trajectory_in_return) else pd.DataFrame(),
    )


def _match_workspace_graphs_to_metadata(
    graphs: dict,
    meta_df: pd.DataFrame,
    *,
    group_col: str,
    healthy_values: str,
    sz_values: str,
) -> dict[str, object]:
    """Match loaded workspace graphs to metadata and split into HC / SZ."""
    healthy_set = _token_set(healthy_values)
    sz_set = _token_set(sz_values)

    matched_rows: list[dict] = []
    hc_gids: list[str] = []
    sz_gids: list[str] = []
    unmatched_gids: list[str] = []

    meta_key_to_row = {
        str(row["__meta_subject_key"]): row
        for row in meta_df.to_dict(orient="records")
        if str(row.get("__meta_subject_key", "")).strip()
    }

    for gid, entry in graphs.items():
        candidates = _metadata_match_candidates(entry, str(gid))

        row = None
        matched_key = ""
        for key in candidates:
            hit = meta_key_to_row.get(key)
            if hit is not None:
                row = dict(hit)
                matched_key = key
                break

        if row is None:
            # Final fallback: allow metadata subject id to appear as a suffix/token inside
            # the graph label. This is slower, so we only do it after exact-key attempts.
            cand_set = set(candidates)
            for meta_key, meta_row in meta_key_to_row.items():
                if meta_key in cand_set:
                    row = dict(meta_row)
                    matched_key = meta_key
                    break
                if any(meta_key and (meta_key in cand or cand in meta_key) for cand in cand_set):
                    row = dict(meta_row)
                    matched_key = meta_key
                    break

        if row is None:
            unmatched_gids.append(gid)
            continue

        group_token = _norm_meta_token(row.get(group_col, ""))
        subject_id = str(row.get("__meta_subject_key", matched_key or gid))

        rec = {
            "graph_id": str(gid),
            "graph_name": str(entry.name),
            "graph_source": str(entry.source),
            "subject_id": subject_id,
            "group_value_raw": row.get(group_col, ""),
            "group_value_norm": group_token,
        }

        matched_rows.append(rec)

        if group_token in healthy_set:
            hc_gids.append(gid)
        elif group_token in sz_set:
            sz_gids.append(gid)

    matched_df = pd.DataFrame(matched_rows)
    hc_meta_df = matched_df[matched_df["graph_id"].isin(hc_gids)].copy() if not matched_df.empty else pd.DataFrame()
    sz_meta_df = matched_df[matched_df["graph_id"].isin(sz_gids)].copy() if not matched_df.empty else pd.DataFrame()

    return {
        "matched_df": matched_df,
        "hc_gids": hc_gids,
        "sz_gids": sz_gids,
        "unmatched_gids": unmatched_gids,
        "hc_meta_df": hc_meta_df,
        "sz_meta_df": sz_meta_df,
    }


def _compute_metrics_df_for_graph_ids(
    graph_ids: list[str],
    *,
    graphs: dict,
    min_conf: float,
    min_weight: float,
    analysis_mode: str,
    eff_k: int,
    seed: int,
    compute_curvature: bool,
    metric_names: list[str] | None = None,
    progress_cb=None,
    timeout_seconds_per_graph: float = 0.0,
) -> pd.DataFrame:
    """Compute baseline metrics table for selected loaded graphs.

    Дополнительно сохраняет audit-информацию по каждому графу:
    graph_name/source/status/error/elapsed/N/E/density...
    """
    _mset = set(str(m) for m in (metric_names or []))
    _needs_spectral = not _mset or bool(
        _mset & {"mod", "l2_lcc", "lmax", "thresh", "tau_lcc", "tau_relax", "algebraic_connectivity"}
    )
    _needs_diameter = not _mset or bool(_mset & {"diameter_approx"})
    _needs_assortativity = not _mset or bool(_mset & {"assortativity"})
    _needs_clustering = not _mset or bool(_mset & {"clustering"})

    rows = []
    total = len(graph_ids)

    for idx, gid in enumerate(graph_ids):
        entry = graphs[gid]
        t0 = time.perf_counter()

        base_row = {
            "graph_id": str(gid),
            "graph_name": str(getattr(entry, "name", "")),
            "graph_source": str(getattr(entry, "source", "")),
            "status": "unknown",
            "error": "",
            "elapsed_sec": np.nan,
        }

        try:
            if callable(progress_cb):
                try:
                    progress_cb(idx, total, str(gid), "build+metrics:start")
                except Exception:
                    pass

            row = _run_with_timeout(
                compute_metrics_safe,
                entry.edges,
                entry.src_col,
                entry.dst_col,
                min_conf=float(min_conf),
                min_weight=float(min_weight),
                analysis_mode=str(analysis_mode),
                eff_k=int(eff_k),
                seed=int(seed),
                compute_curvature=bool(compute_curvature),
                needs_spectral=bool(_needs_spectral),
                needs_clustering=bool(_needs_clustering),
                needs_assortativity=bool(_needs_assortativity),
                needs_diameter=bool(_needs_diameter),
                graph_name=str(entry.name),
                timeout_seconds=float(timeout_seconds_per_graph or 0.0),
            )
            row = dict(row)
            row.update(base_row)
            row["status"] = "ok"
            row["elapsed_sec"] = float(time.perf_counter() - t0)

        except TimeoutError as exc:
            row = dict(base_row)
            row["status"] = "timeout"
            row["error"] = str(exc)
            row["elapsed_sec"] = float(time.perf_counter() - t0)

        except Exception as exc:
            row = dict(base_row)
            row["status"] = "error"
            row["error"] = f"{type(exc).__name__}: {exc}"
            row["elapsed_sec"] = float(time.perf_counter() - t0)

        rows.append(row)

        if callable(progress_cb):
            try:
                extra = str(row.get("status", "ok"))
                if row.get("status") == "ok":
                    extra += (
                        f" · name={row.get('graph_name', '')} "
                        f"· N={row.get('N', 'na')} "
                        f"E={row.get('E', 'na')} "
                        f"density={row.get('density', np.nan):.4f} "
                        f"t={row.get('elapsed_sec', np.nan):.1f}s"
                    )
                elif row.get("error"):
                    extra += (
                        f" · name={row.get('graph_name', '')} "
                        f"· error={row.get('error', '')}"
                    )
                progress_cb(idx + 1, total, str(gid), extra)
            except Exception:
                pass

    return pd.DataFrame(rows)


def _build_current_graph_for_entry(
    entry: GraphEntry,
    *,
    min_conf: float,
    min_weight: float,
    analysis_mode: str,
) -> nx.Graph:
    """Собрать граф для конкретного entry с текущими UI-фильтрами."""
    return GraphService.build_graph(
        entry.edges,
        entry.src_col,
        entry.dst_col,
        float(min_conf),
        float(min_weight),
        str(analysis_mode),
    )


def _compute_one_graph_metrics_payload(
    entry: GraphEntry,
    *,
    min_conf: float,
    min_weight: float,
    analysis_mode: str,
    eff_k: int,
    seed: int,
    compute_curvature: bool,
    needs_spectral: bool,
    needs_clustering: bool,
    needs_assortativity: bool,
    needs_diameter: bool,
) -> dict:
    """Build one graph and compute one baseline metrics row in an isolated worker."""
    G = _build_current_graph_for_entry(
        entry,
        min_conf=float(min_conf),
        min_weight=float(min_weight),
        analysis_mode=str(analysis_mode),
    )
    met = calculate_metrics(
        G,
        int(eff_k),
        int(seed),
        bool(compute_curvature),
        curvature_sample_edges=80,
        compute_heavy=True,
        skip_spectral=not bool(needs_spectral),
        skip_clustering=not bool(needs_clustering),
        skip_assortativity=not bool(needs_assortativity),
        diameter_samples=8 if bool(needs_diameter) else 0,
    )
    row = {
        "graph_id": str(getattr(entry, "gid", getattr(entry, "name", ""))),
        "graph_name": str(getattr(entry, "name", "")),
        "status": "ok",
    }
    row.update(met)
    return row


def _run_with_timeout(fn, *args, timeout_seconds: float = 0.0, **kwargs):
    """Thin wrapper over the streamlit-safe multiprocessing timeout runner."""
    return _run_with_timeout_safe(fn, *args, timeout_seconds=timeout_seconds, **kwargs)


def _needs_curvature_for_metrics(metrics: list[str]) -> bool:
    """Нужен ли пересчет curvature для выбранного набора метрик."""
    return any(str(m).startswith("kappa_") or str(m) == "fragility_kappa" for m in metrics)


def _guess_hc_like_graph_ids(graphs: dict, active_graph_id: str | None) -> list[str]:
    """Эвристика: предложить healthy/control графы по имени и источнику."""
    hc_ids: list[str] = []
    for gid, entry in graphs.items():
        if gid == active_graph_id:
            continue
        txt = f"{getattr(entry, 'name', '')} {getattr(entry, 'source', '')}".lower()
        if any(tok in txt for tok in ("hc", "healthy", "control", "norm", "норма", "контроль")):
            hc_ids.append(gid)
    return hc_ids


def _guess_sz_like_graph_ids(graphs: dict, active_graph_id: str | None) -> list[str]:
    """Эвристика: предложить SZ/patient графы по имени и источнику."""
    sz_ids: list[str] = []
    for gid, entry in graphs.items():
        if gid == active_graph_id:
            continue
        txt = f"{getattr(entry, 'name', '')} {getattr(entry, 'source', '')}".lower()
        if any(tok in txt for tok in ("sz", "schiz", "schizo", "patient", "case", "пациент", "шиз")):
            sz_ids.append(gid)
    return sz_ids


def _mixfrac_result_to_history(res: dict) -> pd.DataFrame:
    """Преобразовать результат mix_frac* в одно-строчную history-таблицу эксперимента."""
    vals = [float(v) for v in res.get("mix_frac_values", []) if np.isfinite(v)]
    dists = [float(v) for v in res.get("distances", []) if np.isfinite(v)]
    return pd.DataFrame(
        [
            {
                "mix_frac_star": float(res.get("mix_frac_star", np.nan)),
                "ci_low": float(res.get("ci_low", np.nan)),
                "ci_high": float(res.get("ci_high", np.nan)),
                "distance_median": float(res.get("distance_median", np.nan)),
                "distance_mean": float(np.mean(dists)) if dists else float("nan"),
                "healthy_n": int(res.get("healthy_n", 0)),
                "match_mode": str(res.get("match_mode", "")),
                "replace_from": str(res.get("replace_from", "")),
                "used_metrics": ",".join([str(x) for x in res.get("used_metrics", [])]),
                "mix_frac_values_n": int(len(vals)),
            }
        ]
    )


def _live_history_preview(max_rows: int = 12):
    """Build a tiny live table that shows the latest attack-trajectory rows."""
    holder = st.empty()
    rows: list[dict] = []

    def _row_cb(row: dict, i: int, total: int) -> None:
        _ = (i, total)  # kept for signature compatibility with attack callbacks
        rows.append(dict(row))
        df = pd.DataFrame(rows[-max_rows:])
        holder.dataframe(df, width="stretch", height=260)

    return holder, _row_cb

def _extract_removed_order(aux):
    if isinstance(aux, dict):
        for k in ["removed_nodes", "removed_order", "order", "removal_order", "removed"]:
            v = aux.get(k)
            if isinstance(v, (list, tuple)) and v:
                return list(v)
    if isinstance(aux, (list, tuple)) and aux:
        if not isinstance(aux[0], (pd.DataFrame, np.ndarray, dict, list, tuple)):
            return list(aux)
    return None

def _fallback_removal_order(G: nx.Graph, kind: str, seed: int):
    """
    Fallback для 3D-декомпозиции, если src.attacks не вернул порядок удаления.
    ВАЖНО: это не адаптивная атака, только визуальный fallback.
    """
    if G.number_of_nodes() == 0:
        return []

    rng = np.random.default_rng(int(seed))
    H = as_simple_undirected(G)
    nodes = list(H.nodes())

    if kind in ("random",):
        rng.shuffle(nodes)
        return nodes

    if kind in ("degree",):
        nodes.sort(key=lambda n: H.degree(n), reverse=True)
        return nodes

    if kind in ("low_degree",):  
        nodes.sort(key=lambda n: H.degree(n))
        return nodes

    if kind in ("weak_strength",): 
        nodes.sort(key=lambda n: get_node_strength(H, n))
        return nodes

    if kind in ("betweenness",):
        if H.number_of_nodes() > 5000:
            nodes.sort(key=lambda n: H.degree(n), reverse=True)
            return nodes
        b = _cached_betweenness(H)
        nodes.sort(key=lambda n: b.get(n, 0.0), reverse=True)
        return nodes

    if kind in ("kcore",):
        core = nx.core_number(H)
        nodes.sort(key=lambda n: core.get(n, 0), reverse=True)
        return nodes

    if kind in ("richclub_top",):
        nodes.sort(key=lambda n: get_node_strength(H, n), reverse=True)
        return nodes

    rng.shuffle(nodes)
    return nodes

def render_null_models(G_view: nx.Graph | None, G_full: nx.Graph | None, met: dict, active_entry: GraphEntry, seed_val: int, add_graph_callback) -> None:
    """Render the null models tab."""
    if G_view is None:
        return

    st.header("🧪 Нулевые модели и синтетика")

    nm_col1, nm_col2 = st.columns([1, 2])

    with nm_col1:
        st.subheader("Параметры")
        null_kind = st.selectbox("Тип модели", ["ER G(n,m)", "Configuration Model", "Mix/Rewire (p)"])

        mix_p = 0.0
        if null_kind == "Mix/Rewire (p)":
            mix_p = st.slider("p (rewiring probability)", 0.0, 1.0, 0.2, 0.05, help=help_icon("Mix/Rewire"))

        nm_seed = st.number_input("Seed генерации", value=int(seed_val), step=1)
        new_name_suffix = st.text_input("Суффикс имени", value="_null")

        if st.button("⚙️ Создать и добавить", type="primary"):
            with st.spinner("Генерация..."):
                if null_kind == "ER G(n,m)":
                    G_new = make_er_gnm(G_full.number_of_nodes(), G_full.number_of_edges(), seed=int(nm_seed))
                    src_tag = "ER"
                elif null_kind == "Configuration Model":
                    G_new = make_configuration_model(G_full, seed=int(nm_seed))
                    src_tag = "CFG"
                else:
                    G_new = rewire_mix(G_full, p=float(mix_p), seed=int(nm_seed))
                    src_tag = f"MIX(p={mix_p})"

                edges = [[u, v, 1.0, 1.0] for u, v in as_simple_undirected(G_new).edges()]
                df_new = pd.DataFrame(edges, columns=["src", "dst", "weight", "confidence"])

                add_graph_callback(
                    f"{active_entry.name}{new_name_suffix}",
                    df_new,
                    f"null:{src_tag}",
                    "src",
                    "dst",
                )
                st.success("Граф создан. Переключаюсь на него...")
                st.rerun()

    with nm_col2:
        st.info("Быстрая проверка против ER-ожиданий (очень грубо):")
        N = G_view.number_of_nodes()
        M = G_view.number_of_edges()
        er_density = 2 * M / (N * (N - 1)) if N > 1 else 0.0
        er_clustering = er_density

        met_light = met
        cmp_df = pd.DataFrame({
            "Metric": ["Avg Degree", "Density", "Clustering (C)", "Modularity (примерно)"],
            "Active Graph": [met_light.get("avg_degree", np.nan), met_light.get("density", np.nan), met_light.get("clustering", np.nan), met_light.get("mod", np.nan)],
            "ER Expected": [met_light.get("avg_degree", np.nan), er_density, er_clustering, 0.0],
        })
        st.dataframe(cmp_df, width="stretch")

def render_phenotype_matching_tab(
    *,
    active_entry: GraphEntry,
    seed_val: int,
    min_conf: float,
    min_weight: float,
    analysis_mode: str,
) -> None:
    """Render standalone HC→SZ phenotype-matching UI tab."""
    st.header("🧬 HC → SZ phenotype matching")
    st.caption(
        "Отдельная вкладка для: разрушать HC-графы, считать trajectory по всем выбранным атакам "
        "и сравнивать с целевым SZ-профилем."
    )

    graphs = st.session_state["graphs"]
    active_gid = st.session_state.get("active_graph_id")
    hc_guess = _guess_hc_like_graph_ids(graphs, active_gid)

    pm_col1, pm_col2 = st.columns([1, 2])

    with pm_col1:
        st.caption("Можно либо выбрать HC вручную, либо дать metadata и собрать HC/SZ автоматически из workspace.")

        pm_meta_file = st.file_uploader(
            "Metadata file (CSV/XLSX)",
            type=["csv", "tsv", "xlsx", "xls"],
            key="pm_meta_file",
        )

        pm_hc_gids = st.multiselect(
            "HC графы (ручной режим / fallback)",
            options=list(graphs.keys()),
            default=hc_guess,
            format_func=lambda gid: f"{graphs[gid].name} ({graphs[gid].source})",
            key="pm_hc_gids",
        )

        pm_sz_file = st.file_uploader(
            "SZ metrics table (CSV/XLSX, optional if metadata is loaded)",
            type=["csv", "tsv", "xlsx", "xls"],
            key="pm_sz_file",
        )
        pm_hc_base_file = st.file_uploader(
            "HC baseline metrics table (optional; иначе посчитается автоматически)",
            type=["csv", "tsv", "xlsx", "xls"],
            key="pm_hc_base_file",
        )

        st.markdown("#### Вывод и управление")
        pm_out_dir = st.text_input(
            "Папка вывода",
            value=str(st.session_state.get("pm_out_dir", "./phenotype_runs")),
            key="pm_out_dir",
            help="Относительный путь считается от корня репозитория. По умолчанию: ./phenotype_runs",
        )
        pm_out_dir_resolved = _pm_resolve_out_dir(pm_out_dir)
        try:
            pm_out_dir_resolved.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            st.error(f"Не удалось создать папку вывода: {type(exc).__name__}: {exc}")
        st.caption(f"Физически сохраняется в: {pm_out_dir_resolved}")
        pm_run_label = st.text_input(
            "Название run",
            value=str(st.session_state.get("pm_run_label", f"hc_sz_{time.strftime('%Y%m%d_%H%M%S')}")),
            key="pm_run_label",
        )
        pm_skip_done = st.checkbox(
            "Проверять, сделано ли уже",
            value=True,
            key="pm_skip_done",
        )
        pm_start_index = st.number_input(
            "Начать с HC-индекса",
            min_value=0,
            value=int(st.session_state.get("pm_start_index", 0)),
            step=1,
            key="pm_start_index",
        )
        pm_timeout_per_subject = st.number_input(
            "Скип по времени на HC-граф (сек, 0=без лимита)",
            min_value=0,
            value=int(st.session_state.get("pm_timeout_per_subject", 0)),
            step=10,
            key="pm_timeout_per_subject",
        )
        pm_timeout_per_stage = st.number_input(
            "Скип по времени на любой prepare/build-этап (сек)",
            min_value=0,
            value=int(st.session_state.get("pm_timeout_per_stage", 600)),
            step=10,
            key="pm_timeout_per_stage",
            help="Если один baseline/build этап длится слишком долго, он помечается как timeout и run идёт дальше.",
        )
        pm_exclude_ids_raw = st.text_area(
            "Exclude graph/subject ids",
            value="",
            help="Список через запятую/новую строку. Эти graph_id или subject_id будут исключены из HC run.",
        )

        pm_attack_kinds = st.multiselect(
            "Модели деградации",
            options=DEGRADATION_KIND_OPTIONS,
            default=[
                "weight_noise",
                "inter_module_removal",
                "intra_module_removal",
                "weak_edges_by_weight",
                "strong_edges_by_weight",
            ],
            key="pm_attack_kinds",
        )

        # Фильтруем default по фактическим options, чтобы избежать падения UI,
        # если registry и UI-список временно рассинхронизировались.
        pm_metric_options = list(DEGRADATION_METRIC_OPTIONS)
        pm_metric_default = [
            m for m in get_default_metrics_for_regime("full_weighted_signed_hybrid")
            if m in pm_metric_options
        ]

        # Страховка от пустого default: всегда оставляем минимальный рабочий набор.
        if not pm_metric_default:
            pm_metric_default = ["l2_lcc", "H_rw", "fragility_H", "mod"]

        # Если в session_state остались старые/битые метрики, очищаем их заранее,
        # чтобы st.multiselect не падал при инициализации значения key.
        existing_pm_metrics = st.session_state.get("pm_metrics")
        if isinstance(existing_pm_metrics, (list, tuple)):
            st.session_state["pm_metrics"] = [
                m for m in existing_pm_metrics if m in pm_metric_options
            ]

        pm_metrics = st.multiselect(
            "Метрики distance",
            options=pm_metric_options,
            default=pm_metric_default,
            key="pm_metrics",
        )

        regime_metric_info = describe_metrics_for_regime("full_weighted_signed_hybrid")
        with st.expander("Какие метрики валидны для signed-hybrid full weighted regime"):
            st.markdown(
                "**Core:** " + ", ".join(regime_metric_info["core"]) + "\n\n"
                + "**Secondary:** " + ", ".join(regime_metric_info["secondary"]) + "\n\n"
                + "**Discouraged:** " + ", ".join(regime_metric_info["discouraged"]) + "\n\n"
                + "**Guardrail only:** " + ", ".join(regime_metric_info["guardrail"])
            )

        st.info(
            "Signed-hybrid режим: raw signed weight сохраняется для sign-aware метрик, "
            "а operational weight=|raw_weight| используется для distance/random-walk/spectral частей. "
            "Нулевые веса удаляются."
        )

        with st.expander("Phenotype matching advanced"):
            pm_meta_id_col = st.text_input(
                "Metadata ID column (optional)",
                value="",
                key="pm_meta_id_col",
            )
            pm_meta_group_col = st.text_input(
                "Metadata group column (optional)",
                value="",
                key="pm_meta_group_col",
            )
            pm_meta_hc_values = st.text_input(
                "Healthy/control values",
                value="0,healthy,control,hc,false",
                key="pm_meta_hc_values",
            )
            pm_meta_sz_values = st.text_input(
                "SZ values",
                value="1,sz,schizophrenia,patient,case,true",
                key="pm_meta_sz_values",
            )
            pm_compute_curv = st.checkbox(
                "Считать curvature в baseline",
                value=_needs_curvature_for_metrics(pm_metrics),
                key="pm_compute_curv",
            )
            pm_steps = st.slider("Шаги trajectory", 5, 40, 12, key="pm_steps")
            pm_frac = st.slider("Макс damage frac", 0.05, 0.95, 0.5, 0.05, key="pm_frac")
            pm_seed = st.number_input("Seed (matching)", value=int(seed_val), step=1, key="pm_seed")
            pm_effk = st.slider("Efficiency k (matching)", 8, 256, 32, key="pm_effk")
            pm_heavy = st.slider("Heavy every N (matching)", 1, 10, 2, key="pm_heavy")
            pm_sigma = st.slider("sigma_max (weight_noise)", 0.01, 2.0, 0.5, 0.01, key="pm_sigma")
            pm_keep_density = st.checkbox("Keep density from baseline", value=True, key="pm_keep_density")
            pm_recompute_modules = st.checkbox("Recompute modules each step", value=False, key="pm_recompute_modules")
            # Явно управляем политикой обработки знака/веса в positive-weight pipeline.
            pm_weight_policy_ui = st.selectbox(
                "Политика обработки знака весов",
                options=[
                    "signed_split",
                    "drop_nonpositive",
                    "abs",
                    "shift",
                ],
                index=0,
                key="pm_weight_policy_ui",
                help=(
                    "signed_split: сохранять raw signed вес отдельно и считать операционный вес как |w|; "
                    "drop_nonpositive: удалить отрицательные и нулевые веса; "
                    "abs: взять модуль (не рекомендуется как дефолт); "
                    "shift: сдвинуть веса на константу и обрезать снизу."
                ),
            )

            pm_weight_shift_ui = st.number_input(
                "Shift для весов (если выбран shift)",
                value=float(st.session_state.get("pm_weight_shift_ui", 0.0)),
                step=0.01,
                key="pm_weight_shift_ui",
            )
            pm_module_resolution = st.slider("Module resolution", 0.2, 3.0, 1.0, 0.1, key="pm_module_resolution")
            pm_removal_mode = st.selectbox(
                "Removal mode",
                ["random", "weak_weight", "strong_weight"],
                index=0,
                key="pm_removal_mode",
            )
            pm_strict_streaming = st.checkbox(
                "Жёсткая потоковость (без автозагрузки trajectory в UI)",
                value=True,
                key="pm_strict_streaming",
                help="Быстрее и стабильнее на больших run: UI не перечитывает trajectory целиком после завершения.",
            )
            pm_export_subject_xlsx = st.checkbox(
                "Экспортировать subject_bundle.xlsx для каждого HC",
                value=False,
                key="pm_export_subject_xlsx",
                help="Замедляет run, потому что в конце каждого HC приходится перечитывать trajectory и собирать Excel.",
            )
            pm_export_run_xlsx = st.checkbox(
                "Экспортировать общий run_bundle.xlsx",
                value=False,
                key="pm_export_run_xlsx",
                help="Полезно для отчёта, но это не потоковая операция и может тормозить на больших trajectory.",
            )

        # Подсказка по выбранной политике знака, чтобы избежать неявной интерпретации весов.
        if str(pm_weight_policy_ui) == "signed_split":
            st.info(
                "Сейчас выбран режим signed_split: знак сохраняется в raw_weight/weight_signed, "
                "а операционные расчёты используют |w|."
            )
        elif str(pm_weight_policy_ui) == "abs":
            st.warning(
                "Сейчас выбран режим abs: отрицательные связи превращаются в положительные по модулю. "
                "Это уничтожает знак и может исказить интерпретацию functional connectivity."
            )
        elif str(pm_weight_policy_ui) == "drop_nonpositive":
            st.info(
                "Сейчас выбран режим drop_nonpositive: отрицательные связи не анализируются, "
                "но и не превращаются искусственно в положительные."
            )
        elif str(pm_weight_policy_ui) == "shift":
            st.warning(
                "Сейчас выбран режим shift: знак не сохраняется как знак, а кодируется через общий сдвиг. "
                "Используй только осознанно."
            )

        pm_meta = None
        pm_meta_info = {}
        pm_match = None
        if pm_meta_file is not None:
            try:
                pm_meta, pm_meta_info = _prepare_uploaded_metadata(
                    pm_meta_file,
                    id_col=str(st.session_state.get("pm_meta_id_col", "") or ""),
                    group_col=str(st.session_state.get("pm_meta_group_col", "") or ""),
                )
                pm_match = _match_workspace_graphs_to_metadata(
                    graphs,
                    pm_meta,
                    group_col=str(pm_meta_info["group_col"]),
                    healthy_values=str(st.session_state.get("pm_meta_hc_values", "0,healthy,control,hc,false")),
                    sz_values=str(st.session_state.get("pm_meta_sz_values", "1,sz,schizophrenia,patient,case,true")),
                )

                st.success(
                    f"Metadata detected: id_col='{pm_meta_info['id_col']}', "
                    f"group_col='{pm_meta_info['group_col']}'. "
                    f"Matched={len(pm_match['matched_df'])}, HC={len(pm_match['hc_gids'])}, SZ={len(pm_match['sz_gids'])}, "
                    f"unmatched={len(pm_match['unmatched_gids'])}"
                )
                if pm_match["unmatched_gids"]:
                    st.warning(
                        "Не сматчились: "
                        + ", ".join(str(x) for x in pm_match["unmatched_gids"][:10])
                        + (" ..." if len(pm_match["unmatched_gids"]) > 10 else "")
                    )
            except Exception as exc:
                st.error(f"Metadata parse/match error: {type(exc).__name__}: {exc}")

        use_metadata_preview = (
            pm_match is not None
            and len(pm_match.get("hc_gids", [])) > 0
            and len(pm_match.get("sz_gids", [])) > 0
        )

        preview_hc_n = len(pm_match.get("hc_gids", [])) if use_metadata_preview else len(pm_hc_gids)
        preview_sz_n = len(pm_match.get("sz_gids", [])) if use_metadata_preview else (0 if pm_sz_file is not None else 0)

        # Estimate graph density from the active graph for runtime prediction.
        _density_est = 0.0
        _n_edges_est = 0
        _n_nodes_est = 0
        try:
            _edges_df = active_entry.edges
            _n_edges_est = int(len(_edges_df))
            _n_nodes_est = int(
                len(set(_edges_df[active_entry.src_col].tolist() + _edges_df[active_entry.dst_col].tolist()))
            )
            _max_e_est = _n_nodes_est * (_n_nodes_est - 1) // 2 if _n_nodes_est > 1 else 1
            _density_est = float(_n_edges_est) / float(max(1, _max_e_est))
        except Exception:
            # Best-effort only: keep UI responsive even if the preview graph is malformed.
            pass

        preflight = _pm_build_preflight_summary(
            hc_n=int(preview_hc_n),
            sz_n=int(preview_sz_n),
            attack_kinds=[str(x) for x in pm_attack_kinds],
            metrics=[str(x) for x in pm_metrics],
            steps=int(pm_steps),
            compute_curv=bool(pm_compute_curv),
            export_subject_xlsx=bool(pm_export_subject_xlsx),
            export_run_xlsx=bool(pm_export_run_xlsx),
            density_estimate=float(_density_est),
            n_edges_approx=int(_n_edges_est),
            n_nodes_approx=int(_n_nodes_est),
            heavy_every=int(pm_heavy),
        )

        with st.expander("Что будет посчитано / preflight", expanded=True):
            st.markdown(
                "\n".join(
                    [
                        f"- HC графов: **{preflight['hc_n']}**",
                        f"- SZ графов: **{preflight['sz_n']}**",
                        f"- Моделей деградации: **{preflight['attacks_n']}**",
                        f"- Метрик distance: **{preflight['metrics_n']}**",
                        f"- Шагов trajectory на атаку: **{preflight['steps']}**",
                        f"- Prepare graph builds: **{preflight['baseline_graph_builds']}**",
                        f"- Prepare metric passes: **{preflight['baseline_metric_passes']}**",
                        f"- Всего trajectory steps: **{preflight['trajectory_steps_total']}**",
                        f"- Curvature/Ricci в baseline: **{'ON' if preflight['curvature_baseline'] else 'OFF'}**",
                        f"- Curvature/Ricci в trajectory: **{'ON' if preflight['curvature_trajectory'] else 'OFF'}**",
                        f"- Тяжёлые метрики: **{', '.join(preflight['heavy_metrics']) if preflight['heavy_metrics'] else 'нет'}**",
                        f"- Тяжёлые атаки: **{', '.join(preflight['heavy_attacks']) if preflight['heavy_attacks'] else 'нет'}**",
                        f"- subject_bundle.xlsx: **{'ON' if preflight['export_subject_xlsx'] else 'OFF'}**",
                        f"- run_bundle.xlsx: **{'ON' if preflight['export_run_xlsx'] else 'OFF'}**",
                        f"- Оценка нагрузки: **{preflight['estimated_cost'].upper()}**",
                    ]
                )
            )

            if preflight.get("density_estimate", 0) > 0.01:
                _est_sub = preflight.get("est_time_per_subject_sec", 0.0)
                _est_total = preflight.get("est_total_sec", 0.0)
                st.markdown(
                    "\n".join(
                        [
                            f"- Плотность графа (оценка): **{preflight['density_estimate']:.2f}** "
                            f"({preflight['n_nodes_approx']} узлов, {preflight['n_edges_approx']} рёбер)",
                            f"- Оценка времени на 1 HC: **~{_est_sub / 60:.1f} мин**",
                            f"- Оценка общего времени: **~{_est_total / 3600:.1f} ч** "
                            f"(baseline + {preflight['hc_n']} HC)",
                        ]
                    )
                )

            if preflight["curvature_baseline"] or preflight["curvature_trajectory"]:
                st.warning("Curvature/Ricci включён. Такой run будет заметно тяжелее.")
            else:
                st.info("Curvature/Ricci сейчас не должен считаться, если ты не включила его другими настройками/атаками.")

            if preflight.get("mix_on_dense_graph"):
                st.error(
                    "⚠️ Mix-атаки (mix_default, mix_degree_preserving) выбраны при density > 0.50. "
                    "На плотных графах edge swap и replacement нефункциональны: "
                    "double_edge_swap не находит валидных свопов, "
                    "_replace_edges_from_source попадает в collision loop. "
                    "Эти атаки будут автоматически пропущены при density > 0.80. "
                    "Рекомендация: убрать mix_default и mix_degree_preserving из списка."
                )

            if preflight.get("density_estimate", 0) > 0.5 and int(pm_heavy) < 3:
                st.warning(
                    f"При density={preflight['density_estimate']:.2f} рекомендуется heavy_every ≥ 3 "
                    f"(сейчас {pm_heavy}). Это уменьшит количество тяжёлых шагов с метриками "
                    f"и ускорит run примерно на ~30%."
                )

            if preflight.get("density_estimate", 0) > 0.80:
                dead_defaults = [m for m in ["density", "clustering", "lcc_frac"] if m in pm_metrics]
                if dead_defaults:
                    st.warning(
                        "Для почти полного графа выбраны метрики, которые обычно близки к константе: "
                        + ", ".join(dead_defaults)
                        + ". Они будут автоматически исключены из phenotype-distance по baseline variability audit, "
                        "но лучше убрать их сразу из UI-набора."
                    )

        if st.button("🚀 RUN HC→SZ MATCHING", type="primary", width="stretch", key="pm_run"):
            # Settings заморожен, поэтому для быстрого hotfix используем object.__setattr__.
            object.__setattr__(settings, "WEIGHT_POLICY", str(pm_weight_policy_ui))
            if str(pm_weight_policy_ui) == "shift":
                object.__setattr__(settings, "WEIGHT_SHIFT", float(pm_weight_shift_ui))
            else:
                object.__setattr__(settings, "WEIGHT_SHIFT", 0.0)
            if not pm_attack_kinds:
                st.error("Выбери хотя бы одну модель деградации.")
            elif not pm_metrics:
                st.error("Выбери хотя бы одну метрику.")
            else:
                use_metadata_mode = (
                    pm_match is not None
                    and len(pm_match.get("hc_gids", [])) > 0
                    and len(pm_match.get("sz_gids", [])) > 0
                )

                if use_metadata_mode:
                    hc_gids_effective = list(pm_match["hc_gids"])
                    sz_gids_effective = list(pm_match["sz_gids"])
                else:
                    hc_gids_effective = list(pm_hc_gids)
                    sz_gids_effective = []

                if not hc_gids_effective:
                    st.error("Не найдено HC-графов: ни по metadata, ни вручную.")
                    st.stop()

                metric_list = [str(x) for x in pm_metrics]

                out_root = _pm_resolve_out_dir(pm_out_dir)
                out_root.mkdir(parents=True, exist_ok=True)
                run_name = _pm_safe_stem(str(pm_run_label), f"hc_sz_{time.strftime('%Y%m%d_%H%M%S')}")
                run_dir = out_root / run_name
                (run_dir / "aggregate").mkdir(parents=True, exist_ok=True)
                (run_dir / "per_subject").mkdir(parents=True, exist_ok=True)
                _pm_write_run_location_note(run_dir)
                st.session_state["last_degmatch_run_dir"] = str(run_dir)

                overall_bar = st.progress(0.0, text="Общий прогресс: prepare 0%")
                subject_bar = st.progress(0.0, text="Текущий HC: ожидание")
                phase_box = st.empty()
                status_box = st.empty()
                save_box = st.empty()
                files_box = st.empty()
                event_box = st.empty()

                ui_events: list[str] = []

                def _push_ui_event(msg: str) -> None:
                    stamp = time.strftime("%H:%M:%S")
                    line = f"[{stamp}] {msg}"
                    ui_events.append(line)
                    event_box.code("\n".join(ui_events[-14:]))

                save_box.info(f"Run dir: {run_dir}")
                _push_ui_event(f"Run created: {run_dir}")

                _pm_emit_progress(
                    run_dir,
                    {
                        "status": "running",
                        "phase": "prepare_init",
                        "run_dir": str(run_dir),
                        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "total_subjects": 0,
                        "completed_subjects": 0,
                        "current_subject_id": "",
                        "current_attack_kind": "",
                        "current_step": 0,
                        "current_step_total": 0,
                        "phase_progress": 0.0,
                    },
                    write_event=True,
                )

                phase_box.info("Prepare phase: инициализация run")
                status_box.caption("Подготовка конфигурации и входов...")
                overall_bar.progress(0.02, text="Общий прогресс: prepare 2%")
                subject_bar.progress(0.0, text="Текущий HC: ожидание")

                _pm_save_run_inputs(
                    run_dir,
                    config={
                        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "run_dir": str(run_dir),
                        "metrics": metric_list,
                        "metric_families": normalize_metric_families(metric_list, None),
                        "attack_kinds": [str(x) for x in pm_attack_kinds],
                        "steps": int(pm_steps),
                        "frac": float(pm_frac),
                        "seed": int(pm_seed),
                        "eff_k": int(pm_effk),
                        "heavy_every": int(pm_heavy),
                        "sigma_max": float(pm_sigma),
                        "keep_density_from_baseline": bool(pm_keep_density),
                        "recompute_modules": bool(pm_recompute_modules),
                        "module_resolution": float(pm_module_resolution),
                        "removal_mode": str(pm_removal_mode),
                        "compute_curvature": bool(pm_compute_curv),
                        "timeout_per_subject": float(pm_timeout_per_subject or 0),
                        "timeout_per_stage": float(pm_timeout_per_stage or 0),
                        "distance_mode": "raw",
                        "graph_regime": "full_weighted_signed_hybrid",
                        "weight_policy": str(pm_weight_policy_ui),
                        "weight_shift": float(pm_weight_shift_ui),
                    },
                    metadata_upload=pm_meta_file,
                    sz_upload=pm_sz_file,
                    hc_baseline_upload=pm_hc_base_file,
                )

                # ---------------- PREPARE: SZ baseline ----------------
                if use_metadata_mode:
                    phase_box.info("Prepare phase: SZ baseline metrics")
                    status_box.caption(f"Считаю baseline-метрики для SZ-группы ({len(sz_gids_effective)} графов)")
                    _push_ui_event(f"Prepare SZ baseline: {len(sz_gids_effective)} graphs")
                    overall_bar.progress(0.07, text="Общий прогресс: prepare 7%")

                    _pm_emit_progress(
                        run_dir,
                        {
                            "status": "running",
                            "phase": "prepare_sz_baseline",
                            "run_dir": str(run_dir),
                            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "total_subjects": 0,
                            "completed_subjects": 0,
                            "current_subject_id": "",
                            "current_attack_kind": "",
                            "current_step": 0,
                            "current_step_total": 0,
                            "phase_progress": 0.35,
                            "prepare_note": f"compute SZ baseline for {len(sz_gids_effective)} graphs",
                        },
                        write_event=True,
                    )

                    def _sz_progress(done, total, gid, stage="metrics"):
                        frac = 0.07 + 0.05 * (float(done) / float(max(1, total)))
                        overall_bar.progress(frac, text=f"Общий прогресс: SZ baseline {done}/{total} ({int(frac*100)}%)")
                        status_box.caption(f"SZ baseline: граф {done}/{total} · {gid} · {stage}")
                        _push_ui_event(f"SZ baseline: {done}/{total} · {gid} · {stage}")

                    sz_group_metrics_df = _compute_metrics_df_for_graph_ids(
                        sz_gids_effective,
                        graphs=graphs,
                        min_conf=float(min_conf),
                        min_weight=float(min_weight),
                        analysis_mode=str(analysis_mode),
                        eff_k=int(pm_effk),
                        seed=int(pm_seed),
                        compute_curvature=bool(pm_compute_curv),
                        metric_names=metric_list,
                        progress_cb=_sz_progress,
                        timeout_seconds_per_graph=float(pm_timeout_per_stage or 0),
                    )
                else:
                    if pm_sz_file is None:
                        st.error("Либо загрузи metadata, либо загрузи таблицу метрик SZ-группы.")
                        st.stop()
                    phase_box.info("Prepare phase: reading uploaded SZ metrics")
                    status_box.caption("Читаю загруженную таблицу метрик SZ")
                    _push_ui_event("Read uploaded SZ metrics table")
                    overall_bar.progress(0.07, text="Общий прогресс: prepare 7%")
                    sz_group_metrics_df = _read_uploaded_metrics_table(pm_sz_file)

                # ---------------- PREPARE: HC baseline ----------------
                phase_box.info("Prepare phase: HC baseline metrics")
                status_box.caption(f"Считаю baseline-метрики для HC-группы ({len(hc_gids_effective)} графов)")
                _push_ui_event(f"Prepare HC baseline: {len(hc_gids_effective)} graphs")
                overall_bar.progress(0.14, text="Общий прогресс: prepare 14%")

                _pm_emit_progress(
                    run_dir,
                    {
                        "status": "running",
                        "phase": "prepare_hc_baseline",
                        "run_dir": str(run_dir),
                        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "total_subjects": 0,
                        "completed_subjects": 0,
                        "current_subject_id": "",
                        "current_attack_kind": "",
                        "current_step": 0,
                        "current_step_total": 0,
                        "phase_progress": 0.70,
                        "prepare_note": f"compute HC baseline for {len(hc_gids_effective)} graphs",
                    },
                    write_event=True,
                )

                if pm_hc_base_file is not None:
                    hc_baseline_metrics_df = _read_uploaded_metrics_table(pm_hc_base_file)
                else:
                    def _hc_progress(done, total, gid, stage="metrics"):
                        frac = 0.14 + 0.04 * (float(done) / float(max(1, total)))
                        overall_bar.progress(frac, text=f"Общий прогресс: HC baseline {done}/{total} ({int(frac*100)}%)")
                        status_box.caption(f"HC baseline: граф {done}/{total} · {gid} · {stage}")
                        _push_ui_event(f"HC baseline: {done}/{total} · {gid} · {stage}")

                    hc_baseline_metrics_df = _compute_metrics_df_for_graph_ids(
                        hc_gids_effective,
                        graphs=graphs,
                        min_conf=float(min_conf),
                        min_weight=float(min_weight),
                        analysis_mode=str(analysis_mode),
                        eff_k=int(pm_effk),
                        seed=int(pm_seed),
                        compute_curvature=bool(pm_compute_curv),
                        metric_names=metric_list,
                        progress_cb=_hc_progress,
                        timeout_seconds_per_graph=float(pm_timeout_per_stage or 0),
                    )

                try:
                    sz_group_metrics_df.to_csv(run_dir / "aggregate" / "sz_baseline_graph_audit.csv", index=False)
                except Exception:
                    pass

                try:
                    hc_baseline_metrics_df.to_csv(run_dir / "aggregate" / "hc_baseline_graph_audit.csv", index=False)
                except Exception:
                    pass

                phase_box.info("Prepare phase: target/scales")
                status_box.caption("Собираю target vector, scales и metric families")
                _push_ui_event("Build target vector and scales")
                overall_bar.progress(0.18, text="Общий прогресс: prepare 18%")

                sz_status = sz_group_metrics_df.get("status")
                if sz_status is not None:
                    sz_ok_mask = sz_status.astype(str).eq("ok")
                    sz_failed = int((~sz_ok_mask).sum())
                    if sz_failed > 0:
                        st.warning(
                            f"⚠️ SZ baseline: {sz_failed} из {len(sz_group_metrics_df)} графов завершились с ошибкой/timeout. "
                            f"Target vector будет рассчитан по {int(sz_ok_mask.sum())} валидным графам."
                        )

                        # Показываем расширенный audit по проблемным графам до фильтрации ok-only.
                        sz_bad_df = sz_group_metrics_df.loc[
                            ~sz_ok_mask,
                            [c for c in [
                                "graph_id",
                                "graph_name",
                                "graph_source",
                                "status",
                                "error",
                                "elapsed_sec",
                            ] if c in sz_group_metrics_df.columns]
                        ].copy()

                        if "elapsed_sec" in sz_bad_df.columns:
                            sz_bad_df["elapsed_sec"] = pd.to_numeric(
                                sz_bad_df["elapsed_sec"], errors="coerce"
                            ).round(2)

                        st.markdown("**Проблемные графы SZ baseline**")
                        st.dataframe(sz_bad_df, use_container_width=True)

                        try:
                            for _, bad_row in sz_bad_df.iterrows():
                                _push_ui_event(
                                    "SZ baseline failed: "
                                    f"gid={bad_row.get('graph_id', '')} · "
                                    f"name={bad_row.get('graph_name', '')} · "
                                    f"status={bad_row.get('status', '')} · "
                                    f"error={bad_row.get('error', '')}"
                                )
                        except Exception:
                            pass

                    sz_group_metrics_df = sz_group_metrics_df[sz_ok_mask].copy()

                hc_status = hc_baseline_metrics_df.get("status")
                if hc_status is not None:
                    hc_ok_mask = hc_status.astype(str).eq("ok")
                    hc_failed = int((~hc_ok_mask).sum())
                    if hc_failed > 0:
                        st.warning(
                            f"⚠️ HC baseline: {hc_failed} из {len(hc_baseline_metrics_df)} графов завершились с ошибкой/timeout. "
                            f"Scale-вектор будет рассчитан по {int(hc_ok_mask.sum())} валидным графам."
                        )

                        # Показываем расширенный audit по проблемным графам до фильтрации ok-only.
                        hc_bad_df = hc_baseline_metrics_df.loc[
                            ~hc_ok_mask,
                            [c for c in [
                                "graph_id",
                                "graph_name",
                                "graph_source",
                                "status",
                                "error",
                                "elapsed_sec",
                            ] if c in hc_baseline_metrics_df.columns]
                        ].copy()

                        if "elapsed_sec" in hc_bad_df.columns:
                            hc_bad_df["elapsed_sec"] = pd.to_numeric(
                                hc_bad_df["elapsed_sec"], errors="coerce"
                            ).round(2)

                        st.markdown("**Проблемные графы HC baseline**")
                        st.dataframe(hc_bad_df, use_container_width=True)

                        try:
                            for _, bad_row in hc_bad_df.iterrows():
                                _push_ui_event(
                                    "HC baseline failed: "
                                    f"gid={bad_row.get('graph_id', '')} · "
                                    f"name={bad_row.get('graph_name', '')} · "
                                    f"status={bad_row.get('status', '')} · "
                                    f"error={bad_row.get('error', '')}"
                                )
                        except Exception:
                            pass

                    hc_baseline_metrics_df = hc_baseline_metrics_df[hc_ok_mask].copy()

                target_vector = build_group_target_vector(sz_group_metrics_df, metrics=metric_list)
                scale_resolved = resolve_metric_scales(hc_baseline_metrics_df, metrics=metric_list)
                scales = {str(k): float(v) for k, v in (scale_resolved.get("scales") or {}).items()}
                normalized_families = normalize_metric_families(metric_list, None)

                excluded_from_distance = list(scale_resolved.get("excluded_metrics", []) or [])
                if excluded_from_distance:
                    st.warning(
                        "⚠️ Метрики исключены из distance (low HC baseline variance): "
                        f"{', '.join(excluded_from_distance)}. "
                        f"Distance будет рассчитан по {len(scales)} из {len(metric_list)} метрик."
                    )

                _pm_save_run_inputs(
                    run_dir,
                    config={
                        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "run_dir": str(run_dir),
                        "metrics": metric_list,
                        "metric_families": normalized_families,
                        "attack_kinds": [str(x) for x in pm_attack_kinds],
                        "steps": int(pm_steps),
                        "frac": float(pm_frac),
                        "seed": int(pm_seed),
                        "eff_k": int(pm_effk),
                        "heavy_every": int(pm_heavy),
                        "sigma_max": float(pm_sigma),
                        "keep_density_from_baseline": bool(pm_keep_density),
                        "recompute_modules": bool(pm_recompute_modules),
                        "module_resolution": float(pm_module_resolution),
                        "removal_mode": str(pm_removal_mode),
                        "compute_curvature": bool(pm_compute_curv),
                        "timeout_per_subject": float(pm_timeout_per_subject or 0),
                        "timeout_per_stage": float(pm_timeout_per_stage or 0),
                        "distance_mode": "raw",
                        "graph_regime": "full_weighted_signed_hybrid",
                        "weight_policy": str(pm_weight_policy_ui),
                        "weight_shift": float(pm_weight_shift_ui),
                        "target_vector": target_vector,
                        "scales": scales,
                    },
                    metadata_upload=pm_meta_file,
                    sz_upload=pm_sz_file,
                    hc_baseline_upload=pm_hc_base_file,
                )

                subject_ids: list[str] = []
                subject_meta_map: dict[str, dict] = {}

                if use_metadata_mode:
                    hc_meta_df = pm_match.get("hc_meta_df", pd.DataFrame()).copy()
                    for gid in hc_gids_effective:
                        sub = hc_meta_df.loc[hc_meta_df["graph_id"] == gid]
                        if sub.empty:
                            subject_id = str(gid)
                            meta_row = {
                                "subject_id": subject_id,
                                "graph_id": str(gid),
                                "graph_name": str(graphs[gid].name),
                                "group_value": "",
                            }
                        else:
                            row = sub.iloc[0]
                            subject_id = str(row.get("subject_id", gid))
                            meta_row = {
                                "subject_id": subject_id,
                                "graph_id": str(gid),
                                "graph_name": str(graphs[gid].name),
                                "group_value": row.get("group_value_raw", ""),
                            }
                        subject_ids.append(subject_id)
                        subject_meta_map[str(gid)] = meta_row
                else:
                    subject_ids = [str(gid) for gid in hc_gids_effective]
                    for gid in hc_gids_effective:
                        subject_meta_map[str(gid)] = {
                            "subject_id": str(gid),
                            "graph_id": str(gid),
                            "graph_name": str(graphs[gid].name),
                        }

                start_idx = int(pm_start_index or 0)
                selected_pairs = list(zip(hc_gids_effective, subject_ids))[start_idx:]

                exclude_ids = _parse_token_list(pm_exclude_ids_raw)
                if exclude_ids:
                    selected_pairs = [
                        (gid, sid)
                        for gid, sid in selected_pairs
                        if str(gid) not in exclude_ids and str(sid) not in exclude_ids
                    ]
                    _push_ui_event(f"Excluded ids: {sorted(exclude_ids)}")

                total_subjects = max(1, len(selected_pairs))
                manifest_path = run_dir / "manifest.csv"

                phase_box.success("Run phase: processing HC subjects")
                status_box.caption(f"Будет обработано HC-графов: {len(selected_pairs)}")
                _push_ui_event(f"Run phase started: {len(selected_pairs)} HC subjects")

                _pm_emit_progress(
                    run_dir,
                    {
                        "status": "running",
                        "phase": "run_subjects",
                        "run_dir": str(run_dir),
                        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "total_subjects": int(len(selected_pairs)),
                        "completed_subjects": 0,
                        "current_subject_id": "",
                        "current_attack_kind": "",
                        "current_step": 0,
                        "current_step_total": 0,
                        "phase_progress": 1.0,
                    },
                    write_event=True,
                )

                for pos, (gid, subject_id) in enumerate(selected_pairs, start=1):
                    entry = graphs[gid]
                    subject_meta = dict(
                        subject_meta_map.get(
                            str(gid),
                            {
                                "subject_id": subject_id,
                                "graph_id": str(gid),
                                "graph_name": str(entry.name),
                            },
                        )
                    )

                    overall_frac = 0.20 + 0.75 * (float(pos - 1) / float(total_subjects))
                    overall_bar.progress(
                        overall_frac,
                        text=f"Общий прогресс: {pos - 1}/{total_subjects} HC ({int(round(overall_frac * 100))}%)",
                    )
                    subject_bar.progress(0.0, text=f"Текущий HC: {subject_id} · 0%")
                    phase_box.info(f"Run phase: subject {pos}/{total_subjects}")
                    status_box.caption(f"[{pos}/{total_subjects}] {subject_id} · build graph")
                    files_box.caption(f"Сохранение: {run_dir} · subject dir: {_pm_subject_dir(run_dir, subject_id)}")
                    _push_ui_event(f"Start subject {subject_id} ({pos}/{total_subjects})")

                    _pm_emit_progress(
                        run_dir,
                        {
                            "status": "running",
                            "phase": "subject_build_graph",
                            "run_dir": str(run_dir),
                            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "total_subjects": int(len(selected_pairs)),
                            "completed_subjects": int(pos - 1),
                            "current_subject_id": str(subject_id),
                            "current_attack_kind": "",
                            "current_step": 0,
                            "current_step_total": 0,
                            "subject_position": int(pos),
                            "phase_progress": 0.0,
                        },
                        write_event=True,
                    )

                    if bool(pm_skip_done) and _pm_subject_done(run_dir, subject_id):
                        _pm_append_csv(
                            manifest_path,
                            {
                                "subject_id": str(subject_id),
                                "graph_id": str(gid),
                                "status": "skipped_done",
                                "elapsed_sec": 0.0,
                                "saved_dir": str(_pm_subject_dir(run_dir, subject_id)),
                            },
                        )
                        _push_ui_event(f"Skip already done: {subject_id}")
                        overall_bar.progress(
                            0.20 + 0.75 * (float(pos) / float(total_subjects)),
                            text=f"Общий прогресс: {pos}/{total_subjects} HC",
                        )
                        subject_bar.progress(1.0, text=f"Текущий HC: {subject_id} · skipped")
                        continue

                    t0 = time.perf_counter()
                    stage_name = "build_graph"
                    attack_name = ""
                    graph_stats = {}

                    try:
                        stage_name = "build_graph"
                        _build_t0 = time.perf_counter()

                        _push_ui_event(f"Build graph started: {subject_id} · gid={gid}")

                        G = _run_with_timeout(
                            build_graph_safe,
                            entry.edges,
                            entry.src_col,
                            entry.dst_col,
                            float(min_conf),
                            float(min_weight),
                            str(analysis_mode),
                            timeout_seconds=float(pm_timeout_per_stage or 0),
                        )

                        graph_stats = {
                            "N": int(G.number_of_nodes()),
                            "E": int(G.number_of_edges()),
                            "density": float(nx.density(G)) if G.number_of_nodes() > 1 else 0.0,
                            "build_graph_sec": float(time.perf_counter() - _build_t0),
                        }
                        _push_ui_event(
                            f"Graph built: {subject_id} · "
                            f"N={graph_stats['N']} E={graph_stats['E']} "
                            f"density={graph_stats['density']:.4f} "
                            f"t={graph_stats['build_graph_sec']:.1f}s"
                        )

                        status_box.caption(f"[{pos}/{total_subjects}] {subject_id} · trajectory")
                        stage_name = "trajectory"

                        def _ui_progress(done_attacks: int, total_attacks: int, step_i: int, step_total: int, attack_kind: str, current_value=None) -> None:
                            nonlocal attack_name
                            attack_name = str(attack_kind)

                            attack_frac = 0.0 if total_attacks <= 0 else float(done_attacks) / float(total_attacks)
                            step_frac = 0.0 if step_total <= 0 else float(step_i) / float(step_total)
                            total_frac = min(1.0, attack_frac + step_frac / max(1, total_attacks))

                            subject_bar.progress(
                                total_frac,
                                text=(
                                    f"Текущий HC: {subject_id} · {attack_kind} · "
                                    f"{int(round(total_frac * 100))}%"
                                ),
                            )

                            suffix = f" · value={current_value}" if current_value is not None else ""
                            status_box.caption(
                                f"[{pos}/{total_subjects}] {subject_id} · {attack_kind} · step {step_i}/{step_total}{suffix}"
                            )

                            should_log = (
                                int(step_i) <= 1
                                or int(step_i) >= int(step_total)
                                or (int(step_i) % max(1, int(step_total) // 4) == 0)
                            )

                            _pm_emit_progress(
                                run_dir,
                                {
                                    "status": "running",
                                    "phase": "subject_trajectory",
                                    "run_dir": str(run_dir),
                                    "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                                    "total_subjects": int(len(selected_pairs)),
                                    "completed_subjects": int(pos - 1),
                                    "current_subject_id": str(subject_id),
                                    "current_attack_kind": str(attack_kind),
                                    "current_step": int(step_i),
                                    "current_step_total": int(step_total),
                                    "current_value": None if current_value is None else float(current_value),
                                    "subject_position": int(pos),
                                    "subject_progress": float(total_frac),
                                },
                                write_event=bool(should_log),
                            )

                            if should_log:
                                _push_ui_event(
                                    f"Heartbeat {subject_id}: attack={attack_kind} step={step_i}/{step_total}"
                                )

                            time.sleep(0.01)

                        _pm_stream_subject(
                            run_dir=run_dir,
                            G=G,
                            subject_id=str(subject_id),
                            subject_idx=pos - 1 + start_idx,
                            attack_kinds=[str(x) for x in pm_attack_kinds],
                            metric_list=metric_list,
                            target_vector=target_vector,
                            scales=scales,
                            normalized_families=normalized_families,
                            steps=int(pm_steps),
                            frac=float(pm_frac),
                            seed=int(pm_seed),
                            eff_sources_k=int(pm_effk),
                            compute_heavy_every=int(pm_heavy),
                            compute_curvature=bool(pm_compute_curv),
                            curvature_sample_edges=80,
                            noise_sigma_max=float(pm_sigma),
                            keep_density_from_baseline=bool(pm_keep_density),
                            recompute_modules=bool(pm_recompute_modules),
                            module_resolution=float(pm_module_resolution),
                            removal_mode=str(pm_removal_mode),
                            fast_mode=False,
                            subject_meta=subject_meta,
                            timeout_seconds=float(pm_timeout_per_subject or 0),
                            attack_timeout_seconds=float(pm_timeout_per_stage or 0),
                            progress_cb=_ui_progress,
                            export_subject_excel=bool(pm_export_subject_xlsx),
                            include_subject_trajectory_in_return=False,
                        )

                        elapsed = float(time.perf_counter() - t0)
                        saved_subject_dir = _pm_subject_dir(run_dir, subject_id)

                        _pm_append_csv(
                            manifest_path,
                            {
                                "subject_id": str(subject_id),
                                "graph_id": str(gid),
                                "graph_name": str(getattr(entry, "name", "")),
                                "graph_source": str(getattr(entry, "source", "")),
                                "status": "ok",
                                "stage": "done",
                                "last_attack_kind": str(attack_name),
                                "elapsed_sec": elapsed,
                                "saved_dir": str(saved_subject_dir),
                                "N": graph_stats.get("N", np.nan),
                                "E": graph_stats.get("E", np.nan),
                                "density": graph_stats.get("density", np.nan),
                                "build_graph_sec": graph_stats.get("build_graph_sec", np.nan),
                            },
                        )

                        files_box.caption(f"Сохранено: {saved_subject_dir}")
                        _push_ui_event(f"Done subject {subject_id} in {elapsed:.1f}s")

                        del G
                        gc.collect()

                    except TimeoutError as exc:
                        elapsed = float(time.perf_counter() - t0)
                        _pm_append_csv(
                            manifest_path,
                            {
                                "subject_id": str(subject_id),
                                "graph_id": str(gid),
                                "graph_name": str(getattr(entry, "name", "")),
                                "graph_source": str(getattr(entry, "source", "")),
                                "status": "timeout",
                                "stage": str(stage_name),
                                "last_attack_kind": str(attack_name),
                                "elapsed_sec": elapsed,
                                "error": str(exc),
                                "N": graph_stats.get("N", np.nan),
                                "E": graph_stats.get("E", np.nan),
                                "density": graph_stats.get("density", np.nan),
                                "build_graph_sec": graph_stats.get("build_graph_sec", np.nan),
                            },
                        )
                        _pm_subject_file(run_dir, subject_id, "traceback.txt", ensure_dir=True).write_text(
                            f"TimeoutError: {exc}\n",
                            encoding="utf-8",
                        )
                        _push_ui_event(
                            f"TIMEOUT subject {subject_id}: stage={stage_name} attack={attack_name or '-'} "
                            f"N={graph_stats.get('N', 'na')} E={graph_stats.get('E', 'na')} "
                            f"density={graph_stats.get('density', np.nan):.4f} err={exc}"
                        )
                        _pm_emit_progress(
                            run_dir,
                            {
                                "status": "running",
                                "phase": "subject_timeout",
                                "run_dir": str(run_dir),
                                "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                                "total_subjects": int(len(selected_pairs)),
                                "completed_subjects": int(pos - 1),
                                "current_subject_id": str(subject_id),
                                "current_attack_kind": "",
                                "current_step": 0,
                                "current_step_total": 0,
                                "error": str(exc),
                            },
                            write_event=True,
                        )
                        gc.collect()

                    except Exception as exc:
                        elapsed = float(time.perf_counter() - t0)
                        tb = traceback.format_exc()

                        _pm_append_csv(
                            manifest_path,
                            {
                                "subject_id": str(subject_id),
                                "graph_id": str(gid),
                                "graph_name": str(getattr(entry, "name", "")),
                                "graph_source": str(getattr(entry, "source", "")),
                                "status": "error",
                                "stage": str(stage_name),
                                "last_attack_kind": str(attack_name),
                                "elapsed_sec": elapsed,
                                "error": f"{type(exc).__name__}: {exc}",
                                "N": graph_stats.get("N", np.nan),
                                "E": graph_stats.get("E", np.nan),
                                "density": graph_stats.get("density", np.nan),
                                "build_graph_sec": graph_stats.get("build_graph_sec", np.nan),
                            },
                        )
                        _pm_subject_file(run_dir, subject_id, "traceback.txt", ensure_dir=True).write_text(
                            tb,
                            encoding="utf-8",
                        )
                        _push_ui_event(
                            f"ERROR subject {subject_id}: stage={stage_name} attack={attack_name or '-'} "
                            f"N={graph_stats.get('N', 'na')} E={graph_stats.get('E', 'na')} "
                            f"density={graph_stats.get('density', np.nan):.4f} "
                            f"err={type(exc).__name__}: {exc}"
                        )
                        _pm_emit_progress(
                            run_dir,
                            {
                                "status": "running",
                                "phase": "subject_error",
                                "run_dir": str(run_dir),
                                "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                                "total_subjects": int(len(selected_pairs)),
                                "completed_subjects": int(pos - 1),
                                "current_subject_id": str(subject_id),
                                "current_attack_kind": "",
                                "current_step": 0,
                                "current_step_total": 0,
                                "error": f"{type(exc).__name__}: {exc}",
                            },
                            write_event=True,
                        )
                        gc.collect()

                    overall_frac = 0.20 + 0.75 * (float(pos) / float(total_subjects))
                    overall_bar.progress(
                        overall_frac,
                        text=f"Общий прогресс: {pos}/{total_subjects} HC ({int(round(overall_frac * 100))}%)",
                    )
                    subject_bar.progress(1.0, text=f"Текущий HC: {subject_id} · 100%")

                    _pm_emit_progress(
                        run_dir,
                        {
                            "status": "running",
                            "phase": "subject_done",
                            "run_dir": str(run_dir),
                            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "total_subjects": int(len(selected_pairs)),
                            "completed_subjects": int(pos),
                            "current_subject_id": str(subject_id),
                            "current_attack_kind": "",
                            "current_step": 0,
                            "current_step_total": 0,
                            "subject_position": int(pos),
                        },
                        write_event=True,
                    )

                # ---------------- FINALIZE ----------------
                phase_box.info("Finalize phase: aggregate + export")
                status_box.caption("Финализирую aggregate-таблицы и optional export")
                _push_ui_event("Finalize aggregate results")
                overall_bar.progress(0.96, text="Общий прогресс: finalize 96%")
                subject_bar.progress(1.0, text="Текущий HC: завершение")

                _pm_emit_progress(
                    run_dir,
                    {
                        "status": "running",
                        "phase": "finalize",
                        "run_dir": str(run_dir),
                        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "total_subjects": int(len(selected_pairs)),
                        "completed_subjects": int(len(selected_pairs)),
                        "current_subject_id": "",
                        "current_attack_kind": "",
                        "current_step": 0,
                        "current_step_total": 0,
                    },
                    write_event=True,
                )

                _pm_finalize_scalar_aggregates(run_dir, metrics=metric_list)
                pm_res = _pm_load_run_result(
                    run_dir,
                    include_trajectory=not bool(pm_strict_streaming),
                )

                if bool(pm_export_run_xlsx):
                    try:
                        export_phenotype_match_excel(
                            _pm_load_run_result(run_dir, include_trajectory=True),
                            run_dir / "aggregate" / "run_bundle.xlsx",
                        )
                        _push_ui_event("run_bundle.xlsx exported")
                    except Exception as exc:
                        logger.warning("Failed to export aggregate phenotype workbook: %s", exc)
                        _push_ui_event(f"run_bundle.xlsx export failed: {type(exc).__name__}: {exc}")

                overall_bar.progress(1.0, text="Общий прогресс: 100%")
                subject_bar.progress(1.0, text="Текущий HC: done")
                phase_box.success("Run finished")
                status_box.caption("Готово")

                _pm_emit_progress(
                    run_dir,
                    {
                        "status": "done",
                        "phase": "done",
                        "run_dir": str(run_dir),
                        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "total_subjects": int(len(selected_pairs)),
                        "completed_subjects": int(len(selected_pairs)),
                    },
                    write_event=True,
                )

                st.session_state["last_degmatch_result"] = pm_res
                st.session_state["last_degmatch_run_dir"] = str(run_dir)
                st.success(f"Готово. Результаты сохранены в: {run_dir}")
                st.info(f"Абсолютный путь записи: {run_dir}")

    with pm_col2:
        pm_res = st.session_state.get("last_degmatch_result")
        last_saved_run = str(st.session_state.get("last_degmatch_run_dir", "") or "").strip()
        if last_saved_run:
            st.caption(f"Последний сохранённый run: {last_saved_run}")
            run_path = Path(last_saved_run)
            if run_path.exists():
                agg_dir = run_path / "aggregate"
                per_dir = run_path / "per_subject"
                st.caption(
                    f"Папка существует. aggregate={agg_dir.exists()} · per_subject={per_dir.exists()}"
                )
        if pm_res:
            st.markdown("### Winner attacks per HC")
            winners_df = pm_res.get("winner_results", pd.DataFrame())
            subject_df = pm_res.get("subject_results", pd.DataFrame())
            traj_df = pm_res.get("trajectory_results", pd.DataFrame())
            summary_pack = build_paper_ready_summary(pm_res)
            summary_attack_df = summary_pack["summary_attack"]
            summary_winners_df = summary_pack["summary_winners"]
            target_vector_df = summary_pack["target_vector"]
            scales_df = summary_pack["scales"]
            stats_overall_df = summary_pack["stats_overall"]
            stats_pairwise_df = summary_pack["stats_pairwise"]
            stats_winner_df = summary_pack["stats_winners"]
            scalar_subject_df = summary_pack["scalar_subject_results"]
            scalar_winners_df = summary_pack["scalar_winners"]
            scalar_summary_df = summary_pack["scalar_summary"]
            scalar_inference_df = summary_pack.get("scalar_inference", pd.DataFrame())
            family_inference_df = summary_pack.get("family_inference", pd.DataFrame())
            claim_readiness_df = summary_pack.get("core_claim_readiness", pd.DataFrame())

            if isinstance(summary_winners_df, pd.DataFrame) and not summary_winners_df.empty:
                st.markdown("#### Winner summary")
                st.dataframe(summary_winners_df, width="stretch")

            if isinstance(summary_attack_df, pd.DataFrame) and not summary_attack_df.empty:
                st.markdown("#### Attack summary")
                st.dataframe(summary_attack_df, width="stretch")

            if isinstance(scalar_summary_df, pd.DataFrame) and not scalar_summary_df.empty:
                st.markdown("#### Scalar metric-wise summary")
                st.dataframe(scalar_summary_df, width="stretch")

            if isinstance(scalar_winners_df, pd.DataFrame) and not scalar_winners_df.empty:
                with st.expander("Scalar winners (best attack per subject × metric)"):
                    st.dataframe(scalar_winners_df, width="stretch")

            if isinstance(scalar_subject_df, pd.DataFrame) and not scalar_subject_df.empty:
                with st.expander("Scalar subject results"):
                    st.dataframe(scalar_subject_df, width="stretch")

            if isinstance(stats_overall_df, pd.DataFrame) and not stats_overall_df.empty:
                st.markdown("#### Overall stats")
                st.dataframe(stats_overall_df, width="stretch")

            if isinstance(stats_pairwise_df, pd.DataFrame) and not stats_pairwise_df.empty:
                with st.expander("Pairwise comparisons"):
                    st.dataframe(stats_pairwise_df, width="stretch")

            if isinstance(scalar_inference_df, pd.DataFrame) and not scalar_inference_df.empty:
                with st.expander("Scalar inference (FDR + effect size)"):
                    st.dataframe(scalar_inference_df, width="stretch")

            if isinstance(family_inference_df, pd.DataFrame) and not family_inference_df.empty:
                with st.expander("Family inference (FDR + effect size)"):
                    st.dataframe(family_inference_df, width="stretch")

            if isinstance(stats_winner_df, pd.DataFrame) and not stats_winner_df.empty:
                with st.expander("Winner-count stats"):
                    st.dataframe(stats_winner_df, width="stretch")

            if isinstance(winners_df, pd.DataFrame) and not winners_df.empty:
                with st.expander("Winner rows"):
                    st.dataframe(winners_df, width="stretch")

            if isinstance(subject_df, pd.DataFrame) and not subject_df.empty:
                with st.expander("Subject results"):
                    st.dataframe(subject_df, width="stretch")

            if isinstance(traj_df, pd.DataFrame) and not traj_df.empty:
                with st.expander("Trajectory results"):
                    st.dataframe(traj_df.head(500), width="stretch")

            export_col1, export_col2, export_col3 = st.columns(3)

            with export_col1:
                if isinstance(summary_winners_df, pd.DataFrame) and not summary_winners_df.empty:
                    st.download_button(
                        "Скачать summary_winners.csv",
                        summary_winners_df.to_csv(index=False).encode("utf-8"),
                        file_name="phenotype_match_summary_winners.csv",
                        mime="text/csv",
                        width="stretch",
                    )

            with export_col2:
                if isinstance(summary_attack_df, pd.DataFrame) and not summary_attack_df.empty:
                    st.download_button(
                        "Скачать summary_attack.csv",
                        summary_attack_df.to_csv(index=False).encode("utf-8"),
                        file_name="phenotype_match_summary_attack.csv",
                        mime="text/csv",
                        width="stretch",
                    )

            with export_col3:
                import io

                bio = io.BytesIO()
                with pd.ExcelWriter(bio, engine="openpyxl") as writer:
                    (winners_df if isinstance(winners_df, pd.DataFrame) else pd.DataFrame()).to_excel(
                        writer, index=False, sheet_name="winners"
                    )
                    (subject_df if isinstance(subject_df, pd.DataFrame) else pd.DataFrame()).to_excel(
                        writer, index=False, sheet_name="subject_results"
                    )
                    (traj_df if isinstance(traj_df, pd.DataFrame) else pd.DataFrame()).to_excel(
                        writer, index=False, sheet_name="trajectories"
                    )
                    (summary_attack_df if isinstance(summary_attack_df, pd.DataFrame) else pd.DataFrame()).to_excel(
                        writer, index=False, sheet_name="summary_attack"
                    )
                    (summary_winners_df if isinstance(summary_winners_df, pd.DataFrame) else pd.DataFrame()).to_excel(
                        writer, index=False, sheet_name="summary_winners"
                    )
                    (target_vector_df if isinstance(target_vector_df, pd.DataFrame) else pd.DataFrame()).to_excel(
                        writer, index=False, sheet_name="target_vector"
                    )
                    (scales_df if isinstance(scales_df, pd.DataFrame) else pd.DataFrame()).to_excel(
                        writer, index=False, sheet_name="scales"
                    )
                    (stats_overall_df if isinstance(stats_overall_df, pd.DataFrame) else pd.DataFrame()).to_excel(
                        writer, index=False, sheet_name="stats_overall"
                    )
                    (stats_pairwise_df if isinstance(stats_pairwise_df, pd.DataFrame) else pd.DataFrame()).to_excel(
                        writer, index=False, sheet_name="stats_pairwise"
                    )
                    (stats_winner_df if isinstance(stats_winner_df, pd.DataFrame) else pd.DataFrame()).to_excel(
                        writer, index=False, sheet_name="stats_winners"
                    )
                    (scalar_subject_df if isinstance(scalar_subject_df, pd.DataFrame) else pd.DataFrame()).to_excel(
                        writer, index=False, sheet_name="scalar_subject"
                    )
                    (scalar_winners_df if isinstance(scalar_winners_df, pd.DataFrame) else pd.DataFrame()).to_excel(
                        writer, index=False, sheet_name="scalar_winners"
                    )
                    (scalar_summary_df if isinstance(scalar_summary_df, pd.DataFrame) else pd.DataFrame()).to_excel(
                        writer, index=False, sheet_name="scalar_summary"
                    )
                    (scalar_inference_df if isinstance(scalar_inference_df, pd.DataFrame) else pd.DataFrame()).to_excel(
                        writer, index=False, sheet_name="scalar_inference"
                    )
                    (family_inference_df if isinstance(family_inference_df, pd.DataFrame) else pd.DataFrame()).to_excel(
                        writer, index=False, sheet_name="family_inference"
                    )
                    (claim_readiness_df if isinstance(claim_readiness_df, pd.DataFrame) else pd.DataFrame()).to_excel(
                        writer, index=False, sheet_name="claim_readiness"
                    )
                bio.seek(0)

                st.download_button(
                    "Скачать один XLSX",
                    data=bio.getvalue(),
                    file_name="phenotype_match_full.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    width="stretch",
                )

            with st.expander("Target vector / scales"):
                tv_col, sc_col = st.columns(2)
                with tv_col:
                    st.markdown("**Target vector (SZ median)**")
                    st.dataframe(target_vector_df, width="stretch")
                with sc_col:
                    st.markdown("**Scales**")
                    st.dataframe(scales_df, width="stretch")

            if isinstance(traj_df, pd.DataFrame) and not traj_df.empty and {"attack_kind", "step", "distance_to_target"}.issubset(traj_df.columns):
                plot_df = traj_df.copy()
                plot_df["step"] = pd.to_numeric(plot_df["step"], errors="coerce")
                plot_df["distance_to_target"] = pd.to_numeric(plot_df["distance_to_target"], errors="coerce")
                plot_df = plot_df.dropna(subset=["step", "distance_to_target"])
                if not plot_df.empty:
                    agg = (
                        plot_df.groupby(["attack_kind", "step"], dropna=False)["distance_to_target"]
                        .median()
                        .reset_index()
                    )
                    fig_pm = px.line(
                        agg,
                        x="step",
                        y="distance_to_target",
                        color="attack_kind",
                        title="Median distance to SZ target vs step",
                    )
                    fig_pm.update_layout(template="plotly_dark")
                    fig_pm = _apply_plot_defaults(fig_pm, height=520)
                    st.plotly_chart(fig_pm, width="stretch", key="plot_pm_distance")
        else:
            st.info("Запусти HC→SZ matching слева.")


def render_sz_ml_ready_tab(
    *,
    active_entry: GraphEntry,
    seed_val: int,
    min_conf: float,
    min_weight: float,
    analysis_mode: str,
) -> None:
    """SZ-only ML-ready export with streaming CSV append."""
    st.header("🧬 SZ → ML")
    st.caption(
        "Считает только фиксированный ML-набор метрик для SZ-графов и потоково пишет одну wide-таблицу CSV."
    )

    graphs = st.session_state["graphs"]
    active_gid = st.session_state.get("active_graph_id")
    sz_guess = _guess_sz_like_graph_ids(graphs, active_gid)

    metric_options = list(DEGRADATION_METRIC_OPTIONS)
    metric_list = [m for m in SZ_ML_METRICS if m in metric_options]
    if not metric_list:
        metric_list = ["l2_lcc", "H_rw", "fragility_H", "mod"]

    left_col, right_col = st.columns([1, 2])

    with left_col:
        szml_meta_file = st.file_uploader(
            "Metadata file (CSV/XLSX, optional)",
            type=["csv", "tsv", "xlsx", "xls"],
            key="szml_meta_file",
        )

        szml_sz_gids = st.multiselect(
            "SZ графы (ручной режим / fallback)",
            options=list(graphs.keys()),
            default=sz_guess,
            format_func=lambda gid: f"{graphs[gid].name} ({graphs[gid].source})",
            key="szml_sz_gids",
        )

        with st.expander("Metadata matching"):
            szml_meta_id_col = st.text_input(
                "Metadata ID column (optional)",
                value="",
                key="szml_meta_id_col",
            )
            szml_meta_group_col = st.text_input(
                "Metadata group column (optional)",
                value="",
                key="szml_meta_group_col",
            )
            szml_meta_sz_values = st.text_input(
                "SZ values",
                value="1,sz,schizophrenia,patient,case,true",
                key="szml_meta_sz_values",
            )

        szml_out_dir = st.text_input(
            "Папка вывода",
            value=str(st.session_state.get("szml_out_dir", "./phenotype_runs")),
            key="szml_out_dir",
        )
        szml_run_label = st.text_input(
            "Название run",
            value=str(st.session_state.get("szml_run_label", f"sz_ml_{time.strftime('%Y%m%d_%H%M%S')}")),
            key="szml_run_label",
        )
        szml_timeout_per_graph = st.number_input(
            "Скип по времени на SZ-граф (сек, 0=без лимита)",
            min_value=0,
            value=int(st.session_state.get("szml_timeout_per_graph", 600)),
            step=10,
            key="szml_timeout_per_graph",
        )
        szml_skip_existing = st.checkbox(
            "Пропускать уже посчитанные graph_id в CSV",
            value=True,
            key="szml_skip_existing",
        )
        szml_compute_curv = st.checkbox(
            "Считать curvature в baseline",
            value=_needs_curvature_for_metrics(metric_list),
            key="szml_compute_curv",
        )
        szml_effk = st.slider(
            "Efficiency k",
            8,
            256,
            32,
            key="szml_effk",
        )

        st.markdown("#### ML-метрики")
        st.code("\n".join(metric_list), language="text")

        meta_df = None
        meta_info = {}
        match = None
        if szml_meta_file is not None:
            try:
                meta_df, meta_info = _prepare_uploaded_metadata(
                    szml_meta_file,
                    id_col=str(szml_meta_id_col or ""),
                    group_col=str(szml_meta_group_col or ""),
                )
                match = _match_workspace_graphs_to_metadata(
                    graphs,
                    meta_df,
                    group_col=str(meta_info["group_col"]),
                    healthy_values="0,healthy,control,hc,false",
                    sz_values=str(szml_meta_sz_values or "1,sz,schizophrenia,patient,case,true"),
                )
                st.success(
                    f"Matched={len(match['matched_df'])}, "
                    f"SZ={len(match['sz_gids'])}, "
                    f"unmatched={len(match['unmatched_gids'])}"
                )
            except Exception as exc:
                st.error(f"Metadata parse/match error: {type(exc).__name__}: {exc}")

        use_metadata_mode = match is not None and len(match.get("sz_gids", [])) > 0
        preview_sz_n = len(match.get("sz_gids", [])) if use_metadata_mode else len(szml_sz_gids)

        if st.button("🚀 BUILD SZ ML TABLE", type="primary", width="stretch", key="szml_run"):
            if use_metadata_mode:
                sz_gids_effective = list(match["sz_gids"])
                meta_map_df = match.get("sz_meta_df", pd.DataFrame()).copy()
            else:
                sz_gids_effective = list(szml_sz_gids)
                meta_map_df = pd.DataFrame()

            if not sz_gids_effective:
                st.error("Не найдено SZ-графов: ни по metadata, ни вручную.")
                st.stop()

            out_root = _pm_resolve_out_dir(szml_out_dir)
            out_root.mkdir(parents=True, exist_ok=True)
            run_name = _pm_safe_stem(str(szml_run_label), f"sz_ml_{time.strftime('%Y%m%d_%H%M%S')}")
            run_dir = out_root / run_name
            agg_dir = run_dir / "aggregate"
            agg_dir.mkdir(parents=True, exist_ok=True)
            _pm_write_run_location_note(run_dir)
            st.session_state["last_szml_run_dir"] = str(run_dir)

            ml_path = agg_dir / "sz_ml_ready.csv"
            log_path = agg_dir / "sz_ml_progress_log.csv"

            _pm_save_run_inputs(
                run_dir,
                config={
                    "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "run_dir": str(run_dir),
                    "mode": "sz_ml_ready_streaming",
                    "metrics": metric_list,
                    "compute_curvature": bool(szml_compute_curv),
                    "timeout_per_graph": float(szml_timeout_per_graph or 0),
                    "graph_regime": "full_weighted_signed_hybrid",
                    "weight_policy": str(getattr(settings, "WEIGHT_POLICY", "signed_split")),
                    "weight_shift": float(getattr(settings, "WEIGHT_SHIFT", 0.0)),
                    "eff_k": int(szml_effk),
                },
                metadata_upload=szml_meta_file,
                sz_upload=None,
                hc_baseline_upload=None,
            )

            already_done: set[str] = set()
            if bool(szml_skip_existing) and ml_path.exists() and ml_path.stat().st_size > 0:
                try:
                    prev_df = pd.read_csv(ml_path, usecols=["graph_id"])
                    already_done = set(prev_df["graph_id"].astype(str).tolist())
                except Exception:
                    already_done = set()

            progress_bar = st.progress(0.0, text="SZ→ML: старт")
            event_box = st.empty()
            preview_box = st.empty()
            ui_events: list[str] = []
            preview_rows: list[dict] = []

            def _push_ui_event(msg: str) -> None:
                stamp = time.strftime("%H:%M:%S")
                ui_events.append(f"[{stamp}] {msg}")
                event_box.code("\n".join(ui_events[-18:]))

            total = len(sz_gids_effective)
            done = 0
            ok_n = 0
            err_n = 0
            skip_n = 0

            def _meta_row_for_gid(gid: str) -> dict:
                if meta_map_df is None or meta_map_df.empty or "graph_id" not in meta_map_df.columns:
                    return {}
                hit = meta_map_df.loc[meta_map_df["graph_id"].astype(str) == str(gid)]
                if hit.empty:
                    return {}
                row = hit.iloc[0].to_dict()
                return {
                    "subject_id": row.get("subject_id", gid),
                    "group_value_raw": row.get("group_value_raw", "sz"),
                    "group_value_norm": row.get("group_value_norm", "sz"),
                }

            _push_ui_event(f"Run dir: {run_dir}")
            _push_ui_event(f"SZ graphs selected: {total}")
            _push_ui_event(f"Output CSV: {ml_path.name}")

            for pos, gid in enumerate(sz_gids_effective, start=1):
                gid = str(gid)
                entry = graphs[gid]
                progress_bar.progress(done / max(1, total), text=f"SZ→ML: {done}/{total} · {gid}")

                if bool(szml_skip_existing) and gid in already_done:
                    skip_n += 1
                    done += 1
                    _push_ui_event(f"{pos}/{total} SKIP existing · {gid}")
                    _pm_append_csv(
                        log_path,
                        {
                            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "graph_id": gid,
                            "subject_id": gid,
                            "status": "skip_existing",
                            "error": "",
                            "elapsed_sec": np.nan,
                        },
                    )
                    continue

                t0 = time.perf_counter()
                base_info = {
                    "graph_id": gid,
                    "graph_name": str(getattr(entry, "name", "")),
                    "graph_source": str(getattr(entry, "source", "")),
                    "label": 1,
                    "status": "unknown",
                    "error": "",
                    "elapsed_sec": np.nan,
                    "N": np.nan,
                    "E": np.nan,
                    "density": np.nan,
                }
                base_info.update(_meta_row_for_gid(gid))
                if "subject_id" not in base_info:
                    base_info["subject_id"] = gid
                if "group_value_raw" not in base_info:
                    base_info["group_value_raw"] = "sz"
                if "group_value_norm" not in base_info:
                    base_info["group_value_norm"] = "sz"

                try:
                    row_df = _compute_metrics_df_for_graph_ids(
                        [gid],
                        graphs=graphs,
                        min_conf=float(min_conf),
                        min_weight=float(min_weight),
                        analysis_mode=str(analysis_mode),
                        eff_k=int(szml_effk),
                        seed=int(seed_val),
                        compute_curvature=bool(szml_compute_curv),
                        metric_names=metric_list,
                        progress_cb=None,
                        timeout_seconds_per_graph=float(szml_timeout_per_graph or 0),
                    )

                    if row_df.empty:
                        row = dict(base_info)
                        row["status"] = "error"
                        row["error"] = "empty metrics row"
                    else:
                        got = row_df.iloc[0].to_dict()
                        row = dict(base_info)
                        row.update(got)
                        row["subject_id"] = base_info["subject_id"]
                        row["group_value_raw"] = base_info["group_value_raw"]
                        row["group_value_norm"] = base_info["group_value_norm"]
                        row["label"] = 1

                    row["elapsed_sec"] = float(row.get("elapsed_sec", np.nan))

                except Exception as exc:
                    row = dict(base_info)
                    row["status"] = "error"
                    row["error"] = f"{type(exc).__name__}: {exc}"
                    row["elapsed_sec"] = float(time.perf_counter() - t0)
                    for m in metric_list:
                        row[m] = np.nan

                for m in metric_list:
                    if m not in row:
                        row[m] = np.nan

                preferred_cols = [
                    "subject_id",
                    "graph_id",
                    "graph_name",
                    "graph_source",
                    "group_value_raw",
                    "group_value_norm",
                    "label",
                    "status",
                    "error",
                    "elapsed_sec",
                    "N",
                    "E",
                    "density",
                ]
                metric_cols = list(metric_list)
                ordered = preferred_cols + [c for c in metric_cols if c not in preferred_cols]
                other_cols = [c for c in row.keys() if c not in ordered]
                row_df_out = pd.DataFrame([{k: row.get(k, np.nan) for k in ordered + other_cols}])

                _pm_append_csv(ml_path, row_df_out)
                _pm_append_csv(
                    log_path,
                    {
                        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "graph_id": row.get("graph_id", gid),
                        "subject_id": row.get("subject_id", gid),
                        "status": row.get("status", "unknown"),
                        "error": row.get("error", ""),
                        "elapsed_sec": row.get("elapsed_sec", np.nan),
                    },
                )

                preview_rows.append(row_df_out.iloc[0].to_dict())
                preview_rows = preview_rows[-20:]
                preview_box.dataframe(pd.DataFrame(preview_rows), use_container_width=True)

                status_now = str(row.get("status", "unknown"))
                if status_now == "ok":
                    ok_n += 1
                else:
                    err_n += 1

                done += 1
                progress_bar.progress(
                    done / max(1, total),
                    text=f"SZ→ML: {done}/{total} · ok={ok_n} · err={err_n} · skip={skip_n}",
                )
                _push_ui_event(
                    f"{pos}/{total} {status_now.upper()} · {gid} · "
                    f"N={row.get('N', 'na')} E={row.get('E', 'na')} "
                    f"t={float(pd.to_numeric(row.get('elapsed_sec', np.nan), errors='coerce')):.1f}s"
                )

            st.success(
                f"Готово. CSV обновлён: {ml_path}. "
                f"OK={ok_n}, error/timeout={err_n}, skipped={skip_n}."
            )

            if ml_path.exists() and ml_path.stat().st_size > 0:
                try:
                    final_df = pd.read_csv(ml_path)
                    st.markdown("### Текущий CSV")
                    st.dataframe(final_df.tail(50), use_container_width=True)
                except Exception as exc:
                    st.warning(f"CSV записан, но перечитать предпросмотр не удалось: {type(exc).__name__}: {exc}")

    with right_col:
        st.markdown("### Что делает вкладка")
        st.markdown(
            "\n".join([
                f"- SZ графов к запуску: **{preview_sz_n}**",
                f"- Метрик: **{len(metric_list)}**",
                "- Только SZ",
                "- Только ML-ready wide CSV",
                "- Потоковая запись: по одному графу",
                "- Resume через skip existing graph_id",
                "- Без HC→SZ matching pipeline",
                "- Без trajectory / winner / distance",
            ])
        )

        st.markdown("### Основной файл")
        st.code("aggregate/sz_ml_ready.csv", language="text")

        st.markdown("### Структура строки")
        st.code(
            "subject_id, graph_id, graph_name, graph_source, "
            "group_value_raw, group_value_norm, label, status, error, elapsed_sec, "
            "N, E, density, "
            + ", ".join(metric_list),
            language="text",
        )


def render_attack_lab(G_view: nx.Graph | None, active_entry: GraphEntry, seed_val: int, src_col: str, dst_col: str, min_conf: float, min_weight: float, analysis_mode: str, save_experiment_callback) -> None:
    """Render the Attack Lab tab."""
    if G_view is None:
        return

    # Совместимость по сигнатуре колбэка сохранения эксперимента.
    # В app.py сейчас: save_experiment_to_state(name, gid, kind, params, df_hist)
    # В старых/других местах мог быть вариант с keyword graph_id=...
    def _save_experiment(*, name: str, graph_id: str, kind: str, params: dict, df_hist):
        try:
            return save_experiment_callback(
                name=name,
                graph_id=graph_id,
                kind=kind,
                params=params,
                df_hist=df_hist,
            )
        except TypeError:
            # fallback на (name, gid, ...)
            return save_experiment_callback(name, graph_id, kind, params, df_hist)

    st.header("💥 Attack Lab (node + edge + weak)")

    # --------------------------
    # SINGLE RUN
    # --------------------------
    st.subheader("Single run")
    family = st.radio(
        "Тип атаки",
        [
            "Node (узлы)",
            "Edge (рёбра: слабые/сильные)",
            "Mix/Entropy (Hrish)",
            "Degradation (HC→SZ models)",
        ],
        horizontal=True,
    )

    col_setup, _ = st.columns([1, 2])

    with col_setup:
        with st.container(border=True):
            st.markdown("### Параметры")

            frac = st.slider("Доля удаления", 0.05, 0.95, 0.5, 0.05)
            steps = st.slider("Шаги", 5, 150, 30)
            seed_run = st.number_input("Seed", value=int(seed_val), step=1)

            with st.expander("Дополнительно"):
                eff_k = st.slider("Efficiency samples (k)", 8, 256, 32)
                heavy_freq = st.slider("Тяжёлые метрики каждые N шагов", 1, 10, 2)
                fast_mode = st.checkbox("⚡ Fast Mode (approx)", value=True, help="Сильно ускоряет расчет за счет снижения точности на промежуточных шагах.")

                tag = st.text_input("Тег", "")

            if family.startswith("Node"):
                attack_ui = st.selectbox(
                    "Стратегия (узлы)",
                    [
                        "random",
                        "degree (Hubs)",
                        "betweenness (Bridges)",
                        "kcore (Deep Core)",
                        "richclub_top (Top Strength)",
                        "low_degree (Weak nodes)",
                        "weak_strength (Weak strength)",
                    ],
                )
                kind_map = {
                    "random": "random",
                    "degree (Hubs)": "degree",
                    "betweenness (Bridges)": "betweenness",
                    "kcore (Deep Core)": "kcore",
                    "richclub_top (Top Strength)": "richclub_top",
                    "low_degree (Weak nodes)": "low_degree",
                    "weak_strength (Weak strength)": "weak_strength",
                }
                kind = kind_map.get(attack_ui, "random")

            elif family.startswith("Edge"):
                attack_ui = st.selectbox(
                    "Стратегия (рёбра)",
                    [
                        "weak_edges_by_weight",
                        "weak_edges_by_confidence",
                        "strong_edges_by_weight",
                        "strong_edges_by_confidence",
                        "ricci_most_negative (κ min)",
                        "ricci_most_positive (κ max)",
                        "ricci_abs_max (|κ| max)",
                        "flux_high_rw",
                        "flux_high_evo",
                        "flux_high_rw_x_neg_ricci",
                    ],
                    help=help_icon("Weak edges")
                )
                kind = str(attack_ui).split(" ")[0]

            elif family.startswith("Degradation"):
                kind = st.selectbox(
                    "Модель деградации",
                    DEGRADATION_KIND_OPTIONS,
                    index=0,
                    help="Новые модели деградации для HC→SZ phenotype matching."
                )

                compute_curv_deg = st.checkbox(
                    "Считать curvature в trajectory",
                    value=False,
                    help="Включай только если реально нужны kappa_* в distance."
                )

                if kind == "weight_noise":
                    noise_sigma_max = st.slider("sigma_max", 0.01, 2.0, 0.5, 0.01)
                    keep_density_from_baseline = st.checkbox(
                        "Держать damage через baseline edge-count",
                        value=True,
                        help="После noise сохраняем top-k рёбер, чтобы damage был контролируем."
                    )
                    recompute_modules = False
                    module_resolution = 1.0
                    removal_mode = "random"
                else:
                    noise_sigma_max = 0.5
                    keep_density_from_baseline = True
                    if kind in {"inter_module_removal", "intra_module_removal"}:
                        removal_mode = st.selectbox(
                            "Порядок удаления",
                            ["random", "weak_weight", "strong_weight"],
                            index=0,
                        )
                        recompute_modules = st.checkbox(
                            "Пересчитывать модули на каждом шаге",
                            value=False,
                            help="По умолчанию лучше OFF: baseline-модули интерпретируемее."
                        )
                        module_resolution = st.slider("Louvain resolution", 0.2, 3.0, 1.0, 0.1)
                    else:
                        removal_mode = "random"
                        recompute_modules = False
                        module_resolution = 1.0

            else:
                kind = st.selectbox(
                    "Режим Hrish",
                    [
                        "hrish_mix",
                        "mix_degree_preserving",
                        "mix_weightconf_preserving",
                    ],
                    help="hrish_mix = rewire (degree-preserving) + replace из нулевой модели.",
                )
                replace_from = st.selectbox("Replace source", ["ER", "CFG"], index=0)
                alpha_rewire = st.slider("alpha (rewire)", 0.0, 1.0, 0.6, 0.05)
                beta_replace = st.slider("beta (replace)", 0.0, 1.0, 0.4, 0.05)
                swaps_per_edge = st.slider("swaps_per_edge", 0.0, 3.0, 0.5, 0.1)
                st.caption("Ось X здесь: mix_frac (0..1), а не removed_frac.")

            if st.button("🚀 RUN", type="primary", width="stretch"):
                if family.startswith("Mix/Entropy"):
                    with st.spinner(f"Mix attack: {kind}"):
                        bar = st.progress(0.0)
                        msg = st.empty()
                        st.caption("Промежуточные шаги")
                        preview_holder, row_cb = _live_history_preview()

                        def _cb(i, total, x=None):
                            frac_ = 0.0 if total <= 0 else min(1.0, max(0.0, i / total))
                            bar.progress(frac_)
                            if x is not None:
                                msg.caption(f"mix: {i}/{total}  mix_frac={x:.3f}")

                        df_hist, aux = run_mix_attack(
                            G_view,
                            kind=str(kind),
                            steps=int(steps),
                            seed=int(seed_run),
                            eff_sources_k=int(eff_k),
                            heavy_every=int(heavy_freq),
                            alpha_rewire=float(alpha_rewire),
                            beta_replace=float(beta_replace),
                            swaps_per_edge=float(swaps_per_edge),
                            replace_from=str(replace_from),
                            progress_cb=_cb,
                            row_cb=row_cb,
                            fast_mode=fast_mode,
                        )
                        bar.empty(); msg.empty()
                        preview_holder.empty()
                        df_hist = _forward_fill_heavy(df_hist)
                        phase_info = classify_phase_transition(
                            df_hist.rename(columns={"mix_frac": "removed_frac"})
                        )

                        label = f"{active_entry.name} | mix:{kind} | seed={seed_run}"
                        if tag:
                            label += f" [{tag}]"

                        _save_experiment(
                            name=label,
                            graph_id=active_entry.id,
                            kind=str(kind),
                            params={
                                "attack_family": "mix",
                                "steps": int(steps),
                                "seed": int(seed_run),
                                "phase": phase_info,
                                "eff_k": int(eff_k),
                                "heavy_every": int(heavy_freq),
                                **aux,
                            },
                            df_hist=df_hist,
                        )
                    st.success("Готово.")
                    st.rerun()

                elif family.startswith("Node"):
                    with st.spinner(f"Node attack: {kind}"):
                        # TODO: stabilize the progress indicator updates in Streamlit.
                        bar = st.progress(0.0)
                        msg = st.empty()
                        st.caption("Промежуточные шаги")
                        preview_holder, row_cb = _live_history_preview()

                        def _cb(i, total, k=None):
                            frac_ = 0.0 if total <= 0 else min(1.0, max(0.0, i / total))
                            bar.progress(frac_)
                            if k is not None:
                                msg.caption(f"node attack: {i}/{total}  target_k={k}")

                        df_hist, aux = run_attack(
                            G_view, kind, float(frac), int(steps), int(seed_run), int(eff_k),
                            rc_frac=0.1, compute_heavy_every=int(heavy_freq),
                            fast_mode=bool(fast_mode),
                            progress_cb=_cb,
                            row_cb=row_cb,
                        )
                        bar.empty(); msg.empty()
                        preview_holder.empty()
                        df_hist = _forward_fill_heavy(df_hist)
                        removed_order = _extract_removed_order(aux) or _fallback_removal_order(G_view, kind, int(seed_run))
                        phase_info = classify_phase_transition(df_hist)

                        label = f"{active_entry.name} | node:{kind} | seed={seed_run}"
                        if tag:
                            label += f" [{tag}]"

                        _save_experiment(
                            name=label,
                            graph_id=active_entry.id,
                            kind=kind,
                            params={
                                "attack_family": "node",
                                "frac": float(frac),
                                "steps": int(steps),
                                "seed": int(seed_run),
                                "phase": phase_info,
                                "compute_heavy_every": int(heavy_freq),
                                "eff_k": int(eff_k),
                                "removed_order": removed_order,
                                "mode": "src_run_attack_or_fallback",
                            },
                            df_hist=df_hist
                        )
                    st.success("Готово.")
                    st.rerun()

                elif family.startswith("Degradation"):
                    with st.spinner(f"Degradation: {kind}"):
                        bar = st.progress(0.0)
                        msg = st.empty()
                        st.caption("Промежуточные шаги")
                        preview_holder, row_cb = _live_history_preview()

                        def _cb(i, total, x=None):
                            frac_ = 0.0 if total <= 0 else min(1.0, max(0.0, i / total))
                            bar.progress(frac_)
                            if x is not None:
                                msg.caption(f"degradation: {i}/{total}  x={x}")

                        df_hist, aux = run_degradation_trajectory(
                            G_view,
                            kind=str(kind),
                            steps=int(steps),
                            frac=float(frac),
                            seed=int(seed_run),
                            eff_sources_k=int(eff_k),
                            compute_heavy_every=int(heavy_freq),
                            compute_curvature=bool(compute_curv_deg),
                            curvature_sample_edges=80,
                            noise_sigma_max=float(noise_sigma_max),
                            keep_density_from_baseline=bool(keep_density_from_baseline),
                            recompute_modules=bool(recompute_modules),
                            module_resolution=float(module_resolution),
                            removal_mode=str(removal_mode),
                            progress_cb=_cb,
                            row_cb=row_cb,
                            fast_mode=bool(fast_mode),
                        )
                        bar.empty(); msg.empty()
                        preview_holder.empty()

                        df_hist = _forward_fill_heavy(df_hist)
                        phase_info = classify_phase_transition(
                            df_hist.rename(columns={"damage_frac": "removed_frac"})
                        )

                        label = f"{active_entry.name} | degradation:{kind} | seed={seed_run}"
                        if tag:
                            label += f" [{tag}]"

                        _save_experiment(
                            name=label,
                            graph_id=active_entry.id,
                            kind=str(kind),
                            params={
                                "attack_family": "degradation",
                                "steps": int(steps),
                                "seed": int(seed_run),
                                "phase": phase_info,
                                "eff_k": int(eff_k),
                                "heavy_every": int(heavy_freq),
                                "noise_sigma_max": float(noise_sigma_max),
                                "keep_density_from_baseline": bool(keep_density_from_baseline),
                                "recompute_modules": bool(recompute_modules),
                                "module_resolution": float(module_resolution),
                                "removal_mode": str(removal_mode),
                                **aux,
                            },
                            df_hist=df_hist,
                        )
                    st.success("Готово.")
                    st.rerun()

                else:
                    with st.spinner(f"Edge attack: {kind}"):
                        bar = st.progress(0.0)
                        msg = st.empty()
                        st.caption("Промежуточные шаги")
                        preview_holder, row_cb = _live_history_preview()

                        def _cb(i, total, k=None):
                            # i=0..total; на больших графах это прям спасает психику
                            frac_ = 0.0 if total <= 0 else min(1.0, max(0.0, i / total))
                            bar.progress(frac_)
                            if k is not None:
                                msg.caption(f"edge attack: {i}/{total}  target_edges={k}")

                        df_hist, aux = run_edge_attack(
                            G_view, kind, float(frac), int(steps), int(seed_run), int(eff_k),
                            compute_heavy_every=int(heavy_freq),
                            compute_curvature=bool(st.session_state.get("__compute_curvature", False)),
                            curvature_sample_edges=int(st.session_state.get("__curvature_sample_edges", 80)),
                            fast_mode=bool(fast_mode),
                            progress_cb=_cb,
                            row_cb=row_cb,
                        )
                        bar.empty(); msg.empty()
                        preview_holder.empty()
                        df_hist = _forward_fill_heavy(df_hist)
                        phase_info = classify_phase_transition(df_hist)

                        label = f"{active_entry.name} | edge:{kind} | seed={seed_run}"
                        if tag:
                            label += f" [{tag}]"

                        _save_experiment(
                            name=label,
                            graph_id=active_entry.id,
                            kind=kind,
                            params={
                                "attack_family": "edge",
                                "frac": float(frac),
                                "steps": int(steps),
                                "seed": int(seed_run),
                                "phase": phase_info,
                                "compute_heavy_every": int(heavy_freq),
                                "eff_k": int(eff_k),
                                "removed_edges_order": aux.get("removed_edges_order", []),
                                "total_edges": aux.get("total_edges", None),
                            },
                            df_hist=df_hist
                        )
                    st.success("Готово.")
                    st.rerun()

    st.markdown("---")
    st.markdown("## Последний результат (для текущего графа)")

    exps_here = [e for e in st.session_state["experiments"] if e.graph_id == active_entry.id]
    mixfrac_here = [
        e for e in exps_here
        if (e.params or {}).get("attack_family") == "mixfrac"
    ]
    if mixfrac_here:
        mixfrac_here.sort(key=lambda x: x.created_at, reverse=True)
        last_mixfrac = mixfrac_here[0]
        mp = last_mixfrac.params or {}
        with st.expander("Последний сохранённый mix_frac*", expanded=False):
            st.write(
                {
                    "mix_frac_star": mp.get("mix_frac_star"),
                    "ci_low": mp.get("ci_low"),
                    "ci_high": mp.get("ci_high"),
                    "distance_median": mp.get("distance_median"),
                    "replace_from": mp.get("replace_from"),
                    "healthy_n": mp.get("healthy_n"),
                    "used_metrics": mp.get("used_metrics", []),
                    "match_mode": mp.get("match_mode"),
                }
            )

    st.markdown("---")
    st.info("HC → SZ phenotype matching вынесен в отдельную вкладку «🧬 HC→SZ».")

    # Визуализация "Последний результат" работает только по стандартным атакам.
    exps_here = [
        e for e in exps_here
        if (e.params or {}).get("attack_family") in {"node", "edge", "mix", "degradation"}
    ]
    if not exps_here:
        st.info("Нет экспериментов. Запусти сверху.")
    else:
        exps_here.sort(key=lambda x: x.created_at, reverse=True)
        last_exp = exps_here[0]
        df_res = _forward_fill_heavy(last_exp.history.copy())
        params = last_exp.params or {}
        fam = params.get("attack_family", "node")
        if fam == "mix" and "mix_frac" in df_res.columns:
            xcol = "mix_frac"
        elif fam == "degradation" and "damage_frac" in df_res.columns:
            xcol = "damage_frac"
        else:
            xcol = "removed_frac"

        ph = last_exp.params.get("phase", {}) if last_exp.params else {}
        if ph:
            st.caption(
                f"Phase: {'🔥 Abrupt' if ph.get('is_abrupt') else '🌊 Continuous'}"
                f" | critical_x ≈ {float(ph.get('critical_x', 0.0)):.3f}"
            )

        attack_tabs = ["📉 Curves", "🌀 Phase views", "🧊 3D step-by-step"]
        # Stateful selector avoids tab resets when animation uses st.rerun().
        selected_attack_tab = st.radio(
            "Просмотр результатов",
            attack_tabs,
            horizontal=True,
            key="attack_results_tab",
        )

        if selected_attack_tab == attack_tabs[0]:
            with st.expander("❔ Что означают метрики на графиках", expanded=False):
                st.markdown(
                    "- **lcc_frac**: доля узлов в гигантской компоненте (порядковый параметр перколяции)\n"
                    "- **eff_w**: глобальная эффективность (в среднем насколько короткие пути; выше = сеть “связнее”)\n"
                    "- **l2_lcc**: λ₂ (алгебраическая связность) для LCC; близко к 0 = “на грани распада”\n"
                    "- **mod**: модульность сообществ; рост часто означает фрагментацию на кластеры\n"
                    "- **H_***: энтропии распределений (рост “случайности” структуры)\n"
                )
            fig = fig_metrics_over_steps(
                df_res,
                title="Метрики по шагам",
                normalize_mode=st.session_state["norm_mode"],
                height=st.session_state["plot_height"],
            )
            fig.update_layout(template="plotly_dark")
            fig.update_traces(mode="lines+markers")
            fig.update_traces(line_width=3)
            fig = _apply_plot_defaults(fig, height=st.session_state["plot_height"])
            st.plotly_chart(fig, width="stretch", key="plot_attack_metrics")

            st.markdown("#### AUC (robustness) по выбранной метрике")
            y_axis = st.selectbox(
                "Метрика для AUC",
                [c for c in ["lcc_frac", "eff_w", "l2_lcc", "mod", "H_deg", "H_w", "H_conf", "H_tri"] if c in df_res.columns],
                index=0,
                key="auc_y_single",
            )
            st.caption(METRIC_HELP.get(y_axis, ""))

            if y_axis in df_res.columns and xcol in df_res.columns:
                xs = pd.to_numeric(df_res[xcol], errors="coerce")
                ys = pd.to_numeric(df_res[y_axis], errors="coerce")
                mask = xs.notna() & ys.notna()
                if mask.sum() >= 2:
                    auc_val = float(AUC_TRAP(ys[mask].to_numpy(), xs[mask].to_numpy()))
                    st.metric("AUC", f"{auc_val:.6f}")
                else:
                    st.info("Недостаточно точек для AUC.")

            st.markdown("#### Resistance summary")
            base_res = graph_resistance_summary(G_view)
            attack_sum = attack_trajectory_summary(df_res, attack_kind=str(last_exp.attack_kind))
            rc1, rc2, rc3, rc4 = st.columns(4)
            rc1.metric("Giant comp frac", f"{float(base_res.get('giant_component_frac', 0.0)):.3f}")
            rc2.metric("Algebraic conn.", f"{float(base_res.get('algebraic_connectivity', float('nan'))):.3f}" if pd.notna(base_res.get('algebraic_connectivity')) else "—")
            rc3.metric("Final LCC frac", f"{float(attack_sum.get('final_lcc_frac', 0.0)):.3f}" if pd.notna(attack_sum.get('final_lcc_frac')) else "—")
            rc4.metric("Collapse 50%", f"{float(attack_sum.get('collapse_step_50', float('nan'))):.3f}" if pd.notna(attack_sum.get('collapse_step_50')) else "—")
            rc5, rc6, rc7, rc8 = st.columns(4)
            rc5.metric("Edge conn.", f"{float(base_res.get('edge_connectivity', float('nan'))):.3f}" if pd.notna(base_res.get('edge_connectivity')) else "—")
            rc6.metric("Node conn.", f"{float(base_res.get('node_connectivity', float('nan'))):.3f}" if pd.notna(base_res.get('node_connectivity')) else "—")
            rc7.metric("AUC LCC", f"{float(attack_sum.get('auc_lcc_frac', 0.0)):.3f}" if pd.notna(attack_sum.get('auc_lcc_frac')) else "—")
            rc8.metric("AUC eff", f"{float(attack_sum.get('auc_eff_w', 0.0)):.3f}" if pd.notna(attack_sum.get('auc_eff_w')) else "—")

            exp1, exp2, exp3 = st.columns(3)
            exp1.download_button(
                "Скачать trajectory (.csv)",
                df_res.to_csv(index=False).encode("utf-8"),
                file_name=f"{active_entry.name}_{last_exp.attack_kind}_trajectory.csv",
                mime="text/csv",
                width="stretch",
            )
            exp2.download_button(
                "Скачать attack summary (.json)",
                json.dumps(attack_sum, ensure_ascii=False, indent=2).encode("utf-8"),
                file_name=f"{active_entry.name}_{last_exp.attack_kind}_attack_summary.json",
                mime="application/json",
                width="stretch",
            )
            exp3.download_button(
                "Скачать graph resistance (.json)",
                json.dumps(base_res, ensure_ascii=False, indent=2).encode("utf-8"),
                file_name=f"{active_entry.name}_graph_resistance.json",
                mime="application/json",
                width="stretch",
            )

            with st.expander("❓ Что на этих графиках", expanded=False):
                txt = """
                Ось X:
                  - removed_frac: доля удалённых узлов/рёбер (атаки).
                  - mix_frac: уровень энтропизации (Hrish mix), 0..1.

                Ось Y:
                  - lcc_frac: доля LCC (перколяция).
                  - eff_w: эффективность (качество глобальной связности путей).
                  - l2_lcc: λ₂ (спектральная связность LCC).
                  - mod: модульность (структура сообществ).
                  - H_*: энтропии распределений (рост “случайности”).
                """
                st.text(textwrap.dedent(txt).strip())

        elif selected_attack_tab == attack_tabs[1]:
            if xcol in df_res.columns and "lcc_frac" in df_res.columns:
                fig_lcc = px.line(df_res, x=xcol, y="lcc_frac", title="Order parameter: LCC fraction vs removed fraction")
                fig_lcc.update_layout(template="plotly_dark")
                fig_lcc = _apply_plot_defaults(fig_lcc, height=780, y_range=_auto_y_range(df_res["lcc_frac"]))
                st.plotly_chart(fig_lcc, width="stretch", key="plot_phase_lcc")

            if xcol in df_res.columns and "lcc_frac" in df_res.columns:
                dfp = df_res.sort_values(xcol).copy()
                dx = pd.to_numeric(dfp[xcol], errors="coerce").diff()
                dy = pd.to_numeric(dfp["lcc_frac"], errors="coerce").diff()
                dfp["suscep"] = (dy / dx).replace([np.inf, -np.inf], np.nan)
                fig_s = px.line(dfp, x=xcol, y="suscep", title="Susceptibility proxy: d(LCC)/dx")
                fig_s.update_layout(template="plotly_dark")
                fig_s = _apply_plot_defaults(fig_s, height=780, y_range=_auto_y_range(dfp["suscep"]))
                st.plotly_chart(fig_s, width="stretch", key="plot_phase_suscep")

            if "mod" in df_res.columns and "l2_lcc" in df_res.columns:
                dfp2 = df_res.copy()
                dfp2["mod"] = pd.to_numeric(dfp2["mod"], errors="coerce")
                dfp2["l2_lcc"] = pd.to_numeric(dfp2["l2_lcc"], errors="coerce")
                dfp2 = dfp2.dropna(subset=["mod", "l2_lcc"])
                if not dfp2.empty:
                    fig_phase = px.line(dfp2, x="l2_lcc", y="mod", title="Phase portrait (trajectory): Q vs λ₂")
                    fig_phase.update_layout(template="plotly_dark")
                    fig_phase = _apply_plot_defaults(fig_phase, height=780)
                    st.plotly_chart(fig_phase, width="stretch", key="plot_phase_portrait")

        elif selected_attack_tab == attack_tabs[2]:
            edge_overlay_ui = st.selectbox(
                "Разметка рёбер (3D step-by-step)",
                [
                    "Ricci sign (κ<0/κ>0)",
                    "Energy flux (RW)",
                    "Energy flux (Demetrius)",
                    "Weight (log10)",
                    "Confidence",
                    "None",
                ],
                index=0,
                key="edge_overlay_tabc",
            )
            edge_overlay = "ricci"
            flow_mode = "rw"
            if edge_overlay_ui.startswith("Energy flux"):
                edge_overlay = "flux"
                flow_mode = "evo" if "Demetrius" in edge_overlay_ui else "rw"
            elif edge_overlay_ui.startswith("Weight"):
                edge_overlay = "weight"
            elif edge_overlay_ui.startswith("Confidence"):
                edge_overlay = "confidence"
            elif edge_overlay_ui.startswith("None"):
                edge_overlay = "none"

            base_seed = int(seed_val) + int(st.session_state.get("layout_seed_bump", 0))
            pos_base = _layout_cached(
                active_entry.edges,
                src_col,
                dst_col,
                float(min_conf),
                float(min_weight),
                analysis_mode,
                base_seed,
            )

            if fam == "mix":
                st.info("Для Mix/Entropy 3D-декомпозиция не поддерживается (нет порядка удаления).")
            elif fam == "node":
                removed_order = params.get("removed_order") or []
                if not removed_order:
                    st.warning("Нет removed_order для 3D. (src.run_attack не дал, а fallback не сохранился.)")
                else:
                    max_steps = max(1, len(df_res) - 1)
                    step_val = st.slider(
                        "Шаг (3D)",
                        0,
                        max_steps,
                        int(st.session_state.get("__decomp_step", 0)),
                        key="__decomp_step_slider",
                    )
                    st.session_state["__decomp_step"] = int(step_val)

                    play = st.toggle("▶ Play", value=False, key="play3d")
                    fps = st.slider("FPS", 1, 10, 3, key="fps3d")

                    frac_here = float(df_res.iloc[int(step_val)]["removed_frac"]) if "removed_frac" in df_res.columns else (step_val / max_steps)
                    k_remove = int(round(frac_here * G_view.number_of_nodes()))
                    k_remove = max(0, min(k_remove, len(removed_order)))

                    removed_set = set(removed_order[:k_remove])
                    H = as_simple_undirected(G_view).copy()
                    H.remove_nodes_from([n for n in removed_set if H.has_node(n)])

                    pos_k = {n: pos_base[n] for n in H.nodes() if n in pos_base}
                    edge_traces, node_trace = make_3d_traces(
                        H,
                        pos_k,
                        show_scale=True,
                        edge_overlay=edge_overlay,
                        flow_mode=flow_mode,
                    )

                    if node_trace is not None:
                        fig = go.Figure(data=[*edge_traces, node_trace])
                        fig.update_layout(template="plotly_dark", height=860, showlegend=False)
                        fig.update_layout(title=f"Node removal | step={step_val}/{max_steps} | removed~{k_remove} | frac={frac_here:.3f}")
                        st.plotly_chart(fig, width="stretch", key="plot_attack_3d_node_step")
                    else:
                        st.info("На этом шаге граф пуст.")

                    if play:
                        time.sleep(1.0 / float(fps))
                        nxt = int(step_val) + 1
                        if nxt > max_steps:
                            nxt = 0
                        st.session_state["__decomp_step"] = nxt
                        st.rerun()

            else:
                removed_edges_order = params.get("removed_edges_order") or []
                total_edges = (
                    params.get("total_edges")
                    or params.get("candidate_edges_total")
                    or len(as_simple_undirected(G_view).edges())
                )
                if not removed_edges_order:
                    st.warning("Нет removed_edges_order для 3D.")
                else:
                    max_steps = max(1, len(df_res) - 1)
                    step_val = st.slider(
                        "Шаг (3D)",
                        0,
                        max_steps,
                        int(st.session_state.get("__decomp_step", 0)),
                        key="__decomp_step_slider_edge",
                    )
                    st.session_state["__decomp_step"] = int(step_val)

                    play = st.toggle("▶ Play", value=False, key="play3d_edge")
                    fps = st.slider("FPS", 1, 10, 3, key="fps3d_edge")

                    frac_here = float(df_res.iloc[int(step_val)]["removed_frac"]) if "removed_frac" in df_res.columns else (step_val / max_steps)
                    k_remove = int(round(frac_here * float(total_edges)))
                    k_remove = max(0, min(k_remove, len(removed_edges_order)))

                    H = as_simple_undirected(G_view).copy()
                    for (u, v) in removed_edges_order[:k_remove]:
                        if H.has_edge(u, v):
                            H.remove_edge(u, v)

                    pos_k = {n: pos_base[n] for n in H.nodes() if n in pos_base}
                    edge_traces, node_trace = make_3d_traces(
                        H,
                        pos_k,
                        show_scale=True,
                        edge_overlay=edge_overlay,
                        flow_mode=flow_mode,
                    )

                    if node_trace is not None:
                        fig = go.Figure(data=[*edge_traces, node_trace])
                        fig.update_layout(template="plotly_dark", height=860, showlegend=False)
                        fig.update_layout(title=f"Edge removal | step={step_val}/{max_steps} | removed~{k_remove} edges | frac={frac_here:.3f}")
                        st.plotly_chart(fig, width="stretch", key="plot_attack_3d_edge_step")
                    else:
                        st.info("На этом шаге граф пуст.")

                    if play:
                        time.sleep(1.0 / float(fps))
                        nxt = int(step_val) + 1
                        if nxt > max_steps:
                            nxt = 0
                        st.session_state["__decomp_step"] = nxt
                        st.rerun()

    st.markdown("---")

    # --------------------------
    # MIX_FRAC* ESTIMATOR
    # --------------------------
    st.subheader("mix_frac* estimator")
    st.caption(
        "Оценка: на какой точке randomization trajectory текущий граф "
        "становится наиболее похож на patient-like профиль."
    )

    graphs = st.session_state["graphs"]
    all_gids = list(graphs.keys())
    hc_guess = _guess_hc_like_graph_ids(graphs, active_entry.id)

    mf_col1, mf_col2 = st.columns([1, 1.2])

    with mf_col1:
        with st.container(border=True):
            st.markdown("### Параметры mix_frac*")

            healthy_gids = st.multiselect(
                "Healthy / reference graphs",
                [gid for gid in all_gids if gid != active_entry.id],
                default=hc_guess,
                format_func=lambda gid: f"{graphs[gid].name} ({graphs[gid].source})",
                key="mixfrac_hc_gids",
            )

            match_mode = st.radio(
                "Match mode",
                ["nearest", "interpolate"],
                horizontal=True,
                key="mixfrac_match_mode",
                help="nearest = matching по вектору метрик; interpolate = старый одномерный режим.",
            )

            if match_mode == "nearest":
                selected_metrics = st.multiselect(
                    "Метрики сопоставления",
                    MIX_FRAC_METRIC_OPTIONS,
                    default=["kappa_mean", "kappa_frac_negative", "clustering"],
                    key="mixfrac_metrics_multi",
                )
            else:
                one_metric = st.selectbox(
                    "Метрика сопоставления",
                    MIX_FRAC_METRIC_OPTIONS,
                    index=0,
                    key="mixfrac_metric_single",
                )
                selected_metrics = [one_metric]

            mf_steps = st.slider("Trajectory steps", 4, 50, 20, 1, key="mixfrac_steps")
            mf_replace_from = st.selectbox(
                "Replace from",
                ["CFG", "ER"],
                index=0,
                key="mixfrac_replace_from",
            )
            mf_effk = st.slider("Efficiency k", 8, 256, 32, key="mixfrac_effk")
            mf_seed = st.number_input("Seed (mix_frac*)", value=int(seed_val), step=1, key="mixfrac_seed")
            mf_n_boot = st.slider("Bootstrap n", 100, 5000, 1000, 100, key="mixfrac_n_boot")

            mf_btn1, mf_btn2 = st.columns(2)
            run_mixfrac = mf_btn1.button(
                "🧭 Estimate",
                type="primary",
                width="stretch",
                key="mixfrac_run",
            )
            save_mixfrac = mf_btn2.button(
                "💾 Save result",
                width="stretch",
                key="mixfrac_save",
            )

    with mf_col2:
        mixfrac_res = st.session_state.get("__mix_frac_star_result")
        if mixfrac_res:
            m1, m2, m3 = st.columns(3)
            star = mixfrac_res.get("mix_frac_star", float("nan"))
            ci_low = mixfrac_res.get("ci_low", float("nan"))
            ci_high = mixfrac_res.get("ci_high", float("nan"))
            med_dist = mixfrac_res.get("distance_median", float("nan"))

            m1.metric("mix_frac*", f"{star:.4f}" if np.isfinite(star) else "NaN")
            m2.metric(
                "95% CI",
                f"[{ci_low:.4f}, {ci_high:.4f}]" if np.isfinite(ci_low) and np.isfinite(ci_high) else "NaN",
            )
            m3.metric("median distance", f"{med_dist:.4f}" if np.isfinite(med_dist) else "NaN")

            st.write(
                {
                    "match_mode": mixfrac_res.get("match_mode"),
                    "used_metrics": mixfrac_res.get("used_metrics", []),
                    "replace_from": mixfrac_res.get("replace_from"),
                    "healthy_n": mixfrac_res.get("healthy_n"),
                    "skipped_graphs": mixfrac_res.get("skipped_graphs", []),
                }
            )

            vals = [float(v) for v in mixfrac_res.get("mix_frac_values", []) if np.isfinite(v)]
            dists = [float(v) for v in mixfrac_res.get("distances", []) if np.isfinite(v)]
            if vals:
                st.markdown("#### По healthy-кривым")
                df_show = pd.DataFrame(
                    {
                        "mix_frac_value": vals,
                        "distance": dists[: len(vals)] if dists else [np.nan] * len(vals),
                    }
                )
                st.dataframe(df_show, width="stretch")

                fig_vals = px.histogram(
                    df_show,
                    x="mix_frac_value",
                    nbins=min(20, max(5, len(df_show))),
                    title="Distribution of mix_frac values",
                )
                fig_vals.update_layout(template="plotly_dark")
                st.plotly_chart(fig_vals, width="stretch", key="mixfrac_hist_vals")

                if np.isfinite(df_show["distance"]).any():
                    fig_dist = px.histogram(
                        df_show,
                        x="distance",
                        nbins=min(20, max(5, len(df_show))),
                        title="Distribution of matching distances",
                    )
                    fig_dist.update_layout(template="plotly_dark")
                    st.plotly_chart(fig_dist, width="stretch", key="mixfrac_hist_dist")
        else:
            st.info("Выбери healthy-графы и запусти оценку.")

    if run_mixfrac:
        if not healthy_gids:
            st.error("Нужен хотя бы один healthy/reference graph.")
        elif not selected_metrics:
            st.error("Выбери хотя бы одну метрику.")
        else:
            needs_curv = _needs_curvature_for_metrics(selected_metrics)
            curv_edges = int(st.session_state.get("__curvature_sample_edges", 120))

            with st.spinner("Считаю patient profile и healthy trajectories..."):
                patient_graph = _build_current_graph_for_entry(
                    active_entry,
                    min_conf=float(min_conf),
                    min_weight=float(min_weight),
                    analysis_mode=str(analysis_mode),
                )

                patient_metrics = calculate_metrics(
                    patient_graph,
                    eff_sources_k=int(mf_effk),
                    seed=int(mf_seed),
                    compute_curvature=bool(needs_curv),
                    curvature_sample_edges=int(curv_edges),
                )

                healthy_graphs = []
                skipped = []
                for gid in healthy_gids:
                    entry = graphs[gid]
                    try:
                        g_h = _build_current_graph_for_entry(
                            entry,
                            min_conf=float(min_conf),
                            min_weight=float(min_weight),
                            analysis_mode=str(analysis_mode),
                        )
                        if g_h.number_of_nodes() > 0 and g_h.number_of_edges() > 0:
                            healthy_graphs.append(g_h)
                        else:
                            skipped.append(entry.name)
                    except Exception:
                        # Один кривой граф не должен ломать весь расчет.
                        skipped.append(entry.name)

                if not healthy_graphs:
                    st.error("После фильтрации не осталось пригодных healthy-графов.")
                else:
                    res = estimate_mix_frac_star(
                        healthy_graphs,
                        patient_metrics,
                        target_metric=selected_metrics if match_mode == "nearest" else selected_metrics[0],
                        match_mode=str(match_mode),
                        steps=int(mf_steps),
                        seed=int(mf_seed),
                        eff_sources_k=int(mf_effk),
                        replace_from=str(mf_replace_from),
                        n_boot=int(mf_n_boot),
                    )

                    dists = [float(v) for v in res.get("distances", []) if np.isfinite(v)]
                    res["distance_median"] = float(np.median(dists)) if dists else float("nan")
                    res["replace_from"] = str(mf_replace_from)
                    res["healthy_n"] = int(len(healthy_graphs))
                    res["skipped_graphs"] = skipped
                    st.session_state["__mix_frac_star_result"] = res
                    st.success("mix_frac* посчитан.")
                    st.rerun()

    if save_mixfrac:
        mixfrac_res = st.session_state.get("__mix_frac_star_result")
        if not mixfrac_res:
            st.error("Сначала посчитай mix_frac*.")
        else:
            label = (
                f"{active_entry.name} | mix_frac* | "
                f"{mixfrac_res.get('match_mode', 'nearest')} | "
                f"{mixfrac_res.get('replace_from', 'CFG')}"
            )
            _save_experiment(
                name=label,
                graph_id=active_entry.id,
                kind="mix_frac_estimate",
                params={
                    "attack_family": "mixfrac",
                    "mix_frac_star": float(mixfrac_res.get("mix_frac_star", np.nan)),
                    "ci_low": float(mixfrac_res.get("ci_low", np.nan)),
                    "ci_high": float(mixfrac_res.get("ci_high", np.nan)),
                    "distance_median": float(mixfrac_res.get("distance_median", np.nan)),
                    "replace_from": str(mixfrac_res.get("replace_from", "")),
                    "healthy_n": int(mixfrac_res.get("healthy_n", 0)),
                    "used_metrics": list(mixfrac_res.get("used_metrics", [])),
                    "match_mode": str(mixfrac_res.get("match_mode", "")),
                    "skipped_graphs": list(mixfrac_res.get("skipped_graphs", [])),
                },
                df_hist=_mixfrac_result_to_history(mixfrac_res),
            )
            st.success("mix_frac* result saved to experiments.")
            st.rerun()

    # --------------------------
    # PRESET BATCH (same graph)
    # --------------------------
    st.subheader("Preset batch (на одном графе)")
    bcol1, bcol2 = st.columns([1, 2])

    with bcol1:
        batch_family = st.radio("Batch тип", ["Node presets", "Edge presets"], horizontal=True, key="batch_family")

        if batch_family.startswith("Node"):
            preset_name = st.selectbox("Preset", list(ATTACK_PRESETS_NODE.keys()), key="preset_node")
            preset = ATTACK_PRESETS_NODE[preset_name]
        else:
            preset_name = st.selectbox("Preset", list(ATTACK_PRESETS_EDGE.keys()), key="preset_edge")
            preset = ATTACK_PRESETS_EDGE[preset_name]

        frac_b = st.slider("Доля удаления (batch)", 0.05, 0.95, 0.5, 0.05, key="batch_frac")
        steps_b = st.slider("Шаги (batch)", 5, 150, 30, key="batch_steps")
        seed_b = st.number_input("Base seed (batch)", value=123, step=1, key="batch_seed")

        with st.expander("Batch advanced"):
            eff_k_b = st.slider("Efficiency k", 8, 256, 32, key="batch_effk")
            heavy_b = st.slider("Heavy every N", 1, 10, 2, key="batch_heavy")
            tag_b = st.text_input("Тег batch", "", key="batch_tag")

        if st.button("🚀 RUN PRESET SUITE", type="primary", width="stretch", key="run_suite"):
            with st.spinner(f"Running preset: {preset_name}"):
                if batch_family.startswith("Node"):
                    curves = run_node_attack_suite(
                        G_view, active_entry, preset,
                        frac=float(frac_b), steps=int(steps_b), base_seed=int(seed_b),
                        eff_k=int(eff_k_b), heavy_freq=int(heavy_b),
                        rc_frac=0.1, tag=tag_b
                    )
                else:
                    curves = run_edge_attack_suite(
                        G_view, active_entry, preset,
                        frac=float(frac_b), steps=int(steps_b), base_seed=int(seed_b),
                        eff_k=int(eff_k_b), heavy_freq=int(heavy_b),
                        tag=tag_b
                    )

            st.session_state["last_suite_curves"] = curves
            st.success(f"Готово: {len(curves)} прогонов сохранено.")
            st.rerun()

    with bcol2:
        curves = st.session_state.get("last_suite_curves")
        if curves:
            st.markdown("### Сравнение suite")
            y_axis = st.selectbox("Y", ["lcc_frac", "eff_w", "l2_lcc", "mod"], index=0, key="suite_y")
            fig = fig_compare_attacks(
                curves,
                "removed_frac",
                y_axis,
                f"Suite compare: {y_axis}",
                normalize_mode=st.session_state["norm_mode"],
                height=st.session_state["plot_height"],
            )
            fig.update_layout(template="plotly_dark")
            all_y = pd.concat([pd.to_numeric(df[y_axis], errors="coerce") for _, df in curves if y_axis in df.columns], ignore_index=True)
            fig = _apply_plot_defaults(fig, height=st.session_state["plot_height"], y_range=_auto_y_range(all_y))
            st.plotly_chart(fig, width="stretch", key="plot_suite_compare")

            st.markdown("#### AUC ranking")
            rows = []
            for name, df in curves:
                if "removed_frac" in df.columns and y_axis in df.columns:
                    xs = pd.to_numeric(df["removed_frac"], errors="coerce")
                    ys = pd.to_numeric(df[y_axis], errors="coerce")
                    mask = xs.notna() & ys.notna()
                    if mask.sum() >= 2:
                        rows.append({"run": name, "AUC": float(AUC_TRAP(ys[mask].to_numpy(), xs[mask].to_numpy()))})
            if rows:
                df_auc = pd.DataFrame(rows).sort_values("AUC", ascending=False)
                st.dataframe(df_auc, width="stretch")
        else:
            st.info("Запусти suite слева, чтобы увидеть сравнение.")

    st.markdown("---")

    # --------------------------
    # MULTI-GRAPH BATCH
    # --------------------------
    st.subheader("Multi-graph batch (на нескольких графах)")
    graphs = st.session_state["graphs"]
    gid_list = list(graphs.keys())

    mg_col1, mg_col2 = st.columns([1, 2])

    with mg_col1:
        mg_family = st.radio("Multi тип", ["Node presets", "Edge presets"], horizontal=True, key="mg_family")

        sel_gids = st.selectbox(
            "Графы (multi) — выбери несколько в списке ниже",
            options=["(выбрать ниже)"],
            index=0,
            help="Основной выбор — в multiselect ниже"
        )

        sel_gids = st.multiselect(
            "Выбери графы",
            gid_list,
            default=[st.session_state["active_graph_id"]] if st.session_state["active_graph_id"] else [],
            format_func=lambda gid: f"{graphs[gid].name} ({graphs[gid].source})",
            key="mg_gids"
        )

        if mg_family.startswith("Node"):
            preset_name_mg = st.selectbox("Preset (multi)", list(ATTACK_PRESETS_NODE.keys()), key="mg_preset_node")
            preset_mg = ATTACK_PRESETS_NODE[preset_name_mg]
        else:
            preset_name_mg = st.selectbox("Preset (multi)", list(ATTACK_PRESETS_EDGE.keys()), key="mg_preset_edge")
            preset_mg = ATTACK_PRESETS_EDGE[preset_name_mg]

        mg_frac = st.slider("Доля удаления", 0.05, 0.95, 0.5, 0.05, key="mg_frac")
        mg_steps = st.slider("Шаги", 5, 150, 30, key="mg_steps")
        mg_seed = st.number_input("Base seed", value=321, step=1, key="mg_seed")

        with st.expander("Multi advanced"):
            mg_effk = st.slider("Efficiency k", 8, 256, 32, key="mg_effk")
            mg_heavy = st.slider("Heavy every N", 1, 10, 2, key="mg_heavy")
            mg_tag = st.text_input("Тег multi", "", key="mg_tag")

        if st.button("🚀 RUN MULTI-GRAPH SUITE", type="primary", width="stretch", key="run_mg"):
            if not sel_gids:
                st.error("Выбери хотя бы один граф.")
            else:
                all_curves = []
                with st.spinner("Running multi-graph suite..."):
                    for gid in sel_gids:
                        entry = graphs[gid]
                        _df = filter_edges(
                            entry.edges,
                            entry.src_col,
                            entry.dst_col,
                            min_conf, min_weight
                        )
                        _G = build_graph_from_edges(_df, entry.src_col, entry.dst_col)
                        if analysis_mode.startswith("LCC"):
                            _G = lcc_subgraph(_G)

                        if mg_family.startswith("Node"):
                            curves = run_node_attack_suite(
                                _G, entry, preset_mg,
                                frac=float(mg_frac), steps=int(mg_steps),
                                base_seed=int(mg_seed), eff_k=int(mg_effk),
                                heavy_freq=int(mg_heavy),
                                rc_frac=0.1,
                                tag=f"MG:{mg_tag}"
                            )
                        else:
                            curves = run_edge_attack_suite(
                                _G, entry, preset_mg,
                                frac=float(mg_frac), steps=int(mg_steps),
                                base_seed=int(mg_seed), eff_k=int(mg_effk),
                                heavy_freq=int(mg_heavy),
                                tag=f"MG:{mg_tag}"
                            )

                        all_curves.extend(curves)

                st.session_state["last_multi_curves"] = all_curves
                st.success(f"Готово: {len(all_curves)} прогонов.")
                st.rerun()

    with mg_col2:
        multi_curves = st.session_state.get("last_multi_curves")
        if multi_curves:
            st.markdown("### Multi сравнение")
            y = st.selectbox("Y (multi)", ["lcc_frac", "eff_w", "l2_lcc", "mod"], index=0, key="mg_y")
            fig = fig_compare_attacks(
                multi_curves,
                "removed_frac",
                y,
                f"Multi compare: {y}",
                normalize_mode=st.session_state["norm_mode"],
                height=st.session_state["plot_height"],
            )
            fig.update_layout(template="plotly_dark")
            all_y = pd.concat([pd.to_numeric(df[y], errors="coerce") for _, df in multi_curves if y in df.columns], ignore_index=True)
            fig = _apply_plot_defaults(fig, height=st.session_state["plot_height"], y_range=_auto_y_range(all_y))
            st.plotly_chart(fig, width="stretch", key="plot_multi_compare")
        else:
            st.info("Запусти multi suite слева, чтобы увидеть сравнение.")
