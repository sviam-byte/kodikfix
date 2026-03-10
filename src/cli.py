from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterable

import networkx as nx
import pandas as pd
from joblib import Parallel, delayed

from .attacks import run_attack, run_edge_attack
from .attacks_mix import run_mix_attack
from .config import settings
from .degradation import run_degradation_trajectory
from .exporters import (
    export_metrics_payload,
    infer_output_format,
    payload_to_dataframe,
    payload_to_flat_row,
)
from .graph_build import build_graph_from_edges, graph_summary, lcc_subgraph
from .matrix_import import load_matrix, matrix_to_graph
from .metrics import calculate_metrics
from .mix_frac_estimator import estimate_mix_frac_star
from .phenotype_controls import effective_metrics_for_run_type, run_control_suite
from .phenotype_matching import compare_degradation_models, summarize_best_attack
from .phenotype_preflight import build_run_manifest, run_phenotype_preflight, save_run_bundle
from .phenotype_reporting import (
    build_paper_ready_summary,
    build_warning_flags,
    export_phenotype_match_excel,
)
from .preprocess import coerce_fixed_format, filter_edges

SUPPORTED_EDGE_EXTS = {".csv", ".tsv", ".txt", ".xlsx", ".xls"}
SUPPORTED_MATRIX_EXTS = {".mat", ".npy", ".npz"}
SUPPORTED_INPUT_EXTS = SUPPORTED_EDGE_EXTS | SUPPORTED_MATRIX_EXTS


def _load_table(path: Path) -> pd.DataFrame:
    """Load a CSV/Excel file into a DataFrame."""
    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(path)
    sep = "\t" if suffix == ".tsv" else None
    return pd.read_csv(path, sep=sep, engine="python", encoding_errors="replace")


def _json_default(obj):
    """JSON fallback serializer for numpy/pandas scalar objects."""
    try:
        return float(obj)
    except Exception:
        return str(obj)


def _safe_stem(value: str) -> str:
    """Create a filename-safe stem to avoid unsafe chars in batch output names."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    cleaned = cleaned.strip("._")
    return cleaned or "item"


def _iter_input_files(
    input_dir: Path,
    *,
    recursive: bool,
    pattern: str,
    limit: int = 0,
) -> list[Path]:
    """Collect supported input files from directory using glob/rglob pattern."""
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    iterator: Iterable[Path] = input_dir.rglob(pattern) if recursive else input_dir.glob(pattern)
    files = [path for path in sorted(iterator) if path.is_file() and path.suffix.lower() in SUPPORTED_INPUT_EXTS]

    if int(limit) > 0:
        files = files[: int(limit)]
    if not files:
        raise FileNotFoundError(f"No supported input files found in {input_dir} with pattern {pattern!r}")
    return files


def _load_edges_df(
    path: Path,
    *,
    fixed: bool,
    src: str,
    dst: str,
    min_conf: float,
    min_weight: float,
) -> tuple[pd.DataFrame, str, str]:
    """Load and normalize edge table according to fixed/custom column mode."""
    df_any = _load_table(path)
    if fixed:
        df_edges, meta = coerce_fixed_format(df_any)
        return df_edges, meta["src_col"], meta["dst_col"]
    df_edges = filter_edges(df_any.copy(), src, dst, min_conf, min_weight)
    return df_edges, src, dst


def _build_graph_from_cli(
    path: Path,
    *,
    fixed: bool,
    src: str,
    dst: str,
    min_conf: float,
    min_weight: float,
    lcc: bool,
    input_kind: str = "auto",
    mat_key: str = "",
    sign_policy: str = "abs",
    threshold_mode: str = "density",
    threshold_value: float = 0.15,
    shift: float = 0.0,
):
    """Build graph from file using either edge-list or matrix loader."""
    suffix = path.suffix.lower()
    use_matrix = input_kind == "matrix" or (input_kind == "auto" and suffix in SUPPORTED_MATRIX_EXTS)

    if use_matrix:
        corr = load_matrix(path, key=(mat_key or None))
        return matrix_to_graph(
            corr,
            sign_policy=str(sign_policy),
            threshold_mode=str(threshold_mode),
            threshold_value=float(threshold_value),
            shift=float(shift),
            labels=None,
            use_lcc=bool(lcc),
        )

    df_edges, src_col, dst_col = _load_edges_df(
        path,
        fixed=fixed,
        src=src,
        dst=dst,
        min_conf=min_conf,
        min_weight=min_weight,
    )
    graph = build_graph_from_edges(df_edges, src_col, dst_col, strict=True)
    if lcc:
        graph = lcc_subgraph(graph)
    return graph


def _write_json(payload: dict, out: str) -> None:
    """Write JSON payload to stdout or file."""
    txt = json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default)
    if out == "-":
        print(txt)
    else:
        Path(out).write_text(txt, encoding="utf-8")


def _add_common_graph_args(parser: argparse.ArgumentParser) -> None:
    """Add common graph-loading arguments used across subcommands."""
    parser.add_argument(
        "--fixed",
        action="store_true",
        help="Use fixed-format loader (src,dst at col 0/1; conf at 8; weight at 9)",
    )
    parser.add_argument("--src", type=str, default="src", help="Source column name")
    parser.add_argument("--dst", type=str, default="dst", help="Target column name")
    parser.add_argument("--min-conf", type=float, default=0.0)
    parser.add_argument("--min-weight", type=float, default=0.0)
    parser.add_argument("--lcc", action="store_true", help="Restrict to largest connected component")
    parser.add_argument(
        "--input-kind",
        type=str,
        default="auto",
        choices=["auto", "edge", "matrix"],
        help="auto: infer by extension; matrix for .mat/.npy/.npz; edge for edge lists",
    )
    parser.add_argument("--mat-key", type=str, default="", help="Variable name inside .mat file")
    parser.add_argument(
        "--sign-policy",
        type=str,
        default="abs",
        choices=["abs", "positive_only", "shift"],
        help="How to handle signed matrix weights",
    )
    parser.add_argument(
        "--threshold-mode",
        type=str,
        default="density",
        choices=["density", "absolute"],
        help="Thresholding mode for matrix inputs",
    )
    parser.add_argument("--threshold-value", type=float, default=0.15, help="Density or absolute threshold for matrix inputs")
    parser.add_argument("--shift", type=float, default=0.0, help="Shift added to matrix weights when sign-policy=shift")


def _metrics_payload_from_graph(args, graph, *, input_label: str) -> dict:
    """Build metrics payload from an already constructed graph."""
    met = calculate_metrics(
        graph,
        int(args.eff_k),
        int(args.seed),
        bool(args.compute_curvature),
        curvature_sample_edges=int(args.curvature_sample_edges),
        compute_heavy=bool(getattr(args, "compute_heavy", True)),
        skip_spectral=bool(getattr(args, "skip_spectral", False)),
        diameter_samples=int(getattr(args, "diameter_samples", 16)),
        ricci_n_jobs=int(getattr(args, "n_jobs", 0)) or None,
    )
    return {
        "mode": "metrics",
        "input": str(input_label),
        "summary": graph_summary(graph),
        "settings": {
            "seed": int(args.seed),
            "weight_policy": settings.WEIGHT_POLICY,
            "weight_eps": settings.WEIGHT_EPS,
            "compute_curvature": bool(args.compute_curvature),
            "curvature_sample_edges": int(args.curvature_sample_edges),
            "compute_heavy": bool(getattr(args, "compute_heavy", True)),
            "skip_spectral": bool(getattr(args, "skip_spectral", False)),
        },
        "metrics": met,
    }




def _build_metrics_payload(args, path: Path) -> dict:
    """Build metrics payload for one input graph without writing to disk."""
    graph = _build_graph_from_cli(
        path,
        fixed=bool(args.fixed),
        src=str(args.src),
        dst=str(args.dst),
        min_conf=float(args.min_conf),
        min_weight=float(args.min_weight),
        lcc=bool(args.lcc),
        input_kind=str(getattr(args, "input_kind", "auto")),
        mat_key=str(getattr(args, "mat_key", "")),
        sign_policy=str(getattr(args, "sign_policy", "abs")),
        threshold_mode=str(getattr(args, "threshold_mode", "density")),
        threshold_value=float(getattr(args, "threshold_value", 0.15)),
        shift=float(getattr(args, "shift", 0.0)),
    )
    return _metrics_payload_from_graph(args, graph, input_label=str(path))


def _load_metrics_table(path: Path) -> pd.DataFrame:
    """Load CSV/TSV/XLSX metrics table."""
    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(path)
    sep = "	" if suffix == ".tsv" else None
    return pd.read_csv(path, sep=sep, engine="python", encoding_errors="replace")


def _compute_metrics_row_for_graph(args, graph, *, label: str) -> dict:
    """Compute one flat metrics row suitable for phenotype matching tables."""
    met = calculate_metrics(
        graph,
        int(args.eff_k),
        int(args.seed),
        bool(getattr(args, "compute_curvature", False)),
        curvature_sample_edges=int(getattr(args, "curvature_sample_edges", 80)),
        compute_heavy=True,
        skip_spectral=False,
        diameter_samples=16,
        ricci_n_jobs=None,
    )
    row = {"input": str(label)}
    row.update(met)
    return row


def _build_graphs_for_paths(args, paths: list[str]) -> list[nx.Graph]:
    """Load many graphs from CLI paths using current graph-loading options."""
    graphs: list[nx.Graph] = []
    for item in paths:
        g = _build_graph_from_cli(
            Path(item),
            fixed=bool(args.fixed),
            src=str(args.src),
            dst=str(args.dst),
            min_conf=float(args.min_conf),
            min_weight=float(args.min_weight),
            lcc=bool(args.lcc),
            input_kind=str(getattr(args, "input_kind", "auto")),
            mat_key=str(getattr(args, "mat_key", "")),
            sign_policy=str(getattr(args, "sign_policy", "abs")),
            threshold_mode=str(getattr(args, "threshold_mode", "density")),
            threshold_value=float(getattr(args, "threshold_value", 0.15)),
            shift=float(getattr(args, "shift", 0.0)),
        )
        graphs.append(g)
    return graphs


def _compute_baseline_metrics_df(args, paths: list[str]) -> pd.DataFrame:
    """Compute HC baseline metrics table from graph files."""
    rows = []
    for item in paths:
        graph = _build_graph_from_cli(
            Path(item),
            fixed=bool(args.fixed),
            src=str(args.src),
            dst=str(args.dst),
            min_conf=float(args.min_conf),
            min_weight=float(args.min_weight),
            lcc=bool(args.lcc),
            input_kind=str(getattr(args, "input_kind", "auto")),
            mat_key=str(getattr(args, "mat_key", "")),
            sign_policy=str(getattr(args, "sign_policy", "abs")),
            threshold_mode=str(getattr(args, "threshold_mode", "density")),
            threshold_value=float(getattr(args, "threshold_value", 0.15)),
            shift=float(getattr(args, "shift", 0.0)),
        )
        rows.append(_compute_metrics_row_for_graph(args, graph, label=item))
    return pd.DataFrame(rows)


def _attack_payload_from_graph(args, graph, *, input_label: str) -> tuple[dict, pd.DataFrame]:
    """Execute one attack experiment for an already constructed graph."""
    family = str(args.family)

    if family == "node":
        history, aux = run_attack(
            graph,
            str(args.kind),
            float(args.frac),
            int(args.steps),
            int(args.seed),
            int(args.eff_k),
            compute_heavy_every=int(args.heavy_every),
            fast_mode=bool(args.fast_mode),
        )

    elif family == "edge":
        history, aux = run_edge_attack(
            graph,
            str(args.kind),
            float(args.frac),
            int(args.steps),
            int(args.seed),
            int(args.eff_k),
            compute_heavy_every=int(args.heavy_every),
            compute_curvature=bool(args.compute_curvature),
            curvature_sample_edges=int(args.curvature_sample_edges),
        )

    elif family == "mix":
        history, aux = run_mix_attack(
            graph,
            kind=str(args.kind),
            steps=int(args.steps),
            seed=int(args.seed),
            eff_sources_k=int(args.eff_k),
            heavy_every=int(args.heavy_every),
            alpha_rewire=float(args.alpha_rewire),
            beta_replace=float(args.beta_replace),
            swaps_per_edge=float(args.swaps_per_edge),
            replace_from=str(args.replace_from),
            fast_mode=bool(args.fast_mode),
        )

    elif family == "degradation":
        history, aux = run_degradation_trajectory(
            graph,
            kind=str(args.kind),
            steps=int(args.steps),
            frac=float(args.frac),
            seed=int(args.seed),
            eff_sources_k=int(args.eff_k),
            compute_heavy_every=int(args.heavy_every),
            compute_curvature=bool(args.compute_curvature),
            curvature_sample_edges=int(args.curvature_sample_edges),
            noise_sigma_max=float(args.noise_sigma_max),
            keep_density_from_baseline=bool(args.keep_density_from_baseline),
            recompute_modules=bool(args.recompute_modules),
            module_resolution=float(args.module_resolution),
            removal_mode=str(args.removal_mode),
            fast_mode=bool(args.fast_mode),
        )

    else:
        raise ValueError(f"Unsupported family: {family}")

    payload = {
        "mode": "attack",
        "family": family,
        "kind": str(args.kind),
        "input": str(input_label),
        "summary": graph_summary(graph),
        "params": {
            "seed": int(args.seed),
            "steps": int(args.steps),
            "frac": float(args.frac),
            "eff_k": int(args.eff_k),
            "heavy_every": int(args.heavy_every),
            "replace_from": getattr(args, "replace_from", None),
            "alpha_rewire": getattr(args, "alpha_rewire", None),
            "beta_replace": getattr(args, "beta_replace", None),
            "swaps_per_edge": getattr(args, "swaps_per_edge", None),
            "noise_sigma_max": getattr(args, "noise_sigma_max", None),
            "keep_density_from_baseline": getattr(args, "keep_density_from_baseline", None),
            "recompute_modules": getattr(args, "recompute_modules", None),
            "module_resolution": getattr(args, "module_resolution", None),
            "removal_mode": getattr(args, "removal_mode", None),
        },
        "aux": aux,
        "final_row": history.iloc[-1].to_dict() if not history.empty else {},
    }
    return payload, history


def _run_attack_payload(args, path: Path) -> tuple[dict, pd.DataFrame]:
    """Execute one attack experiment and return payload + full history frame."""
    graph = _build_graph_from_cli(
        path,
        fixed=bool(args.fixed),
        src=str(args.src),
        dst=str(args.dst),
        min_conf=float(args.min_conf),
        min_weight=float(args.min_weight),
        lcc=bool(args.lcc),
        input_kind=str(getattr(args, "input_kind", "auto")),
        mat_key=str(getattr(args, "mat_key", "")),
        sign_policy=str(getattr(args, "sign_policy", "abs")),
        threshold_mode=str(getattr(args, "threshold_mode", "density")),
        threshold_value=float(getattr(args, "threshold_value", 0.15)),
        shift=float(getattr(args, "shift", 0.0)),
    )
    return _attack_payload_from_graph(args, graph, input_label=str(path))


def _cmd_metrics(args) -> int:
    """Run scalar metrics command."""
    payload = _build_metrics_payload(args, Path(args.input))

    if args.out == "-":
        fmt = infer_output_format("stdout.json", args.out_format)
        if fmt == "json":
            txt = json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default)
            print(txt)
        elif fmt == "csv":
            print(payload_to_dataframe(payload).to_csv(index=False))
        else:
            raise ValueError("For XLSX output, specify a file path in --out instead of stdout.")
    else:
        export_metrics_payload(payload, args.out, args.out_format)
    return 0


def _cmd_attack(args) -> int:
    """Run node/edge/mix attack command."""
    payload, history = _run_attack_payload(args, Path(args.input))
    _write_json(payload, args.out)
    if args.history_out:
        Path(args.history_out).parent.mkdir(parents=True, exist_ok=True)
        history.to_csv(args.history_out, index=False)
    return 0


def _cmd_mixfrac(args) -> int:
    """Run mix_frac* estimation command for patient + healthy graphs."""
    patient_graph = _build_graph_from_cli(
        Path(args.patient),
        fixed=bool(args.fixed),
        src=str(args.src),
        dst=str(args.dst),
        min_conf=float(args.min_conf),
        min_weight=float(args.min_weight),
        lcc=bool(args.lcc),
        input_kind=str(getattr(args, "input_kind", "auto")),
        mat_key=str(getattr(args, "mat_key", "")),
        sign_policy=str(getattr(args, "sign_policy", "abs")),
        threshold_mode=str(getattr(args, "threshold_mode", "density")),
        threshold_value=float(getattr(args, "threshold_value", 0.15)),
        shift=float(getattr(args, "shift", 0.0)),
    )
    metrics = [m.strip() for m in str(args.metrics).split(",") if m.strip()]
    if not metrics:
        raise ValueError("--metrics must contain at least one metric")

    need_curv = any(m.startswith("kappa_") or m == "fragility_kappa" for m in metrics)
    patient_metrics = calculate_metrics(
        patient_graph,
        int(args.eff_k),
        int(args.seed),
        bool(need_curv),
        curvature_sample_edges=int(args.curvature_sample_edges),
        compute_heavy=True,
    )

    healthy_graphs = [
        _build_graph_from_cli(
            Path(hp),
            fixed=bool(args.fixed),
            src=str(args.src),
            dst=str(args.dst),
            min_conf=float(args.min_conf),
            min_weight=float(args.min_weight),
            lcc=bool(args.lcc),
            input_kind=str(getattr(args, "input_kind", "auto")),
            mat_key=str(getattr(args, "mat_key", "")),
            sign_policy=str(getattr(args, "sign_policy", "abs")),
            threshold_mode=str(getattr(args, "threshold_mode", "density")),
            threshold_value=float(getattr(args, "threshold_value", 0.15)),
            shift=float(getattr(args, "shift", 0.0)),
        )
        for hp in args.healthy
    ]

    result = estimate_mix_frac_star(
        healthy_graphs,
        patient_metrics,
        target_metric=metrics if args.match_mode == "nearest" else metrics[0],
        match_mode=str(args.match_mode),
        steps=int(args.steps),
        seed=int(args.seed),
        eff_sources_k=int(args.eff_k),
        replace_from=str(args.replace_from),
        n_boot=int(args.n_boot),
    )
    payload = {
        "mode": "mixfrac",
        "patient": str(args.patient),
        "healthy": [str(x) for x in args.healthy],
        "result": result,
    }
    _write_json(payload, args.out)
    return 0



def _cmd_phenotype_match(args) -> int:
    """Compare degradation models HC -> SZ using a SZ metrics table."""
    attack_kinds = [x.strip() for x in str(args.attack_kinds).split(",") if x.strip()]
    if not attack_kinds:
        raise ValueError("--attack-kinds must contain at least one attack kind")

    metrics = [x.strip() for x in str(args.metrics).split(",") if x.strip()]
    if not metrics:
        raise ValueError("--metrics must contain at least one metric")

    metric_families = None
    distance_mode = str(getattr(args, "distance_mode", "raw"))
    if bool(getattr(args, "family_balance", False)) and distance_mode == "raw":
        distance_mode = "family_balanced"
    run_type = str(getattr(args, "run_type", "primary_run"))

    sz_group_metrics_df = _load_metrics_table(Path(args.sz_metrics))
    if args.hc_baseline_metrics:
        hc_baseline_metrics_df = _load_metrics_table(Path(args.hc_baseline_metrics))
    else:
        hc_baseline_metrics_df = _compute_baseline_metrics_df(args, list(args.hc))

    hc_paths = [Path(x) for x in args.hc]
    sid_source = str(getattr(args, "subject_id_source", "basename"))
    if sid_source == "fullpath":
        subject_ids = [str(p) for p in hc_paths]
    elif sid_source == "index":
        subject_ids = [f"hc_{i:04d}" for i, _ in enumerate(hc_paths)]
    else:
        subject_ids = [_safe_stem(p.stem) for p in hc_paths]

    effective_metrics_requested = effective_metrics_for_run_type(metrics, run_type=run_type, metric_families=metric_families)
    manifest = build_run_manifest(
        run_type=run_type,
        hc_paths=[str(p) for p in hc_paths],
        sz_metrics_path=str(args.sz_metrics),
        hc_baseline_metrics_path=str(getattr(args, "hc_baseline_metrics", "")),
        metric_list=effective_metrics_requested,
        metric_families=metric_families,
        attack_kinds=attack_kinds,
        steps=int(args.steps),
        frac=float(args.frac),
        seed=int(args.seed),
        distance_mode=distance_mode,
        module_resolution=float(args.module_resolution),
        recompute_modules=bool(args.recompute_modules),
        removal_mode=str(args.removal_mode),
        notes=str(getattr(args, "notes", "")),
    )
    preflight = run_phenotype_preflight(
        sz_group_metrics_df=sz_group_metrics_df,
        hc_baseline_metrics_df=hc_baseline_metrics_df,
        metrics=effective_metrics_requested,
        subject_ids=subject_ids,
        metric_families=metric_families,
    )
    if not preflight.get("ok", False):
        raise ValueError("Phenotype preflight failed: " + "; ".join(preflight.get("fatal_errors", [])))
    effective_metrics = list(preflight.get("metrics_effective", effective_metrics_requested))

    hc_graphs = _build_graphs_for_paths(args, [str(p) for p in hc_paths])
    compare_kwargs = dict(
        steps=int(args.steps), frac=float(args.frac), seed=int(args.seed), eff_sources_k=int(args.eff_k),
        compute_heavy_every=int(args.heavy_every), compute_curvature=bool(args.compute_curvature),
        curvature_sample_edges=int(args.curvature_sample_edges), noise_sigma_max=float(args.noise_sigma_max),
        keep_density_from_baseline=bool(args.keep_density_from_baseline), recompute_modules=bool(args.recompute_modules),
        module_resolution=float(args.module_resolution), removal_mode=str(args.removal_mode),
        subject_ids=subject_ids, distance_mode=distance_mode,
    )

    if bool(getattr(args, "control_suite", False)):
        resolution_tokens = [x.strip() for x in str(getattr(args, "modularity_resolutions", "0.5,1.0,1.5")).split(",") if x.strip()]
        resolutions = [float(x) for x in resolution_tokens] if resolution_tokens else [0.5, 1.0, 1.5]
        recompute_modes = [bool(int(x.strip())) for x in str(getattr(args, "modularity_recompute_options", "0,1")).split(",") if x.strip()]
        suite = run_control_suite(
            hc_graphs=hc_graphs,
            sz_group_metrics_df=sz_group_metrics_df,
            hc_baseline_metrics_df=hc_baseline_metrics_df,
            metrics=effective_metrics,
            attack_kinds=attack_kinds,
            metric_families=metric_families,
            compare_kwargs=compare_kwargs,
            modularity_resolutions=resolutions,
            modularity_recompute_options=recompute_modes,
            target_bootstrap_reps=int(getattr(args, "target_bootstrap_reps", 16)),
        )
        result = suite["primary_result"]
    else:
        result = compare_degradation_models(
            hc_graphs=hc_graphs,
            sz_group_metrics_df=sz_group_metrics_df,
            hc_baseline_metrics_df=hc_baseline_metrics_df,
            attack_kinds=attack_kinds,
            metrics=effective_metrics,
            metric_families=metric_families,
            **compare_kwargs,
        )
        suite = None

    compact = summarize_best_attack(result)
    paper_summary = build_paper_ready_summary(result)
    winners_df = result["winner_results"]
    subject_df = result["subject_results"]
    traj_df = result["trajectory_results"]

    if args.winners_out:
        Path(args.winners_out).parent.mkdir(parents=True, exist_ok=True)
        winners_df.to_csv(args.winners_out, index=False)
    if args.subject_out:
        Path(args.subject_out).parent.mkdir(parents=True, exist_ok=True)
        subject_df.to_csv(args.subject_out, index=False)
    if args.traj_out:
        Path(args.traj_out).parent.mkdir(parents=True, exist_ok=True)
        traj_df.to_csv(args.traj_out, index=False)
    if args.xlsx_out:
        export_phenotype_match_excel(result, args.xlsx_out)

    if getattr(args, "out_dir", ""):
        extra_tables = {
            "summary_attack.csv": paper_summary["summary_attack"],
            "summary_winners.csv": paper_summary["summary_winners"],
            "target_vector.csv": paper_summary["target_vector"],
            "scales.csv": paper_summary["scales"],
            "metric_families.csv": paper_summary["metric_families"],
            "family_summary.csv": paper_summary["family_summary"],
            "warning_flags.csv": suite.get("warning_flags", paper_summary["warning_flags"]) if suite is not None else paper_summary["warning_flags"],
            "stats_overall.csv": paper_summary["stats_overall"],
            "stats_pairwise.csv": paper_summary["stats_pairwise"],
            "stats_pairwise_matched_delta_density.csv": paper_summary.get("stats_pairwise_matched_delta_density", pd.DataFrame()),
            "stats_pairwise_matched_delta_total_weight.csv": paper_summary.get("stats_pairwise_matched_delta_total_weight", pd.DataFrame()),
            "stats_winners.csv": paper_summary["stats_winners"],
        }
        if suite is not None:
            extra_tables["control_suite_summary.csv"] = suite.get("suite_summary", pd.DataFrame())
            extra_tables["modularity_sensitivity_summary.csv"] = suite.get("modularity_sensitivity_summary", pd.DataFrame())
            extra_tables["target_stability_summary.csv"] = suite.get("target_stability_summary", pd.DataFrame())
            extra_tables["target_winner_stability_summary.csv"] = suite.get("target_winner_stability_summary", pd.DataFrame())
            extra_tables["target_attack_distance_stability_summary.csv"] = suite.get("target_attack_distance_stability_summary", pd.DataFrame())
            extra_tables["target_family_stability_summary.csv"] = suite.get("target_family_stability_summary", pd.DataFrame())
            extra_tables["target_scalar_stability_summary.csv"] = suite.get("target_scalar_stability_summary", pd.DataFrame())
            extra_tables["null_severity_density_summary.csv"] = suite.get("null_severity_density_summary", pd.DataFrame())
            extra_tables["null_severity_density_detail.csv"] = suite.get("null_severity_density_detail", pd.DataFrame())
            extra_tables["null_severity_total_weight_summary.csv"] = suite.get("null_severity_total_weight_summary", pd.DataFrame())
            extra_tables["null_severity_total_weight_detail.csv"] = suite.get("null_severity_total_weight_detail", pd.DataFrame())
            for name, df in suite.get("modularity_detail_tables", {}).items():
                extra_tables[name] = df
            for name, df in suite.get("target_stability_detail_tables", {}).items():
                extra_tables[name] = df
        save_run_bundle(
            out_dir=args.out_dir,
            result=result,
            manifest=manifest,
            preflight_report=preflight,
            extra_tables=extra_tables,
        )

    payload = {
        "mode": "phenotype-match",
        "run_type": manifest["run_type"],
        "distance_mode": distance_mode,
        "hc_n": int(len(hc_graphs)),
        "attack_kinds": attack_kinds,
        "metrics_requested": metrics,
        "metrics_effective": effective_metrics,
        "target_vector": result["target_vector"],
        "scales": result["scales"],
        "preflight": preflight,
        "winner_rows": winners_df.to_dict(orient="records"),
        "subject_rows_n": int(len(subject_df)),
        "trajectory_rows_n": int(len(traj_df)),
        "control_suite": bool(getattr(args, "control_suite", False)),
        "compact_summary": compact,
        "summary_attack_rows": paper_summary["summary_attack"].to_dict(orient="records"),
        "summary_winner_rows": paper_summary["summary_winners"].to_dict(orient="records"),
        "family_summary_rows": paper_summary["family_summary"].to_dict(orient="records"),
        "stats_overall_rows": paper_summary["stats_overall"].to_dict(orient="records"),
        "stats_pairwise_rows": paper_summary["stats_pairwise"].to_dict(orient="records"),
        "stats_pairwise_matched_delta_density_rows": paper_summary.get("stats_pairwise_matched_delta_density", pd.DataFrame()).to_dict(orient="records"),
        "stats_pairwise_matched_delta_total_weight_rows": paper_summary.get("stats_pairwise_matched_delta_total_weight", pd.DataFrame()).to_dict(orient="records"),
        "stats_winner_rows": paper_summary["stats_winners"].to_dict(orient="records"),
        "scalar_subject_rows_n": int(len(result.get("scalar_subject_results", pd.DataFrame()))),
        "scalar_winner_rows_n": int(len(result.get("scalar_winners", pd.DataFrame()))),
        "scalar_summary_rows": result.get("scalar_summary", pd.DataFrame()).to_dict(orient="records") if isinstance(result.get("scalar_summary", pd.DataFrame()), pd.DataFrame) else [],
        "control_suite_rows": suite.get("suite_summary", pd.DataFrame()).to_dict(orient="records") if suite is not None else [],
        "modularity_sensitivity_rows": suite.get("modularity_sensitivity_summary", pd.DataFrame()).to_dict(orient="records") if suite is not None else [],
        "target_stability_rows": suite.get("target_stability_summary", pd.DataFrame()).to_dict(orient="records") if suite is not None else [],
        "target_winner_stability_rows": suite.get("target_winner_stability_summary", pd.DataFrame()).to_dict(orient="records") if suite is not None else [],
        "target_attack_distance_stability_rows": suite.get("target_attack_distance_stability_summary", pd.DataFrame()).to_dict(orient="records") if suite is not None else [],
        "target_family_stability_rows": suite.get("target_family_stability_summary", pd.DataFrame()).to_dict(orient="records") if suite is not None else [],
        "target_scalar_stability_rows": suite.get("target_scalar_stability_summary", pd.DataFrame()).to_dict(orient="records") if suite is not None else [],
        "null_severity_density_rows": suite.get("null_severity_density_summary", pd.DataFrame()).to_dict(orient="records") if suite is not None else [],
        "null_severity_total_weight_rows": suite.get("null_severity_total_weight_summary", pd.DataFrame()).to_dict(orient="records") if suite is not None else [],
        "warning_flags_rows": suite.get("warning_flags", build_warning_flags(result)).to_dict(orient="records") if suite is not None else build_warning_flags(result).to_dict(orient="records"),
    }
    _write_json(payload, args.out)
    return 0


def _cmd_batch_metrics(args) -> int:
    """Run metrics for many inputs from a folder and save per-item + summary files."""
    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    per_item_dir = out_dir / "per_item"
    per_item_dir.mkdir(parents=True, exist_ok=True)

    files = _iter_input_files(
        input_dir,
        recursive=bool(args.recursive),
        pattern=str(args.pattern),
        limit=int(args.limit),
    )
    total = len(files)
    print(f"[batch-metrics] found {total} files", flush=True)

    def _job(idx: int, path: Path) -> dict:
        print(f"[{idx}/{total}] metrics :: {path.name}", flush=True)
        try:
            payload = _build_metrics_payload(args, path)
            item_name = _safe_stem(path.stem)
            export_metrics_payload(payload, str(per_item_dir / f"{item_name}.json"), "json")
            row = payload_to_flat_row(payload)
            row["status"] = "ok"
            row["error"] = ""
            return row
        except Exception as exc:  # pylint: disable=broad-except
            print(f"  ERROR :: {path.name} :: {type(exc).__name__}: {exc}", flush=True)
            return {
                "mode": "metrics",
                "input": str(path),
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }

    if int(args.n_jobs) == 1:
        rows = [_job(i, path) for i, path in enumerate(files, start=1)]
    else:
        rows = Parallel(n_jobs=int(args.n_jobs), prefer="processes")(delayed(_job)(i, path) for i, path in enumerate(files, start=1))

    df = pd.DataFrame(rows)
    summary_csv = Path(args.summary_out) if args.summary_out else out_dir / "metrics_summary.csv"
    summary_xlsx = summary_csv.with_suffix(".xlsx")
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(summary_csv, index=False)
    with pd.ExcelWriter(summary_xlsx, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="metrics", index=False)

    print(f"[batch-metrics] saved {len(df)} rows -> {summary_csv}", flush=True)
    print(f"[batch-metrics] per-item JSON -> {per_item_dir}", flush=True)
    return 0


def _cmd_batch_attack(args) -> int:
    """Run the same attack setup over many inputs from a folder."""
    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    json_dir = out_dir / "payloads"
    hist_dir = out_dir / "histories"
    json_dir.mkdir(parents=True, exist_ok=True)
    hist_dir.mkdir(parents=True, exist_ok=True)

    files = _iter_input_files(
        input_dir,
        recursive=bool(args.recursive),
        pattern=str(args.pattern),
        limit=int(args.limit),
    )
    total = len(files)
    print(f"[batch-attack] found {total} files", flush=True)

    def _job(idx: int, path: Path) -> dict:
        print(f"[{idx}/{total}] attack :: {path.name}", flush=True)
        try:
            payload, history = _run_attack_payload(args, path)
            item_name = _safe_stem(path.stem)
            export_metrics_payload(payload, str(json_dir / f"{item_name}.json"), "json")
            history.to_csv(hist_dir / f"{item_name}.csv", index=False)
            return {
                "mode": "attack",
                "input": str(path),
                "family": payload.get("family"),
                "kind": payload.get("kind"),
                "status": "ok",
                "error": "",
                **payload.get("summary", {}),
                **{f"final__{k}": v for k, v in payload.get("final_row", {}).items()},
            }
        except Exception as exc:  # pylint: disable=broad-except
            print(f"  ERROR :: {path.name} :: {type(exc).__name__}: {exc}", flush=True)
            return {
                "mode": "attack",
                "input": str(path),
                "family": str(args.family),
                "kind": str(args.kind),
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }

    if int(args.n_jobs) == 1:
        rows = [_job(i, path) for i, path in enumerate(files, start=1)]
    else:
        rows = Parallel(n_jobs=int(args.n_jobs), prefer="processes")(delayed(_job)(i, path) for i, path in enumerate(files, start=1))

    df = pd.DataFrame(rows)
    summary_csv = Path(args.summary_out) if args.summary_out else out_dir / "attack_summary.csv"
    summary_xlsx = summary_csv.with_suffix(".xlsx")
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(summary_csv, index=False)
    with pd.ExcelWriter(summary_xlsx, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="attacks", index=False)

    print(f"[batch-attack] saved {len(df)} rows -> {summary_csv}", flush=True)
    print(f"[batch-attack] payloads -> {json_dir}", flush=True)
    print(f"[batch-attack] histories -> {hist_dir}", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build root parser with subcommands for local runs."""
    parser = argparse.ArgumentParser(
        prog="kodiklab",
        description="Kodik Lab local runner: metrics / attack / mixfrac / phenotype-match / batch-metrics / batch-attack.",
    )
    subparsers = parser.add_subparsers(dest="command")

    p_metrics = subparsers.add_parser("metrics", help="Compute graph metrics for one graph")
    p_metrics.add_argument("input", type=str, help="Path to edge list or matrix file (.mat/.npy/.npz)")
    _add_common_graph_args(p_metrics)
    p_metrics.add_argument("--seed", type=int, default=settings.DEFAULT_SEED)
    p_metrics.add_argument("--eff-k", type=int, default=32)
    p_metrics.add_argument("--compute-curvature", action="store_true")
    p_metrics.add_argument("--curvature-sample-edges", type=int, default=120)
    p_metrics.set_defaults(compute_heavy=True)
    p_metrics.add_argument("--compute-heavy", dest="compute_heavy", action="store_true", help="Enable heavy metrics (default: enabled)")
    p_metrics.add_argument("--no-compute-heavy", dest="compute_heavy", action="store_false", help="Disable heavy metrics for faster runs")
    p_metrics.add_argument("--skip-spectral", action="store_true")
    p_metrics.add_argument("--diameter-samples", type=int, default=16)
    p_metrics.add_argument("--out", type=str, default="-")
    p_metrics.add_argument(
        "--out-format",
        type=str,
        default="auto",
        choices=["auto", "json", "csv", "xlsx"],
        help="Output format. auto = infer from --out extension",
    )

    p_attack = subparsers.add_parser("attack", help="Run node/edge/mix attack locally")
    p_attack.add_argument("input", type=str, help="Path to edge list or matrix file (.mat/.npy/.npz)")
    _add_common_graph_args(p_attack)
    p_attack.add_argument("--family", choices=["node", "edge", "mix", "degradation"], required=True)
    p_attack.add_argument("--kind", type=str, required=True)
    p_attack.add_argument("--frac", type=float, default=0.5)
    p_attack.add_argument("--steps", type=int, default=30)
    p_attack.add_argument("--seed", type=int, default=settings.DEFAULT_SEED)
    p_attack.add_argument("--eff-k", type=int, default=32)
    p_attack.add_argument("--heavy-every", type=int, default=2)
    p_attack.add_argument("--fast-mode", action="store_true")
    p_attack.add_argument("--compute-curvature", action="store_true")
    p_attack.add_argument("--curvature-sample-edges", type=int, default=80)
    p_attack.add_argument("--replace-from", choices=["ER", "CFG"], default="CFG")
    p_attack.add_argument("--alpha-rewire", type=float, default=0.6)
    p_attack.add_argument("--beta-replace", type=float, default=0.4)
    p_attack.add_argument("--swaps-per-edge", type=float, default=0.5)
    p_attack.add_argument("--out", type=str, default="-")
    p_attack.add_argument("--history-out", type=str, default="")
    p_attack.add_argument("--noise-sigma-max", type=float, default=0.5)
    p_attack.add_argument("--keep-density-from-baseline", action="store_true")
    p_attack.add_argument("--recompute-modules", action="store_true")
    p_attack.add_argument("--module-resolution", type=float, default=1.0)
    p_attack.add_argument("--removal-mode", choices=["random", "weak_weight", "strong_weight"], default="random")

    p_mix = subparsers.add_parser("mixfrac", help="Estimate mix_frac* from patient + healthy graphs")
    p_mix.add_argument("--patient", required=True, type=str, help="Patient graph edge list or matrix file")
    p_mix.add_argument(
        "--healthy",
        required=True,
        nargs="+",
        type=str,
        help="One or more healthy/reference graph files",
    )
    _add_common_graph_args(p_mix)
    p_mix.add_argument("--metrics", type=str, default="kappa_mean,kappa_frac_negative,clustering")
    p_mix.add_argument("--match-mode", choices=["nearest", "interpolate"], default="nearest")
    p_mix.add_argument("--steps", type=int, default=20)
    p_mix.add_argument("--seed", type=int, default=settings.DEFAULT_SEED)
    p_mix.add_argument("--eff-k", type=int, default=32)
    p_mix.add_argument("--replace-from", choices=["ER", "CFG"], default="CFG")
    p_mix.add_argument("--curvature-sample-edges", type=int, default=120)
    p_mix.add_argument("--n-boot", type=int, default=1000)
    p_mix.add_argument("--out", type=str, default="-")


    p_pm = subparsers.add_parser("phenotype-match", help="HC -> SZ degradation-model matching")
    p_pm.add_argument("--hc", required=True, nargs="+", type=str, help="Healthy/control graph files")
    p_pm.add_argument("--sz-metrics", required=True, type=str, help="CSV/XLSX table with SZ group metrics")
    p_pm.add_argument("--hc-baseline-metrics", type=str, default="", help="Optional CSV/XLSX table with HC baseline metrics")
    _add_common_graph_args(p_pm)
    p_pm.add_argument("--attack-kinds", type=str, default="weight_noise,inter_module_removal,intra_module_removal,weak_edges_by_weight,strong_edges_by_weight,mix_default,mix_degree_preserving")
    p_pm.add_argument("--metrics", type=str, default="density,clustering,mod,l2_lcc,H_rw,fragility_H,eff_w,lcc_frac")
    p_pm.add_argument("--steps", type=int, default=12)
    p_pm.add_argument("--frac", type=float, default=0.5)
    p_pm.add_argument("--seed", type=int, default=settings.DEFAULT_SEED)
    p_pm.add_argument("--eff-k", type=int, default=32)
    p_pm.add_argument("--heavy-every", type=int, default=2)
    p_pm.add_argument("--compute-curvature", action="store_true")
    p_pm.add_argument("--curvature-sample-edges", type=int, default=80)
    p_pm.add_argument("--noise-sigma-max", type=float, default=0.5)
    p_pm.add_argument("--keep-density-from-baseline", action="store_true")
    p_pm.add_argument("--recompute-modules", action="store_true")
    p_pm.add_argument("--module-resolution", type=float, default=1.0)
    p_pm.add_argument("--removal-mode", choices=["random", "weak_weight", "strong_weight"], default="random")
    p_pm.add_argument("--distance-mode", type=str, default="raw", choices=["raw", "family_balanced"], help="Distance aggregation mode")
    p_pm.add_argument("--family-balance", action="store_true", help="Shortcut for --distance-mode family_balanced")
    p_pm.add_argument("--subject-id-source", type=str, default="basename", choices=["basename", "fullpath", "index"], help="How to derive subject IDs for HC inputs")
    p_pm.add_argument("--run-type", type=str, default="primary_run")
    p_pm.add_argument("--notes", type=str, default="")
    p_pm.add_argument("--control-suite", action="store_true", help="Run primary + density-control + random-edge null + modularity sensitivity suite")
    p_pm.add_argument("--modularity-resolutions", type=str, default="0.5,1.0,1.5", help="Comma-separated module resolutions for control suite")
    p_pm.add_argument("--modularity-recompute-options", type=str, default="0,1", help="Comma-separated booleans (0/1) for recompute-modules in control suite")
    p_pm.add_argument("--target-bootstrap-reps", type=int, default=16, help="Bootstrap repetitions for target stability control suite")
    p_pm.add_argument("--out", type=str, default="-")
    p_pm.add_argument("--out-dir", type=str, default="", help="Optional directory for manifest, preflight, and CSV bundle")
    p_pm.add_argument("--winners-out", type=str, default="")
    p_pm.add_argument("--subject-out", type=str, default="")
    p_pm.add_argument("--traj-out", type=str, default="")
    p_pm.add_argument("--xlsx-out", type=str, default="", help="Optional XLSX export with all phenotype-match tables")

    p_batch_m = subparsers.add_parser("batch-metrics", help="Run metrics for all graphs in a folder")
    p_batch_m.add_argument("--input-dir", required=True, type=str)
    p_batch_m.add_argument("--out-dir", type=str, default="batch_metrics_out")
    p_batch_m.add_argument("--summary-out", type=str, default="")
    p_batch_m.add_argument("--pattern", type=str, default="*")
    p_batch_m.add_argument("--recursive", action="store_true")
    p_batch_m.add_argument("--limit", type=int, default=0)
    p_batch_m.add_argument("--n-jobs", type=int, default=1)
    _add_common_graph_args(p_batch_m)
    p_batch_m.add_argument("--seed", type=int, default=settings.DEFAULT_SEED)
    p_batch_m.add_argument("--eff-k", type=int, default=32)
    p_batch_m.add_argument("--compute-curvature", action="store_true")
    p_batch_m.add_argument("--curvature-sample-edges", type=int, default=120)
    p_batch_m.set_defaults(compute_heavy=True)
    p_batch_m.add_argument("--compute-heavy", dest="compute_heavy", action="store_true", help="Enable heavy metrics (default: enabled)")
    p_batch_m.add_argument("--no-compute-heavy", dest="compute_heavy", action="store_false", help="Disable heavy metrics for faster runs")
    p_batch_m.add_argument("--skip-spectral", action="store_true")
    p_batch_m.add_argument("--diameter-samples", type=int, default=16)

    p_batch_a = subparsers.add_parser("batch-attack", help="Run the same attack for all graphs in a folder")
    p_batch_a.add_argument("--input-dir", required=True, type=str)
    p_batch_a.add_argument("--out-dir", type=str, default="batch_attack_out")
    p_batch_a.add_argument("--summary-out", type=str, default="")
    p_batch_a.add_argument("--pattern", type=str, default="*")
    p_batch_a.add_argument("--recursive", action="store_true")
    p_batch_a.add_argument("--limit", type=int, default=0)
    p_batch_a.add_argument("--n-jobs", type=int, default=1)
    _add_common_graph_args(p_batch_a)
    p_batch_a.add_argument("--family", choices=["node", "edge", "mix", "degradation"], required=True)
    p_batch_a.add_argument("--kind", type=str, required=True)
    p_batch_a.add_argument("--frac", type=float, default=0.5)
    p_batch_a.add_argument("--steps", type=int, default=30)
    p_batch_a.add_argument("--seed", type=int, default=settings.DEFAULT_SEED)
    p_batch_a.add_argument("--eff-k", type=int, default=32)
    p_batch_a.add_argument("--heavy-every", type=int, default=2)
    p_batch_a.add_argument("--fast-mode", action="store_true")
    p_batch_a.add_argument("--compute-curvature", action="store_true")
    p_batch_a.add_argument("--curvature-sample-edges", type=int, default=80)
    p_batch_a.add_argument("--replace-from", choices=["ER", "CFG"], default="CFG")
    p_batch_a.add_argument("--alpha-rewire", type=float, default=0.6)
    p_batch_a.add_argument("--beta-replace", type=float, default=0.4)
    p_batch_a.add_argument("--swaps-per-edge", type=float, default=0.5)
    p_batch_a.add_argument("--noise-sigma-max", type=float, default=0.5)
    p_batch_a.add_argument("--keep-density-from-baseline", action="store_true")
    p_batch_a.add_argument("--recompute-modules", action="store_true")
    p_batch_a.add_argument("--module-resolution", type=float, default=1.0)
    p_batch_a.add_argument("--removal-mode", choices=["random", "weak_weight", "strong_weight"], default="random")

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point with backwards-compatible default to `metrics`."""
    argv = list(sys.argv[1:] if argv is None else argv)
    subcommands = {"metrics", "attack", "mixfrac", "phenotype-match", "batch-metrics", "batch-attack"}

    if argv and argv[0] not in subcommands and not argv[0].startswith("-"):
        argv = ["metrics"] + argv

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command in (None, "metrics"):
        return _cmd_metrics(args)
    if args.command == "attack":
        return _cmd_attack(args)
    if args.command == "mixfrac":
        return _cmd_mixfrac(args)
    if args.command == "phenotype-match":
        return _cmd_phenotype_match(args)
    if args.command == "batch-metrics":
        return _cmd_batch_metrics(args)
    if args.command == "batch-attack":
        return _cmd_batch_attack(args)

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
