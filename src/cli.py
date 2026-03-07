from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd
from joblib import Parallel, delayed

from .attacks import run_attack, run_edge_attack
from .attacks_mix import run_mix_attack
from .config import settings
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
    met = calculate_metrics(
        graph,
        int(args.eff_k),
        int(args.seed),
        bool(args.compute_curvature),
        curvature_sample_edges=int(args.curvature_sample_edges),
        compute_heavy=bool(getattr(args, "compute_heavy", True)),
        skip_spectral=bool(getattr(args, "skip_spectral", False)),
        diameter_samples=int(getattr(args, "diameter_samples", 16)),
    )
    return {
        "mode": "metrics",
        "input": str(path),
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
    else:
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

    payload = {
        "mode": "attack",
        "family": family,
        "kind": str(args.kind),
        "input": str(path),
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
        },
        "aux": aux,
        "final_row": history.iloc[-1].to_dict() if not history.empty else {},
    }
    return payload, history


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
        description="Kodik Lab local runner: metrics / attack / mixfrac / batch-metrics / batch-attack.",
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
    p_attack.add_argument("--family", choices=["node", "edge", "mix"], required=True)
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
    p_batch_a.add_argument("--family", choices=["node", "edge", "mix"], required=True)
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

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point with backwards-compatible default to `metrics`."""
    argv = list(sys.argv[1:] if argv is None else argv)
    subcommands = {"metrics", "attack", "mixfrac", "batch-metrics", "batch-attack"}

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
    if args.command == "batch-metrics":
        return _cmd_batch_metrics(args)
    if args.command == "batch-attack":
        return _cmd_batch_attack(args)

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
