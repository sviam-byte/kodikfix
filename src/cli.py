from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from .attacks import run_attack, run_edge_attack
from .attacks_mix import run_mix_attack
from .config import settings
from .exporters import export_metrics_payload, infer_output_format, payload_to_dataframe
from .graph_build import build_graph_from_edges, graph_summary, lcc_subgraph
from .metrics import calculate_metrics
from .mix_frac_estimator import estimate_mix_frac_star
from .preprocess import coerce_fixed_format, filter_edges


def _load_table(path: Path) -> pd.DataFrame:
    """Load a CSV/Excel file into a DataFrame."""
    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(path)
    return pd.read_csv(path, sep=None, engine="python", encoding_errors="replace")


def _json_default(obj):
    """JSON fallback serializer for numpy/pandas scalar objects."""
    try:
        return float(obj)
    except Exception:
        return str(obj)


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
):
    """Build graph from file using CLI filters and optional LCC extraction."""
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


def _cmd_metrics(args) -> int:
    """Run scalar metrics command."""
    path = Path(args.input)
    graph = _build_graph_from_cli(
        path,
        fixed=bool(args.fixed),
        src=str(args.src),
        dst=str(args.dst),
        min_conf=float(args.min_conf),
        min_weight=float(args.min_weight),
        lcc=bool(args.lcc),
    )
    met = calculate_metrics(
        graph,
        int(args.eff_k),
        int(args.seed),
        bool(args.compute_curvature),
        curvature_sample_edges=int(args.curvature_sample_edges),
        compute_heavy=True,
    )
    payload = {
        "mode": "metrics",
        "input": str(path),
        "summary": graph_summary(graph),
        "settings": {
            "seed": int(args.seed),
            "weight_policy": settings.WEIGHT_POLICY,
            "weight_eps": settings.WEIGHT_EPS,
        },
        "metrics": met,
    }

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
    path = Path(args.input)
    graph = _build_graph_from_cli(
        path,
        fixed=bool(args.fixed),
        src=str(args.src),
        dst=str(args.dst),
        min_conf=float(args.min_conf),
        min_weight=float(args.min_weight),
        lcc=bool(args.lcc),
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
    _write_json(payload, args.out)
    if args.history_out:
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


def build_parser() -> argparse.ArgumentParser:
    """Build root parser with subcommands for local runs."""
    parser = argparse.ArgumentParser(
        prog="kodiklab",
        description="Kodik Lab local runner: metrics / attack / mixfrac.",
    )
    subparsers = parser.add_subparsers(dest="command")

    p_metrics = subparsers.add_parser("metrics", help="Compute graph metrics for one graph")
    p_metrics.add_argument("input", type=str, help="Path to CSV/Excel edge list")
    _add_common_graph_args(p_metrics)
    p_metrics.add_argument("--seed", type=int, default=settings.DEFAULT_SEED)
    p_metrics.add_argument("--eff-k", type=int, default=32)
    p_metrics.add_argument("--compute-curvature", action="store_true")
    p_metrics.add_argument("--curvature-sample-edges", type=int, default=120)
    p_metrics.add_argument("--out", type=str, default="-")
    p_metrics.add_argument(
        "--out-format",
        type=str,
        default="auto",
        choices=["auto", "json", "csv", "xlsx"],
        help="Output format. auto = infer from --out extension",
    )

    p_attack = subparsers.add_parser("attack", help="Run node/edge/mix attack locally")
    p_attack.add_argument("input", type=str, help="Path to CSV/Excel edge list")
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
    p_mix.add_argument("--patient", required=True, type=str, help="Patient graph CSV/Excel")
    p_mix.add_argument(
        "--healthy",
        required=True,
        nargs="+",
        type=str,
        help="One or more healthy/reference graph CSV/Excel files",
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

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point with backwards-compatible default to `metrics`."""
    argv = list(sys.argv[1:] if argv is None else argv)
    subcommands = {"metrics", "attack", "mixfrac"}

    # Backward compatibility: old form `python -m src.cli data.csv ...`.
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

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
