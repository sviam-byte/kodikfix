from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .config import settings
from .graph_build import build_graph_from_edges, graph_summary, lcc_subgraph
from .metrics import calculate_metrics
from .preprocess import coerce_fixed_format, filter_edges


def _load_table(path: Path) -> pd.DataFrame:
    """Load a CSV/Excel file into a DataFrame."""
    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(path)
    return pd.read_csv(path, sep=None, engine="python", encoding_errors="replace")


def main(argv: list[str] | None = None) -> int:
    """Run metrics offline for reproducible CLI experiments."""
    p = argparse.ArgumentParser(
        prog="kodiklab",
        description="Kodik Lab: offline runner (metrics / summary) for reproducible experiments.",
    )
    p.add_argument("input", type=str, help="Path to CSV/Excel edge list")
    p.add_argument(
        "--fixed",
        action="store_true",
        help="Use fixed-format loader (src,dst at col 0/1; conf at 8; weight at 9)",
    )
    p.add_argument("--src", type=str, default="src", help="Source column name")
    p.add_argument("--dst", type=str, default="dst", help="Target column name")
    p.add_argument("--min-conf", type=float, default=0.0)
    p.add_argument("--min-weight", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=settings.DEFAULT_SEED)
    p.add_argument("--eff-k", type=int, default=32)
    p.add_argument("--lcc", action="store_true", help="Restrict to largest connected component")
    p.add_argument("--out", type=str, default="-", help="Output path for metrics JSON (default: stdout)")

    args = p.parse_args(argv)

    path = Path(args.input)
    df_any = _load_table(path)

    if args.fixed:
        df_edges, meta = coerce_fixed_format(df_any)
        src_col = meta["src_col"]
        dst_col = meta["dst_col"]
    else:
        df_edges = df_any.copy()
        src_col = args.src
        dst_col = args.dst
        df_edges = filter_edges(df_edges, src_col, dst_col, args.min_conf, args.min_weight)

    G = build_graph_from_edges(df_edges, src_col, dst_col, strict=True)
    if args.lcc:
        G = lcc_subgraph(G)

    met = calculate_metrics(G, int(args.eff_k), int(args.seed), False, compute_heavy=True)
    payload = {
        "summary": graph_summary(G),
        "settings": {
            "seed": int(args.seed),
            "weight_policy": settings.WEIGHT_POLICY,
            "weight_eps": settings.WEIGHT_EPS,
        },
        "metrics": met,
    }

    txt = json.dumps(payload, ensure_ascii=False, indent=2, default=float)
    if args.out == "-":
        print(txt)
    else:
        Path(args.out).write_text(txt, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
