"""
Batch pipeline: папка матриц → таблица метрик.

Использование:
    python -m src.batch \
        --data-dir ./cobre_matrices/ \
        --meta ./cobre_meta.csv \
        --sign-policy abs \
        --densities 0.05,0.10,0.15,0.20,0.25,0.30 \
        --output results.csv \
        --compute-curvature
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd

from .core.graph_ops import calculate_metrics
from .matrix_import import load_matrix, matrix_to_graph


def run_single_subject(
    matrix_path: Path,
    *,
    sign_policy: str = "abs",
    threshold_mode: str = "density",
    threshold_value: float = 0.15,
    labels: Optional[list] = None,
    compute_curvature: bool = True,
    curvature_sample_edges: int = 150,
    seed: int = 42,
    eff_sources_k: int = 20,
) -> dict:
    """Один субъект: матрица → dict метрик."""
    corr = load_matrix(matrix_path)
    graph = matrix_to_graph(
        corr,
        sign_policy=sign_policy,
        threshold_mode=threshold_mode,
        threshold_value=threshold_value,
        labels=labels,
        use_lcc=True,
    )
    return calculate_metrics(
        graph,
        eff_sources_k=eff_sources_k,
        seed=seed,
        compute_curvature=compute_curvature,
        curvature_sample_edges=curvature_sample_edges,
    )


def run_batch(
    data_dir: Path,
    meta_csv: Optional[Path] = None,
    *,
    sign_policy: str = "abs",
    densities: Optional[list[float]] = None,
    compute_curvature: bool = True,
    curvature_sample_edges: int = 150,
    seed: int = 42,
    eff_sources_k: int = 20,
) -> pd.DataFrame:
    """
    Batch-обработка всех матриц в папке.

    Parameters
    ----------
    data_dir : папка с .npy/.mat/.csv матрицами
    meta_csv : CSV с колонками [filename, group, ...] — если есть
    densities : список порогов (по умолчанию [0.15])
    """
    if densities is None:
        densities = [0.15]

    data_dir = Path(data_dir)
    matrix_files = sorted(
        path for path in data_dir.iterdir() if path.suffix in (".npy", ".npz", ".mat", ".csv")
    )
    if not matrix_files:
        raise FileNotFoundError(f"No matrix files in {data_dir}")

    meta = None
    if meta_csv is not None:
        meta = pd.read_csv(meta_csv)
        if "filename" not in meta.columns:
            raise ValueError("meta CSV must have 'filename' column")

    rows = []
    total = len(matrix_files) * len(densities)
    done = 0

    for matrix_file in matrix_files:
        for density in densities:
            done += 1
            print(f"[{done}/{total}] {matrix_file.name} @ density={density:.2f}", flush=True)
            try:
                metrics = run_single_subject(
                    matrix_file,
                    sign_policy=sign_policy,
                    threshold_mode="density",
                    threshold_value=density,
                    compute_curvature=compute_curvature,
                    curvature_sample_edges=curvature_sample_edges,
                    seed=seed,
                    eff_sources_k=eff_sources_k,
                )
                row = {"filename": matrix_file.name, "density": density, **metrics}
            except Exception as error:  # pylint: disable=broad-except
                print(f"  ERROR: {error}", flush=True)
                row = {"filename": matrix_file.name, "density": density, "error": str(error)}

            # Дополняем метаданными субъекта (если есть запись по filename).
            if meta is not None:
                meta_row = meta[meta["filename"] == matrix_file.name]
                if not meta_row.empty:
                    for column in meta_row.columns:
                        if column != "filename":
                            row[column] = meta_row.iloc[0][column]

            rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    """CLI-обёртка для запуска batch-пайплайна."""
    parser = argparse.ArgumentParser(description="Batch metrics pipeline")
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--meta", type=Path, default=None)
    parser.add_argument("--sign-policy", default="abs", choices=["abs", "positive_only", "shift"])
    parser.add_argument("--densities", default="0.15", help="Comma-separated densities")
    parser.add_argument("--output", default="batch_results.csv", type=Path)
    parser.add_argument("--compute-curvature", action="store_true")
    parser.add_argument("--curvature-edges", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    densities = [float(value.strip()) for value in args.densities.split(",")]
    df = run_batch(
        args.data_dir,
        meta_csv=args.meta,
        sign_policy=args.sign_policy,
        densities=densities,
        compute_curvature=args.compute_curvature,
        curvature_sample_edges=args.curvature_edges,
        seed=args.seed,
    )
    df.to_csv(args.output, index=False)
    print(f"Saved {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()
