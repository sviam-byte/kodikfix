from pathlib import Path

import pandas as pd

from src.batch_ops import build_ui_args, make_run_dir, run_batch_attack, run_batch_metrics


def _write_edges(path: Path) -> None:
    df = pd.DataFrame(
        {
            "src": ["a", "b", "c", "d", "a", "c"],
            "dst": ["b", "c", "d", "a", "c", "a"],
            "weight": [1, 1, 1, 1, 1, 1],
            "confidence": [1, 1, 1, 1, 1, 1],
        }
    )
    df.to_csv(path, index=False)


def test_run_batch_metrics_from_ui_args(tmp_path: Path):
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    _write_edges(input_dir / "graph.csv")

    out_root = tmp_path / "runs"
    run_dir = make_run_dir(out_root, mode="batch_metrics", seed=123, run_label="ui")
    args = build_ui_args(input_dir=str(input_dir), out_dir=str(run_dir), pattern="*.csv")

    out_dir, df = run_batch_metrics(args)

    assert out_dir == run_dir
    assert len(df) == 1
    assert (run_dir / "metrics_summary.csv").exists()
    assert (run_dir / "metrics_summary.xlsx").exists()
    assert (run_dir / "manifest.json").exists()


def test_run_batch_attack_from_ui_args(tmp_path: Path):
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    _write_edges(input_dir / "graph.csv")

    out_root = tmp_path / "runs"
    run_dir = make_run_dir(out_root, mode="batch_attack", seed=123, run_label="ui")
    args = build_ui_args(
        input_dir=str(input_dir),
        out_dir=str(run_dir),
        pattern="*.csv",
        family="node",
        kind="degree",
        steps=3,
    )

    out_dir, df = run_batch_attack(args)

    assert out_dir == run_dir
    assert len(df) == 1
    assert (run_dir / "attack_summary.csv").exists()
    assert (run_dir / "attack_summary.xlsx").exists()
    assert (run_dir / "manifest.json").exists()
