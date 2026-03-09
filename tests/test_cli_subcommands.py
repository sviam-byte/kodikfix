import json
from pathlib import Path

import pandas as pd

from src import cli


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


def test_cli_metrics_backward_compatible_no_subcommand(tmp_path: Path):
    data = tmp_path / "graph.csv"
    out = tmp_path / "metrics.json"
    _write_edges(data)

    code = cli.main([str(data), "--src", "src", "--dst", "dst", "--out", str(out)])
    assert code == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["mode"] == "metrics"
    assert "metrics" in payload


def test_cli_attack_and_mixfrac_subcommands(tmp_path: Path):
    patient = tmp_path / "patient.csv"
    hc1 = tmp_path / "hc1.csv"
    hc2 = tmp_path / "hc2.csv"
    attack_out = tmp_path / "attack.json"
    history_out = tmp_path / "attack_history.csv"
    mix_out = tmp_path / "mixfrac.json"
    for p in [patient, hc1, hc2]:
        _write_edges(p)

    code_attack = cli.main(
        [
            "attack",
            str(patient),
            "--family",
            "mix",
            "--kind",
            "hrish_mix",
            "--steps",
            "5",
            "--out",
            str(attack_out),
            "--history-out",
            str(history_out),
        ]
    )
    assert code_attack == 0
    assert attack_out.exists()
    assert history_out.exists()

    code_mix = cli.main(
        [
            "mixfrac",
            "--patient",
            str(patient),
            "--healthy",
            str(hc1),
            str(hc2),
            "--metrics",
            "clustering",
            "--match-mode",
            "interpolate",
            "--steps",
            "5",
            "--out",
            str(mix_out),
        ]
    )
    assert code_mix == 0
    payload_mix = json.loads(mix_out.read_text(encoding="utf-8"))
    assert payload_mix["mode"] == "mixfrac"
    assert "result" in payload_mix


def test_cli_metrics_out_format_csv(tmp_path: Path):
    data = tmp_path / "graph.csv"
    out = tmp_path / "metrics.csv"
    _write_edges(data)

    code = cli.main(["metrics", str(data), "--out", str(out), "--out-format", "csv"])
    assert code == 0
    df = pd.read_csv(out)
    assert "summary" in df.columns or "summary__N" in df.columns
    assert "settings__seed" in df.columns


def test_cli_batch_metrics_and_no_compute_heavy(tmp_path: Path):
    in_dir = tmp_path / "inputs"
    in_dir.mkdir()
    _write_edges(in_dir / "g1.csv")
    _write_edges(in_dir / "g2.csv")

    out_dir = tmp_path / "batch_out"
    code = cli.main(
        [
            "batch-metrics",
            "--input-dir",
            str(in_dir),
            "--out-dir",
            str(out_dir),
            "--pattern",
            "*.csv",
            "--no-compute-heavy",
        ]
    )

    assert code == 0
    summary_csv = out_dir / "metrics_summary.csv"
    assert summary_csv.exists()
    df = pd.read_csv(summary_csv)
    assert len(df) == 2
    assert set(df["status"].tolist()) == {"ok"}


def test_cli_phenotype_match_subcommand(tmp_path: Path):
    hc1 = tmp_path / "hc1.csv"
    hc2 = tmp_path / "hc2.csv"
    out = tmp_path / "pm.json"
    winners = tmp_path / "winners.csv"
    subject = tmp_path / "subject.csv"
    traj = tmp_path / "traj.csv"
    xlsx = tmp_path / "pm.xlsx"

    for p in [hc1, hc2]:
        _write_edges(p)

    sz_df = pd.DataFrame(
        [
            {
                "density": 0.5,
                "clustering": 0.3,
                "mod": 0.2,
                "l2_lcc": 0.1,
                "H_rw": 1.0,
                "fragility_H": 0.4,
                "eff_w": 0.5,
                "lcc_frac": 1.0,
            }
        ]
    )
    sz_path = tmp_path / "sz_metrics.csv"
    sz_df.to_csv(sz_path, index=False)

    code = cli.main(
        [
            "phenotype-match",
            "--hc",
            str(hc1),
            str(hc2),
            "--sz-metrics",
            str(sz_path),
            "--attack-kinds",
            "weight_noise,inter_module_removal",
            "--metrics",
            "density,clustering,mod,l2_lcc,H_rw,fragility_H,eff_w,lcc_frac",
            "--steps",
            "4",
            "--out",
            str(out),
            "--winners-out",
            str(winners),
            "--subject-out",
            str(subject),
            "--traj-out",
            str(traj),
            "--xlsx-out",
            str(xlsx),
        ]
    )

    assert code == 0
    assert out.exists()
    assert winners.exists()
    assert subject.exists()
    assert traj.exists()
    assert xlsx.exists()

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["mode"] == "phenotype-match"
    assert "compact_summary" in payload
    assert "summary_attack_rows" in payload
    assert "summary_winner_rows" in payload
