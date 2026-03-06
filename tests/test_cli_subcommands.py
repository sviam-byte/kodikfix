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
