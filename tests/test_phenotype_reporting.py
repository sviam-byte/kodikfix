from pathlib import Path

import pandas as pd

from src.phenotype_reporting import (
    build_attack_summary,
    build_winner_summary,
    export_phenotype_match_excel,
)


def _toy_result():
    subject_df = pd.DataFrame(
        [
            {"subject_idx": 0, "attack_kind": "inter_module_removal", "best_step": 3, "best_damage_frac": 0.3, "best_distance": 1.0},
            {"subject_idx": 1, "attack_kind": "inter_module_removal", "best_step": 4, "best_damage_frac": 0.4, "best_distance": 1.2},
            {"subject_idx": 0, "attack_kind": "weight_noise", "best_step": 2, "best_damage_frac": 0.2, "best_distance": 1.5},
            {"subject_idx": 1, "attack_kind": "weight_noise", "best_step": 2, "best_damage_frac": 0.2, "best_distance": 1.7},
        ]
    )
    winners_df = pd.DataFrame(
        [
            {"subject_idx": 0, "attack_kind": "inter_module_removal", "best_step": 3, "best_damage_frac": 0.3, "best_distance": 1.0},
            {"subject_idx": 1, "attack_kind": "inter_module_removal", "best_step": 4, "best_damage_frac": 0.4, "best_distance": 1.2},
        ]
    )
    traj_df = pd.DataFrame(
        [
            {"subject_idx": 0, "attack_kind": "inter_module_removal", "step": 0, "distance_to_target": 2.0},
            {"subject_idx": 0, "attack_kind": "inter_module_removal", "step": 3, "distance_to_target": 1.0},
        ]
    )
    return {
        "subject_results": subject_df,
        "winner_results": winners_df,
        "trajectory_results": traj_df,
        "target_vector": {"density": 0.4, "clustering": 0.3},
        "scales": {"density": 0.1, "clustering": 0.2},
    }


def test_build_attack_summary():
    result = _toy_result()
    out = build_attack_summary(result["subject_results"])
    assert not out.empty
    assert "attack_kind" in out.columns
    assert "best_distance_median" in out.columns


def test_build_winner_summary():
    result = _toy_result()
    out = build_winner_summary(result["winner_results"], total_subjects=2)
    assert not out.empty
    assert out.iloc[0]["attack_kind"] == "inter_module_removal"
    assert float(out.iloc[0]["win_rate"]) == 1.0


def test_export_phenotype_match_excel(tmp_path: Path):
    result = _toy_result()
    out_path = tmp_path / "phenotype_match.xlsx"
    export_phenotype_match_excel(result, out_path)
    assert out_path.exists()
