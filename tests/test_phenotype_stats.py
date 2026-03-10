import pandas as pd

from src.phenotype_stats import (
    friedman_test_from_subject_results,
    pairwise_wilcoxon_like,
    matched_severity_pairwise_stats,
    build_stats_tables,
)


def _toy_subject_df():
    return pd.DataFrame(
        [
            {"subject_idx": 0, "attack_kind": "A", "best_distance": 1.0, "best_step": 2, "best_damage_frac": 0.2, "best_delta_density": 0.10, "best_delta_total_weight": -0.10},
            {"subject_idx": 0, "attack_kind": "B", "best_distance": 2.0, "best_step": 3, "best_damage_frac": 0.3, "best_delta_density": 0.11, "best_delta_total_weight": -0.09},
            {"subject_idx": 0, "attack_kind": "C", "best_distance": 3.0, "best_step": 4, "best_damage_frac": 0.4, "best_delta_density": 0.40, "best_delta_total_weight": -0.30},
            {"subject_idx": 1, "attack_kind": "A", "best_distance": 1.2, "best_step": 2, "best_damage_frac": 0.2, "best_delta_density": 0.12, "best_delta_total_weight": -0.11},
            {"subject_idx": 1, "attack_kind": "B", "best_distance": 2.1, "best_step": 3, "best_damage_frac": 0.3, "best_delta_density": 0.13, "best_delta_total_weight": -0.10},
            {"subject_idx": 1, "attack_kind": "C", "best_distance": 3.1, "best_step": 4, "best_damage_frac": 0.4, "best_delta_density": 0.41, "best_delta_total_weight": -0.29},
            {"subject_idx": 2, "attack_kind": "A", "best_distance": 0.9, "best_step": 2, "best_damage_frac": 0.2, "best_delta_density": 0.09, "best_delta_total_weight": -0.12},
            {"subject_idx": 2, "attack_kind": "B", "best_distance": 2.2, "best_step": 3, "best_damage_frac": 0.3, "best_delta_density": 0.10, "best_delta_total_weight": -0.11},
            {"subject_idx": 2, "attack_kind": "C", "best_distance": 3.2, "best_step": 4, "best_damage_frac": 0.4, "best_delta_density": 0.42, "best_delta_total_weight": -0.31},
        ]
    )


def test_friedman_test_from_subject_results():
    df = _toy_subject_df()
    out = friedman_test_from_subject_results(df)
    assert out["test"] == "friedman"
    assert out["n_subjects"] == 3
    assert out["k_attacks"] == 3


def test_pairwise_wilcoxon_like():
    df = _toy_subject_df()
    out = pairwise_wilcoxon_like(df)
    assert not out.empty
    assert {"attack_a", "attack_b", "sign_test_pvalue"}.issubset(out.columns)


def test_matched_severity_pairwise_stats():
    df = _toy_subject_df().copy()
    df["best_delta_density"] = [0.10, 0.11, 0.40, 0.12, 0.13, 0.41, 0.09, 0.10, 0.42]
    out = matched_severity_pairwise_stats(df, severity_col="best_delta_density")
    assert not out.empty
    assert {"severity_col", "n_pairs_matched", "sign_test_pvalue"}.issubset(out.columns)


def test_build_stats_tables():
    df = _toy_subject_df()
    winners = pd.DataFrame(
        [
            {"subject_idx": 0, "attack_kind": "A", "best_distance": 1.0},
            {"subject_idx": 1, "attack_kind": "A", "best_distance": 1.2},
            {"subject_idx": 2, "attack_kind": "A", "best_distance": 0.9},
        ]
    )
    result = {
        "subject_results": df,
        "winner_results": winners,
    }
    pack = build_stats_tables(result)
    assert "stats_overall" in pack
    assert "stats_pairwise" in pack
    assert "stats_winners" in pack
    assert "stats_pairwise_matched_delta_density" in pack
    assert "stats_pairwise_matched_delta_total_weight" in pack
