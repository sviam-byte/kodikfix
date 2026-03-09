import pandas as pd

from src.phenotype_stats import (
    friedman_test_from_subject_results,
    pairwise_wilcoxon_like,
    build_stats_tables,
)


def _toy_subject_df():
    return pd.DataFrame(
        [
            {"subject_idx": 0, "attack_kind": "A", "best_distance": 1.0, "best_step": 2, "best_damage_frac": 0.2},
            {"subject_idx": 0, "attack_kind": "B", "best_distance": 2.0, "best_step": 3, "best_damage_frac": 0.3},
            {"subject_idx": 0, "attack_kind": "C", "best_distance": 3.0, "best_step": 4, "best_damage_frac": 0.4},
            {"subject_idx": 1, "attack_kind": "A", "best_distance": 1.2, "best_step": 2, "best_damage_frac": 0.2},
            {"subject_idx": 1, "attack_kind": "B", "best_distance": 2.1, "best_step": 3, "best_damage_frac": 0.3},
            {"subject_idx": 1, "attack_kind": "C", "best_distance": 3.1, "best_step": 4, "best_damage_frac": 0.4},
            {"subject_idx": 2, "attack_kind": "A", "best_distance": 0.9, "best_step": 2, "best_damage_frac": 0.2},
            {"subject_idx": 2, "attack_kind": "B", "best_distance": 2.2, "best_step": 3, "best_damage_frac": 0.3},
            {"subject_idx": 2, "attack_kind": "C", "best_distance": 3.2, "best_step": 4, "best_damage_frac": 0.4},
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
