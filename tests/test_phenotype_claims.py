import pandas as pd

from src.phenotype_claims import benjamini_hochberg, build_core_claim_readiness, build_family_inference_table, build_scalar_inference_table


def _toy_result():
    scalar_subject = pd.DataFrame(
        [
            {"subject_idx": 0, "attack_kind": "inter", "metric": "mod", "best_scalar_error": 0.1},
            {"subject_idx": 0, "attack_kind": "random", "metric": "mod", "best_scalar_error": 0.4},
            {"subject_idx": 1, "attack_kind": "inter", "metric": "mod", "best_scalar_error": 0.2},
            {"subject_idx": 1, "attack_kind": "random", "metric": "mod", "best_scalar_error": 0.6},
            {"subject_idx": 0, "attack_kind": "inter", "metric": "eff_w", "best_scalar_error": 0.2},
            {"subject_idx": 0, "attack_kind": "random", "metric": "eff_w", "best_scalar_error": 0.5},
            {"subject_idx": 1, "attack_kind": "inter", "metric": "eff_w", "best_scalar_error": 0.25},
            {"subject_idx": 1, "attack_kind": "random", "metric": "eff_w", "best_scalar_error": 0.55},
        ]
    )
    return {
        "scalar_subject_results": scalar_subject,
        "metric_family_map": {"mod": "modularity", "eff_w": "integration"},
        "winner_results": pd.DataFrame([{"subject_idx": 0, "attack_kind": "inter"}, {"subject_idx": 1, "attack_kind": "inter"}]),
        "summary_winners": pd.DataFrame([{"attack_kind": "inter", "win_rate": 1.0}]),
        "warning_flags": pd.DataFrame([{"flag": "density_axis_dominance_flag", "value": False}]),
    }


def test_bh_monotone():
    q = benjamini_hochberg([0.01, 0.02, 0.20])
    assert len(q) == 3
    assert q[0] <= q[1] <= q[2]


def test_scalar_and_family_inference_nonempty():
    result = _toy_result()
    s = build_scalar_inference_table(result)
    f = build_family_inference_table(result)
    assert not s.empty
    assert not f.empty
    assert "fdr_q_value" in s.columns
    assert "cliffs_delta" in f.columns


def test_claim_readiness_shape():
    ready = build_core_claim_readiness(_toy_result())
    assert not ready.empty
    assert "claim_status" in ready.columns
    assert int(ready.iloc[0]["readiness_score"]) >= 1
