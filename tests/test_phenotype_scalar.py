import pandas as pd

from src.phenotype_scalar import (
    build_scalar_summary,
    build_scalar_winners,
    find_best_scalar_match,
)


def test_find_best_scalar_match():
    traj = pd.DataFrame(
        [
            {"step": 0, "damage_frac": 0.0, "density": 0.90, "mod": 0.10},
            {"step": 1, "damage_frac": 0.2, "density": 0.60, "mod": 0.30},
            {"step": 2, "damage_frac": 0.4, "density": 0.30, "mod": 0.60},
        ]
    )
    target = {"density": 0.58, "mod": 0.28}
    scales = {"density": 0.1, "mod": 0.1}

    out = find_best_scalar_match(
        traj,
        target_vector=target,
        metrics=["density", "mod"],
        scales=scales,
    )
    assert out["density"]["best_step"] == 1
    assert out["mod"]["best_step"] == 1


def test_build_scalar_winners_and_summary():
    scalar_subject = pd.DataFrame(
        [
            {"subject_idx": 0, "attack_kind": "A", "metric": "density", "best_step": 1, "best_damage_frac": 0.2, "best_scalar_error": 0.1, "best_value": 0.6},
            {"subject_idx": 0, "attack_kind": "B", "metric": "density", "best_step": 2, "best_damage_frac": 0.4, "best_scalar_error": 0.3, "best_value": 0.4},
            {"subject_idx": 1, "attack_kind": "A", "metric": "density", "best_step": 1, "best_damage_frac": 0.2, "best_scalar_error": 0.2, "best_value": 0.59},
            {"subject_idx": 1, "attack_kind": "B", "metric": "density", "best_step": 2, "best_damage_frac": 0.4, "best_scalar_error": 0.4, "best_value": 0.35},
        ]
    )
    winners = build_scalar_winners(scalar_subject)
    summary = build_scalar_summary(scalar_subject, winners)

    assert len(winners) == 2
    assert (winners["attack_kind"] == "A").all()
    assert not summary.empty
