import networkx as nx
import numpy as np

from src.attacks_mix import run_mix_attack
from src.mix_frac_estimator import estimate_mix_frac_star


def _weighted_cycle(n: int) -> nx.Graph:
    g = nx.cycle_graph(n)
    for u, v in g.edges():
        g[u][v]["weight"] = 1.0
    return g


def test_run_mix_attack_reports_effective_progress():
    graph = _weighted_cycle(20)
    df, aux = run_mix_attack(
        graph,
        kind="mix_weightconf_preserving",
        steps=6,
        seed=42,
        eff_sources_k=8,
        replace_from="ER",
        heavy_every=1,
    )
    assert "mix_frac_effective" in df.columns
    assert "replaced_done_step" in df.columns
    assert float(df["mix_frac_effective"].iloc[-1]) >= 0.0
    assert int(aux["total_replaced_done"]) >= 0


def test_estimate_mix_frac_star_nearest_multimetric_returns_ci():
    healthy = [_weighted_cycle(14), _weighted_cycle(16), _weighted_cycle(18)]
    patient_metrics = {
        "clustering": 0.05,
        "avg_degree": 2.0,
    }
    res = estimate_mix_frac_star(
        healthy,
        patient_metrics,
        target_metric=["clustering", "avg_degree"],
        match_mode="nearest",
        steps=6,
        seed=7,
        eff_sources_k=8,
        replace_from="ER",
        n_boot=200,
    )
    assert "ci_low" in res and "ci_high" in res
    assert res["match_mode"] == "nearest"
    assert len(res["used_metrics"]) >= 1
    assert np.isfinite(res["mix_frac_star"])
    assert res["ci_low"] <= res["mix_frac_star"] <= res["ci_high"]
