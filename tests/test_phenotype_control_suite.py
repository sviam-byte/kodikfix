import networkx as nx
import pandas as pd

from src.degradation import run_degradation_trajectory
from src.phenotype_controls import effective_metrics_for_run_type, run_control_suite


def _toy_graph(weight: float = 1.0):
    G = nx.Graph()
    G.add_edge(0, 1, weight=weight)
    G.add_edge(1, 2, weight=weight)
    G.add_edge(2, 3, weight=weight)
    G.add_edge(0, 3, weight=weight)
    G.add_edge(0, 2, weight=0.5 * weight)
    G.add_edge(1, 3, weight=0.5 * weight)
    return G


def test_random_edges_degradation_kind_runs():
    df, aux = run_degradation_trajectory(_toy_graph(), kind="random_edges", steps=3, frac=0.5, seed=42, eff_sources_k=4, compute_heavy_every=1)
    assert aux["kind"] == "random_edges"
    assert df["attack_kind"].iloc[0] == "random_edges"
    assert float(df["damage_frac"].iloc[-1]) > 0.0


def test_effective_metrics_for_density_control_removes_density_family():
    metrics = ["density", "avg_degree", "mod", "eff_w"]
    families = {"density": ["density", "avg_degree"], "integration": ["eff_w"], "modularity": ["mod"]}
    out = effective_metrics_for_run_type(metrics, run_type="density_control_run", metric_families=families)
    assert out == ["mod", "eff_w"]


def test_run_control_suite_returns_all_core_tables():
    sz = pd.DataFrame({"density": [0.7, 0.65], "mod": [0.2, 0.25], "eff_w": [0.4, 0.35]})
    hc = pd.DataFrame({"density": [0.9, 0.92], "mod": [0.05, 0.07], "eff_w": [0.7, 0.72]})
    suite = run_control_suite(
        hc_graphs=[_toy_graph(1.0), _toy_graph(1.1)],
        sz_group_metrics_df=sz,
        hc_baseline_metrics_df=hc,
        metrics=["density", "mod", "eff_w"],
        attack_kinds=["weak_edges_by_weight", "inter_module_removal", "intra_module_removal"],
        metric_families={"density": ["density"], "integration": ["eff_w"], "modularity": ["mod"]},
        compare_kwargs={"steps": 2, "frac": 0.5, "seed": 42, "eff_sources_k": 4, "compute_heavy_every": 1, "distance_mode": "family_balanced", "subject_ids": ["s1", "s2"]},
        modularity_resolutions=[1.0],
        modularity_recompute_options=[False, True],
    )
    assert not suite["suite_summary"].empty
    assert not suite["modularity_sensitivity_summary"].empty
    assert set(suite["suite_metrics"].keys()) == {"primary_run", "density_control_run", "null_attack_run", "target_stability_run"}
    assert "random_edges" in suite["null_attack_result"]["subject_results"]["attack_kind"].unique().tolist()


def test_run_control_suite_includes_target_stability_and_warning_flags():
    g1 = _toy_graph(1.0)
    g2 = _toy_graph(1.05)
    sz = pd.DataFrame({"density": [0.7, 0.65, 0.68], "mod": [0.2, 0.25, 0.22], "eff_w": [0.4, 0.35, 0.38]})
    hc = pd.DataFrame({"density": [0.9, 0.92], "mod": [0.05, 0.07], "eff_w": [0.7, 0.72]})
    suite = run_control_suite(
        hc_graphs=[g1, g2],
        sz_group_metrics_df=sz,
        hc_baseline_metrics_df=hc,
        metrics=["density", "mod", "eff_w"],
        attack_kinds=["weak_edges_by_weight", "inter_module_removal", "intra_module_removal"],
        metric_families={"density": ["density"], "integration": ["eff_w"], "modularity": ["mod"]},
        compare_kwargs={
            "steps": 2,
            "frac": 0.5,
            "seed": 42,
            "eff_sources_k": 4,
            "compute_heavy_every": 1,
            "distance_mode": "family_balanced",
            "subject_ids": ["s1", "s2"],
        },
        modularity_resolutions=[1.0],
        modularity_recompute_options=[False],
        target_bootstrap_reps=3,
    )
    assert not suite["target_stability_summary"].empty
    assert "target_stability_run" in suite["suite_metrics"]
    assert set(suite["warning_flags"]["flag"].tolist()) >= {"density_axis_dominance_flag", "module_partition_instability_flag", "target_instability_flag"}


def test_run_control_suite_includes_family_and_null_severity_summaries():
    g1 = _toy_graph(1.0)
    g2 = _toy_graph(1.2)
    sz = pd.DataFrame({"density": [0.7, 0.65], "mod": [0.2, 0.25], "eff_w": [0.4, 0.35]})
    hc = pd.DataFrame({"density": [0.9, 0.92], "mod": [0.05, 0.07], "eff_w": [0.7, 0.72]})
    suite = run_control_suite(
        hc_graphs=[g1, g2],
        sz_group_metrics_df=sz,
        hc_baseline_metrics_df=hc,
        metrics=["density", "mod", "eff_w"],
        attack_kinds=["weak_edges_by_weight", "inter_module_removal", "intra_module_removal"],
        metric_families={"density": ["density"], "integration": ["eff_w"], "modularity": ["mod"]},
        compare_kwargs={
            "steps": 2,
            "frac": 0.5,
            "seed": 42,
            "eff_sources_k": 4,
            "compute_heavy_every": 1,
            "distance_mode": "family_balanced",
            "subject_ids": ["s1", "s2"],
        },
        modularity_resolutions=[1.0],
        modularity_recompute_options=[False],
        target_bootstrap_reps=4,
    )
    assert not suite["target_family_stability_summary"].empty
    assert not suite["target_scalar_stability_summary"].empty
    assert not suite["target_attack_distance_stability_summary"].empty
    assert not suite["null_severity_density_summary"].empty
    assert not suite["null_severity_total_weight_summary"].empty
