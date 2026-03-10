import networkx as nx
import pandas as pd

from src.phenotype_matching import compare_degradation_models, compute_profile_distance
from src.phenotype_preflight import run_phenotype_preflight


def _toy_graph(weight: float = 1.0):
    G = nx.Graph()
    G.add_edge(0, 1, weight=weight)
    G.add_edge(1, 2, weight=weight)
    G.add_edge(0, 2, weight=weight)
    return G


def test_family_balanced_distance_differs_from_raw_when_family_sizes_are_uneven():
    row = {"density": 0.2, "avg_degree": 2.0, "mod": 0.5}
    target = {"density": 0.0, "avg_degree": 0.0, "mod": 0.0}
    scales = {"density": 1.0, "avg_degree": 1.0, "mod": 1.0}
    metrics = ["density", "avg_degree", "mod"]
    families = {"density": ["density", "avg_degree"], "modularity": ["mod"]}
    raw = compute_profile_distance(row, target, metrics=metrics, scales=scales, distance_mode="raw", metric_families=families)
    balanced = compute_profile_distance(row, target, metrics=metrics, scales=scales, distance_mode="family_balanced", metric_families=families)
    assert raw["distance"] != balanced["distance"]
    assert set(balanced["family_distances"].keys()) == {"density", "modularity"}


def test_preflight_catches_duplicate_subject_ids_and_missing_metrics():
    sz = pd.DataFrame({"density": [0.1, 0.2], "mod": [0.3, 0.4]})
    hc = pd.DataFrame({"density": [0.4, 0.5], "subject_id": ["a", "a"]})
    rep = run_phenotype_preflight(sz_group_metrics_df=sz, hc_baseline_metrics_df=hc, metrics=["density", "mod"], subject_ids=["x", "x"])
    assert rep["ok"] is False
    assert rep["missing_in_hc"] == ["mod"]
    assert rep["duplicate_subject_ids"] == ["x"]


def test_compare_degradation_models_propagates_subject_id():
    sz = pd.DataFrame({"density": [1.0], "mod": [0.0]})
    hc_baseline = pd.DataFrame({"density": [1.0], "mod": [0.0]})
    result = compare_degradation_models(
        [_toy_graph()],
        sz_group_metrics_df=sz,
        hc_baseline_metrics_df=hc_baseline,
        attack_kinds=["weak_edges_by_weight"],
        metrics=["density", "mod"],
        steps=1,
        frac=0.2,
        subject_ids=["subj_001"],
        compute_heavy_every=1,
        distance_mode="family_balanced",
        metric_families={"density": ["density"], "modularity": ["mod"]},
    )
    assert result["subject_results"]["subject_id"].iloc[0] == "subj_001"
    assert result["winner_results"]["subject_id"].iloc[0] == "subj_001"
    assert "family_dist__density" in result["trajectory_results"].columns


def test_compare_degradation_models_exposes_best_severity_columns():
    sz = pd.DataFrame({"density": [1.0], "mod": [0.0]})
    hc_baseline = pd.DataFrame({"density": [1.0], "mod": [0.0]})
    result = compare_degradation_models(
        [_toy_graph()],
        sz_group_metrics_df=sz,
        hc_baseline_metrics_df=hc_baseline,
        attack_kinds=["weak_edges_by_weight"],
        metrics=["density", "mod"],
        steps=1,
        frac=0.2,
        subject_ids=["subj_001"],
        compute_heavy_every=1,
        distance_mode="family_balanced",
        metric_families={"density": ["density"], "modularity": ["mod"]},
    )
    row = result["subject_results"].iloc[0]
    assert "best_delta_density" in result["subject_results"].columns
    assert "best_delta_total_weight" in result["subject_results"].columns
    assert pd.notna(row["best_delta_total_weight"])
