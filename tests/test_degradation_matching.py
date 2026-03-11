import networkx as nx
import numpy as np
import pandas as pd

from src.degradation import prepare_module_info, run_degradation_trajectory
from src.phenotype_matching import find_best_match_to_target


def _toy_two_module_graph():
    G = nx.Graph()
    G.add_edge(0, 1, weight=3.0)
    G.add_edge(1, 2, weight=3.0)
    G.add_edge(0, 2, weight=3.0)
    G.add_edge(3, 4, weight=3.0)
    G.add_edge(4, 5, weight=3.0)
    G.add_edge(3, 5, weight=3.0)
    G.add_edge(2, 3, weight=0.2)
    G.add_edge(1, 4, weight=0.2)
    return G


def test_inter_module_removal_removes_only_inter_edges():
    G = _toy_two_module_graph()
    mi = prepare_module_info(G, seed=42)

    df, aux = run_degradation_trajectory(
        G,
        kind="inter_module_removal",
        steps=4,
        frac=1.0,
        seed=42,
        eff_sources_k=4,
        compute_heavy_every=1,
        module_info=mi,
        recompute_modules=False,
        removal_mode="random",
    )

    removed = aux["removed_edges_order"]
    assert removed, "inter-module attack must remove something on this graph"
    for u, v in removed:
        assert mi.membership[u] != mi.membership[v]
    assert df["E"].iloc[-1] == G.number_of_edges() - len(removed)


def test_weight_noise_preserves_nodes_and_tracks_damage_frac():
    G = _toy_two_module_graph()
    df, aux = run_degradation_trajectory(
        G,
        kind="weight_noise",
        steps=5,
        frac=0.5,
        seed=42,
        eff_sources_k=4,
        compute_heavy_every=1,
        noise_sigma_max=0.8,
        keep_density_from_baseline=True,
    )

    assert aux["kind"] == "weight_noise"
    assert (df["N"] == G.number_of_nodes()).all()
    assert np.isfinite(df["damage_frac"]).all()
    assert float(df["damage_frac"].iloc[-1]) > 0.0
    assert float(df["damage_frac"].iloc[-1]) <= 0.5 + 1e-9


def test_find_best_match_to_target_selects_expected_step():
    traj = pd.DataFrame(
        [
            {"step": 0, "damage_frac": 0.0, "density": 0.90, "clustering": 0.85},
            {"step": 1, "damage_frac": 0.2, "density": 0.60, "clustering": 0.55},
            {"step": 2, "damage_frac": 0.4, "density": 0.30, "clustering": 0.25},
        ]
    )
    target = {"density": 0.58, "clustering": 0.50}
    scales = {"density": 0.1, "clustering": 0.1}

    out = find_best_match_to_target(
        traj,
        target_vector=target,
        metrics=["density", "clustering"],
        scales=scales,
    )
    assert out["best_step"] == 1


def test_run_degradation_trajectory_reports_attack_family_and_graph_audit():
    """Trajectory aux should expose regime-aware audit metadata for downstream logs."""
    G = _toy_two_module_graph()
    df, aux = run_degradation_trajectory(
        G,
        kind="weight_noise",
        steps=2,
        frac=0.2,
        seed=7,
        eff_sources_k=4,
        compute_heavy_every=1,
        graph_regime="full_weighted_unsigned",
    )
    assert not df.empty
    assert aux.get("attack_family") == "weight"
    graph_audit = aux.get("graph_audit", {})
    assert graph_audit.get("graph_regime") == "full_weighted_unsigned"
    assert int(graph_audit.get("n_nodes", -1)) == G.number_of_nodes()
