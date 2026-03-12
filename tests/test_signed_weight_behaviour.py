import networkx as nx
import pandas as pd

from src.config import settings
from src.degradation import _rank_edges_with_noise, run_degradation_trajectory
from src.core_math import signed_laplacian_spectrum
from src.core.graph_ops import calculate_metrics
from src.preprocess import filter_edges


def test_filter_edges_signed_split_uses_abs_threshold():
    """signed_split must apply min_weight threshold to |w|, not signed value."""
    prev = settings.WEIGHT_POLICY
    object.__setattr__(settings, "WEIGHT_POLICY", "signed_split")
    try:
        df = pd.DataFrame(
            {
                "src": [1, 1, 2],
                "dst": [2, 3, 3],
                "confidence": [1.0, 1.0, 1.0],
                "weight": [-0.5, 0.2, -0.05],
            }
        )
        out = filter_edges(df, "src", "dst", min_conf=0.0, min_weight=0.1)
        kept = set(zip(out["src"], out["dst"], out["weight"], strict=False))
        assert (1, 2, -0.5) in kept
        assert (1, 3, 0.2) in kept
        assert (2, 3, -0.05) not in kept
    finally:
        object.__setattr__(settings, "WEIGHT_POLICY", prev)


def test_rank_edges_with_noise_uses_signed_base_without_floor_bias():
    """Noisy ranking should perturb signed weights and keep signed noisy value."""
    G = nx.Graph()
    for i in range(12):
        G.add_edge(i, i + 1, weight=0.01, raw_weight=0.01, weight_signed=0.01)

    ranked = _rank_edges_with_noise(G, sigma=4.0, seed=7)
    assert all(len(item) == 5 for item in ranked)
    # With many weak edges at high sigma, at least one perturbed signed weight
    # should cross zero (unlike the old max(1e-12, w+eps) behavior).
    assert any(item[4] < 0 for item in ranked)


def test_strong_negative_edges_attack_targets_negative_edges_only():
    """Signed attack must remove only most negative edges from candidate pool."""
    G = nx.Graph()
    G.add_edge(0, 1, weight=0.8, raw_weight=0.8, weight_signed=0.8, sign=1.0)
    G.add_edge(1, 2, weight=0.9, raw_weight=-0.9, weight_signed=-0.9, sign=-1.0)
    G.add_edge(2, 3, weight=0.6, raw_weight=-0.6, weight_signed=-0.6, sign=-1.0)
    G.add_edge(0, 3, weight=0.4, raw_weight=0.4, weight_signed=0.4, sign=1.0)

    _df, aux = run_degradation_trajectory(
        G,
        kind="strong_negative_edges",
        frac=1.0,
        steps=2,
        seed=42,
        eff_sources_k=4,
        compute_heavy_every=1,
    )

    removed = aux["removed_edges_order"]
    assert removed == [(1, 2), (2, 3)]


def test_negative_edges_by_magnitude_alias_behaves_like_negative_edges_only():
    """Compatibility alias should route to the same signed-negative ranking."""
    G = nx.Graph()
    G.add_edge(0, 1, weight=0.8, raw_weight=-0.8, weight_signed=-0.8, sign=-1.0)
    G.add_edge(1, 2, weight=0.4, raw_weight=-0.4, weight_signed=-0.4, sign=-1.0)
    G.add_edge(2, 3, weight=0.3, raw_weight=0.3, weight_signed=0.3, sign=1.0)

    _df, aux = run_degradation_trajectory(
        G,
        kind="negative_edges_by_magnitude",
        frac=1.0,
        steps=2,
        seed=1,
        eff_sources_k=4,
        compute_heavy_every=1,
    )
    assert aux["removed_edges_order"] == [(0, 1), (1, 2)]


def test_signed_laplacian_metrics_are_computed_on_heavy_pass():
    """Signed spectral metrics should be exposed in calculate_metrics output."""
    G = nx.Graph()
    G.add_edge(0, 1, weight=0.7, raw_weight=0.7, weight_signed=0.7, sign=1.0)
    G.add_edge(1, 2, weight=0.6, raw_weight=-0.6, weight_signed=-0.6, sign=-1.0)
    G.add_edge(0, 2, weight=0.5, raw_weight=0.5, weight_signed=0.5, sign=1.0)

    sl = signed_laplacian_spectrum(G)
    assert "signed_lambda_min" in sl and "frustration_index" in sl

    metrics = calculate_metrics(G, eff_sources_k=3, seed=1, compute_heavy=True)
    assert "signed_lambda_min" in metrics
    assert "strength_pos_mean" in metrics
