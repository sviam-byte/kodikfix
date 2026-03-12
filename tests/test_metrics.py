import sys
from pathlib import Path

import networkx as nx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.core_math import entropy_degree  # noqa: E402


def test_entropy_degree_complete_graph():
    """Complete graph has uniform degree distribution, so entropy should be 0."""
    K5 = nx.complete_graph(5)
    ent = entropy_degree(K5)
    assert np.isclose(ent, 0.0), f"Entropy of K5 should be 0, got {ent}"


def test_entropy_star_graph():
    """Star graph has mixed degrees, so entropy should be > 0."""
    S = nx.star_graph(4)
    ent = entropy_degree(S)
    assert ent > 0


from src.core.graph_ops import calculate_metrics  # noqa: E402


def test_calculate_metrics_exposes_signed_weight_summary():
    """Signed-hybrid metrics should expose raw signed-weight summary fields."""
    G = nx.Graph()
    G.add_edge(0, 1, weight=0.5, raw_weight=0.5)
    G.add_edge(1, 2, weight=0.4, raw_weight=-0.4)
    G.add_edge(0, 2, weight=0.2, raw_weight=0.2)
    m = calculate_metrics(G, eff_sources_k=4, seed=0, compute_curvature=False, compute_heavy=True)
    assert np.isclose(m["frac_negative_weight"], 1.0 / 3.0)
    assert m["signed_std_weight"] > 0
    assert np.isfinite(m["signed_balance_weight"])
