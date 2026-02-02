from pathlib import Path
import sys

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
