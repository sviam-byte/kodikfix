"""Regression tests for core_math entropy helpers."""

import networkx as nx
import numpy as np

from src.core_math import entropy_degree, network_entropy_rate


def test_entropy_degree_complete_graph() -> None:
    """У полного графа все степени равны, энтропия должна быть 0."""
    k5 = nx.complete_graph(5)
    ent = entropy_degree(k5)
    assert np.isclose(ent, 0.0), f"Entropy of K5 should be 0, got {ent}"


def test_entropy_nonuniform_graph() -> None:
    """Неоднородное распределение степеней должно давать энтропию > 0."""
    graph = nx.barabasi_albert_graph(30, 1, seed=1)
    ent = entropy_degree(graph)
    assert ent > 0.0


def test_entropy_rate_empty() -> None:
    """Пустой граф не должен вызывать краш."""
    graph = nx.Graph()
    rate = network_entropy_rate(graph)
    assert rate == 0.0
