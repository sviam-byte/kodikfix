from __future__ import annotations

import time

import networkx as nx
import pytest

from src.timeout_worker import (
    compute_metrics_from_graph_safe,
    graph_resistance_from_graph_safe,
    run_with_timeout,
)


def _sleep_then_return(seconds: float) -> str:
    """Sleep in child process to validate hard-timeout enforcement."""
    time.sleep(float(seconds))
    return "ok"


def test_compute_metrics_from_graph_safe_returns_ok_status() -> None:
    """Metrics helper should return a flat row with status for subprocess path."""
    graph = nx.path_graph(6)
    row = compute_metrics_from_graph_safe(
        graph,
        eff_k=8,
        seed=42,
        compute_curvature=False,
        curvature_sample_edges=0,
        curvature_max_support=0,
        compute_heavy=False,
        skip_spectral=True,
        skip_clustering=True,
        skip_assortativity=True,
        diameter_samples=0,
        graph_name="toy_graph",
    )

    assert isinstance(row, dict)
    assert row["status"] == "ok"
    assert row["graph_name"] == "toy_graph"
    assert "N" in row


def test_graph_resistance_from_graph_safe_returns_ok_status() -> None:
    """Resistance helper should return summary row with explicit success marker."""
    graph = nx.cycle_graph(8)
    row = graph_resistance_from_graph_safe(graph)

    assert isinstance(row, dict)
    assert row["status"] == "ok"
    assert row["n_nodes"] == 8


def test_run_with_timeout_raises_timeout_error() -> None:
    """Hard-timeout runner must terminate slow stage and raise TimeoutError."""
    with pytest.raises(TimeoutError):
        run_with_timeout(_sleep_then_return, 0.5, timeout_seconds=0.1)


def test_run_with_timeout_passes_result_without_timeout() -> None:
    """Hard-timeout runner should pass results for fast stages."""
    out = run_with_timeout(_sleep_then_return, 0.01, timeout_seconds=5.0)
    assert out == "ok"
