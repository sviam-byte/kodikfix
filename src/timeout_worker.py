"""Subprocess-safe computation functions for phenotype matching.

CRITICAL: This module MUST NOT import streamlit or any module that imports streamlit.
On Windows, multiprocessing uses ``spawn`` so the child process re-imports this module.
Any direct or transitive ``streamlit`` import here can crash child processes.

This module contains:
1. Pure computation functions (``build_graph_safe``, ``compute_metrics_safe``)
2. The subprocess worker (``_mp_worker``) that the child process actually runs
3. The timeout runner (``run_with_timeout``) that the parent process calls
"""

from __future__ import annotations

import multiprocessing as mp
import traceback as _tb_mod
from typing import Any

import networkx as nx
import pandas as pd

# Streamlit-free imports only (safe for multiprocessing "spawn")
from src.metrics import calculate_metrics
from src.services.graph_service import GraphService

# ---------------------------------------------------------------------------
# Pure computation functions (called inside the child process)
# ---------------------------------------------------------------------------

def build_graph_safe(
    edges: pd.DataFrame,
    src_col: str,
    dst_col: str,
    min_conf: float,
    min_weight: float,
    analysis_mode: str,
) -> nx.Graph:
    """Build a graph from an edge DataFrame in a subprocess-safe way."""
    return GraphService.build_graph(
        edges,
        src_col,
        dst_col,
        float(min_conf),
        float(min_weight),
        str(analysis_mode),
    )


def compute_metrics_safe(
    edges: pd.DataFrame,
    src_col: str,
    dst_col: str,
    *,
    min_conf: float,
    min_weight: float,
    analysis_mode: str,
    eff_k: int,
    seed: int,
    compute_curvature: bool,
    needs_spectral: bool,
    needs_clustering: bool,
    needs_assortativity: bool,
    needs_diameter: bool,
    graph_name: str = "",
) -> dict:
    """Build graph and compute metrics using only pickleable arguments."""
    graph = GraphService.build_graph(
        edges,
        src_col,
        dst_col,
        float(min_conf),
        float(min_weight),
        str(analysis_mode),
    )
    metrics = calculate_metrics(
        graph,
        int(eff_k),
        int(seed),
        bool(compute_curvature),
        curvature_sample_edges=80,
        compute_heavy=True,
        skip_spectral=not bool(needs_spectral),
        skip_clustering=not bool(needs_clustering),
        skip_assortativity=not bool(needs_assortativity),
        diameter_samples=8 if bool(needs_diameter) else 0,
    )
    row: dict[str, Any] = {"graph_name": str(graph_name), "status": "ok"}
    row.update(metrics)
    return row


# ---------------------------------------------------------------------------
# Subprocess machinery (must stay streamlit-free)
# ---------------------------------------------------------------------------

def _mp_worker(
    result_q: mp.Queue,
    error_q: mp.Queue,
    fn,
    args: tuple,
    kwargs: dict,
) -> None:
    """Execute ``fn(*args, **kwargs)`` and post result/error into queues."""
    try:
        result = fn(*args, **kwargs)
        result_q.put(result)
    except Exception as exc:  # pragma: no cover - worker-side exception reporting
        error_q.put((type(exc).__name__, str(exc), _tb_mod.format_exc()))


def run_with_timeout(fn, *args, timeout_seconds: float = 0.0, **kwargs):
    """Run a callable with a hard timeout via ``multiprocessing.Process``.

    Notes:
    - ``fn`` must be a top-level function from a streamlit-free module.
    - With timeout <= 0, execution falls back to direct in-process call.
    """
    timeout_seconds = float(timeout_seconds or 0.0)
    if timeout_seconds <= 0:
        return fn(*args, **kwargs)

    ctx = mp.get_context("spawn")
    result_q: mp.Queue = ctx.Queue(maxsize=1)
    error_q: mp.Queue = ctx.Queue(maxsize=1)

    proc = ctx.Process(
        target=_mp_worker,
        args=(result_q, error_q, fn, args, kwargs),
        daemon=True,
    )
    proc.start()
    proc.join(timeout=timeout_seconds)

    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=5)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=2)
        raise TimeoutError(f"stage timeout > {timeout_seconds:.1f}s")

    if not error_q.empty():
        exc_type, exc_msg, exc_tb = error_q.get_nowait()
        raise RuntimeError(f"{exc_type}: {exc_msg}\n--- child traceback ---\n{exc_tb}")

    if not result_q.empty():
        return result_q.get_nowait()

    exitcode = proc.exitcode
    raise RuntimeError(
        f"subprocess exited with code {exitcode} without producing a result. "
        "This usually means a segfault or import error in the child process."
    )
