from __future__ import annotations

from typing import Callable, Tuple

import networkx as nx
import pandas as pd

from ..config import settings
from ..graph_build import build_graph_from_edges, lcc_subgraph
from ..graph_wrapper import GraphWrapper
from ..core.graph_ops import calculate_metrics, compute_3d_layout
from ..core.physics import simulate_energy_flow
from ..core_math import fragility_from_curvature, ollivier_ricci_summary
from ..preprocess import filter_edges


class GraphService:
    @staticmethod
    def filter_edges(
        edges: pd.DataFrame,
        src_col: str,
        dst_col: str,
        min_conf: float,
        min_weight: float,
    ) -> pd.DataFrame:
        return filter_edges(edges, src_col, dst_col, float(min_conf), float(min_weight))

    @staticmethod
    def build_graph(
        edges: pd.DataFrame,
        src_col: str,
        dst_col: str,
        min_conf: float,
        min_weight: float,
        analysis_mode: str,
    ) -> nx.Graph:
        df_filtered = GraphService.filter_edges(edges, src_col, dst_col, min_conf, min_weight)
        G = build_graph_from_edges(df_filtered, src_col, dst_col)
        if str(analysis_mode).startswith("LCC"):
            G = lcc_subgraph(G)
        return G

    @staticmethod
    def compute_metrics(
        edges: pd.DataFrame,
        src_col: str,
        dst_col: str,
        min_conf: float,
        min_weight: float,
        analysis_mode: str,
        seed: int,
        compute_curvature: bool,
        curvature_sample_edges: int,
        progress_cb: Callable[[float], None] | None = None,
    ) -> dict:
        G = GraphService.build_graph(edges, src_col, dst_col, min_conf, min_weight, analysis_mode)
        return calculate_metrics(
            G,
            eff_sources_k=settings.APPROX_EFFICIENCY_K,
            seed=int(seed),
            compute_curvature=bool(compute_curvature),
            curvature_sample_edges=int(curvature_sample_edges),
            progress_cb=progress_cb,
        )

    @staticmethod
    def compute_layout3d(
        edges: pd.DataFrame,
        src_col: str,
        dst_col: str,
        min_conf: float,
        min_weight: float,
        analysis_mode: str,
        seed: int,
    ) -> dict:
        G = GraphService.build_graph(edges, src_col, dst_col, min_conf, min_weight, analysis_mode)
        return compute_3d_layout(G, seed=int(seed))

    @staticmethod
    def compute_energy_frames(
        edges: pd.DataFrame,
        src_col: str,
        dst_col: str,
        min_conf: float,
        min_weight: float,
        analysis_mode: str,
        *,
        steps: int,
        flow_mode: str,
        damping: float,
        sources: Tuple,
        phys_injection: float,
        phys_leak: float,
        phys_cap_mode: str,
        rw_impulse: bool,
    ) -> tuple[list[dict], list[dict]]:
        G = GraphService.build_graph(edges, src_col, dst_col, min_conf, min_weight, analysis_mode)
        src_list = list(sources) if sources else None
        node_frames, edge_frames = simulate_energy_flow(
            G,
            steps=int(steps),
            flow_mode=str(flow_mode),
            damping=float(damping),
            sources=src_list,
            phys_injection=float(phys_injection),
            phys_leak=float(phys_leak),
            phys_cap_mode=str(phys_cap_mode),
            rw_impulse=bool(rw_impulse),
        )
        return node_frames, edge_frames

    @staticmethod
    def compute_layout2d(wrapper: GraphWrapper, seed: int = 0, dim: int = 2) -> dict:
        return nx.spring_layout(wrapper.G, seed=int(seed), dim=int(dim))

    @staticmethod
    def compute_curvature(
        wrapper: GraphWrapper,
        *,
        sample_edges: int = settings.RICCI_SAMPLE_EDGES,
        seed: int = 0,
    ) -> dict:
        use_seed = int(seed)
        curv = ollivier_ricci_summary(
            wrapper.G,
            sample_edges=int(sample_edges),
            seed=use_seed,
            max_support=settings.RICCI_MAX_SUPPORT,
            cutoff=settings.RICCI_CUTOFF,
        )
        return {
            "summary": {
                "kappa_mean": float(curv.kappa_mean),
                "kappa_median": float(curv.kappa_median),
                "kappa_frac_negative": float(curv.kappa_frac_negative),
                "computed_edges": int(curv.computed_edges),
                "skipped_edges": int(curv.skipped_edges),
            },
            "fragility": float(fragility_from_curvature(curv.kappa_mean)),
        }

    @staticmethod
    def compute_ricci_progress(
        G: nx.Graph,
        *,
        sample_edges: int = settings.RICCI_SAMPLE_EDGES,
        seed: int = 0,
        progress_cb: Callable[[float], None] | None = None,
        status_cb: Callable[[str], None] | None = None,
    ) -> dict:
        """Compute curvature with optional progress callbacks for the UI."""

        def _cb(i: int, total: int, x=None, y=None) -> None:
            if progress_cb is not None:
                frac = float(i) / float(max(1, total))
                progress_cb(min(1.0, max(0.0, frac)))
            if status_cb is not None:
                if x is not None and y is not None:
                    status_cb(f"Ricci: {i}/{total}  ({x}—{y})")
                else:
                    status_cb(f"Ricci: {i}/{total}")

        curv = ollivier_ricci_summary(
            G,
            sample_edges=int(sample_edges),
            seed=int(seed),
            max_support=settings.RICCI_MAX_SUPPORT,
            cutoff=settings.RICCI_CUTOFF,
            progress_cb=_cb if (progress_cb is not None or status_cb is not None) else None,
            force_sequential=True,
        )

        return {
            "summary": {
                "kappa_mean": float(curv.kappa_mean),
                "kappa_median": float(curv.kappa_median),
                "kappa_frac_negative": float(curv.kappa_frac_negative),
                "computed_edges": int(curv.computed_edges),
                "skipped_edges": int(curv.skipped_edges),
            },
            "fragility": float(fragility_from_curvature(curv.kappa_mean)),
        }
