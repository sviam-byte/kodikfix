from __future__ import annotations

import hashlib
from typing import Optional, Tuple

import streamlit as st
import pandas as pd
import networkx as nx

from ..config import settings
from ..preprocess import filter_edges
from ..graph_build import build_graph_from_edges, lcc_subgraph
from ..graph_wrapper import GraphWrapper
from ..core.graph_ops import calculate_metrics, compute_3d_layout
from ..core.physics import simulate_energy_flow
from ..core_math import ollivier_ricci_summary, fragility_from_curvature


def _hash_df_fast(df: pd.DataFrame) -> str:
    if df is None:
        return "None"
    cols = list(df.columns)
    shape = df.shape
    head = df.head(1000)
    hasher = hashlib.md5()
    hasher.update(str(shape).encode())
    hasher.update(str(cols).encode())
    hasher.update(pd.util.hash_pandas_object(head, index=True).values.tobytes())
    return hasher.hexdigest()


class GraphService:
    @staticmethod
    @st.cache_data(show_spinner=False, hash_funcs={pd.DataFrame: _hash_df_fast})
    def filter_edges(
        edges: pd.DataFrame,
        src_col: str,
        dst_col: str,
        min_conf: float,
        min_weight: float,
    ) -> pd.DataFrame:
        return filter_edges(edges, src_col, dst_col, float(min_conf), float(min_weight))

    @staticmethod
    @st.cache_resource(show_spinner=False, hash_funcs={pd.DataFrame: _hash_df_fast})
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
    @st.cache_data(show_spinner=False, hash_funcs={pd.DataFrame: _hash_df_fast})
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
    ) -> dict:
        G = GraphService.build_graph(edges, src_col, dst_col, min_conf, min_weight, analysis_mode)
        return calculate_metrics(
            G,
            eff_sources_k=settings.APPROX_EFFICIENCY_K,
            seed=int(seed),
            compute_curvature=bool(compute_curvature),
            curvature_sample_edges=int(curvature_sample_edges),
        )

    @staticmethod
    @st.cache_data(show_spinner=False, hash_funcs={pd.DataFrame: _hash_df_fast})
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
    @st.cache_data(show_spinner=False, hash_funcs={pd.DataFrame: _hash_df_fast})
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
    @st.cache_data(show_spinner=False, hash_funcs={GraphWrapper: lambda w: w.get_version()})
    def compute_layout2d(wrapper: GraphWrapper, seed: int = 0, dim: int = 2) -> dict:
        return nx.spring_layout(wrapper.G, seed=int(seed), dim=int(dim))

    @staticmethod
    @st.cache_data(show_spinner=False, hash_funcs={GraphWrapper: lambda w: w.get_version()})
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
    ) -> dict:
        # Да, это сервис, и да, тут st.progress. Это сознательный костыль:
        # joblib-параллелизм не даёт нормальный прогресс, а UX без прогресса — боль.
        bar = st.progress(0.0)
        msg = st.empty()

        def _cb(i, total, x=None, y=None):
            frac = float(i) / float(max(1, total))
            bar.progress(min(1.0, max(0.0, frac)))
            if x is not None and y is not None:
                msg.caption(f"Ricci: {i}/{total}  ({x}—{y})")
            else:
                msg.caption(f"Ricci: {i}/{total}")

        curv = ollivier_ricci_summary(
            G,
            sample_edges=int(sample_edges),
            seed=int(seed),
            max_support=settings.RICCI_MAX_SUPPORT,
            cutoff=settings.RICCI_CUTOFF,
            progress_cb=_cb,
            force_sequential=True,
        )

        bar.empty()
        msg.empty()

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



# Совместимость с существующими импортами/именами в app.py
_filter_edges_cached = GraphService.filter_edges
_build_graph_cached = GraphService.build_graph
_metrics_cached = GraphService.compute_metrics
_layout_cached = GraphService.compute_layout3d
_energy_frames_cached = GraphService.compute_energy_frames

compute_layout_cached = GraphService.compute_layout2d
compute_curvature_cached = GraphService.compute_curvature
