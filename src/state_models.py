
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict

import pandas as pd

from .preprocess import filter_edges


@dataclass
class GraphEntry:
    id: str
    name: str
    source: str
    edges: pd.DataFrame
    src_col: str
    dst_col: str
    created_at: float

    def get_filtered_edges(self, min_conf: float, min_weight: float) -> pd.DataFrame:
        """"""
        return filter_edges(self.edges, self.src_col, self.dst_col, min_conf, min_weight)


@dataclass
class ExperimentEntry:

    id: str
    name: str
    graph_id: str
    attack_kind: str
    params: Dict[str, Any]
    history: pd.DataFrame
    created_at: float


def build_graph_entry(
    *,
    name: str,
    source: str,
    edges: pd.DataFrame,
    src_col: str,
    dst_col: str,
    entry_id: str,
    created_at: float | None = None,
) -> GraphEntry:
    return GraphEntry(
        id=entry_id,
        name=name,
        source=source,
        edges=edges.copy(),
        src_col=src_col,
        dst_col=dst_col,
        created_at=time.time() if created_at is None else float(created_at),
    )


def build_experiment_entry(
    *,
    name: str,
    graph_id: str,
    attack_kind: str,
    params: Dict[str, Any],
    history: pd.DataFrame,
    entry_id: str,
    created_at: float | None = None,
) -> ExperimentEntry:
    return ExperimentEntry(
        id=entry_id,
        name=name,
        graph_id=graph_id,
        attack_kind=attack_kind,
        params=params,
        history=history.copy(),
        created_at=time.time() if created_at is None else float(created_at),
    )
