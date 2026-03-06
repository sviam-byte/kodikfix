"""
Импорт корреляционных матриц → edge DataFrame → Graph.

Поддерживает:
- numpy .npy / .npz
- scipy .mat (MATLAB)
- CSV-матрицы (квадратные)

Политики обработки знака:
- 'abs': |corr|, стандарт для функциональных коннектомов
- 'positive_only': отбросить r <= 0
- 'shift': r + shift, затем clip

Порог (threshold):
- 'density': оставить top-k% рёбер по весу
- 'absolute': оставить рёбра с weight >= threshold
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import networkx as nx
import numpy as np
import pandas as pd

from .graph_build import build_graph_from_edges, lcc_subgraph


def load_matrix(path: Union[str, Path]) -> np.ndarray:
    """Загрузить квадратную матрицу из файла."""
    p = Path(path)
    if p.suffix == ".npy":
        return np.load(p)
    if p.suffix == ".npz":
        data = np.load(p)
        # Берём первый массив из контейнера.
        key = list(data.keys())[0]
        return data[key]
    if p.suffix == ".mat":
        from scipy.io import loadmat

        data = loadmat(str(p))
        # Ищем первую квадратную матрицу.
        for _, value in data.items():
            if isinstance(value, np.ndarray) and value.ndim == 2 and value.shape[0] == value.shape[1]:
                return value.astype(float)
        raise ValueError(f"No square matrix found in {p}")

    # Иначе трактуем как CSV с квадратной матрицей.
    df = pd.read_csv(p, header=None)
    return df.values.astype(float)


def matrix_to_edges(
    corr: np.ndarray,
    *,
    sign_policy: str = "abs",
    threshold_mode: str = "density",
    threshold_value: float = 0.15,
    shift: float = 0.0,
    labels: Optional[list] = None,
) -> pd.DataFrame:
    """
    Корреляционная матрица → DataFrame рёбер (src, dst, weight, confidence).

    Parameters
    ----------
    corr : (N, N) array
    sign_policy : 'abs' | 'positive_only' | 'shift'
    threshold_mode : 'density' | 'absolute'
    threshold_value : для density — доля рёбер (0.15 = top 15%);
                      для absolute — минимальный вес
    labels : имена узлов (по умолчанию 0..N-1)
    """
    n = corr.shape[0]
    if corr.shape != (n, n):
        raise ValueError(f"Matrix must be square, got {corr.shape}")

    if labels is None:
        labels = list(range(n))

    # Зануляем диагональ и работаем с копией.
    mat = corr.copy().astype(float)
    np.fill_diagonal(mat, 0.0)

    # Обработка знака по выбранной политике.
    if sign_policy == "abs":
        mat = np.abs(mat)
    elif sign_policy == "positive_only":
        mat[mat <= 0] = 0.0
    elif sign_policy == "shift":
        mat = np.clip(mat + float(shift), 0.0, None)
    else:
        raise ValueError(f"Unknown sign_policy: {sign_policy!r}")

    # Верхний треугольник для неориентированного графа.
    rows, cols = np.triu_indices(n, k=1)
    weights = mat[rows, cols]

    # Отбрасываем нулевые веса.
    mask = weights > 0
    rows, cols, weights = rows[mask], cols[mask], weights[mask]

    # Порогование по плотности или абсолютному значению.
    if threshold_mode == "density":
        if len(weights) == 0:
            rows = rows[:0]
            cols = cols[:0]
            weights = weights[:0]
        else:
            k = max(1, int(len(weights) * float(threshold_value)))
            idx_sorted = np.argsort(weights)[::-1]
            top_idx = idx_sorted[:k]
            keep_mask = np.zeros(len(weights), dtype=bool)
            keep_mask[top_idx] = True
            rows, cols, weights = rows[keep_mask], cols[keep_mask], weights[keep_mask]
    elif threshold_mode == "absolute":
        keep_mask = weights >= float(threshold_value)
        rows, cols, weights = rows[keep_mask], cols[keep_mask], weights[keep_mask]
    else:
        raise ValueError(f"Unknown threshold_mode: {threshold_mode!r}")

    return pd.DataFrame(
        {
            "src": [labels[r] for r in rows],
            "dst": [labels[c] for c in cols],
            "weight": weights,
            # Совместимость с остальным пайплайном: confidence ожидается в данных рёбер.
            "confidence": np.ones(len(weights), dtype=float) * 100.0,
        }
    )


def matrix_to_graph(
    corr: np.ndarray,
    *,
    sign_policy: str = "abs",
    threshold_mode: str = "density",
    threshold_value: float = 0.15,
    shift: float = 0.0,
    labels: Optional[list] = None,
    use_lcc: bool = True,
) -> nx.Graph:
    """Корреляционная матрица → NetworkX Graph (через edge DataFrame)."""
    df = matrix_to_edges(
        corr,
        sign_policy=sign_policy,
        threshold_mode=threshold_mode,
        threshold_value=threshold_value,
        shift=shift,
        labels=labels,
    )
    if df.empty:
        raise ValueError("No edges after thresholding")

    graph = build_graph_from_edges(df, "src", "dst", strict=False)
    if use_lcc:
        graph = lcc_subgraph(graph)
    return graph
