from __future__ import annotations

import io
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PackedMatSubject:
    index: int
    subject_name: str
    packed_edges: np.ndarray


@dataclass(frozen=True)
class PackedMatBundle:
    n_subjects: int
    n_nodes: int
    packed_edge_count: int
    subjects: list[PackedMatSubject]


def packed_edge_count_to_n(m: int) -> int:
    """Recover node count n from packed upper-triangle size m=n*(n-1)/2."""
    if int(m) <= 0:
        raise ValueError("Packed edge count must be > 0")

    disc = 1 + 8 * int(m)
    root = int(math.isqrt(disc))
    if root * root != disc:
        raise ValueError(
            f"Column count {m} is not compatible with packed upper triangle; "
            "expected m=n*(n-1)/2"
        )

    n = (1 + root) // 2
    if n * (n - 1) // 2 != int(m):
        raise ValueError(f"Column count {m} does not factor as n*(n-1)/2")
    return int(n)


def mat_obj_to_subject_names(obj: np.ndarray | None, n_rows: int) -> list[str]:
    """Convert MATLAB object-array `subj_id` into a flat list of strings."""
    if obj is None:
        return [f"subject_{i:03d}" for i in range(int(n_rows))]

    flat = np.asarray(obj, dtype=object).ravel().tolist()
    out: list[str] = []
    for x in flat:
        if isinstance(x, np.ndarray):
            if x.size == 1:
                out.append(str(x.item()))
            else:
                out.append(" ".join(map(str, x.ravel().tolist())))
        else:
            out.append(str(x))

    if len(out) < int(n_rows):
        out.extend([f"subject_{i:03d}" for i in range(len(out), int(n_rows))])
    return out[: int(n_rows)]


def load_packed_mat_bundle(file_bytes: bytes) -> PackedMatBundle:
    """
    Load COBRE-like MAT with:
      - data: (subjects, packed_edges)
      - subj_id: optional subject ids
    """
    from scipy.io import loadmat

    mat = loadmat(io.BytesIO(file_bytes))
    if "data" not in mat:
        raise ValueError("В .mat не найден ключ 'data'")

    X = np.asarray(mat["data"], dtype=float)
    if X.ndim != 2:
        raise ValueError(f"'data' в .mat должна быть 2D, получено: ndim={X.ndim}")
    if X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError(f"'data' в .mat пуста: shape={X.shape}")
    if not np.isfinite(X).all():
        bad = int(np.size(X) - np.isfinite(X).sum())
        raise ValueError(f"'data' в .mat содержит нечисловые/неfinite значения: {bad} шт.")

    n_subjects, m = map(int, X.shape)
    n_nodes = packed_edge_count_to_n(m)
    subj_names = mat_obj_to_subject_names(mat.get("subj_id"), n_subjects)

    subjects: list[PackedMatSubject] = []
    for i in range(n_subjects):
        row = np.asarray(X[i], dtype=float).reshape(-1)
        if row.size != m:
            raise ValueError(f"Строка субъекта {i} имеет неверную длину: {row.size} != {m}")
        subjects.append(
            PackedMatSubject(
                index=i,
                subject_name=subj_names[i],
                packed_edges=row.copy(),
            )
        )

    return PackedMatBundle(
        n_subjects=n_subjects,
        n_nodes=n_nodes,
        packed_edge_count=m,
        subjects=subjects,
    )


def packed_row_to_matrix(row: np.ndarray | list[float], n_nodes: int) -> np.ndarray:
    """Restore a symmetric dense matrix from upper-triangle packed edges."""
    row_arr = np.asarray(row, dtype=float).reshape(-1)
    expected = int(n_nodes) * (int(n_nodes) - 1) // 2
    if row_arr.size != expected:
        raise ValueError(f"Packed row length mismatch: got {row_arr.size}, expected {expected}")
    if not np.isfinite(row_arr).all():
        bad = int(row_arr.size - np.isfinite(row_arr).sum())
        raise ValueError(f"Packed row contains non-finite values: {bad} шт.")

    mat = np.zeros((int(n_nodes), int(n_nodes)), dtype=float)
    iu, ju = np.triu_indices(int(n_nodes), k=1)
    mat[iu, ju] = row_arr
    mat[ju, iu] = row_arr
    np.fill_diagonal(mat, 0.0)
    return mat


def packed_row_to_edge_df(
    row: np.ndarray | list[float],
    n_nodes: int,
    *,
    drop_nonfinite: bool = True,
    keep_zero_weight: bool = False,
) -> pd.DataFrame:
    """
    Restore an edge table from a packed upper triangle.

    By default keeps all finite edges, including negative ones, because weight policy
    is applied later by the normal graph-building pipeline.
    """
    row_arr = np.asarray(row, dtype=float).reshape(-1)
    expected = int(n_nodes) * (int(n_nodes) - 1) // 2
    if row_arr.size != expected:
        raise ValueError(f"Packed row length mismatch: got {row_arr.size}, expected {expected}")

    iu, ju = np.triu_indices(int(n_nodes), k=1)
    keep = np.ones(row_arr.size, dtype=bool)
    if drop_nonfinite:
        keep &= np.isfinite(row_arr)
    if not keep_zero_weight:
        keep &= row_arr != 0.0

    weights = row_arr[keep].astype(float, copy=False)
    return pd.DataFrame(
        {
            "src": iu[keep].astype(int),
            "dst": ju[keep].astype(int),
            "weight": weights,
            "confidence": np.full(weights.shape[0], 100.0, dtype=float),
        }
    )


def bundle_to_edge_frames(
    file_bytes: bytes,
    *,
    keep_zero_weight: bool = False,
) -> tuple[list[tuple[str, pd.DataFrame]], int]:
    """Convenience adapter for the existing app staging flow."""
    bundle = load_packed_mat_bundle(file_bytes)
    graphs: list[tuple[str, pd.DataFrame]] = []
    for subj in bundle.subjects:
        df = packed_row_to_edge_df(
            subj.packed_edges,
            bundle.n_nodes,
            keep_zero_weight=keep_zero_weight,
        )
        graphs.append((subj.subject_name, df))
    return graphs, int(bundle.n_nodes)
