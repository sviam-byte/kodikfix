import base64
import json

import numpy as np
import pandas as pd

from .state_models import (
    ExperimentEntry,
    GraphEntry,
    build_experiment_entry,
    build_graph_entry,
)


class GraphDataEncoder(json.JSONEncoder):
    """JSON encoder for numpy/pandas-friendly payloads."""
    def default(self, obj):  # noqa: ANN001 - JSONEncoder API uses untyped args.
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (pd.Timestamp, pd.Timedelta)):
            return obj.isoformat()
        if isinstance(obj, set):
            return sorted(list(obj), key=lambda item: str(item))
        if isinstance(obj, pd.Series):
            return obj.to_list()
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="records")
        try:
            if pd.isna(obj):
                return None
        except TypeError:
            pass
        return super().default(obj)


def _json_dumps_bytes(payload: dict) -> bytes:
    """Serialize payload with GraphDataEncoder to UTF-8 bytes."""
    return json.dumps(payload, cls=GraphDataEncoder, ensure_ascii=False, indent=2).encode("utf-8")


def _df_to_b64_csv(df: pd.DataFrame) -> str:
    """Serialize a DataFrame to a base64-encoded CSV string."""
    csv = df.to_csv(index=False).encode("utf-8")
    return base64.b64encode(csv).decode("ascii")


def _b64_csv_to_df(s: str) -> pd.DataFrame:
    """Deserialize a base64-encoded CSV string into a DataFrame."""
    raw = base64.b64decode(s.encode("ascii"))
    return pd.read_csv(pd.io.common.BytesIO(raw))


def export_workspace_json(graphs: dict, experiments: list[ExperimentEntry]) -> bytes:
    """Экспорт workspace (graphs + experiments) в JSON bytes.

    Допускаются два формата в session_state:
    - новый: GraphEntry / ExperimentEntry
    - легаси: dict-пэйлоады
    """
    g_out: dict[str, dict] = {}
    for gid, g in graphs.items():
        if isinstance(g, GraphEntry):
            g_out[gid] = {
                "id": g.id,
                "name": g.name,
                "source": g.source,
                "tags": {"src_col": g.src_col, "dst_col": g.dst_col},
                "created_at": g.created_at,
                "edges_b64": _df_to_b64_csv(g.edges),
            }
        else:
            g_out[gid] = {
                "id": g["id"],
                "name": g["name"],
                "source": g["source"],
                "tags": g.get("tags", {}),
                "created_at": g.get("created_at", 0.0),
                "edges_b64": _df_to_b64_csv(g["edges"]),
            }

    e_out: list[dict] = []
    for e in experiments:
        if isinstance(e, ExperimentEntry):
            e_out.append(
                {
                    "id": e.id,
                    "name": e.name,
                    "graph_id": e.graph_id,
                    "attack_kind": e.attack_kind,
                    "params": e.params or {},
                    "created_at": e.created_at,
                    "history_b64": _df_to_b64_csv(e.history),
                }
            )
        else:
            e_out.append(
                {
                    "id": e["id"],
                    "name": e["name"],
                    "graph_id": e["graph_id"],
                    "attack_kind": e["attack_kind"],
                    "params": e.get("params", {}),
                    "created_at": e.get("created_at", 0.0),
                    "history_b64": _df_to_b64_csv(e["history"]),
                }
            )

    payload = {"graphs": g_out, "experiments": e_out}
    return _json_dumps_bytes(payload)


def import_workspace_json(blob: bytes) -> tuple[dict[str, GraphEntry], list[ExperimentEntry]]:
    """Импорт workspace из JSON bytes."""
    payload = json.loads(blob.decode("utf-8"))
    graphs_raw = payload.get("graphs", {})
    exps_raw = payload.get("experiments", [])

    graphs: dict[str, GraphEntry] = {}
    for gid, g in graphs_raw.items():
        edges = _b64_csv_to_df(g["edges_b64"])
        tags = g.get("tags", {}) or {}
        graphs[gid] = build_graph_entry(
            name=g.get("name", gid),
            source=g.get("source", "import"),
            edges=edges,
            src_col=tags.get("src_col", edges.columns[0]),
            dst_col=tags.get("dst_col", edges.columns[1]),
            entry_id=g.get("id", gid),
            created_at=g.get("created_at", 0.0),
        )

    exps: list[ExperimentEntry] = []
    for e in exps_raw:
        hist = _b64_csv_to_df(e["history_b64"])
        exps.append(
            build_experiment_entry(
                name=e.get("name"),
                graph_id=e.get("graph_id"),
                attack_kind=e.get("attack_kind"),
                params=e.get("params", {}),
                history=hist,
                entry_id=e.get("id"),
                created_at=e.get("created_at", 0.0),
            )
        )

    return graphs, exps


def export_experiments_json(experiments: list[ExperimentEntry]) -> bytes:
    """Export experiments only (without graph storage) as JSON bytes."""
    e_out: list[dict] = []
    for e in experiments:
        if isinstance(e, ExperimentEntry):
            e_out.append(
                {
                    "id": e.id,
                    "name": e.name,
                    "graph_id": e.graph_id,
                    "attack_kind": e.attack_kind,
                    "params": e.params or {},
                    "created_at": e.created_at,
                    "history_b64": _df_to_b64_csv(e.history),
                }
            )
        else:
            e_out.append(
                {
                    "id": e["id"],
                    "name": e["name"],
                    "graph_id": e["graph_id"],
                    "attack_kind": e["attack_kind"],
                    "params": e.get("params", {}),
                    "created_at": e.get("created_at", 0.0),
                    "history_b64": _df_to_b64_csv(e["history"]),
                }
            )
    payload = {"experiments": e_out}
    return _json_dumps_bytes(payload)


def import_experiments_json(blob: bytes) -> list[ExperimentEntry]:
    """Import experiments from JSON bytes."""
    payload = json.loads(blob.decode("utf-8"))
    exps_raw = payload.get("experiments", [])
    exps: list[ExperimentEntry] = []
    for e in exps_raw:
        hist = _b64_csv_to_df(e["history_b64"])
        exps.append(
            build_experiment_entry(
                name=e.get("name"),
                graph_id=e.get("graph_id"),
                attack_kind=e.get("attack_kind"),
                params=e.get("params", {}),
                history=hist,
                entry_id=e.get("id"),
                created_at=e.get("created_at", 0.0),
            )
        )
    return exps
