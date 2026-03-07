from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd


def _json_default(obj: Any):
    """Serialize numpy/pandas scalars safely for JSON export."""
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass
    return str(obj)


def infer_output_format(out_path: str, out_format: str = "auto") -> str:
    """Infer output format from extension unless out_format is explicitly set."""
    fmt = str(out_format).strip().lower()
    if fmt != "auto":
        return fmt
    suffix = Path(out_path).suffix.lower()
    if suffix == ".csv":
        return "csv"
    if suffix in {".xlsx", ".xls"}:
        return "xlsx"
    return "json"


def payload_to_flat_row(payload: dict) -> dict:
    """
    Flatten nested metrics payload into one tabular row.

    Example keys:
      - summary__N, summary__E, ...
      - settings__seed, settings__weight_policy, ...
      - metric columns copied from payload["metrics"]
    """
    row: dict[str, Any] = {}

    summary = payload.get("summary")
    if isinstance(summary, dict):
        for key, value in summary.items():
            row[f"summary__{key}"] = value
    elif summary is not None:
        row["summary"] = summary

    settings = payload.get("settings")
    if isinstance(settings, dict):
        for key, value in settings.items():
            row[f"settings__{key}"] = value
    elif settings is not None:
        row["settings"] = settings

    metrics = payload.get("metrics")
    if isinstance(metrics, dict):
        for key, value in metrics.items():
            row[str(key)] = value
    elif metrics is not None:
        row["metrics"] = metrics

    for key in ("mode", "input", "family", "kind"):
        if key in payload:
            row[key] = payload[key]

    return row


def payload_to_dataframe(payload: dict) -> pd.DataFrame:
    """Convert metrics payload to a single-row DataFrame."""
    return pd.DataFrame([payload_to_flat_row(payload)])


def export_metrics_payload(payload: dict, out: str, out_format: str = "auto") -> str:
    """
    Export metrics payload in json/csv/xlsx format.

    Returns
    -------
    str
        Actual format that was used.
    """
    fmt = infer_output_format(out, out_format)
    path = Path(out)

    if fmt == "json":
        txt = json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default)
        path.write_text(txt, encoding="utf-8")
        return fmt

    df = payload_to_dataframe(payload)
    if fmt == "csv":
        df.to_csv(path, index=False)
        return fmt
    if fmt == "xlsx":
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="metrics", index=False)
        return fmt

    raise ValueError(f"Unsupported output format: {fmt}")


def experiments_to_xlsx_bytes(experiments: list) -> bytes:
    """
    Export experiments to an Excel workbook in-memory.

    Workbook layout:
      - index: metadata table for experiments
      - exp_001, exp_002, ...: one sheet per experiment history
    """
    buf = BytesIO()
    index_rows: list[dict[str, Any]] = []

    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for i, exp in enumerate(experiments, start=1):
            if hasattr(exp, "id"):
                exp_id = exp.id
                name = exp.name
                graph_id = exp.graph_id
                attack_kind = exp.attack_kind
                params = exp.params or {}
                created_at = exp.created_at
                history = exp.history.copy()
            else:
                exp_id = exp["id"]
                name = exp["name"]
                graph_id = exp["graph_id"]
                attack_kind = exp["attack_kind"]
                params = exp.get("params", {})
                created_at = exp.get("created_at", 0.0)
                history = exp["history"].copy()

            sheet_name = f"exp_{i:03d}"
            history.to_excel(writer, sheet_name=sheet_name, index=False)

            row = {
                "sheet": sheet_name,
                "id": exp_id,
                "name": name,
                "graph_id": graph_id,
                "attack_kind": attack_kind,
                "created_at": created_at,
            }
            for k, v in params.items():
                row[f"param__{k}"] = v
            index_rows.append(row)

        pd.DataFrame(index_rows).to_excel(writer, sheet_name="index", index=False)

    buf.seek(0)
    return buf.getvalue()



def export_energy_tables_xlsx(
    energy_nodes_long: pd.DataFrame,
    energy_steps_summary: pd.DataFrame,
    energy_run_summary: dict,
) -> bytes:
    """Export energy spread tables into an in-memory XLSX workbook."""
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        pd.DataFrame([energy_run_summary or {}]).to_excel(writer, sheet_name="energy_run_summary", index=False)
        (energy_steps_summary.copy() if energy_steps_summary is not None else pd.DataFrame()).to_excel(
            writer, sheet_name="energy_steps_summary", index=False
        )
        (energy_nodes_long.copy() if energy_nodes_long is not None else pd.DataFrame()).to_excel(
            writer, sheet_name="energy_nodes_long", index=False
        )
    buf.seek(0)
    return buf.getvalue()


def export_energy_tables_csv_zip(
    energy_nodes_long: pd.DataFrame,
    energy_steps_summary: pd.DataFrame,
    energy_run_summary: dict,
) -> bytes:
    """Export energy spread tables as a zip with CSV files."""
    buf = BytesIO()
    with ZipFile(buf, mode="w", compression=ZIP_DEFLATED) as zf:
        zf.writestr(
            "energy_run_summary.csv",
            pd.DataFrame([energy_run_summary or {}]).to_csv(index=False).encode("utf-8"),
        )
        zf.writestr(
            "energy_steps_summary.csv",
            (energy_steps_summary.copy() if energy_steps_summary is not None else pd.DataFrame()).to_csv(index=False).encode("utf-8"),
        )
        zf.writestr(
            "energy_nodes_long.csv",
            (energy_nodes_long.copy() if energy_nodes_long is not None else pd.DataFrame()).to_csv(index=False).encode("utf-8"),
        )
    buf.seek(0)
    return buf.getvalue()
