from __future__ import annotations

import gc
import json
import platform
import re
import shutil
import tempfile
import time
from datetime import datetime
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Iterable
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd

from .cli import (
    _attack_payload_from_graph,
    _build_metrics_payload,
    _iter_input_files,
    _json_default,
    _metrics_payload_from_graph,
    _run_attack_payload,
)
from .config import settings
from .core.physics import simulate_energy_flow
from .energy_export import (
    energy_run_summary_dict,
    frames_to_energy_nodes_long,
    frames_to_energy_steps_summary,
)
from .exporters import export_energy_tables_xlsx, export_metrics_payload, payload_to_flat_row
from .mat_packed import load_packed_mat_bundle, packed_row_to_matrix
from .matrix_import import matrix_to_graph
from .robustness import graph_resistance_summary

# Колбэк прогресса: done, total, label
ProgressCb = Callable[[int, int, str], None]


def _norm_meta_token(value) -> str:
    """Normalize IDs/group labels for tolerant metadata matching."""
    if value is None:
        return ""
    txt = str(value).strip()
    if not txt or txt.lower() in {"nan", "none", "null"}:
        return ""
    txt = Path(txt).name
    txt = txt.rsplit(".", 1)[0]
    txt = re.sub(r"[^A-Za-z0-9]+", "", txt).lower()
    return txt


def _split_csv_tokens(value: str) -> list[str]:
    """Split comma-separated UI values while ignoring empty chunks."""
    return [tok.strip() for tok in str(value or "").split(",") if tok.strip()]


def _auto_pick_column(columns: list[str], preferred: list[str]) -> str | None:
    """Find the best matching column by exact then substring match."""
    normalized = {str(col).strip().lower(): col for col in columns}
    for cand in preferred:
        if cand in normalized:
            return normalized[cand]
    for col in columns:
        norm = str(col).strip().lower()
        for cand in preferred:
            if cand in norm:
                return col
    return None


def _load_batch_metadata(
    meta_path: str | Path | None,
    *,
    id_col: str = "",
    group_col: str = "",
) -> tuple[pd.DataFrame | None, dict[str, str]]:
    """Load metadata CSV/XLSX and infer subject/group columns when possible."""
    if not str(meta_path or "").strip():
        return None, {}

    path = Path(str(meta_path)).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}")

    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        meta = pd.read_excel(path)
    else:
        meta = pd.read_csv(path)

    if meta.empty:
        raise ValueError(f"Metadata file is empty: {path}")

    meta = meta.copy()
    meta.columns = [str(col) for col in meta.columns]

    id_col_resolved = str(id_col or "").strip() or _auto_pick_column(
        list(meta.columns),
        [
            "subject_id",
            "subject",
            "subjectname",
            "subject_name",
            "participant_id",
            "participant",
            "filename",
            "file",
            "name",
            "id",
            "sz",
            "hc",
            "control_id",
            "patient_id",
        ],
    )
    if not id_col_resolved or id_col_resolved not in meta.columns:
        raise ValueError("Could not infer metadata subject-id column. Set metadata_id_col explicitly.")

    group_col_resolved = str(group_col or "").strip() or _auto_pick_column(
        list(meta.columns),
        ["group", "label", "class", "cohort", "diagnosis", "diag", "dx", "status"],
    )
    if not group_col_resolved or group_col_resolved not in meta.columns:
        raise ValueError("Could not infer metadata group column. Set metadata_group_col explicitly.")

    meta["__meta_subject_key"] = meta[id_col_resolved].map(_norm_meta_token)
    meta["__meta_group_key"] = meta[group_col_resolved].map(_norm_meta_token)
    meta = meta[meta["__meta_subject_key"] != ""].copy()
    meta = meta.drop_duplicates(subset=["__meta_subject_key"], keep="first")

    return meta, {
        "meta_path": str(path),
        "id_col": str(id_col_resolved),
        "group_col": str(group_col_resolved),
    }


def _match_job_metadata(job: dict, meta_df: pd.DataFrame | None) -> dict:
    """Match one batch job against metadata row using tolerant subject/file keys."""
    if meta_df is None or meta_df.empty:
        return {}

    path = Path(job.get("path", ""))
    candidates = [
        job.get("subject_name", ""),
        path.stem,
        path.name,
        job.get("relative_input", ""),
        job.get("input", ""),
    ]

    seen = set()
    for raw in candidates:
        key = _norm_meta_token(raw)
        if not key or key in seen:
            continue
        seen.add(key)
        matched = meta_df.loc[meta_df["__meta_subject_key"] == key]
        if not matched.empty:
            row = matched.iloc[0].to_dict()
            row["__matched_meta_key"] = key
            return row
    return {}


def _healthy_values_set(raw: str | None) -> set[str]:
    """Normalize configured healthy tokens into a comparable set."""
    tokens = set(_split_csv_tokens(str(raw or "healthy,control,hc,0,false")))
    return {_norm_meta_token(tok) for tok in tokens if _norm_meta_token(tok)}


def _job_is_healthy(meta_row: dict, healthy_values: set[str], *, group_col: str) -> bool | None:
    """Return health flag when metadata group is present, else ``None``."""
    if not meta_row or not group_col:
        return None
    if group_col not in meta_row:
        return None
    return _norm_meta_token(meta_row.get(group_col, "")) in healthy_values


def _run_info_txt(out_dir: Path) -> Path:
    """Return path to a human-readable run info file."""
    return Path(out_dir) / "RUN_INFO.txt"


def _run_info_json(out_dir: Path) -> Path:
    """Return path to a machine-readable run info file."""
    return Path(out_dir) / "run_info.json"


def _latest_run_pointer_path(base_dir: Path) -> Path:
    """Return path to a small pointer file with the latest batch run directory."""
    return Path(base_dir) / "LAST_BATCH_RUN.txt"


def write_run_metadata(
    out_dir: Path,
    *,
    base_dir: str | Path | None = None,
    run_label: str = "",
    seed: int | None = None,
    mode: str = "",
    status: str = "created",
    extra: dict | None = None,
) -> dict:
    """Persist explicit run-location metadata so users can always find outputs."""
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    base_path = Path(base_dir).expanduser().resolve() if base_dir is not None else out_dir.parent.resolve()
    info = {
        "status": str(status),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(out_dir),
        "base_dir": str(base_path),
        "run_label": str(run_label or ""),
        "seed": None if seed is None else int(seed),
        "mode": str(mode or ""),
        "platform": platform.platform(),
        "cwd": str(Path.cwd().resolve()),
    }
    if extra:
        info.update(extra)

    _run_info_json(out_dir).write_text(
        json.dumps(info, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    txt_lines = [
        f"status: {info['status']}",
        f"created_at: {info['created_at']}",
        f"run_dir: {info['run_dir']}",
        f"base_dir: {info['base_dir']}",
        f"run_label: {info['run_label']}",
        f"seed: {info['seed']}",
        f"mode: {info['mode']}",
        f"cwd: {info['cwd']}",
    ]
    for key, value in sorted((extra or {}).items()):
        txt_lines.append(f"{key}: {value}")
    _run_info_txt(out_dir).write_text("\n".join(txt_lines) + "\n", encoding="utf-8")
    _latest_run_pointer_path(base_path).write_text(str(out_dir) + "\n", encoding="utf-8")
    return info


def _iter_jobs_stream(files: Iterable[Path], args, *, input_dir: Path):
    """Yield expanded graph jobs lazily without materializing the whole batch."""
    for path in files:
        yield from _iter_expanded_graph_jobs(Path(path), args, input_dir=input_dir)


def _estimate_total_jobs(files: Iterable[Path]) -> int:
    """Estimate expanded job count for progress reporting."""
    total = 0
    for path in files:
        try:
            total += int(inspect_batch_file(Path(path)).get("expanded_graphs") or 1)
        except Exception:
            total += 1
    return total


def _chunk_size_from_args(args) -> int:
    """Return chunk size for intermediate flushes; 0 disables chunk flushing."""
    try:
        value = int(getattr(args, "batch_chunk_size", 10))
    except Exception:
        value = 10
    return max(0, value)


def _flush_chunk_tables(out_dir: Path, *, mode_name: str, chunk_idx: int, rows_chunk: list[dict], manifest_chunk: list[dict]) -> None:
    """Persist one chunk summary to disk so large batch runs can be exported incrementally."""
    if not rows_chunk and not manifest_chunk:
        return
    chunks_dir = Path(out_dir) / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{mode_name}_chunk_{int(chunk_idx):04d}"
    df_rows = pd.DataFrame(rows_chunk)
    df_rows.to_csv(chunks_dir / f"{stem}.csv", index=False)
    with pd.ExcelWriter(chunks_dir / f"{stem}.xlsx", engine="openpyxl") as writer:
        df_rows.to_excel(writer, sheet_name=(f"{mode_name}_chunk")[:31], index=False)
        pd.DataFrame(manifest_chunk).to_excel(writer, sheet_name="manifest", index=False)


def _metrics_payload_xlsx_bytes(payload: dict) -> bytes:
    """Serialize one metrics payload to XLSX bytes.

    The workbook keeps both flattened and structured sections so downstream
    users can inspect the same payload in Excel without opening JSON.
    """
    buf = BytesIO()
    flat_row = payload_to_flat_row(payload)
    summary = payload.get("summary", {}) if isinstance(payload.get("summary"), dict) else {}
    settings = payload.get("settings", {}) if isinstance(payload.get("settings"), dict) else {}
    metrics = payload.get("metrics", {}) if isinstance(payload.get("metrics"), dict) else {}
    batch = payload.get("batch", {}) if isinstance(payload.get("batch"), dict) else {}
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        pd.DataFrame([flat_row]).to_excel(writer, sheet_name="metrics_flat", index=False)
        pd.DataFrame([summary]).to_excel(writer, sheet_name="summary", index=False)
        pd.DataFrame([settings]).to_excel(writer, sheet_name="settings", index=False)
        pd.DataFrame([metrics]).to_excel(writer, sheet_name="metrics", index=False)
        pd.DataFrame([batch]).to_excel(writer, sheet_name="batch", index=False)
    buf.seek(0)
    return buf.getvalue()


def _attack_payload_xlsx_bytes(payload: dict, history: pd.DataFrame) -> bytes:
    """Serialize one attack payload (+history) to XLSX bytes."""
    buf = BytesIO()
    summary = payload.get("summary", {}) if isinstance(payload.get("summary"), dict) else {}
    final_row = payload.get("final_row", {}) if isinstance(payload.get("final_row"), dict) else {}
    settings = payload.get("settings", {}) if isinstance(payload.get("settings"), dict) else {}
    batch = payload.get("batch", {}) if isinstance(payload.get("batch"), dict) else {}
    meta = {
        "mode": payload.get("mode", "attack"),
        "input": payload.get("input", ""),
        "family": payload.get("family", ""),
        "kind": payload.get("kind", ""),
    }
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        pd.DataFrame([meta]).to_excel(writer, sheet_name="meta", index=False)
        pd.DataFrame([summary]).to_excel(writer, sheet_name="summary", index=False)
        pd.DataFrame([final_row]).to_excel(writer, sheet_name="final_row", index=False)
        pd.DataFrame([settings]).to_excel(writer, sheet_name="settings", index=False)
        pd.DataFrame([batch]).to_excel(writer, sheet_name="batch", index=False)
        (history.copy() if history is not None else pd.DataFrame()).to_excel(writer, sheet_name="history", index=False)
    buf.seek(0)
    return buf.getvalue()


def _single_row_xlsx_bytes(row: dict, *, sheet_name: str = "summary") -> bytes:
    """Serialize a single summary row to XLSX bytes."""
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        pd.DataFrame([row or {}]).to_excel(writer, sheet_name=sheet_name[:31], index=False)
    buf.seek(0)
    return buf.getvalue()


def _append_row_csv(csv_path: Path, row: dict) -> None:
    """Append one row immediately so partial results survive crashes."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row or {}]).to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)


def _read_csv_maybe(csv_path: Path) -> pd.DataFrame:
    """Read CSV if it exists and has data, otherwise return an empty DataFrame."""
    csv_path = Path(csv_path)
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(csv_path)


def _rewrite_summary_xlsx(xlsx_path: Path, df: pd.DataFrame, *, sheet_name: str) -> Path:
    """Rewrite summary workbook from an already materialized DataFrame."""
    xlsx_path = Path(xlsx_path)
    xlsx_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        (df.copy() if df is not None else pd.DataFrame()).to_excel(writer, sheet_name=sheet_name[:31], index=False)
    return xlsx_path


def _gc_collect() -> None:
    """Force GC between items to limit memory growth on long batch runs."""
    gc.collect()


def _progress_log_path(out_dir: Path) -> Path:
    """Return path to a per-run text progress log."""
    return Path(out_dir) / "progress.log"


def _append_progress_log(out_dir: Path, message: str) -> None:
    """Append one timestamped line to the batch progress log."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_path = _progress_log_path(out_dir)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(f"[{ts}] {message}\n")


def _pause_if_requested(
    out_dir: Path,
    *,
    progress_cb: ProgressCb | None = None,
    idx: int = 0,
    total: int = 0,
    label: str = "",
) -> None:
    """Pause batch processing while pause marker files exist in output directory."""
    pause_files = [Path(out_dir) / ".pause", Path(out_dir) / "PAUSE"]
    logged = False
    while any(p.exists() for p in pause_files):
        if not logged:
            _append_progress_log(out_dir, f"PAUSED at {idx}/{total} :: {label}")
            if progress_cb:
                progress_cb(idx, total, f"paused :: {label}")
            logged = True
        time.sleep(2.0)
    if logged:
        _append_progress_log(out_dir, f"RESUMED at {idx}/{total} :: {label}")


def _expected_paths_for_item(mode: str, *, out_dir: Path, item_name: str) -> dict[str, Path]:
    """Return expected artifact paths for one batch item in a given mode."""
    out_dir = Path(out_dir)
    if mode == "metrics":
        return {
            "json_path": out_dir / "per_item" / f"{item_name}.json",
            "xlsx_path": out_dir / "per_item" / f"{item_name}.xlsx",
        }
    if mode == "attack":
        return {
            "json_path": out_dir / "payloads" / f"{item_name}.json",
            "history_csv_path": out_dir / "histories" / f"{item_name}.csv",
            "xlsx_path": out_dir / "per_item" / f"{item_name}.xlsx",
        }
    return {"xlsx_path": out_dir / "per_item" / f"{item_name}.xlsx"}


def _item_already_done(mode: str, *, out_dir: Path, item_name: str) -> tuple[bool, dict[str, str]]:
    """Check whether all expected artifacts already exist for one batch item."""
    paths = _expected_paths_for_item(mode, out_dir=out_dir, item_name=item_name)
    done = bool(paths) and all(p.exists() and p.stat().st_size > 0 for p in paths.values())
    return done, {k: str(v.resolve()) for k, v in paths.items()}


def _skipped_existing_row(
    job: dict,
    idx: int,
    mode: str,
    resolved_paths: dict[str, str],
    *,
    extra: dict | None = None,
) -> tuple[dict, dict]:
    """Build summary and manifest rows for items skipped due to existing artifacts."""
    path = Path(job["path"])
    row = {
        "mode": mode,
        "batch_index": idx,
        "input": str(job.get("input", path)),
        "input_file": path.name,
        "input_stem": path.stem,
        "input_suffix": path.suffix.lower(),
        "relative_input": str(job.get("relative_input", "")),
        "subject_name": str(job.get("subject_name") or ""),
        "subject_index": job.get("subject_index"),
        "status": "skipped_existing",
        "error": "",
        **(extra or {}),
        **resolved_paths,
    }
    manifest_item = {
        **_base_manifest_row(job, idx, mode),
        "status": "skipped_existing",
        "error": "",
        "xlsx_path": resolved_paths.get("xlsx_path", ""),
        "json_path": resolved_paths.get("json_path", ""),
        "history_csv_path": resolved_paths.get("history_csv_path", ""),
    }
    return row, manifest_item


def _finalize_batch_outputs(
    out_dir: Path,
    *,
    summary_csv: Path,
    summary_xlsx: Path,
    summary_sheet: str,
    manifest_path: Path,
    files: list[Path],
    args,
    mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Finalize summary/manifest artifacts and return loaded DataFrames."""
    summary_csv = Path(summary_csv)
    manifest_csv = Path(out_dir) / "manifest.csv"
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    if not summary_csv.exists():
        pd.DataFrame().to_csv(summary_csv, index=False)
    if not manifest_csv.exists():
        pd.DataFrame().to_csv(manifest_csv, index=False)
    df = _read_csv_maybe(summary_csv)
    manifest_df = _read_csv_maybe(manifest_csv)
    _rewrite_summary_xlsx(summary_xlsx, df, sheet_name=summary_sheet)
    _rewrite_summary_xlsx(Path(out_dir) / "manifest.xlsx", manifest_df, sheet_name="manifest")
    manifest_path.write_text(
        json.dumps(build_batch_manifest(files=files, args=args, mode=mode), ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    return df, manifest_df


def _write_manifest_tables(out_dir: Path, df: pd.DataFrame, *, prefix: str = "manifest") -> tuple[Path, Path]:
    """Write manifest as CSV + XLSX and return paths to saved files."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{prefix}.csv"
    xlsx_path = out_dir / f"{prefix}.xlsx"
    df_to_save = df.copy() if df is not None else pd.DataFrame()
    df_to_save.to_csv(csv_path, index=False)
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df_to_save.to_excel(writer, sheet_name=prefix[:31], index=False)
    return csv_path, xlsx_path


def _base_manifest_row(job: dict, idx: int, mode: str) -> dict:
    """Return common manifest fields for one expanded graph job."""
    path = Path(job["path"])
    return {
        "mode": mode,
        "batch_index": idx,
        "input": str(job.get("input", path)),
        "input_file": path.name,
        "input_stem": path.stem,
        "input_suffix": path.suffix.lower(),
        "relative_input": str(job.get("relative_input", "")),
        "subject_name": str(job.get("subject_name") or ""),
        "subject_index": job.get("subject_index"),
    }


def zip_tree_to_file(root_dir: Path, zip_path: Path) -> Path:
    """Create a recursive zip archive of ``root_dir`` at ``zip_path``."""
    root_dir = Path(root_dir)
    zip_path = Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(zip_path, mode="w", compression=ZIP_DEFLATED) as zf:
        for path in sorted(root_dir.rglob("*")):
            if path.is_file() and path.resolve() != zip_path.resolve():
                zf.write(path, arcname=str(path.relative_to(root_dir)))
    return zip_path


def inspect_batch_file(path: Path) -> dict:
    """Build preview metadata for a batch file.

    For packed ``.mat`` files, this performs a lightweight parse to estimate
    how many graph jobs will be expanded during batch execution.
    """
    suffix = path.suffix.lower()
    row = {
        "file": str(path),
        "suffix": suffix,
        "kind_guess": "matrix" if suffix in {".mat", ".npy", ".npz"} else "edge",
        "expanded_graphs": 1,
        "packed_mat": False,
        "n_subjects": None,
        "n_nodes": None,
        "preview_error": "",
    }
    if suffix == ".mat":
        try:
            bundle = load_packed_mat_bundle(path.read_bytes())
            row["packed_mat"] = True
            row["expanded_graphs"] = int(bundle.n_subjects)
            row["n_subjects"] = int(bundle.n_subjects)
            row["n_nodes"] = int(bundle.n_nodes)
        except Exception as exc:  # pylint: disable=broad-except
            row["preview_error"] = f"{type(exc).__name__}: {exc}"
    return row


def stage_batch_inputs(
    *,
    source_mode: str,
    input_dir: str | Path | None = None,
    uploaded_files: list | None = None,
    uploaded_zip_name: str = "",
    uploaded_zip_bytes: bytes | None = None,
) -> tuple[Path, list[Path], Callable[[], None]]:
    """Stage batch inputs into a local directory for folder/upload/multi-upload modes."""
    mode = str(source_mode).strip().lower() or "local_folder"

    if mode == "local_folder":
        if input_dir is None or not str(input_dir).strip():
            raise ValueError("Не указана входная папка")
        base = Path(str(input_dir)).expanduser().resolve()
        if not base.exists() or not base.is_dir():
            raise FileNotFoundError(f"Папка не найдена: {base}")
        return base, [], (lambda: None)

    tmpdir = Path(tempfile.mkdtemp(prefix="kodik_batch_"))

    def _cleanup() -> None:
        """Remove staged temporary directory for upload-based modes."""
        shutil.rmtree(tmpdir, ignore_errors=True)

    if mode == "uploaded_files":
        files = list(uploaded_files or [])
        if not files:
            _cleanup()
            raise ValueError("Не загружены файлы для batch-режима")
        staged: list[Path] = []
        for idx, up in enumerate(files, start=1):
            name = Path(getattr(up, "name", f"upload_{idx:04d}")).name
            dst = tmpdir / name
            dst.write_bytes(up.getvalue())
            staged.append(dst)
        return tmpdir, staged, _cleanup

    if mode == "uploaded_zip":
        if not uploaded_zip_bytes:
            _cleanup()
            raise ValueError("Не загружен zip-архив для batch-режима")
        archive_name = Path(uploaded_zip_name or "batch_upload.zip").name
        archive_path = tmpdir / archive_name
        archive_path.write_bytes(uploaded_zip_bytes)
        with ZipFile(archive_path, "r") as zf:
            bad = [n for n in zf.namelist() if Path(n).is_absolute() or ".." in Path(n).parts]
            if bad:
                _cleanup()
                raise ValueError("Zip содержит небезопасные пути")
            zf.extractall(tmpdir / "unzipped")
        return tmpdir / "unzipped", [], _cleanup

    _cleanup()
    raise ValueError(f"Неизвестный source_mode: {source_mode}")


def discover_batch_files(
    *,
    source_mode: str,
    pattern: str,
    recursive: bool,
    limit: int = 0,
    input_dir: str | Path | None = None,
    uploaded_files: list | None = None,
    uploaded_zip_name: str = "",
    uploaded_zip_bytes: bytes | None = None,
) -> tuple[Path, list[Path], Callable[[], None]]:
    """Stage inputs if needed and return the concrete file list for batch UI."""
    staged_root, _, cleanup_cb = stage_batch_inputs(
        source_mode=source_mode,
        input_dir=input_dir,
        uploaded_files=uploaded_files,
        uploaded_zip_name=uploaded_zip_name,
        uploaded_zip_bytes=uploaded_zip_bytes,
    )
    files = _iter_input_files(
        Path(staged_root),
        recursive=bool(recursive),
        pattern=str(pattern),
        limit=int(limit),
    )
    return Path(staged_root), files, cleanup_cb


def _build_graph_from_packed_subject(args, packed_row, n_nodes: int):
    """Restore one subject from a packed MAT row and apply normal matrix policies."""
    corr = packed_row_to_matrix(packed_row, int(n_nodes))
    return matrix_to_graph(
        corr,
        sign_policy=str(getattr(args, "sign_policy", "abs")),
        threshold_mode=str(getattr(args, "threshold_mode", "density")),
        threshold_value=float(getattr(args, "threshold_value", 0.15)),
        shift=float(getattr(args, "shift", 0.0)),
        labels=None,
        use_lcc=bool(getattr(args, "lcc", False)),
    )


def _iter_expanded_graph_jobs(path: Path, args, *, input_dir: Path):
    """Yield one or many graph jobs for a file; packed MAT bundles expand per subject."""
    rel = str(path.relative_to(input_dir))
    suffix = path.suffix.lower()
    use_matrix = str(getattr(args, "input_kind", "auto")) == "matrix" or (
        str(getattr(args, "input_kind", "auto")) == "auto" and suffix == ".mat"
    )
    if suffix == ".mat" and use_matrix:
        try:
            bundle = load_packed_mat_bundle(path.read_bytes())
            for subj in bundle.subjects:
                input_label = f"{path}::{subj.subject_name}"
                yield {
                    "path": path,
                    "input": input_label,
                    "relative_input": rel,
                    "subject_name": str(subj.subject_name),
                    "subject_index": int(subj.index),
                    "graph": _build_graph_from_packed_subject(args, subj.packed_edges, bundle.n_nodes),
                }
            return
        except Exception:  # pylint: disable=broad-except
            pass

    yield {
        "path": path,
        "input": str(path),
        "relative_input": rel,
        "subject_name": "",
        "subject_index": None,
        "graph": None,
    }



def run_batch_energy(args, *, progress_cb: ProgressCb | None = None) -> tuple[Path, pd.DataFrame]:
    """Run energy diffusion batch and return output dir and summary DataFrame."""
    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    per_item_dir = out_dir / "per_item"
    per_item_dir.mkdir(parents=True, exist_ok=True)

    explicit_files = [Path(p) for p in getattr(args, "selected_files", []) or []]
    files = explicit_files or _iter_input_files(
        input_dir,
        recursive=bool(args.recursive),
        pattern=str(args.pattern),
        limit=int(args.limit),
    )
    files = [Path(p) for p in files]
    total = _estimate_total_jobs(files)
    rows_chunk: list[dict] = []
    manifest_chunk: list[dict] = []
    chunk_size = _chunk_size_from_args(args)
    chunk_idx = 0
    summary_csv = Path(args.summary_out) if str(getattr(args, "summary_out", "")).strip() else out_dir / "energy_summary.csv"
    manifest_csv = out_dir / "manifest.csv"
    if summary_csv.exists():
        summary_csv.unlink()
    if manifest_csv.exists():
        manifest_csv.unlink()
    _append_progress_log(out_dir, f"START energy total={total}")

    for idx, job in enumerate(_iter_jobs_stream(files, args, input_dir=input_dir), start=1):
        path = Path(job["path"])
        label = job.get("subject_name") or path.name
        _pause_if_requested(out_dir, progress_cb=progress_cb, idx=idx - 1, total=total, label=str(label))
        if progress_cb:
            progress_cb(idx - 1, total, f"energy :: {label}")
        item_suffix = f"__{safe_stem(job['subject_name'])}" if job.get("subject_name") else ""
        item_name = f"{idx:04d}__{safe_stem(path.stem)}{item_suffix}"
        done, resolved_paths = _item_already_done("energy", out_dir=out_dir, item_name=item_name)
        if bool(getattr(args, "skip_existing", True)) and done:
            row, manifest_item = _skipped_existing_row(job, idx, "energy", resolved_paths)
            _append_row_csv(summary_csv, row)
            _append_row_csv(manifest_csv, manifest_item)
            rows_chunk.append(row)
            manifest_chunk.append(manifest_item)
            _append_progress_log(out_dir, f"[{idx}/{total}] SKIP existing :: {label}")
            del row, manifest_item
            _gc_collect()
            if chunk_size and len(rows_chunk) >= chunk_size:
                chunk_idx += 1
                _flush_chunk_tables(out_dir, mode_name="energy", chunk_idx=chunk_idx, rows_chunk=rows_chunk, manifest_chunk=manifest_chunk)
                rows_chunk = []
                manifest_chunk = []
            continue
        try:
            if job.get("graph") is None:
                raise ValueError("Energy batch currently supports matrix/MAT inputs expanded to graphs")
            graph = job["graph"]
            srcs = []
            if graph.number_of_nodes() > 0:
                deg = dict(graph.degree(weight="weight"))
                srcs = [max(deg, key=deg.get)] if deg else []
            node_frames, edge_frames = simulate_energy_flow(
                graph,
                steps=int(getattr(args, "energy_steps", 50)),
                flow_mode=str(getattr(args, "energy_flow_mode", "rw")),
                damping=float(getattr(args, "energy_damping", 1.0)),
                sources=srcs or None,
                phys_injection=float(getattr(args, "energy_phys_injection", 0.15)),
                phys_leak=float(getattr(args, "energy_phys_leak", 0.02)),
                phys_cap_mode=str(getattr(args, "energy_phys_cap_mode", "strength")),
                rw_impulse=bool(getattr(args, "energy_rw_impulse", True)),
            )
            energy_nodes_long = frames_to_energy_nodes_long(graph, node_frames, sources=srcs)
            energy_steps_summary = frames_to_energy_steps_summary(graph, node_frames, edge_frames, sources=srcs)
            energy_run_summary = energy_run_summary_dict(
                graph, node_frames, edge_frames, sources=srcs, flow_mode=str(getattr(args, "energy_flow_mode", "rw"))
            )
            energy_run_summary.update({
                "mode": "energy",
                "batch_index": idx,
                "input": str(job["input"]),
                "input_file": path.name,
                "input_stem": path.stem,
                "input_suffix": path.suffix.lower(),
                "relative_input": str(job["relative_input"]),
                "subject_name": str(job.get("subject_name") or ""),
                "subject_index": job.get("subject_index"),
                "status": "ok",
                "error": "",
            })
            xlsx_path = per_item_dir / f"{item_name}.xlsx"
            xlsx_path.write_bytes(export_energy_tables_xlsx(energy_nodes_long, energy_steps_summary, energy_run_summary))
            xlsx_abs = str(xlsx_path.resolve())
            row = {**energy_run_summary, "xlsx_path": xlsx_abs}
            manifest_item = {
                **_base_manifest_row(job, idx, "energy"),
                "status": "ok",
                "error": "",
                "xlsx_path": xlsx_abs,
                "json_path": "",
                "history_csv_path": "",
            }
        except Exception as exc:
            row = {
                "mode": "energy",
                "batch_index": idx,
                "input": str(path),
                "input_file": path.name,
                "input_stem": path.stem,
                "input_suffix": path.suffix.lower(),
                "relative_input": str(job["relative_input"]),
                "subject_name": str(job.get("subject_name") or ""),
                "subject_index": job.get("subject_index"),
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }
            manifest_item = {
                **_base_manifest_row(job, idx, "energy"),
                "status": "error",
                "error": row["error"],
                "xlsx_path": "",
                "json_path": "",
                "history_csv_path": "",
            }
        _append_row_csv(summary_csv, row)
        _append_row_csv(manifest_csv, manifest_item)
        rows_chunk.append(row)
        manifest_chunk.append(manifest_item)
        _append_progress_log(out_dir, f"[{idx}/{total}] DONE energy :: {label} :: {row.get('status', 'ok')}")
        del row, manifest_item
        _gc_collect()
        if chunk_size and len(rows_chunk) >= chunk_size:
            chunk_idx += 1
            _flush_chunk_tables(out_dir, mode_name="energy", chunk_idx=chunk_idx, rows_chunk=rows_chunk, manifest_chunk=manifest_chunk)
            rows_chunk = []
            manifest_chunk = []

    if rows_chunk or manifest_chunk:
        chunk_idx += 1
        _flush_chunk_tables(out_dir, mode_name="energy", chunk_idx=chunk_idx, rows_chunk=rows_chunk, manifest_chunk=manifest_chunk)

    summary_xlsx = out_dir / "energy_summary.xlsx"
    manifest_path = out_dir / "manifest.json"
    df, _ = _finalize_batch_outputs(
        out_dir,
        summary_csv=summary_csv,
        summary_xlsx=summary_xlsx,
        summary_sheet="energy_runs",
        manifest_path=manifest_path,
        files=files,
        args=args,
        mode="batch-energy",
    )
    _append_progress_log(out_dir, f"FINISH energy rows={len(df)}")
    if progress_cb:
        progress_cb(total, total, f"saved -> {summary_csv.name}")
    return out_dir, df

def run_batch_resistance(args, *, progress_cb: ProgressCb | None = None) -> tuple[Path, pd.DataFrame]:
    """Run structural resistance batch and return output dir and summary DataFrame."""
    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    per_item_dir = out_dir / "per_item"
    per_item_dir.mkdir(parents=True, exist_ok=True)
    explicit_files = [Path(p) for p in getattr(args, "selected_files", []) or []]
    files = explicit_files or _iter_input_files(
        input_dir,
        recursive=bool(args.recursive),
        pattern=str(args.pattern),
        limit=int(args.limit),
    )
    files = [Path(p) for p in files]
    total = _estimate_total_jobs(files)
    rows_chunk: list[dict] = []
    manifest_chunk: list[dict] = []
    chunk_size = _chunk_size_from_args(args)
    chunk_idx = 0
    summary_csv = Path(args.summary_out) if str(getattr(args, "summary_out", "")).strip() else out_dir / "resistance_summary.csv"
    manifest_csv = out_dir / "manifest.csv"
    if summary_csv.exists():
        summary_csv.unlink()
    if manifest_csv.exists():
        manifest_csv.unlink()
    _append_progress_log(out_dir, f"START resistance total={total}")

    for idx, job in enumerate(_iter_jobs_stream(files, args, input_dir=input_dir), start=1):
        path = Path(job["path"])
        label = job.get("subject_name") or path.name
        _pause_if_requested(out_dir, progress_cb=progress_cb, idx=idx - 1, total=total, label=str(label))
        if progress_cb:
            progress_cb(idx - 1, total, f"resistance :: {label}")
        item_suffix = f"__{safe_stem(job['subject_name'])}" if job.get("subject_name") else ""
        item_name = f"{idx:04d}__{safe_stem(path.stem)}{item_suffix}"
        done, resolved_paths = _item_already_done("resistance", out_dir=out_dir, item_name=item_name)
        if bool(getattr(args, "skip_existing", True)) and done:
            row, manifest_item = _skipped_existing_row(job, idx, "resistance", resolved_paths)
            _append_row_csv(summary_csv, row)
            _append_row_csv(manifest_csv, manifest_item)
            rows_chunk.append(row)
            manifest_chunk.append(manifest_item)
            _append_progress_log(out_dir, f"[{idx}/{total}] SKIP existing :: {label}")
            del row, manifest_item
            _gc_collect()
            if chunk_size and len(rows_chunk) >= chunk_size:
                chunk_idx += 1
                _flush_chunk_tables(out_dir, mode_name="resistance", chunk_idx=chunk_idx, rows_chunk=rows_chunk, manifest_chunk=manifest_chunk)
                rows_chunk = []
                manifest_chunk = []
            continue
        try:
            if job.get("graph") is None:
                raise ValueError("Resistance batch currently supports matrix/MAT inputs expanded to graphs")
            graph = job["graph"]
            res = graph_resistance_summary(graph)
            row = {
                "mode": "resistance",
                "batch_index": idx,
                "input": str(job["input"]),
                "input_file": path.name,
                "input_stem": path.stem,
                "input_suffix": path.suffix.lower(),
                "relative_input": str(job["relative_input"]),
                "subject_name": str(job.get("subject_name") or ""),
                "subject_index": job.get("subject_index"),
                "status": "ok",
                "error": "",
                **res,
            }
            xlsx_path = per_item_dir / f"{item_name}.xlsx"
            xlsx_path.write_bytes(_single_row_xlsx_bytes(row, sheet_name="resistance"))
            xlsx_abs = str(xlsx_path.resolve())
            row["xlsx_path"] = xlsx_abs
            manifest_item = {
                **_base_manifest_row(job, idx, "resistance"),
                "status": "ok",
                "error": "",
                "xlsx_path": xlsx_abs,
                "json_path": "",
                "history_csv_path": "",
            }
        except Exception as exc:
            row = {
                "mode": "resistance",
                "batch_index": idx,
                "input": str(path),
                "input_file": path.name,
                "input_stem": path.stem,
                "input_suffix": path.suffix.lower(),
                "relative_input": str(job["relative_input"]),
                "subject_name": str(job.get("subject_name") or ""),
                "subject_index": job.get("subject_index"),
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }
            manifest_item = {
                **_base_manifest_row(job, idx, "resistance"),
                "status": "error",
                "error": row["error"],
                "xlsx_path": "",
                "json_path": "",
                "history_csv_path": "",
            }
        _append_row_csv(summary_csv, row)
        _append_row_csv(manifest_csv, manifest_item)
        rows_chunk.append(row)
        manifest_chunk.append(manifest_item)
        _append_progress_log(out_dir, f"[{idx}/{total}] DONE resistance :: {label} :: {row.get('status', 'ok')}")
        del row, manifest_item
        _gc_collect()
        if chunk_size and len(rows_chunk) >= chunk_size:
            chunk_idx += 1
            _flush_chunk_tables(out_dir, mode_name="resistance", chunk_idx=chunk_idx, rows_chunk=rows_chunk, manifest_chunk=manifest_chunk)
            rows_chunk = []
            manifest_chunk = []
    if rows_chunk or manifest_chunk:
        chunk_idx += 1
        _flush_chunk_tables(out_dir, mode_name="resistance", chunk_idx=chunk_idx, rows_chunk=rows_chunk, manifest_chunk=manifest_chunk)

    summary_xlsx = out_dir / "resistance_summary.xlsx"
    manifest_path = out_dir / "manifest.json"
    df, _ = _finalize_batch_outputs(
        out_dir,
        summary_csv=summary_csv,
        summary_xlsx=summary_xlsx,
        summary_sheet="robustness_summary",
        manifest_path=manifest_path,
        files=files,
        args=args,
        mode="batch-resistance",
    )
    _append_progress_log(out_dir, f"FINISH resistance rows={len(df)}")
    if progress_cb:
        progress_cb(total, total, f"saved -> {summary_csv.name}")
    return out_dir, df

def run_batch_plan(args, *, progress_cb: ProgressCb | None = None) -> tuple[Path, dict[str, pd.DataFrame]]:
    """Run selected batch tasks (metrics/attack/energy/resistance) and return per-mode DataFrames."""
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    write_run_metadata(
        out_dir,
        base_dir=out_dir.parent,
        run_label=Path(out_dir).name,
        seed=getattr(args, "seed", None),
        mode="batch_plan",
        status="running",
        extra={
            "input_dir": str(Path(getattr(args, "input_dir", ".")).expanduser().resolve()),
            "selected_files_count": len(list(getattr(args, "selected_files", []) or [])),
        },
    )

    modes: list[str] = []
    if bool(getattr(args, "run_metrics", False)):
        modes.append("metrics")
    if bool(getattr(args, "run_attack", False)):
        modes.append("attack")
    if bool(getattr(args, "run_energy", False)):
        modes.append("energy")
    if bool(getattr(args, "run_resistance", False)):
        modes.append("resistance")
    if not modes:
        raise ValueError("Не выбрано ни одного расчёта")

    total_modes = len(modes)
    result_frames: dict[str, pd.DataFrame] = {}
    manifest_frames: list[pd.DataFrame] = []

    def _wrap_progress(mode_idx: int, mode_name: str):
        """Map per-mode progress into one global progress stream for UI."""

        def _cb(done: int, total: int, label: str) -> None:
            if progress_cb is None:
                return
            total_ = max(1, int(total))
            frac_local = min(1.0, max(0.0, float(done) / float(total_)))
            done_global = (mode_idx - 1) + frac_local
            progress_cb(done_global, total_modes, f"{mode_name}: {label}")

        return _cb

    for mode_idx, mode_name in enumerate(modes, start=1):
        mode_out_dir = out_dir / mode_name
        mode_args = build_ui_args(**vars(args))
        mode_args.out_dir = str(mode_out_dir)
        if mode_name == "metrics":
            _, df_mode = run_batch_metrics(mode_args, progress_cb=_wrap_progress(mode_idx, "metrics"))
        elif mode_name == "attack":
            _, df_mode = run_batch_attack(mode_args, progress_cb=_wrap_progress(mode_idx, "attack"))
        elif mode_name == "energy":
            _, df_mode = run_batch_energy(mode_args, progress_cb=_wrap_progress(mode_idx, "energy"))
        else:
            _, df_mode = run_batch_resistance(mode_args, progress_cb=_wrap_progress(mode_idx, "resistance"))
        result_frames[mode_name] = df_mode

        mode_manifest_csv = mode_out_dir / "manifest.csv"
        if mode_manifest_csv.exists():
            try:
                manifest_frames.append(pd.read_csv(mode_manifest_csv))
            except Exception:
                pass

    combined_xlsx = out_dir / "batch_plan_summary.xlsx"
    with pd.ExcelWriter(combined_xlsx, engine="openpyxl") as writer:
        for mode_name, df_mode in result_frames.items():
            df_mode.to_excel(writer, sheet_name=mode_name[:31], index=False)

    if manifest_frames:
        combined_manifest_df = pd.concat(manifest_frames, ignore_index=True, sort=False)
    else:
        combined_manifest_df = pd.DataFrame()
    _write_manifest_tables(out_dir, combined_manifest_df, prefix="batch_plan_manifest")

    bundle_zip = out_dir / "batch_plan_bundle.zip"
    saved_label = combined_xlsx.name
    if bool(getattr(args, "write_full_bundle", False)):
        zip_tree_to_file(out_dir, bundle_zip)
        saved_label = bundle_zip.name

    write_run_metadata(
        out_dir,
        base_dir=out_dir.parent,
        run_label=Path(out_dir).name,
        seed=getattr(args, "seed", None),
        mode="batch_plan",
        status="finished",
        extra={
            "saved_label": saved_label,
            "modes": ",".join(modes),
            "result_frames": ",".join(sorted(result_frames.keys())),
        },
    )
    if progress_cb is not None:
        progress_cb(total_modes, total_modes, f"saved -> {saved_label}")
    return out_dir, result_frames


def safe_stem(value: str) -> str:
    """Sanitize a value for safe use in output filenames."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    cleaned = cleaned.strip("._")
    return cleaned or "item"


def make_run_dir(base_dir: str | Path, *, mode: str, seed: int, run_label: str = "") -> Path:
    """Create a timestamped run directory under ``base_dir`` and persist run-location metadata."""
    base = Path(base_dir).expanduser().resolve()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    label = safe_stem(run_label) if str(run_label).strip() else mode
    run_dir = base / f"{label}__seed_{int(seed)}__{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    write_run_metadata(run_dir, base_dir=base, run_label=label, seed=int(seed), mode=str(mode), status="created")
    return run_dir


def build_batch_manifest(*, files: Iterable[Path], args, mode: str) -> dict:
    """Build a manifest with input list and runtime settings for traceability."""
    files_list = list(files)
    return {
        "mode": mode,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "seed": int(getattr(args, "seed", 0)),
        "input_dir": str(Path(getattr(args, "input_dir", ".")).expanduser().resolve()),
        "pattern": str(getattr(args, "pattern", "*")),
        "recursive": bool(getattr(args, "recursive", False)),
        "count": len(files_list),
        "files": [str(Path(p)) for p in files_list],
        "settings": {k: v for k, v in vars(args).items() if k not in {"summary_out"}},
    }


def build_ui_args(**kwargs):
    """Build an ``argparse``-like namespace for batch runners called from UI."""
    defaults = dict(
        fixed=False,
        src="src",
        dst="dst",
        min_conf=0.0,
        min_weight=0.0,
        lcc=False,
        input_kind="auto",
        mat_key="",
        sign_policy="abs",
        threshold_mode="density",
        threshold_value=0.15,
        shift=0.0,
        seed=42,
        eff_k=32,
        compute_curvature=False,
        curvature_sample_edges=120,
        curvature_max_support=settings.RICCI_MAX_SUPPORT,
        compute_heavy=True,
        skip_spectral=False,
        diameter_samples=16,
        n_jobs=1,
        pattern="*",
        recursive=False,
        limit=0,
        input_dir=".",
        out_dir="./batch_out",
        summary_out="",
        family="node",
        kind="degree",
        frac=0.5,
        steps=30,
        heavy_every=5,
        fast_mode=False,
        alpha_rewire=0.6,
        beta_replace=0.4,
        swaps_per_edge=0.5,
        replace_from="CFG",
        run_metrics=True,
        run_attack=False,
        run_energy=False,
        run_resistance=False,
        energy_steps=50,
        energy_flow_mode="rw",
        energy_damping=1.0,
        energy_phys_injection=0.15,
        energy_phys_leak=0.02,
        energy_phys_cap_mode="strength",
        energy_rw_impulse=True,
        source_mode="local_folder",
        selected_files=None,
        batch_chunk_size=10,
        write_full_bundle=False,
        metadata_path="",
        metadata_id_col="",
        metadata_group_col="",
        healthy_group_values="healthy,control,hc,0,false",
        attack_only_healthy=False,
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def run_batch_metrics(args, *, progress_cb: ProgressCb | None = None) -> tuple[Path, pd.DataFrame]:
    """Run metrics batch and return output dir and summary DataFrame."""
    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    per_item_dir = out_dir / "per_item"
    per_item_dir.mkdir(parents=True, exist_ok=True)

    explicit_files = [Path(p) for p in getattr(args, "selected_files", []) or []]
    files = explicit_files or _iter_input_files(
        input_dir,
        recursive=bool(args.recursive),
        pattern=str(args.pattern),
        limit=int(args.limit),
    )
    files = [Path(p) for p in files]
    total = _estimate_total_jobs(files)
    rows_chunk: list[dict] = []
    manifest_chunk: list[dict] = []
    chunk_size = _chunk_size_from_args(args)
    chunk_idx = 0
    summary_csv = Path(args.summary_out) if str(getattr(args, "summary_out", "")).strip() else out_dir / "metrics_summary.csv"
    manifest_csv = out_dir / "manifest.csv"
    if summary_csv.exists():
        summary_csv.unlink()
    if manifest_csv.exists():
        manifest_csv.unlink()
    _append_progress_log(out_dir, f"START metrics total={total}")

    for idx, job in enumerate(_iter_jobs_stream(files, args, input_dir=input_dir), start=1):
        path = Path(job["path"])
        label = job.get("subject_name") or path.name
        _pause_if_requested(out_dir, progress_cb=progress_cb, idx=idx - 1, total=total, label=str(label))
        if progress_cb:
            progress_cb(idx - 1, total, f"metrics :: {label}")
        item_suffix = f"__{safe_stem(job['subject_name'])}" if job.get("subject_name") else ""
        item_name = f"{idx:04d}__{safe_stem(path.stem)}{item_suffix}"
        done, resolved_paths = _item_already_done("metrics", out_dir=out_dir, item_name=item_name)
        if bool(getattr(args, "skip_existing", True)) and done:
            row, manifest_item = _skipped_existing_row(job, idx, "metrics", resolved_paths)
            _append_row_csv(summary_csv, row)
            _append_row_csv(manifest_csv, manifest_item)
            rows_chunk.append(row)
            manifest_chunk.append(manifest_item)
            _append_progress_log(out_dir, f"[{idx}/{total}] SKIP existing :: {label}")
            del row, manifest_item
            _gc_collect()
            if chunk_size and len(rows_chunk) >= chunk_size:
                chunk_idx += 1
                _flush_chunk_tables(out_dir, mode_name="metrics", chunk_idx=chunk_idx, rows_chunk=rows_chunk, manifest_chunk=manifest_chunk)
                rows_chunk = []
                manifest_chunk = []
            continue
        try:
            if job.get("graph") is not None:
                payload = _metrics_payload_from_graph(args, job["graph"], input_label=str(job["input"]))
            else:
                payload = _build_metrics_payload(args, path)
            payload["batch"] = {
                "index": idx,
                "input_file": path.name,
                "input_stem": path.stem,
                "input_suffix": path.suffix.lower(),
                "relative_input": str(job["relative_input"]),
                "subject_name": str(job.get("subject_name") or ""),
                "subject_index": job.get("subject_index"),
            }
            json_path = per_item_dir / f"{item_name}.json"
            export_metrics_payload(payload, str(json_path), "json")
            xlsx_path = per_item_dir / f"{item_name}.xlsx"
            xlsx_path.write_bytes(_metrics_payload_xlsx_bytes(payload))
            json_abs = str(json_path.resolve())
            xlsx_abs = str(xlsx_path.resolve())

            row = payload_to_flat_row(payload)
            row.update({
                "batch_index": idx,
                "input_file": path.name,
                "input_stem": path.stem,
                "input_suffix": path.suffix.lower(),
                "relative_input": str(job["relative_input"]),
                "subject_name": str(job.get("subject_name") or ""),
                "subject_index": job.get("subject_index"),
                "json_path": json_abs,
                "xlsx_path": xlsx_abs,
                "status": "ok",
                "error": "",
            })
            manifest_item = {
                **_base_manifest_row(job, idx, "metrics"),
                "status": "ok",
                "error": "",
                "xlsx_path": xlsx_abs,
                "json_path": json_abs,
                "history_csv_path": "",
            }
        except Exception as exc:  # pylint: disable=broad-except
            row = {
                "mode": "metrics",
                "batch_index": idx,
                "input": str(path),
                "input_file": path.name,
                "input_stem": path.stem,
                "input_suffix": path.suffix.lower(),
                "relative_input": str(job["relative_input"]),
                "subject_name": str(job.get("subject_name") or ""),
                "subject_index": job.get("subject_index"),
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }
            manifest_item = {
                **_base_manifest_row(job, idx, "metrics"),
                "status": "error",
                "error": row["error"],
                "xlsx_path": "",
                "json_path": "",
                "history_csv_path": "",
            }
        _append_row_csv(summary_csv, row)
        _append_row_csv(manifest_csv, manifest_item)
        rows_chunk.append(row)
        manifest_chunk.append(manifest_item)
        _append_progress_log(out_dir, f"[{idx}/{total}] DONE metrics :: {label} :: {row.get('status', 'ok')}")
        del row, manifest_item
        _gc_collect()
        if chunk_size and len(rows_chunk) >= chunk_size:
            chunk_idx += 1
            _flush_chunk_tables(out_dir, mode_name="metrics", chunk_idx=chunk_idx, rows_chunk=rows_chunk, manifest_chunk=manifest_chunk)
            rows_chunk = []
            manifest_chunk = []

    if rows_chunk or manifest_chunk:
        chunk_idx += 1
        _flush_chunk_tables(out_dir, mode_name="metrics", chunk_idx=chunk_idx, rows_chunk=rows_chunk, manifest_chunk=manifest_chunk)

    summary_xlsx = out_dir / "metrics_summary.xlsx"
    manifest_path = out_dir / "manifest.json"
    df, _ = _finalize_batch_outputs(
        out_dir,
        summary_csv=summary_csv,
        summary_xlsx=summary_xlsx,
        summary_sheet="metrics",
        manifest_path=manifest_path,
        files=files,
        args=args,
        mode="batch-metrics",
    )
    _append_progress_log(out_dir, f"FINISH metrics rows={len(df)}")

    if progress_cb:
        progress_cb(total, total, f"saved -> {summary_csv.name}")
    return out_dir, df

def run_batch_attack(args, *, progress_cb: ProgressCb | None = None) -> tuple[Path, pd.DataFrame]:
    """Run attack batch and return output dir and summary DataFrame."""
    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    json_dir = out_dir / "payloads"
    hist_dir = out_dir / "histories"
    xlsx_dir = out_dir / "per_item"
    json_dir.mkdir(parents=True, exist_ok=True)
    hist_dir.mkdir(parents=True, exist_ok=True)
    xlsx_dir.mkdir(parents=True, exist_ok=True)

    explicit_files = [Path(p) for p in getattr(args, "selected_files", []) or []]
    files = explicit_files or _iter_input_files(
        input_dir,
        recursive=bool(args.recursive),
        pattern=str(args.pattern),
        limit=int(args.limit),
    )
    files = [Path(p) for p in files]
    total = _estimate_total_jobs(files)
    rows_chunk: list[dict] = []
    manifest_chunk: list[dict] = []
    chunk_size = _chunk_size_from_args(args)
    chunk_idx = 0
    summary_csv = Path(args.summary_out) if str(getattr(args, "summary_out", "")).strip() else out_dir / "attack_summary.csv"
    manifest_csv = out_dir / "manifest.csv"
    if summary_csv.exists():
        summary_csv.unlink()
    if manifest_csv.exists():
        manifest_csv.unlink()
    _append_progress_log(out_dir, f"START attack total={total}")
    meta_df, meta_info = _load_batch_metadata(
        getattr(args, "metadata_path", ""),
        id_col=str(getattr(args, "metadata_id_col", "") or ""),
        group_col=str(getattr(args, "metadata_group_col", "") or ""),
    )
    healthy_values = _healthy_values_set(getattr(args, "healthy_group_values", ""))
    meta_group_col = str(meta_info.get("group_col", ""))

    for idx, job in enumerate(_iter_jobs_stream(files, args, input_dir=input_dir), start=1):
        path = Path(job["path"])
        label = job.get("subject_name") or path.name
        _pause_if_requested(out_dir, progress_cb=progress_cb, idx=idx - 1, total=total, label=str(label))
        if progress_cb:
            progress_cb(idx - 1, total, f"attack :: {label}")
        item_suffix = f"__{safe_stem(job['subject_name'])}" if job.get("subject_name") else ""
        item_name = f"{idx:04d}__{safe_stem(path.stem)}{item_suffix}"

        meta_row = _match_job_metadata(job, meta_df)
        healthy_flag = _job_is_healthy(meta_row, healthy_values, group_col=meta_group_col)
        meta_export = {k: v for k, v in meta_row.items() if not str(k).startswith("__")}

        if bool(getattr(args, "attack_only_healthy", False)) and healthy_flag is not True:
            group_value = meta_export.get(meta_group_col, "") if meta_group_col else ""
            reason = "metadata_missing" if not meta_export else f"non_healthy:{group_value}"

            row = {
                "mode": "attack",
                "batch_index": idx,
                "input": str(path),
                "input_file": path.name,
                "input_stem": path.stem,
                "input_suffix": path.suffix.lower(),
                "relative_input": str(job["relative_input"]),
                "subject_name": str(job.get("subject_name") or ""),
                "subject_index": job.get("subject_index"),
                "family": str(args.family),
                "kind": str(args.kind),
                "status": "skipped_non_healthy",
                "error": reason,
                "metadata_matched": bool(meta_export),
                "metadata_match_key": meta_row.get("__matched_meta_key", "") if meta_row else "",
                "metadata_group_value": group_value,
                **meta_export,
            }
            manifest_item = {
                **_base_manifest_row(job, idx, "attack"),
                "status": "skipped_non_healthy",
                "error": reason,
                "xlsx_path": "",
                "json_path": "",
                "history_csv_path": "",
            }

            _append_row_csv(summary_csv, row)
            _append_row_csv(manifest_csv, manifest_item)
            rows_chunk.append(row)
            manifest_chunk.append(manifest_item)
            _append_progress_log(out_dir, f"[{idx}/{total}] SKIP non-healthy :: {label} :: {reason}")

            del row, manifest_item
            _gc_collect()

            if chunk_size and len(rows_chunk) >= chunk_size:
                chunk_idx += 1
                _flush_chunk_tables(
                    out_dir,
                    mode_name="attack",
                    chunk_idx=chunk_idx,
                    rows_chunk=rows_chunk,
                    manifest_chunk=manifest_chunk,
                )
                rows_chunk = []
                manifest_chunk = []

            continue

        done, resolved_paths = _item_already_done("attack", out_dir=out_dir, item_name=item_name)
        if bool(getattr(args, "skip_existing", True)) and done:
            row, manifest_item = _skipped_existing_row(
                job,
                idx,
                "attack",
                resolved_paths,
                extra={"family": str(args.family), "kind": str(args.kind)},
            )
            _append_row_csv(summary_csv, row)
            _append_row_csv(manifest_csv, manifest_item)
            rows_chunk.append(row)
            manifest_chunk.append(manifest_item)
            _append_progress_log(out_dir, f"[{idx}/{total}] SKIP existing :: {label}")
            del row, manifest_item
            _gc_collect()
            if chunk_size and len(rows_chunk) >= chunk_size:
                chunk_idx += 1
                _flush_chunk_tables(out_dir, mode_name="attack", chunk_idx=chunk_idx, rows_chunk=rows_chunk, manifest_chunk=manifest_chunk)
                rows_chunk = []
                manifest_chunk = []
            continue
        try:
            if job.get("graph") is not None:
                payload, history = _attack_payload_from_graph(args, job["graph"], input_label=str(job["input"]))
            else:
                payload, history = _run_attack_payload(args, path)
            payload["batch"] = {
                "index": idx,
                "input_file": path.name,
                "input_stem": path.stem,
                "input_suffix": path.suffix.lower(),
                "relative_input": str(job["relative_input"]),
                "subject_name": str(job.get("subject_name") or ""),
                "subject_index": job.get("subject_index"),
            }
            payload_json_path = json_dir / f"{item_name}.json"
            export_metrics_payload(payload, str(payload_json_path), "json")
            history_path = hist_dir / f"{item_name}.csv"
            history.to_csv(history_path, index=False)
            xlsx_path = xlsx_dir / f"{item_name}.xlsx"
            xlsx_path.write_bytes(_attack_payload_xlsx_bytes(payload, history))
            json_abs = str(payload_json_path.resolve())
            hist_abs = str(history_path.resolve())
            xlsx_abs = str(xlsx_path.resolve())

            row = {
                "mode": "attack",
                "batch_index": idx,
                "input": str(path),
                "input_file": path.name,
                "input_stem": path.stem,
                "input_suffix": path.suffix.lower(),
                "relative_input": str(job["relative_input"]),
                "subject_name": str(job.get("subject_name") or ""),
                "subject_index": job.get("subject_index"),
                "family": payload.get("family"),
                "kind": payload.get("kind"),
                "payload_json_path": json_abs,
                "history_csv_path": hist_abs,
                "xlsx_path": xlsx_abs,
                "status": "ok",
                "error": "",
                "metadata_matched": bool(meta_export),
                "metadata_match_key": meta_row.get("__matched_meta_key", "") if meta_row else "",
                "metadata_group_value": meta_export.get(meta_group_col, "") if meta_group_col else "",
                **meta_export,
                **payload.get("summary", {}),
                **{f"final__{k}": v for k, v in payload.get("final_row", {}).items()},
            }
            manifest_item = {
                **_base_manifest_row(job, idx, "attack"),
                "status": "ok",
                "error": "",
                "xlsx_path": xlsx_abs,
                "json_path": json_abs,
                "history_csv_path": hist_abs,
            }
        except Exception as exc:  # pylint: disable=broad-except
            row = {
                "mode": "attack",
                "batch_index": idx,
                "input": str(path),
                "input_file": path.name,
                "input_stem": path.stem,
                "input_suffix": path.suffix.lower(),
                "relative_input": str(job["relative_input"]),
                "subject_name": str(job.get("subject_name") or ""),
                "subject_index": job.get("subject_index"),
                "family": str(args.family),
                "kind": str(args.kind),
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "metadata_matched": bool(meta_export),
                "metadata_match_key": meta_row.get("__matched_meta_key", "") if meta_row else "",
                "metadata_group_value": meta_export.get(meta_group_col, "") if meta_group_col else "",
                **meta_export,
            }
            manifest_item = {
                **_base_manifest_row(job, idx, "attack"),
                "status": "error",
                "error": row["error"],
                "xlsx_path": "",
                "json_path": "",
                "history_csv_path": "",
            }
        _append_row_csv(summary_csv, row)
        _append_row_csv(manifest_csv, manifest_item)
        rows_chunk.append(row)
        manifest_chunk.append(manifest_item)
        _append_progress_log(out_dir, f"[{idx}/{total}] DONE attack :: {label} :: {row.get('status', 'ok')}")
        del row, manifest_item
        _gc_collect()
        if chunk_size and len(rows_chunk) >= chunk_size:
            chunk_idx += 1
            _flush_chunk_tables(out_dir, mode_name="attack", chunk_idx=chunk_idx, rows_chunk=rows_chunk, manifest_chunk=manifest_chunk)
            rows_chunk = []
            manifest_chunk = []

    if rows_chunk or manifest_chunk:
        chunk_idx += 1
        _flush_chunk_tables(out_dir, mode_name="attack", chunk_idx=chunk_idx, rows_chunk=rows_chunk, manifest_chunk=manifest_chunk)

    summary_xlsx = out_dir / "attack_summary.xlsx"
    manifest_path = out_dir / "manifest.json"
    df, _ = _finalize_batch_outputs(
        out_dir,
        summary_csv=summary_csv,
        summary_xlsx=summary_xlsx,
        summary_sheet="attacks",
        manifest_path=manifest_path,
        files=files,
        args=args,
        mode="batch-attack",
    )
    _append_progress_log(out_dir, f"FINISH attack rows={len(df)}")

    if progress_cb:
        progress_cb(total, total, f"saved -> {summary_csv.name}")
    return out_dir, df
