from __future__ import annotations

import json
import re
import shutil
import tempfile
from io import BytesIO
from datetime import datetime
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
    rows: list[dict] = []
    manifest_rows: list[dict] = []
    rows_chunk: list[dict] = []
    manifest_chunk: list[dict] = []
    chunk_size = _chunk_size_from_args(args)
    chunk_idx = 0

    for idx, job in enumerate(_iter_jobs_stream(files, args, input_dir=input_dir), start=1):
        path = Path(job["path"])
        if progress_cb:
            label = job.get("subject_name") or path.name
            progress_cb(idx - 1, total, f"energy :: {label}")
        item_suffix = f"__{safe_stem(job['subject_name'])}" if job.get("subject_name") else ""
        item_name = f"{idx:04d}__{safe_stem(path.stem)}{item_suffix}"
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
        rows.append(row)
        manifest_rows.append(manifest_item)
        rows_chunk.append(row)
        manifest_chunk.append(manifest_item)
        if chunk_size and len(rows_chunk) >= chunk_size:
            chunk_idx += 1
            _flush_chunk_tables(out_dir, mode_name="energy", chunk_idx=chunk_idx, rows_chunk=rows_chunk, manifest_chunk=manifest_chunk)
            rows_chunk = []
            manifest_chunk = []

    if rows_chunk or manifest_chunk:
        chunk_idx += 1
        _flush_chunk_tables(out_dir, mode_name="energy", chunk_idx=chunk_idx, rows_chunk=rows_chunk, manifest_chunk=manifest_chunk)

    df = pd.DataFrame(rows)
    manifest_df = pd.DataFrame(manifest_rows)
    summary_csv = Path(args.summary_out) if str(getattr(args, "summary_out", "")).strip() else out_dir / "energy_summary.csv"
    summary_xlsx = out_dir / "energy_summary.xlsx"
    manifest_path = out_dir / "manifest.json"
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(summary_csv, index=False)
    with pd.ExcelWriter(summary_xlsx, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="energy_runs", index=False)
    _write_manifest_tables(out_dir, manifest_df, prefix="manifest")
    manifest_path.write_text(
        json.dumps(build_batch_manifest(files=files, args=args, mode="batch-energy"), ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
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
    rows: list[dict] = []
    manifest_rows: list[dict] = []
    rows_chunk: list[dict] = []
    manifest_chunk: list[dict] = []
    chunk_size = _chunk_size_from_args(args)
    chunk_idx = 0
    for idx, job in enumerate(_iter_jobs_stream(files, args, input_dir=input_dir), start=1):
        path = Path(job["path"])
        if progress_cb:
            label = job.get("subject_name") or path.name
            progress_cb(idx - 1, total, f"resistance :: {label}")
        item_suffix = f"__{safe_stem(job['subject_name'])}" if job.get("subject_name") else ""
        item_name = f"{idx:04d}__{safe_stem(path.stem)}{item_suffix}"
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
        rows.append(row)
        manifest_rows.append(manifest_item)
        rows_chunk.append(row)
        manifest_chunk.append(manifest_item)
        if chunk_size and len(rows_chunk) >= chunk_size:
            chunk_idx += 1
            _flush_chunk_tables(out_dir, mode_name="resistance", chunk_idx=chunk_idx, rows_chunk=rows_chunk, manifest_chunk=manifest_chunk)
            rows_chunk = []
            manifest_chunk = []
    if rows_chunk or manifest_chunk:
        chunk_idx += 1
        _flush_chunk_tables(out_dir, mode_name="resistance", chunk_idx=chunk_idx, rows_chunk=rows_chunk, manifest_chunk=manifest_chunk)
    df = pd.DataFrame(rows)
    manifest_df = pd.DataFrame(manifest_rows)
    summary_csv = Path(args.summary_out) if str(getattr(args, "summary_out", "")).strip() else out_dir / "resistance_summary.csv"
    summary_xlsx = out_dir / "resistance_summary.xlsx"
    manifest_path = out_dir / "manifest.json"
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(summary_csv, index=False)
    with pd.ExcelWriter(summary_xlsx, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="robustness_summary", index=False)
    _write_manifest_tables(out_dir, manifest_df, prefix="manifest")
    manifest_path.write_text(
        json.dumps(build_batch_manifest(files=files, args=args, mode="batch-resistance"), ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    if progress_cb:
        progress_cb(total, total, f"saved -> {summary_csv.name}")
    return out_dir, df


def run_batch_plan(args, *, progress_cb: ProgressCb | None = None) -> tuple[Path, dict[str, pd.DataFrame]]:
    """Run selected batch tasks (metrics/attack/energy/resistance) and return per-mode DataFrames."""
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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

    if progress_cb is not None:
        progress_cb(total_modes, total_modes, f"saved -> {saved_label}")
    return out_dir, result_frames


def safe_stem(value: str) -> str:
    """Sanitize a value for safe use in output filenames."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    cleaned = cleaned.strip("._")
    return cleaned or "item"


def make_run_dir(base_dir: str | Path, *, mode: str, seed: int, run_label: str = "") -> Path:
    """Create a timestamped run directory under ``base_dir``."""
    base = Path(base_dir).expanduser().resolve()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    label = safe_stem(run_label) if str(run_label).strip() else mode
    run_dir = base / f"{label}__seed_{int(seed)}__{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
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
    rows: list[dict] = []
    manifest_rows: list[dict] = []
    rows_chunk: list[dict] = []
    manifest_chunk: list[dict] = []
    chunk_size = _chunk_size_from_args(args)
    chunk_idx = 0

    for idx, job in enumerate(_iter_jobs_stream(files, args, input_dir=input_dir), start=1):
        path = Path(job["path"])
        if progress_cb:
            label = job.get("subject_name") or path.name
            progress_cb(idx - 1, total, f"metrics :: {label}")
        item_suffix = f"__{safe_stem(job['subject_name'])}" if job.get("subject_name") else ""
        item_name = f"{idx:04d}__{safe_stem(path.stem)}{item_suffix}"
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
        rows.append(row)
        manifest_rows.append(manifest_item)
        rows_chunk.append(row)
        manifest_chunk.append(manifest_item)
        if chunk_size and len(rows_chunk) >= chunk_size:
            chunk_idx += 1
            _flush_chunk_tables(out_dir, mode_name="metrics", chunk_idx=chunk_idx, rows_chunk=rows_chunk, manifest_chunk=manifest_chunk)
            rows_chunk = []
            manifest_chunk = []

    if rows_chunk or manifest_chunk:
        chunk_idx += 1
        _flush_chunk_tables(out_dir, mode_name="metrics", chunk_idx=chunk_idx, rows_chunk=rows_chunk, manifest_chunk=manifest_chunk)

    df = pd.DataFrame(rows)
    manifest_df = pd.DataFrame(manifest_rows)
    summary_csv = Path(args.summary_out) if str(getattr(args, "summary_out", "")).strip() else out_dir / "metrics_summary.csv"
    summary_xlsx = out_dir / "metrics_summary.xlsx"
    manifest_path = out_dir / "manifest.json"

    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(summary_csv, index=False)
    with pd.ExcelWriter(summary_xlsx, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="metrics", index=False)
    _write_manifest_tables(out_dir, manifest_df, prefix="manifest")

    manifest_path.write_text(
        json.dumps(build_batch_manifest(files=files, args=args, mode="batch-metrics"), ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )

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
    rows: list[dict] = []
    manifest_rows: list[dict] = []
    rows_chunk: list[dict] = []
    manifest_chunk: list[dict] = []
    chunk_size = _chunk_size_from_args(args)
    chunk_idx = 0

    for idx, job in enumerate(_iter_jobs_stream(files, args, input_dir=input_dir), start=1):
        path = Path(job["path"])
        if progress_cb:
            label = job.get("subject_name") or path.name
            progress_cb(idx - 1, total, f"attack :: {label}")
        item_suffix = f"__{safe_stem(job['subject_name'])}" if job.get("subject_name") else ""
        item_name = f"{idx:04d}__{safe_stem(path.stem)}{item_suffix}"
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
            }
            manifest_item = {
                **_base_manifest_row(job, idx, "attack"),
                "status": "error",
                "error": row["error"],
                "xlsx_path": "",
                "json_path": "",
                "history_csv_path": "",
            }
        rows.append(row)
        manifest_rows.append(manifest_item)
        rows_chunk.append(row)
        manifest_chunk.append(manifest_item)
        if chunk_size and len(rows_chunk) >= chunk_size:
            chunk_idx += 1
            _flush_chunk_tables(out_dir, mode_name="attack", chunk_idx=chunk_idx, rows_chunk=rows_chunk, manifest_chunk=manifest_chunk)
            rows_chunk = []
            manifest_chunk = []

    if rows_chunk or manifest_chunk:
        chunk_idx += 1
        _flush_chunk_tables(out_dir, mode_name="attack", chunk_idx=chunk_idx, rows_chunk=rows_chunk, manifest_chunk=manifest_chunk)

    df = pd.DataFrame(rows)
    manifest_df = pd.DataFrame(manifest_rows)
    summary_csv = Path(args.summary_out) if str(getattr(args, "summary_out", "")).strip() else out_dir / "attack_summary.csv"
    summary_xlsx = out_dir / "attack_summary.xlsx"
    manifest_path = out_dir / "manifest.json"

    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(summary_csv, index=False)
    with pd.ExcelWriter(summary_xlsx, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="attacks", index=False)
    _write_manifest_tables(out_dir, manifest_df, prefix="manifest")

    manifest_path.write_text(
        json.dumps(build_batch_manifest(files=files, args=args, mode="batch-attack"), ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )

    if progress_cb:
        progress_cb(total, total, f"saved -> {summary_csv.name}")
    return out_dir, df
