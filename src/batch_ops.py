from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Iterable

import pandas as pd

from .cli import (
    _build_metrics_payload,
    _iter_input_files,
    _json_default,
    _run_attack_payload,
)
from .exporters import export_metrics_payload, payload_to_flat_row

# Колбэк прогресса: done, total, label
ProgressCb = Callable[[int, int, str], None]


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
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def run_batch_metrics(args, *, progress_cb: ProgressCb | None = None) -> tuple[Path, pd.DataFrame]:
    """Run metrics batch and return output dir and summary DataFrame."""
    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    per_item_dir = out_dir / "per_item"
    per_item_dir.mkdir(parents=True, exist_ok=True)

    files = _iter_input_files(
        input_dir,
        recursive=bool(args.recursive),
        pattern=str(args.pattern),
        limit=int(args.limit),
    )
    total = len(files)
    rows: list[dict] = []

    for idx, path in enumerate(files, start=1):
        if progress_cb:
            progress_cb(idx - 1, total, f"metrics :: {path.name}")
        item_name = f"{idx:04d}__{safe_stem(path.stem)}"
        try:
            payload = _build_metrics_payload(args, path)
            payload["batch"] = {
                "index": idx,
                "input_file": path.name,
                "input_stem": path.stem,
                "input_suffix": path.suffix.lower(),
                "relative_input": str(path.relative_to(input_dir)),
            }
            json_path = per_item_dir / f"{item_name}.json"
            export_metrics_payload(payload, str(json_path), "json")

            row = payload_to_flat_row(payload)
            row.update(
                {
                    "batch_index": idx,
                    "input_file": path.name,
                    "input_stem": path.stem,
                    "input_suffix": path.suffix.lower(),
                    "relative_input": str(path.relative_to(input_dir)),
                    "json_path": str(json_path.resolve()),
                    "status": "ok",
                    "error": "",
                }
            )
            rows.append(row)
        except Exception as exc:  # pylint: disable=broad-except
            rows.append(
                {
                    "mode": "metrics",
                    "batch_index": idx,
                    "input": str(path),
                    "input_file": path.name,
                    "input_stem": path.stem,
                    "input_suffix": path.suffix.lower(),
                    "relative_input": str(path.relative_to(input_dir)),
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    df = pd.DataFrame(rows)
    summary_csv = Path(args.summary_out) if str(getattr(args, "summary_out", "")).strip() else out_dir / "metrics_summary.csv"
    summary_xlsx = out_dir / "metrics_summary.xlsx"
    manifest_path = out_dir / "manifest.json"

    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(summary_csv, index=False)
    with pd.ExcelWriter(summary_xlsx, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="metrics", index=False)

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
    json_dir.mkdir(parents=True, exist_ok=True)
    hist_dir.mkdir(parents=True, exist_ok=True)

    files = _iter_input_files(
        input_dir,
        recursive=bool(args.recursive),
        pattern=str(args.pattern),
        limit=int(args.limit),
    )
    total = len(files)
    rows: list[dict] = []

    for idx, path in enumerate(files, start=1):
        if progress_cb:
            progress_cb(idx - 1, total, f"attack :: {path.name}")
        item_name = f"{idx:04d}__{safe_stem(path.stem)}"
        try:
            payload, history = _run_attack_payload(args, path)
            payload["batch"] = {
                "index": idx,
                "input_file": path.name,
                "input_stem": path.stem,
                "input_suffix": path.suffix.lower(),
                "relative_input": str(path.relative_to(input_dir)),
            }
            payload_json_path = json_dir / f"{item_name}.json"
            export_metrics_payload(payload, str(payload_json_path), "json")
            history_path = hist_dir / f"{item_name}.csv"
            history.to_csv(history_path, index=False)

            row = {
                "mode": "attack",
                "batch_index": idx,
                "input": str(path),
                "input_file": path.name,
                "input_stem": path.stem,
                "input_suffix": path.suffix.lower(),
                "relative_input": str(path.relative_to(input_dir)),
                "family": payload.get("family"),
                "kind": payload.get("kind"),
                "payload_json_path": str(payload_json_path.resolve()),
                "history_csv_path": str(history_path.resolve()),
                "status": "ok",
                "error": "",
                **payload.get("summary", {}),
                **{f"final__{k}": v for k, v in payload.get("final_row", {}).items()},
            }
            rows.append(row)
        except Exception as exc:  # pylint: disable=broad-except
            rows.append(
                {
                    "mode": "attack",
                    "batch_index": idx,
                    "input": str(path),
                    "input_file": path.name,
                    "input_stem": path.stem,
                    "input_suffix": path.suffix.lower(),
                    "relative_input": str(path.relative_to(input_dir)),
                    "family": str(args.family),
                    "kind": str(args.kind),
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    df = pd.DataFrame(rows)
    summary_csv = Path(args.summary_out) if str(getattr(args, "summary_out", "")).strip() else out_dir / "attack_summary.csv"
    summary_xlsx = out_dir / "attack_summary.xlsx"
    manifest_path = out_dir / "manifest.json"

    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(summary_csv, index=False)
    with pd.ExcelWriter(summary_xlsx, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="attacks", index=False)

    manifest_path.write_text(
        json.dumps(build_batch_manifest(files=files, args=args, mode="batch-attack"), ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )

    if progress_cb:
        progress_cb(total, total, f"saved -> {summary_csv.name}")
    return out_dir, df
