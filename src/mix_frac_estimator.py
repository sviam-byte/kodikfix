"""
mix_frac* estimator: количественная шкала рандомизации.

Старая версия использовала одномерное пересечение по одной метрике.
Это удобно для exploratory-анализа, но для paper-grade оценки слишком хрупко:
- возможны несколько пересечений;
- кривая может быть немонотонной;
- одна метрика плохо идентифицирует положение на траектории.

Новая версия поддерживает два режима:
1) interpolate: обратная совместимость, старое линейное пересечение;
2) nearest: основная рекомендованная оценка. Пациент сопоставляется с
   ближайшей точкой на healthy-trajectory в пространстве нескольких метрик.
   Метрики z-нормируются по разбросу healthy baseline, а затем по всем
   healthy-графам агрегируются медианой и bootstrap CI.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .attacks_mix import run_mix_attack


def build_randomization_curve(
    graph,
    *,
    steps: int = 20,
    seed: int = 42,
    eff_sources_k: int = 16,
    replace_from: str = "CFG",
    kind: str = "mix_default",
) -> pd.DataFrame:
    """
    Одна кривая randomization для одного healthy графа.

    Returns
    -------
    pd.DataFrame
        Таблица с колонками: mix_frac, metric1, metric2, ...
    """
    curve_df, _aux = run_mix_attack(
        graph,
        kind=kind,
        steps=steps,
        seed=seed,
        eff_sources_k=eff_sources_k,
        heavy_every=1,
        replace_from=replace_from,
    )
    return curve_df


def interpolate_mix_frac(curve_df: pd.DataFrame, metric: str, target_value: float) -> float:
    """
    Найти mix_frac, при котором кривая проходит через target_value.

    Используется линейная интерполяция между соседними точками.
    Если target вне диапазона кривой, возвращается NaN.
    """
    if metric not in curve_df.columns or "mix_frac" not in curve_df.columns:
        return float("nan")

    x = curve_df["mix_frac"].values.astype(float)
    y = curve_df[metric].values.astype(float)

    # Убираем неполные точки.
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 2:
        return float("nan")

    # Ищем первое пересечение целевого значения.
    for idx in range(len(y) - 1):
        if (y[idx] - target_value) * (y[idx + 1] - target_value) <= 0:
            dy = y[idx + 1] - y[idx]
            if abs(dy) < 1e-15:
                return float(x[idx])
            t = (target_value - y[idx]) / dy
            return float(x[idx] + t * (x[idx + 1] - x[idx]))

    return float("nan")


def _as_metric_list(target_metric: str | Sequence[str] | None) -> list[str]:
    if target_metric is None:
        return []
    if isinstance(target_metric, str):
        return [target_metric]
    return [str(m) for m in target_metric if str(m)]


def _finite_metric_columns(
    curve_df: pd.DataFrame,
    requested: Sequence[str],
    patient_metrics: dict[str, float],
) -> list[str]:
    cols: list[str] = []
    for metric in requested:
        if metric not in curve_df.columns:
            continue
        pv = patient_metrics.get(metric, float("nan"))
        if not np.isfinite(pv):
            continue
        series = pd.to_numeric(curve_df[metric], errors="coerce").astype(float)
        if np.isfinite(series.to_numpy()).sum() == 0:
            continue
        cols.append(metric)
    return cols


def _healthy_baseline_scale(
    curves: Sequence[pd.DataFrame],
    metrics: Sequence[str],
    eps: float = 1e-8,
) -> dict[str, float]:
    scales: dict[str, float] = {}
    for metric in metrics:
        vals: list[float] = []
        for curve in curves:
            if metric not in curve.columns or curve.empty:
                continue
            v = pd.to_numeric(pd.Series([curve.iloc[0][metric]]), errors="coerce").iloc[0]
            if np.isfinite(v):
                vals.append(float(v))
        if not vals:
            scales[metric] = 1.0
            continue
        arr = np.asarray(vals, dtype=float)
        s = float(np.nanstd(arr, ddof=1)) if arr.size >= 2 else 0.0
        if (not np.isfinite(s)) or s < eps:
            mad = float(np.nanmedian(np.abs(arr - np.nanmedian(arr))))
            s = 1.4826 * mad if np.isfinite(mad) and mad >= eps else 1.0
        scales[metric] = float(max(s, eps))
    return scales


def nearest_mix_frac(
    curve_df: pd.DataFrame,
    patient_metrics: dict[str, float],
    *,
    metrics: Sequence[str],
    scales: dict[str, float] | None = None,
) -> dict:
    """
    Найти ближайшую точку на trajectory в пространстве нескольких метрик.
    Distance = евклидова норма после деления каждой координаты на scale[metric].
    """
    if "mix_frac" not in curve_df.columns:
        return {
            "mix_frac": float("nan"),
            "distance": float("nan"),
            "used_metrics": [],
            "n_used_metrics": 0,
            "curve_index": None,
        }

    used = _finite_metric_columns(curve_df, metrics, patient_metrics)
    if not used:
        return {
            "mix_frac": float("nan"),
            "distance": float("nan"),
            "used_metrics": [],
            "n_used_metrics": 0,
            "curve_index": None,
        }

    x = pd.to_numeric(curve_df["mix_frac"], errors="coerce").to_numpy(dtype=float)
    cols = [pd.to_numeric(curve_df[m], errors="coerce").to_numpy(dtype=float) for m in used]
    Y = np.vstack(cols).T
    patient_vec = np.asarray([float(patient_metrics[m]) for m in used], dtype=float)

    denom = np.asarray(
        [
            float((scales or {}).get(m, 1.0))
            if np.isfinite(float((scales or {}).get(m, 1.0)))
            else 1.0
            for m in used
        ],
        dtype=float,
    )
    denom = np.where(denom > 1e-12, denom, 1.0)

    mask = np.isfinite(x)
    mask &= np.all(np.isfinite(Y), axis=1)
    if not np.any(mask):
        return {
            "mix_frac": float("nan"),
            "distance": float("nan"),
            "used_metrics": used,
            "n_used_metrics": len(used),
            "curve_index": None,
        }

    x_valid = x[mask]
    Y_valid = Y[mask]
    diffs = (Y_valid - patient_vec[None, :]) / denom[None, :]
    dists = np.sqrt(np.sum(diffs * diffs, axis=1))
    idx = int(np.argmin(dists))
    return {
        "mix_frac": float(x_valid[idx]),
        "distance": float(dists[idx]),
        "used_metrics": used,
        "n_used_metrics": int(len(used)),
        "curve_index": int(np.flatnonzero(mask)[idx]),
    }


def _bootstrap_ci(
    values: Sequence[float],
    *,
    seed: int = 42,
    n_boot: int = 1000,
    alpha: float = 0.05,
) -> tuple[float, float]:
    arr = np.asarray([float(v) for v in values if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        return float(arr[0]), float(arr[0])
    rng = np.random.default_rng(int(seed))
    meds = np.empty(int(max(100, n_boot)), dtype=float)
    for i in range(meds.size):
        sample = arr[rng.integers(0, arr.size, size=arr.size)]
        meds[i] = float(np.median(sample))
    lo = float(np.quantile(meds, alpha / 2.0))
    hi = float(np.quantile(meds, 1.0 - alpha / 2.0))
    return lo, hi


def estimate_mix_frac_star(
    healthy_graphs: list,
    patient_metrics: dict[str, float],
    *,
    target_metric: str | Sequence[str] = "kappa_mean",
    match_mode: str = "nearest",
    steps: int = 20,
    seed: int = 42,
    eff_sources_k: int = 16,
    replace_from: str = "CFG",
    n_boot: int = 1000,
) -> dict:
    """
    Оценить mix_frac* для одного пациента относительно группы healthy.

    Returns
    -------
    dict
        Ключи:
        - mix_frac_star: медиана по healthy субъектам
        - mix_frac_values: значение для каждой healthy-кривой
        - target_metric: использованные метрики
        - patient_value: значение метрик пациента
        - ci_low / ci_high: bootstrap CI по медиане
    """
    metric_list = _as_metric_list(target_metric)
    if not metric_list:
        return {
            "mix_frac_star": float("nan"),
            "mix_frac_values": [],
            "target_metric": target_metric,
            "patient_value": float("nan"),
            "used_metrics": [],
            "match_mode": match_mode,
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "distances": [],
        }

    curves: list[pd.DataFrame] = []
    for offset, healthy_graph in enumerate(healthy_graphs):
        curve = build_randomization_curve(
            healthy_graph,
            steps=steps,
            seed=seed + offset,
            eff_sources_k=eff_sources_k,
            replace_from=replace_from,
        )
        curves.append(curve)

    match_mode = str(match_mode).strip().lower()
    if match_mode not in {"nearest", "interpolate"}:
        match_mode = "nearest"

    if match_mode == "interpolate":
        primary_metric = metric_list[0]
        target_value = float(patient_metrics.get(primary_metric, float("nan")))
        if not np.isfinite(target_value):
            return {
                "mix_frac_star": float("nan"),
                "mix_frac_values": [],
                "target_metric": primary_metric,
                "patient_value": float("nan"),
                "used_metrics": [],
                "match_mode": match_mode,
                "ci_low": float("nan"),
                "ci_high": float("nan"),
                "distances": [],
            }
        fracs = [interpolate_mix_frac(curve, primary_metric, target_value) for curve in curves]
        valid = [frac for frac in fracs if np.isfinite(frac)]
        star = float(np.median(valid)) if valid else float("nan")
        ci_low, ci_high = _bootstrap_ci(valid, seed=seed, n_boot=n_boot)
        return {
            "mix_frac_star": star,
            "mix_frac_values": fracs,
            "target_metric": primary_metric,
            "patient_value": target_value,
            "used_metrics": [primary_metric] if valid else [],
            "match_mode": match_mode,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "distances": [float("nan")] * len(fracs),
        }

    union_metrics: list[str] = []
    for curve in curves:
        for metric in _finite_metric_columns(curve, metric_list, patient_metrics):
            if metric not in union_metrics:
                union_metrics.append(metric)

    if not union_metrics:
        return {
            "mix_frac_star": float("nan"),
            "mix_frac_values": [],
            "target_metric": metric_list,
            "patient_value": float("nan"),
            "used_metrics": [],
            "match_mode": match_mode,
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "distances": [],
        }

    scales = _healthy_baseline_scale(curves, union_metrics)
    fracs: list[float] = []
    distances: list[float] = []
    per_curve_used: list[list[str]] = []
    for curve in curves:
        res = nearest_mix_frac(curve, patient_metrics, metrics=union_metrics, scales=scales)
        fracs.append(float(res["mix_frac"]))
        distances.append(float(res["distance"]))
        per_curve_used.append(list(res["used_metrics"]))

    valid = [frac for frac in fracs if np.isfinite(frac)]
    star = float(np.median(valid)) if valid else float("nan")
    ci_low, ci_high = _bootstrap_ci(valid, seed=seed, n_boot=n_boot)

    return {
        "mix_frac_star": star,
        "mix_frac_values": fracs,
        "target_metric": metric_list,
        "patient_value": {m: float(patient_metrics[m]) for m in union_metrics},
        "used_metrics": union_metrics,
        "used_metrics_per_curve": per_curve_used,
        "match_mode": match_mode,
        "metric_scales": scales,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "distances": distances,
    }
