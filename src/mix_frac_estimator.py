"""
mix_frac* estimator: количественная шкала рандомизации.

Для каждого healthy субъекта h:
  1. Прогоняем run_mix_attack(G_h, "mix_default", steps=20, ..., replace_from="CFG")
  2. Получаем кривую metric_h(mix_frac)
  3. Для каждого patient субъекта p с метрикой value_p:
     mix_frac*_{h,p} = линейная интерполяция: при каком mix_frac
                        metric_h(mix_frac) == value_p
  4. mix_frac*_p = медиана по h

Это даёт каждому пациенту число на шкале [0, 1]:
  0 = «как здоровый»
  1 = «как полностью рандомизированный»
"""
from __future__ import annotations

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


def estimate_mix_frac_star(
    healthy_graphs: list,
    patient_metrics: dict[str, float],
    *,
    target_metric: str = "clustering",
    steps: int = 20,
    seed: int = 42,
    eff_sources_k: int = 16,
    replace_from: str = "CFG",
) -> dict:
    """
    Оценить mix_frac* для одного пациента относительно группы healthy.

    Returns
    -------
    dict
        Ключи:
        - mix_frac_star: медиана по healthy субъектам
        - mix_frac_values: значение для каждой healthy-кривой
        - target_metric: использованная метрика
        - patient_value: значение метрики пациента
    """
    target_value = patient_metrics.get(target_metric, float("nan"))
    if not np.isfinite(target_value):
        return {
            "mix_frac_star": float("nan"),
            "mix_frac_values": [],
            "target_metric": target_metric,
            "patient_value": float("nan"),
        }

    fracs: list[float] = []
    for offset, healthy_graph in enumerate(healthy_graphs):
        curve = build_randomization_curve(
            healthy_graph,
            steps=steps,
            seed=seed + offset,
            eff_sources_k=eff_sources_k,
            replace_from=replace_from,
        )
        fracs.append(interpolate_mix_frac(curve, target_metric, target_value))

    valid = [frac for frac in fracs if np.isfinite(frac)]
    star = float(np.median(valid)) if valid else float("nan")

    return {
        "mix_frac_star": star,
        "mix_frac_values": fracs,
        "target_metric": target_metric,
        "patient_value": target_value,
    }
