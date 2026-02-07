from __future__ import annotations

import pandas as pd

from .config import settings
from .weights import policy_from_settings, apply_weight_policy_to_series


def coerce_fixed_format(df_any: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Ожидается:
    - 1-я колонка source id
    - 2-я колонка target id
    - 9-я confidence
    - 10-я weight
    Возвращает df_edges со столбцами:
      [SRC_COL, DST_COL, weight, confidence]
    и мету {src_col, dst_col}
    """
    if df_any.shape[1] < 10:
        raise ValueError("Файл должен содержать минимум 10 колонок (фикс. формат).")

    SRC_COL = df_any.columns[0]
    DST_COL = df_any.columns[1]
    CONF_COL = df_any.columns[8]
    WEIGHT_COL = df_any.columns[9]

    df = df_any.copy()

    df[SRC_COL] = pd.to_numeric(df[SRC_COL], errors="coerce").astype("Int64")
    df[DST_COL] = pd.to_numeric(df[DST_COL], errors="coerce").astype("Int64")

    df[CONF_COL] = pd.to_numeric(df[CONF_COL], errors="coerce")

    if not pd.api.types.is_numeric_dtype(df[WEIGHT_COL]):
        df[WEIGHT_COL] = df[WEIGHT_COL].astype(str).str.replace(",", ".", regex=False)
    df[WEIGHT_COL] = pd.to_numeric(df[WEIGHT_COL], errors="coerce")

    out = df[[SRC_COL, DST_COL, CONF_COL, WEIGHT_COL]].copy()
    out = out.rename(columns={CONF_COL: "confidence", WEIGHT_COL: "weight"})
    out = out.dropna(subset=[SRC_COL, DST_COL, "confidence", "weight"])

    pol = policy_from_settings(settings.WEIGHT_POLICY, settings.WEIGHT_EPS, settings.WEIGHT_SHIFT)
    out["weight"], keep = apply_weight_policy_to_series(out["weight"], pol)
    out = out[keep]

    if out.empty:
        raise ValueError("После очистки данные пустые (проверь numeric confidence/weight и id).")

    meta = {"src_col": SRC_COL, "dst_col": DST_COL}
    return out, meta


def filter_edges(
    df_edges: pd.DataFrame,
    src_col: str,
    dst_col: str,
    min_conf: float,
    min_weight: float,
) -> pd.DataFrame:
    df = df_edges.copy()

    if src_col not in df.columns or dst_col not in df.columns:
        raise ValueError(f"Нет обязательных колонок: {[src_col, dst_col]}")
    if "confidence" not in df.columns:
        df["confidence"] = 100.0
    if "weight" not in df.columns:
        df["weight"] = 1.0

    df = df.dropna(subset=[src_col, dst_col])

    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
    df["weight"] = pd.to_numeric(
        df["weight"].astype(str).str.replace(",", ".", regex=False),
        errors="coerce",
    )

    df = df.dropna(subset=["confidence", "weight"])
    df = df[df["confidence"] >= min_conf]

    pol = policy_from_settings(settings.WEIGHT_POLICY, settings.WEIGHT_EPS, settings.WEIGHT_SHIFT)
    df["weight"], keep = apply_weight_policy_to_series(df["weight"], pol)
    df = df[keep]

    df = df[df["weight"] >= min_weight]

    return df
