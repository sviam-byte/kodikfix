from __future__ import annotations

import io

import pandas as pd
from pandas.errors import ParserError

def load_uploaded_any(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Load uploaded bytes into a DataFrame for CSV or Excel inputs."""
    name = (filename or "").lower()
    bio = io.BytesIO(file_bytes)

    if name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(bio)
    else:
        # Try utf-8 with replacement; fallback to cp1251 for legacy files.
        try:
            df = pd.read_csv(bio, sep=None, engine="python", encoding_errors="replace")
        except (UnicodeDecodeError, ParserError):
            bio.seek(0)
            df = pd.read_csv(
                bio,
                sep=None,
                engine="python",
                encoding="cp1251",
            )

    df.columns = [str(c).strip() for c in df.columns]
    return df


def clean_fixed_format(df_any: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Clean fixed format by position:
      0: src id
      1: dst id
      8: confidence
      9: weight
    """
    if df_any.shape[1] < 10:
        raise ValueError("Need >= 10 columns (fixed format).")

    src_col = df_any.columns[0]
    dst_col = df_any.columns[1]
    conf_col = df_any.columns[8]
    w_col = df_any.columns[9]

    df = df_any.copy()

    # Normalize types and allow for missing/invalid values.
    df[src_col] = pd.to_numeric(df[src_col], errors="coerce").astype("Int64")
    df[dst_col] = pd.to_numeric(df[dst_col], errors="coerce").astype("Int64")
    df[conf_col] = pd.to_numeric(df[conf_col], errors="coerce")
    df[w_col] = pd.to_numeric(
        df[w_col].astype(str).str.replace(",", ".", regex=False),
        errors="coerce",
    )

    df = df.rename(columns={conf_col: "confidence", w_col: "weight"})
    df = df.dropna(subset=[src_col, dst_col, "confidence", "weight"])
    df = df[df["weight"] > 0]

    meta = {
        "SRC_COL": src_col,
        "DST_COL": dst_col,
        "CONF_COL": "confidence",
        "WEIGHT_COL": "weight",
    }
    return df, meta
