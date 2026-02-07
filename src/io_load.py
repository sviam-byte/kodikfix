from __future__ import annotations

import io

import pandas as pd
from pandas.errors import ParserError

from .preprocess import coerce_fixed_format


def load_uploaded_any(file_bytes: bytes, filename: str) -> pd.DataFrame:
    name = (filename or "").lower()
    bio = io.BytesIO(file_bytes)

    if name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(bio)
    else:
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
    """Compatibility wrapper for the legacy fixed-format loader.

    Delegates to :func:`src.preprocess.coerce_fixed_format` so that all weight
    handling stays consistent across the codebase.
    """
    df, meta0 = coerce_fixed_format(df_any)
    meta = {
        "SRC_COL": meta0["src_col"],
        "DST_COL": meta0["dst_col"],
        "CONF_COL": "confidence",
        "WEIGHT_COL": "weight",
    }
    return df, meta
