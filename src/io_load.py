from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
from pandas.errors import ParserError

from .preprocess import coerce_fixed_format


def load_edges(path_or_bytes: str | Path | bytes, filename: str | None = None) -> pd.DataFrame:
    """Load edges table from a file path or uploaded bytes."""
    if isinstance(path_or_bytes, (str, Path)):
        path = Path(path_or_bytes)
        if path.suffix.lower() in (".xlsx", ".xls"):
            df = pd.read_excel(path)
        else:
            df = pd.read_csv(path, sep=None, engine="python", encoding_errors="replace")
        df.columns = [str(c).strip() for c in df.columns]
        return df
    if isinstance(path_or_bytes, (bytes, bytearray)):
        use_name = filename or ""
        return load_uploaded_any(bytes(path_or_bytes), use_name)
    raise TypeError("path_or_bytes must be a file path or raw bytes")


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
