from __future__ import annotations

import pandas as pd

from ..io_load import load_uploaded_any


def load_any(uploaded_file) -> pd.DataFrame:
    return load_uploaded_any(uploaded_file)
