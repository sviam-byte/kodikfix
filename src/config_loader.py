"""Закомментить надо

"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import streamlit as st
import yaml


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@st.cache_data(show_spinner=False)
def load_metrics_info() -> Dict[str, Dict[str, str]]:
    path = _project_root() / "config" / "metrics_info.yaml"
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    # ожидаем {"help_text": {...}, "metric_help": {...}}
    return {
        "help_text": dict(data.get("help_text", {}) or {}),
        "metric_help": dict(data.get("metric_help", {}) or {}),
    }


@st.cache_data(show_spinner=False)
def load_css() -> str:
    path = _project_root() / "assets" / "style.css"
    return path.read_text(encoding="utf-8") if path.exists() else ""
