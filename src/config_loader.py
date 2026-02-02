"""Загрузка конфигов и статических ассетов.

Здесь живёт то, что не должно размазываться по app.py/ui_blocks.py:
- справка по метрикам (yaml)
- общий CSS (assets/style.css)

Streamlit-кэш держит это в памяти между перерендерами.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import streamlit as st
import yaml


def _project_root() -> Path:
    # src/.. -> корень репо
    return Path(__file__).resolve().parents[1]


@st.cache_data(show_spinner=False)
def load_metrics_info() -> Dict[str, Dict[str, str]]:
    """Прочитать config/metrics_info.yaml."""
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
    """Прочитать assets/style.css."""
    path = _project_root() / "assets" / "style.css"
    return path.read_text(encoding="utf-8") if path.exists() else ""
