"""Настройки приложения.

Без pydantic и env-магии: если надо поменять дефолты — правь здесь.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    """Набор параметров по умолчанию для визуализации и расчётов."""

    # Визуал
    PLOT_HEIGHT: int = 800
    PLOT_TEMPLATE: str = "plotly_dark"
    COLOR_PRIMARY: str = "#ff4b4b"
    ANIMATION_DURATION_MS: int = 150

    # Расчёты
    DEFAULT_SEED: int = 42
    RICCI_CUTOFF: float = 8.0
    RICCI_MAX_SUPPORT: int = 60
    RICCI_SAMPLE_EDGES: int = 80
    APPROX_EFFICIENCY_K: int = 32

    # Память/мусор (streamlit любит раздуваться)
    MAX_GRAPHS_IN_MEMORY: int = 8
    MAX_EXPS_IN_MEMORY: int = 40

    # Ricci: сколько ребер по умолчанию сэмпли...
    # Эвристика фазового перехода
    CRITICAL_JUMP_THRESHOLD: float = 0.35

    # Дефолты для энергии
    DEFAULT_DAMPING: float = 0.98
    DEFAULT_INJECTION: float = 1.0
    DEFAULT_LEAK: float = 0.005


settings = Settings()
