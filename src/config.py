"""
дефолтные настройки приложения

"""

from __future__ import annotations

from dataclasses import dataclass

# =========================
# Numerical stability
# =========================
EPS_W: float = 1e-12        # weight -> distance guard
EPS_LOG: float = 1e-20      # log(p) guard

# =========================
# Ricci / W1 approximation
# =========================
RICCI_MASS_SCALE: int = 120_000
RICCI_MAX_SUPPORT: int = 60
RICCI_CUTOFF: float = 8.0
RICCI_SAMPLE_EDGES: int = 80

# =========================
# Phase transition detection
# =========================
JUMP_ALPHA: float = 0.01              # quantile for null model
JUMP_FRACTION_FALLBACK: float = 0.35 # quick mode only
NULL_N_ITER_DEFAULT: int = 25


@dataclass(frozen=True)
class Settings:
    # Визуал
    PLOT_HEIGHT: int = 800
    PLOT_TEMPLATE: str = "plotly_dark"
    COLOR_PRIMARY: str = "#ff4b4b"
    ANIMATION_DURATION_MS: int = 150

    # Расчёты
    DEFAULT_SEED: int = 42

    # Политика весов (важно для корректности и воспроизводимости)
    # Возможные значения:
    # - "drop_nonpositive": удалить рёбра с w<=0 (дефолт, строго для dist=1/w)
    # - "abs": заменить w <- |w| (подходит для корреляций, но теряется знак)
    # - "clip": заменить w <- max(w, eps)
    # - "shift": сдвинуть веса на фиксированную константу (w <- w + shift, затем max(w, eps))
    WEIGHT_POLICY: str = "drop_nonpositive"
    WEIGHT_EPS: float = 1e-9
    WEIGHT_SHIFT: float = 0.0
    RICCI_CUTOFF: float = RICCI_CUTOFF
    RICCI_MAX_SUPPORT: int = RICCI_MAX_SUPPORT
    RICCI_SAMPLE_EDGES: int = RICCI_SAMPLE_EDGES
    APPROX_EFFICIENCY_K: int = 32

    # Ограничения на хранение в памяти для UI.
    MAX_GRAPHS_IN_MEMORY: int = 8
    MAX_EXPS_IN_MEMORY: int = 40

    # Дефолты для энергии
    DEFAULT_DAMPING: float = 0.98
    DEFAULT_INJECTION: float = 1.0
    DEFAULT_LEAK: float = 0.005


settings = Settings()
