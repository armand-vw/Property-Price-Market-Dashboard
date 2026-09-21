"""
config.py
=========
Central configuration for the Real Estate Price Estimator & Market Insights
Dashboard.

Keeping every path, random seed, palette and domain constant in one module means
``data_loader.py``, ``model.py`` and ``app.py`` all agree on the same contract.
Change a value here and the entire project follows.
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Project paths
# --------------------------------------------------------------------------- #
BASE_DIR: Path = Path(__file__).resolve().parent
DATA_DIR: Path = BASE_DIR / "data"
MODEL_DIR: Path = BASE_DIR / "models"

DATA_PATH: Path = DATA_DIR / "housing.csv"
MODEL_PATH: Path = MODEL_DIR / "price_model.joblib"
METRICS_PATH: Path = MODEL_DIR / "metrics.json"
IMPORTANCE_PATH: Path = MODEL_DIR / "feature_importance.csv"

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
RANDOM_SEED: int = 42
N_RECORDS: int = 2_000
TEST_SIZE: float = 0.20

#: Reference year used to derive property age from ``year_built``.
REFERENCE_YEAR: int = 2024

#: Worker threads used by the estimator. Capped (rather than ``-1``) to avoid
#: OpenMP thread oversubscription, which can stall training on containerised or
#: resource-limited hosts. Override via the ``RPE_N_JOBS`` environment variable.
N_JOBS: int = int(os.environ.get("RPE_N_JOBS", min(4, os.cpu_count() or 1)))

# --------------------------------------------------------------------------- #
# Domain schema
# --------------------------------------------------------------------------- #
TARGET_COLUMN: str = "price"
CATEGORICAL_FEATURES: list[str] = ["neighborhood"]
NUMERIC_FEATURES: list[str] = [
    "bedrooms",
    "bathrooms",
    "sqft",
    "lot_size",
    "year_built",
    "has_pool",
    "garage_spaces",
]
FEATURE_COLUMNS: list[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES

#: All columns expected in a raw dataset after synthesis.
RAW_COLUMNS: list[str] = FEATURE_COLUMNS + [TARGET_COLUMN]

#: Plausible operating ranges used for input sliders and validation.
FEATURE_BOUNDS: dict[str, tuple[float, float]] = {
    "bedrooms": (1, 6),
    "bathrooms": (1.0, 5.0),
    "sqft": (500, 7_000),
    "lot_size": (1_000, 40_000),
    "year_built": (1900, 2024),
    "garage_spaces": (0, 3),
}

# --------------------------------------------------------------------------- #
# Synthetic neighbourhood profiles
# --------------------------------------------------------------------------- #
# Each neighbourhood has a base price-per-square-foot and a "prestige" multiplier
# that captures the premium buyers pay purely for the location. Together these
# produce realistic, spatially-clustered price tiers.
NEIGHBORHOODS: dict[str, dict[str, float]] = {
    "Downtown Core": {"base_psf": 465.0, "prestige": 1.10},
    "Lakeview": {"base_psf": 435.0, "prestige": 1.07},
    "Sunset Hills": {"base_psf": 410.0, "prestige": 1.05},
    "Old Town": {"base_psf": 385.0, "prestige": 1.02},
    "Riverside": {"base_psf": 360.0, "prestige": 1.00},
    "Maple Grove": {"base_psf": 338.0, "prestige": 0.975},
    "Cedar Park": {"base_psf": 318.0, "prestige": 0.955},
    "Oakwood": {"base_psf": 300.0, "prestige": 0.94},
}

# --------------------------------------------------------------------------- #
# Visual identity (light, clean corporate theme)
# --------------------------------------------------------------------------- #
COLORS: dict[str, str] = {
    "primary": "#4F46E5",      # indigo
    "primary_dark": "#3730A3",
    "accent": "#0EA5E9",       # sky
    "success": "#059669",      # emerald
    "warning": "#D97706",
    "danger": "#DC2626",
    "ink": "#0F172A",          # slate-900
    "muted": "#64748B",        # slate-500
    "surface": "#FFFFFF",
    "background": "#F8FAFC",   # slate-50
    "border": "#E2E8F0",       # slate-200
}

#: Discrete palette used for categorical Plotly traces.
CHART_SEQUENCE: list[str] = [
    "#4F46E5",
    "#0EA5E9",
    "#14B8A6",
    "#F59E0B",
    "#EF4444",
    "#8B5CF6",
    "#10B981",
    "#F97316",
]

#: Bins + labels for the property-age colour encoding in the analytics scatter.
AGE_BINS: list[int] = [0, 10, 30, 60, 200]
AGE_LABELS: list[str] = [
    "New (0-10 yrs)",
    "Recent (11-30 yrs)",
    "Established (31-60 yrs)",
    "Historic (60+ yrs)",
]
