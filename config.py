"""
config.py
=========
Central configuration for the Real Estate Price Estimator & Market Insights
Dashboard.

Keeping every path, random seed, palette and domain constant in one module means
``market_data.py``, ``data_loader.py``, ``model.py`` and ``app.py`` all agree on
the same contract. Change a value here and the entire project follows.
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
MARKET_DATA_DIR: Path = BASE_DIR / "market_data"

DATA_PATH: Path = DATA_DIR / "housing.csv"
MODEL_PATH: Path = MODEL_DIR / "price_model.joblib"
METRICS_PATH: Path = MODEL_DIR / "metrics.json"
IMPORTANCE_PATH: Path = MODEL_DIR / "feature_importance.csv"

# Committed real-market snapshot (produced by scripts/build_market_snapshot.py).
MARKETS_PATH: Path = MARKET_DATA_DIR / "markets.csv"
MARKET_HISTORY_PATH: Path = MARKET_DATA_DIR / "market_history.csv"
NEIGHBORHOOD_META_PATH: Path = MARKET_DATA_DIR / "neighborhood_meta.csv"
NEIGHBORHOOD_HISTORY_PATH: Path = MARKET_DATA_DIR / "neighborhood_history.csv"
MARKET_RENTS_PATH: Path = MARKET_DATA_DIR / "market_rents.csv"
MARKET_HEALTH_PATH: Path = MARKET_DATA_DIR / "market_health.csv"

# Runtime cache for live Zillow fetches.
MARKET_CACHE_DIR: Path = DATA_DIR / "cache"

# International market data (BIS + HM Land Registry).
INTERNATIONAL_DATA_DIR: Path = MARKET_DATA_DIR / "international"
BIS_INDEX_PATH: Path = INTERNATIONAL_DATA_DIR / "bis_index.csv"
UK_REGIONS_PATH: Path = INTERNATIONAL_DATA_DIR / "uk_regions.csv"

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
RANDOM_SEED: int = 42
TEST_SIZE: float = 0.20

#: Reference year used to derive property age from ``year_built``.
REFERENCE_YEAR: int = 2024

#: Worker threads used by the estimator. Capped (rather than ``-1``) to avoid
#: OpenMP thread oversubscription, which can stall training on containerised or
#: resource-limited hosts. Override via the ``RPE_N_JOBS`` environment variable.
N_JOBS: int = int(os.environ.get("RPE_N_JOBS", min(4, os.cpu_count() or 1)))


def _env_flag(name: str) -> bool:
    """Return ``True`` when an environment variable is set to a truthy value."""
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


#: When set (``RPE_OFFLINE=1``) the app never calls the network and serves the
#: committed Zillow snapshot instead - useful for air-gapped or reproducible runs.
OFFLINE: bool = _env_flag("RPE_OFFLINE")

# --------------------------------------------------------------------------- #
# Live market data
# --------------------------------------------------------------------------- #
# Zillow Research publishes these CSVs for public use; no API key required.
# ``metro`` is small enough to fetch at runtime; ``neighborhood`` is ~100 MB and
# is reduced to a committed snapshot at build time instead.
ZILLOW_METRO_ZHVI_URL: str = (
    "https://files.zillowstatic.com/research/public_csvs/zhvi/"
    "Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv"
)
ZILLOW_NEIGHBORHOOD_ZHVI_URL: str = (
    "https://files.zillowstatic.com/research/public_csvs/zhvi/"
    "Neighborhood_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv"
)
ZILLOW_METRO_ZORI_URL: str = (
    "https://files.zillowstatic.com/research/public_csvs/zori/"
    "Metro_zori_uc_sfrcondomfr_sm_month.csv"
)
# US market-health series (metro level, monthly, free).
ZILLOW_INVENTORY_URL: str = (
    "https://files.zillowstatic.com/research/public_csvs/invt_fs/"
    "Metro_invt_fs_uc_sfrcondo_sm_month.csv"
)
ZILLOW_DAYS_PENDING_URL: str = (
    "https://files.zillowstatic.com/research/public_csvs/med_doz_pending/"
    "Metro_med_doz_pending_uc_sfrcondo_sm_month.csv"
)
ZILLOW_MEDIAN_SALE_PRICE_URL: str = (
    "https://files.zillowstatic.com/research/public_csvs/median_sale_price/"
    "Metro_median_sale_price_uc_sfrcondo_month.csv"
)
ZILLOW_ATTRIBUTION: str = (
    "Market data: Zillow Research (ZHVI home values, ZORI rents), latest published month."
)
BIS_ATTRIBUTION: str = (
    "International: BIS Selected Residential Property Prices (nominal, 2010=100); "
    "UK prices: HM Land Registry (OGL)."
)

#: How many of the largest US metros to expose in the market selector.
TOP_N_MARKETS: int = 15
#: Neighbourhoods retained per market in the snapshot.
NEIGHBORHOODS_PER_MARKET: int = 12
#: Months of history kept in the committed snapshot.
MARKET_HISTORY_MONTHS: int = 180
NEIGHBORHOOD_HISTORY_MONTHS: int = 120
RENT_HISTORY_MONTHS: int = 60
MARKET_HEALTH_MONTHS: int = 60

# --------------------------------------------------------------------------- #
# International markets (BIS residential property prices + UK Land Registry)
# --------------------------------------------------------------------------- #
#: BIS "Selected Residential Property Prices" bulk file (free, no API key).
BIS_SPP_URL: str = "https://data.bis.org/static/bulk/WS_SPP_csv_col.zip"
#: Quarterly observations retained for the international comparison.
INTERNATIONAL_HISTORY_QUARTERS: int = 80
#: Monthly UK nation observations retained.
UK_HISTORY_MONTHS: int = 120

#: Countries exposed by the switcher. ``code`` matches the BIS ``REF_AREA``.
#: A focused set of ten major markets (kept deliberately small for clarity).
COUNTRIES: dict[str, dict[str, str]] = {
    "US": {"name": "United States", "source": "Zillow Research"},
    "GB": {"name": "United Kingdom", "source": "HM Land Registry · BIS"},
    "CA": {"name": "Canada", "source": "BIS"},
    "AU": {"name": "Australia", "source": "BIS"},
    "JP": {"name": "Japan", "source": "BIS"},
    "CN": {"name": "China", "source": "BIS"},
    "DE": {"name": "Germany", "source": "BIS"},
    "FR": {"name": "France", "source": "BIS"},
    "IN": {"name": "India", "source": "BIS"},
    "BR": {"name": "Brazil", "source": "BIS"},
}

#: Display order of the country switcher.
COUNTRY_ORDER: list[str] = list(COUNTRIES)

#: Curated subset shown in cross-country comparison charts (all others remain
#: available in the switcher).
COMPARISON_COUNTRIES: list[str] = ["US", "GB", "CA", "AU", "JP", "CN", "DE", "FR"]

#: UK nations available in the UK regional view (HM Land Registry slugs).
UK_NATIONS: dict[str, str] = {
    "england": "England",
    "scotland": "Scotland",
    "wales": "Wales",
    "northern-ireland": "Northern Ireland",
}
#: Hours a live fetch is cached on disk before being refreshed.
MARKET_CACHE_TTL_HOURS: int = 24
#: Network timeout (seconds) for Zillow fetches.
MARKET_FETCH_TIMEOUT: int = 30

#: Synthetic list-price anchor: assumed median home size used to convert a
#: real median *home value* into a price-per-square-foot anchor.
ASSUMED_MEDIAN_SQFT: float = 2_000.0
#: Synthetic listings generated per neighbourhood.
LISTINGS_PER_NEIGHBORHOOD: int = 40

# --------------------------------------------------------------------------- #
# Domain schema
# --------------------------------------------------------------------------- #
TARGET_COLUMN: str = "price"
CATEGORICAL_FEATURES: list[str] = ["market", "neighborhood"]
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

# --------------------------------------------------------------------------- #
# Plain-English education layer
# --------------------------------------------------------------------------- #
#: One-line definitions shown as tooltips / a glossary for everyday users.
GLOSSARY: dict[str, str] = {
    "zhvi": "Zillow Home Value Index — the typical (median) home value in an area, updated monthly.",
    "zori": "Zillow Observed Rent Index — the typical monthly rent in an area.",
    "price": "The estimated fair market value of a property, in local currency.",
    "yoy": "Year over year — how much a value changed versus the same time last year.",
    "mom": "Month over month — how much a value changed versus the previous month.",
    "change_5y": "The total change over the last five years.",
    "index": "A house-price index. 2010 = 100, so 120 means values are 20% higher than in 2010.",
    "yield": "Gross rental yield — annual rent divided by the property price, before costs.",
    "rent": "The median monthly rent for the area.",
    "inventory": "For-sale inventory — how many homes are currently listed for sale.",
    "days_to_pending": "Days to pending — the typical time from listing to going under contract; lower means a faster, hotter market.",
    "median_sale_price": "The median price of homes that actually sold (closed sales).",
    "temperature": "Market temperature — a simple Hot / Warm / Cool read on current conditions.",
    "r2": "R² — the share of price variation the model explains (1.0 is perfect).",
    "mape": "MAPE — the model's average percentage error. Lower is more accurate.",
}

#: Thresholds for the Hot / Warm / Cool market temperature rating.
TEMPERATURE_HOT: float = 0.25
TEMPERATURE_COOL: float = -0.25
