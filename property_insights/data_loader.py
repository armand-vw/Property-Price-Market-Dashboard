"""
data_loader.py
==============
Data generation, cleaning and aggregation for the Real Estate Price Estimator.

Listing-level data is synthesised, but it is **anchored to real Zillow market
data**: each neighbourhood's price-per-square-foot anchor is derived from its
real median home value (see :mod:`market_data`). This keeps predictive price
levels realistic per market (e.g. San Francisco vs. Detroit) while allowing a
rich feature set (beds, baths, size, age, amenities) that aggregate data cannot
provide.

Run directly to (re)generate the dataset::

    python -m property_insights.data_loader --force
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, market_data


# --------------------------------------------------------------------------- #
# Synthetic data generation (anchored to real market values)
# --------------------------------------------------------------------------- #
def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


def _market_short_name(market: str) -> str:
    """``"Atlanta, GA"`` -> ``"Atlanta"``."""
    return str(market).split(",")[0].strip()


def _generate_neighborhood(
    rng: np.random.Generator,
    anchor: pd.Series,
    n_listings: int,
) -> pd.DataFrame:
    """Generate ``n_listings`` synthetic properties for one neighbourhood.

    The neighbourhood's ``base_price_per_sqft`` anchor is multiplied by the
    same structural and amenity factors used across the dataset, so a home of
    exactly :data:`config.ASSUMED_MEDIAN_SQFT` with neutral characteristics
    prices close to the real neighbourhood median.
    """
    base_psf = float(anchor["base_price_per_sqft"])

    # Square footage: log-normal around the assumed median size.
    sqft = rng.lognormal(
        mean=np.log(config.ASSUMED_MEDIAN_SQFT), sigma=0.40, size=n_listings
    )
    sqft = np.clip(np.round(sqft), 550, 7_000)

    # Bedrooms follow size; bathrooms follow bedrooms.
    bedrooms = np.round(sqft / 620.0 + rng.normal(0, 0.7, size=n_listings))
    bedrooms = np.clip(bedrooms, 1, 6).astype(int)
    bathrooms = np.round(0.6 * bedrooms + rng.normal(0.4, 0.5, size=n_listings), 1)
    bathrooms = np.clip(bathrooms, 1.0, 5.0)

    lot_size = sqft * rng.uniform(2.0, 9.0, size=n_listings)
    lot_size = np.clip(np.round(lot_size), 1_200, 40_000)

    year_built = np.round(
        config.REFERENCE_YEAR - rng.gamma(shape=2.2, scale=16.0, size=n_listings)
    ).astype(int)
    year_built = np.clip(year_built, 1900, config.REFERENCE_YEAR)

    has_pool = rng.random(n_listings) < _sigmoid((sqft - 2_600) / 700.0)
    garage_spaces = np.clip(
        np.round(0.4 * bedrooms + rng.normal(0, 0.8, size=n_listings)), 0, 3
    ).astype(int)

    age = config.REFERENCE_YEAR - year_built
    age_factor = 0.74 + 0.26 * np.exp(-age / 65.0)
    bed_factor = 1.0 + 0.018 * (bedrooms - 3)
    bath_factor = 1.0 + 0.035 * (bathrooms - 2)
    pool_factor = np.where(has_pool, 1.065, 1.0)
    garage_factor = 1.0 + 0.028 * garage_spaces
    lot_factor = 1.0 + 0.000008 * (lot_size - 6_000)
    noise = rng.lognormal(mean=0.0, sigma=0.11, size=n_listings)

    price = (
        base_psf
        * sqft
        * age_factor
        * bed_factor
        * bath_factor
        * pool_factor
        * garage_factor
        * lot_factor
        * noise
    )

    market = str(anchor["market"])
    short = _market_short_name(market)
    location = f"{anchor['neighborhood']} ({short})"

    return pd.DataFrame(
        {
            "market": market,
            "neighborhood": location,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "sqft": sqft.astype(float),
            "lot_size": lot_size.astype(float),
            "year_built": year_built,
            "has_pool": has_pool,
            "garage_spaces": garage_spaces,
            "price": np.round(price, 2),
        }
    )


def generate_synthetic_data(
    listings_per_neighborhood: int = config.LISTINGS_PER_NEIGHBORHOOD,
    seed: int = config.RANDOM_SEED,
) -> pd.DataFrame:
    """Generate a realistic, market-anchored synthetic housing dataset.

    Each neighbourhood from the committed market snapshot contributes
    ``listings_per_neighborhood`` properties. Missing values and duplicate rows
    are injected so the cleaning stage is exercised rather than cosmetic.

    Parameters
    ----------
    listings_per_neighborhood:
        Number of synthetic listings generated per neighbourhood.
    seed:
        Seed for the pseudo-random generator (reproducibility).

    Returns
    -------
    pandas.DataFrame
        Raw, *uncleaned* frame with :data:`config.RAW_COLUMNS`.
    """
    rng = np.random.default_rng(seed)
    anchors = market_data.get_synthetic_anchors()

    frames = [
        _generate_neighborhood(rng, anchor, listings_per_neighborhood)
        for _, anchor in anchors.iterrows()
    ]
    df = pd.concat(frames, ignore_index=True)

    # --- Inject realistic imperfections ---------------------------------- #
    missing_spec = {
        "bathrooms": 0.020,
        "sqft": 0.015,
        "lot_size": 0.020,
        "year_built": 0.010,
    }
    for column, fraction in missing_spec.items():
        n_missing = int(len(df) * fraction)
        if n_missing == 0:
            continue
        idx = rng.choice(len(df), size=n_missing, replace=False)
        df[column] = df[column].astype(float)
        df.loc[idx, column] = np.nan

    duplicate_rows = df.sample(
        n=max(1, int(len(df) * 0.01)), random_state=seed
    )
    df = pd.concat([df, duplicate_rows], ignore_index=True)

    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Cleaning & validation
# --------------------------------------------------------------------------- #
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and validate a raw housing frame.

    Steps: drop duplicates, coerce types, remove impossible rows,
    neighbourhood-median imputation, IQR winsorisation, and derived
    ``price_per_sqft`` / ``property_age`` columns.
    """
    required = set(config.RAW_COLUMNS)
    missing_columns = required.difference(df.columns)
    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing_columns)}"
        )

    data = df.copy()

    # 1. Duplicates -------------------------------------------------------- #
    data = data.drop_duplicates().reset_index(drop=True)

    # 2. Types ------------------------------------------------------------- #
    for column in config.NUMERIC_FEATURES:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data["has_pool"] = (
        data["has_pool"].astype("boolean").fillna(False).astype(bool)
    )
    data[config.TARGET_COLUMN] = pd.to_numeric(
        data[config.TARGET_COLUMN], errors="coerce"
    )

    # 3. Domain-valid rows only ------------------------------------------- #
    data = data.dropna(subset=[config.TARGET_COLUMN])
    data = data[
        (data["sqft"] > 0)
        & (data["price"] > 0)
        & (data["year_built"].between(1800, config.REFERENCE_YEAR))
    ].reset_index(drop=True)

    # 4. Neighbourhood-aware median imputation ---------------------------- #
    for column in ["bathrooms", "sqft", "lot_size", "year_built", "garage_spaces"]:
        if data[column].isna().any():
            group_median = data.groupby("neighborhood")[column].transform("median")
            data[column] = data[column].fillna(group_median)
            data[column] = data[column].fillna(data[column].median())

    # 5. Winsorise heavy-tailed columns at the IQR fences ------------------ #
    for column in ["sqft", "price"]:
        q1, q3 = data[column].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        data[column] = data[column].clip(lower=lower, upper=upper)

    # 6. Final dtype tidy-up ---------------------------------------------- #
    data["bedrooms"] = data["bedrooms"].round().astype(int)
    data["garage_spaces"] = data["garage_spaces"].round().astype(int)
    data["year_built"] = data["year_built"].round().astype(int)
    data["lot_size"] = data["lot_size"].round().astype(int)
    data["sqft"] = data["sqft"].round().astype(int)

    # 7. Derived features -------------------------------------------------- #
    data["price_per_sqft"] = data["price"] / data["sqft"]
    data["property_age"] = (config.REFERENCE_YEAR - data["year_built"]).clip(lower=0)

    return data.reset_index(drop=True)


def validate_schema(df: pd.DataFrame) -> bool:
    """Return ``True`` if ``df`` matches the expected cleaned schema."""
    missing_columns = set(config.RAW_COLUMNS).difference(df.columns)
    if missing_columns:
        raise ValueError(f"Missing columns: {sorted(missing_columns)}")
    if df[config.TARGET_COLUMN].isna().any():
        raise ValueError("Target column contains null values.")
    if df.empty:
        raise ValueError("Dataset is empty.")
    return True


# --------------------------------------------------------------------------- #
# Convenience loaders & aggregates
# --------------------------------------------------------------------------- #
def load_or_create_data(
    path=config.DATA_PATH,
    force_regenerate: bool = False,
) -> pd.DataFrame:
    """Load the cleaned dataset from CSV, synthesising it when necessary."""
    path = Path(path)

    if force_regenerate or not path.exists():
        raw = generate_synthetic_data()
        data = clean_data(raw)
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        data.to_csv(path, index=False)
        return data

    return clean_data(pd.read_csv(path))


def get_location_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-location (market + neighbourhood) market statistics."""
    stats = (
        df.groupby(["market", "neighborhood"])
        .agg(
            listings=("price", "size"),
            median_price=("price", "median"),
            median_price_per_sqft=("price_per_sqft", "median"),
            median_sqft=("sqft", "median"),
        )
        .reset_index()
    )
    return stats


def get_market_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate synthetic listing statistics per market."""
    stats = (
        df.groupby("market")
        .agg(
            listings=("price", "size"),
            median_price=("price", "median"),
            median_price_per_sqft=("price_per_sqft", "median"),
            locations=("neighborhood", "nunique"),
        )
        .reset_index()
        .sort_values("median_price", ascending=False)
    )
    return stats


# --------------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------------- #
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate/clean the housing dataset.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate the CSV even if it already exists.",
    )
    return parser.parse_args()


def main() -> None:
    """Generate, clean and persist the dataset, printing a short summary."""
    _parse_args()
    raw = generate_synthetic_data()
    data = clean_data(raw)
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    data.to_csv(config.DATA_PATH, index=False)

    print("=" * 62)
    print("Real Estate dataset ready")
    print("=" * 62)
    print(f"Raw records          : {len(raw):,}")
    print(f"Clean records        : {len(data):,}")
    print(f"Markets              : {data['market'].nunique()}")
    print(f"Locations            : {data['neighborhood'].nunique()}")
    print(f"Median price         : ${data['price'].median():,.0f}")
    print(f"Saved to             : {config.DATA_PATH}")


if __name__ == "__main__":
    main()
