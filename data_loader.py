"""
data_loader.py
==============
Data acquisition, synthesis and cleaning for the Real Estate Price Estimator.

Responsibilities
----------------
* ``generate_synthetic_data`` - build a realistic, *correlated* housing dataset
  from configurable neighbourhood price profiles. No external data required.
* ``load_or_create_data``    - load ``data/housing.csv`` if it exists, otherwise
  synthesise, clean and persist it.
* ``clean_data``             - de-duplicate, impute, winsorise and validate.
* ``get_neighborhood_stats`` - aggregate statistics powering the dashboard UI.

Run directly to (re)generate the dataset::

    python data_loader.py --force
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import config


# --------------------------------------------------------------------------- #
# Synthetic data generation
# --------------------------------------------------------------------------- #
def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


def generate_synthetic_data(
    n_records: int = config.N_RECORDS,
    seed: int = config.RANDOM_SEED,
) -> pd.DataFrame:
    """Generate a realistic synthetic housing dataset.

    The generator deliberately introduces **correlations** that a model can learn:
    larger homes have more bedrooms/bathrooms and are more likely to have a pool;
    newer and more prestigious neighbourhoods command a higher price per square
    foot. A small amount of missingness and duplicate rows are injected so the
    cleaning stage is exercised rather than cosmetic.

    Parameters
    ----------
    n_records:
        Number of property records to generate.
    seed:
        Seed for the pseudo-random generator (reproducibility).

    Returns
    -------
    pandas.DataFrame
        Raw, *uncleaned* frame with :data:`config.RAW_COLUMNS`.
    """
    rng = np.random.default_rng(seed)

    # --- Location --------------------------------------------------------- #
    names = list(config.NEIGHBORHOODS)
    # Weight demand towards mid-tier suburbs so the sample is not dominated by
    # the most expensive handful of neighbourhoods.
    raw_weights = np.array(
        [config.NEIGHBORHOODS[n]["prestige"] for n in names], dtype=float
    )
    weights = raw_weights / raw_weights.sum()
    neighborhood = rng.choice(names, size=n_records, p=weights)
    base_psf = np.array([config.NEIGHBORHOODS[n]["base_psf"] for n in neighborhood])
    prestige = np.array([config.NEIGHBORHOODS[n]["prestige"] for n in neighborhood])

    # --- Property characteristics ---------------------------------------- #
    # Square footage is the primary metric; log-normal spread gives the
    # right-skewed distribution seen in real listings.
    sqft = rng.lognormal(mean=np.log(2_100.0), sigma=0.46, size=n_records)
    sqft = np.clip(np.round(sqft), 550, 7_000)

    # Bedroom count follows size (bigger homes have more bedrooms) with noise.
    bedrooms = np.round(sqft / 620.0 + rng.normal(0, 0.7, size=n_records))
    bedrooms = np.clip(bedrooms, 1, 6).astype(int)

    # Bathrooms scale with bedrooms with a little noise, then clamp to a
    # realistic 1.0 - 5.0 range.
    bathrooms = np.round(0.6 * bedrooms + rng.normal(0.4, 0.5, size=n_records), 1)
    bathrooms = np.clip(bathrooms, 1.0, 5.0)

    # Lot size correlates with square footage but varies independently.
    lot_size = sqft * rng.uniform(2.0, 9.0, size=n_records)
    lot_size = np.clip(np.round(lot_size), 1_200, 40_000)

    # Newer construction is more common than century-old stock.
    year_built = np.round(
        2024 - rng.gamma(shape=2.2, scale=16.0, size=n_records)
    ).astype(int)
    year_built = np.clip(year_built, 1900, 2024)

    # Pools are more likely in big / upscale homes.
    pool_logit = (sqft - 2_600) / 700.0 + (prestige - 1.0) * 3.0
    has_pool = rng.random(n_records) < _sigmoid(pool_logit)

    garage_spaces = np.clip(
        np.round(0.4 * bedrooms + rng.normal(0, 0.8, size=n_records)), 0, 3
    ).astype(int)

    # --- Price construction ---------------------------------------------- #
    age = config.REFERENCE_YEAR - year_built

    # Homes depreciate with age but historic character stabilises value.
    age_factor = 0.74 + 0.26 * np.exp(-age / 65.0)

    bed_factor = 1.0 + 0.018 * (bedrooms - 3)
    bath_factor = 1.0 + 0.035 * (bathrooms - 2)
    pool_factor = np.where(has_pool, 1.065, 1.0)
    garage_factor = 1.0 + 0.028 * garage_spaces
    lot_factor = 1.0 + 0.000008 * (lot_size - 6_000)

    # Multiplicative log-normal noise -> realistic idiosyncratic dispersion.
    noise = rng.lognormal(mean=0.0, sigma=0.11, size=n_records)

    price = (
        base_psf
        * sqft
        * prestige
        * age_factor
        * bed_factor
        * bath_factor
        * pool_factor
        * garage_factor
        * lot_factor
        * noise
    )

    df = pd.DataFrame(
        {
            "neighborhood": neighborhood,
            "bedrooms": bedrooms.astype(int),
            "bathrooms": bathrooms,
            "sqft": sqft.astype(float),
            "lot_size": lot_size.astype(float),
            "year_built": year_built,
            "has_pool": has_pool,
            "garage_spaces": garage_spaces,
            "price": np.round(price, 2),
        }
    )

    # --- Inject realistic imperfections ---------------------------------- #
    # 1. Missing values (imputed later by clean_data).
    missing_spec = {
        "bathrooms": 0.020,
        "sqft": 0.015,
        "lot_size": 0.020,
        "year_built": 0.010,
    }
    for column, fraction in missing_spec.items():
        n_missing = int(n_records * fraction)
        if n_missing == 0:
            continue
        idx = rng.choice(n_records, size=n_missing, replace=False)
        df[column] = df[column].astype(float)
        df.loc[idx, column] = np.nan

    # 2. A few exact duplicate rows (dropped later).
    duplicate_rows = df.sample(n=max(1, int(n_records * 0.01)), random_state=seed)
    df = pd.concat([df, duplicate_rows], ignore_index=True)

    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Cleaning & validation
# --------------------------------------------------------------------------- #
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and validate a raw housing frame.

    Steps
    -----
    1. Drop exact duplicate records.
    2. Coerce numeric/bool dtypes.
    3. Remove physically impossible rows (non-positive area, bad year).
    4. Median-impute missing numerics, using the *neighbourhood* median when
       available and the global median as a fallback.
    5. Winsorise price and square footage at the IQR fences to tame outliers.
    6. Add derived ``price_per_sqft`` and ``property_age`` columns.

    Parameters
    ----------
    df:
        Raw dataframe containing :data:`config.RAW_COLUMNS`.

    Returns
    -------
    pandas.DataFrame
        Cleaned, analysis-ready frame with a fresh integer index.
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
            neighborhood_median = data.groupby("neighborhood")[column].transform(
                "median"
            )
            data[column] = data[column].fillna(neighborhood_median)
            data[column] = data[column].fillna(data[column].median())

    # 5. Winsorise heavy-tailed columns at the IQR fences ------------------ #
    for column in ["sqft", "price"]:
        q1, q3 = data[column].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        data[column] = data[column].clip(lower=lower, upper=upper)

    # 6. Derived features -------------------------------------------------- #
    data["price_per_sqft"] = data["price"] / data["sqft"]
    data["property_age"] = (config.REFERENCE_YEAR - data["year_built"]).clip(lower=0)

    # Final dtype tidy-up
    data["bedrooms"] = data["bedrooms"].round().astype(int)
    data["garage_spaces"] = data["garage_spaces"].round().astype(int)
    data["year_built"] = data["year_built"].round().astype(int)
    data["lot_size"] = data["lot_size"].round().astype(int)
    data["sqft"] = data["sqft"].round().astype(int)

    return data.reset_index(drop=True)


def validate_schema(df: pd.DataFrame) -> bool:
    """Return ``True`` if ``df`` matches the expected cleaned schema.

    Raises
    ------
    ValueError
        If required columns are absent or the target contains nulls.
    """
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
    """Load the cleaned dataset from CSV, synthesising it when necessary.

    Parameters
    ----------
    path:
        Location of the CSV file.
    force_regenerate:
        When ``True`` the CSV is rebuilt from scratch even if it exists.
    """
    path = pd.io.common.stringify_path(path)

    path = Path(path)

    if force_regenerate or not path.exists():
        raw = generate_synthetic_data()
        data = clean_data(raw)
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        data.to_csv(path, index=False)
        return data

    data = pd.read_csv(path)
    return clean_data(data)


def get_neighborhood_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-neighbourhood market statistics.

    Returns a frame indexed by ``neighborhood`` with listing counts, median
    price, median price-per-sqft and median square footage. Used by the
    dashboard for the live "vs. neighbourhood median" comparison.
    """
    stats = (
        df.groupby("neighborhood")
        .agg(
            listings=("price", "size"),
            median_price=("price", "median"),
            median_price_per_sqft=("price_per_sqft", "median"),
            median_sqft=("sqft", "median"),
        )
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
    args = _parse_args()
    raw = generate_synthetic_data()
    data = clean_data(raw)
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    data.to_csv(config.DATA_PATH, index=False)

    print("=" * 62)
    print("Real Estate dataset ready")
    print("=" * 62)
    print(f"Raw records          : {len(raw):,}")
    print(f"Clean records        : {len(data):,}")
    print(f"Columns              : {', '.join(data.columns)}")
    print(f"Median price         : ${data['price'].median():,.0f}")
    print(f"Median price / sqft  : ${data['price_per_sqft'].median():,.0f}")
    print(f"Saved to             : {config.DATA_PATH}")


if __name__ == "__main__":
    main()
