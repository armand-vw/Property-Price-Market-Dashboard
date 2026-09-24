"""
scripts/fetch_real_dataset.py
=============================
Download the **real** Ames Housing dataset (via OpenML — no API key) and map it
into the project's feature schema as ``datasets/ames.csv``.

The committed CSV is what the real-data benchmark and the notebooks use, so the
repo stays self-contained (no network required at test/run time).

Source
------
Ames, Iowa residential sales (De Cock, 2011), served by OpenML as
``house_prices`` (data id 42165). It is a widely used public teaching dataset.

Usage
-----
    python scripts/fetch_real_dataset.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from property_insights import config  # noqa: E402

DATASETS_DIR = PROJECT_ROOT / "datasets"
AMES_PATH = DATASETS_DIR / "ames.csv"


def fetch_ames() -> pd.DataFrame:
    """Fetch the raw Ames Housing frame from OpenML."""
    bunch = fetch_openml(name="house_prices", as_frame=True)
    frame = bunch.frame.copy()
    if "SalePrice" not in frame.columns:
        frame["SalePrice"] = np.asarray(bunch.target)
    return frame


def map_to_schema(raw: pd.DataFrame) -> pd.DataFrame:
    """Map the raw Ames columns to the project's feature schema."""
    mapped = pd.DataFrame(
        {
            "market": "Ames, IA",
            "neighborhood": raw["Neighborhood"].astype(str),
            "bedrooms": raw["BedroomAbvGr"].astype(int),
            "bathrooms": (raw["FullBath"] + 0.5 * raw["HalfBath"]).astype(float),
            "sqft": raw["GrLivArea"].astype(int),
            "lot_size": raw["LotArea"].astype(int),
            "year_built": raw["YearBuilt"].astype(int),
            "has_pool": raw["PoolArea"].astype(float) > 0,
            "garage_spaces": raw["GarageCars"].fillna(0).astype(int),
            "price": raw["SalePrice"].astype(float),
        }
    )
    mapped = mapped[mapped["sqft"] > 0].reset_index(drop=True)
    return mapped[config.RAW_COLUMNS]


def main() -> None:
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    raw = fetch_ames()
    mapped = map_to_schema(raw)
    mapped.to_csv(AMES_PATH, index=False)

    print("=" * 62)
    print("Real dataset ready (Ames Housing, OpenML house_prices)")
    print("=" * 62)
    print(f"Records        : {len(mapped):,}")
    print(f"Neighborhoods  : {mapped['neighborhood'].nunique()}")
    print(f"Median price   : ${mapped['price'].median():,.0f}")
    print(f"Saved to       : {AMES_PATH}")


if __name__ == "__main__":
    main()
