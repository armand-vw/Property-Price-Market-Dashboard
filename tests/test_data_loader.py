"""Unit tests for the data synthesis and cleaning layer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import config
from data_loader import (
    generate_synthetic_data,
    get_location_stats,
    get_market_stats,
    validate_schema,
)

EXPECTED_DERIVED = ["price_per_sqft", "property_age"]


def test_generate_returns_expected_columns(raw_df: pd.DataFrame) -> None:
    assert set(config.RAW_COLUMNS).issubset(raw_df.columns)
    assert raw_df["market"].nunique() == config.TOP_N_MARKETS


def test_generate_is_reproducible() -> None:
    first = generate_synthetic_data(listings_per_neighborhood=2, seed=123)
    second = generate_synthetic_data(listings_per_neighborhood=2, seed=123)
    pd.testing.assert_frame_equal(first, second)


def test_generate_injects_missing_values(raw_df: pd.DataFrame) -> None:
    assert raw_df[["bathrooms", "sqft", "lot_size", "year_built"]].isna().any().any()


def test_generate_injects_duplicates(raw_df: pd.DataFrame) -> None:
    assert raw_df.duplicated().any()


def test_clean_removes_nulls_and_duplicates(clean_df: pd.DataFrame) -> None:
    assert clean_df.isna().sum().sum() == 0
    assert not clean_df.duplicated().any()


def test_clean_enforces_domain_constraints(clean_df: pd.DataFrame) -> None:
    assert (clean_df["price"] > 0).all()
    assert (clean_df["sqft"] > 0).all()
    assert clean_df["year_built"].between(1800, config.REFERENCE_YEAR).all()
    assert clean_df["bedrooms"].between(1, 6).all()
    assert clean_df["garage_spaces"].between(0, 3).all()


def test_clean_adds_derived_columns(clean_df: pd.DataFrame) -> None:
    for column in EXPECTED_DERIVED:
        assert column in clean_df.columns
    assert (clean_df["property_age"] >= 0).all()
    expected_psf = clean_df["price"] / clean_df["sqft"]
    np.testing.assert_allclose(clean_df["price_per_sqft"], expected_psf)


def test_clean_has_expected_dtypes(clean_df: pd.DataFrame) -> None:
    assert pd.api.types.is_integer_dtype(clean_df["bedrooms"])
    assert pd.api.types.is_integer_dtype(clean_df["sqft"])
    assert pd.api.types.is_integer_dtype(clean_df["garage_spaces"])
    assert pd.api.types.is_bool_dtype(clean_df["has_pool"])
    assert pd.api.types.is_float_dtype(clean_df["price"])


def test_location_stats_are_consistent(clean_df: pd.DataFrame) -> None:
    stats = get_location_stats(clean_df)
    assert stats["listings"].sum() == len(clean_df)
    assert (stats["median_price"] > 0).all()
    assert set(stats["market"]).issubset(set(clean_df["market"]))


def test_market_stats_are_consistent(clean_df: pd.DataFrame) -> None:
    stats = get_market_stats(clean_df)
    assert stats["listings"].sum() == len(clean_df)
    assert len(stats) == clean_df["market"].nunique()
    assert stats["median_price"].is_monotonic_decreasing


def test_validate_schema_accepts_clean_data(clean_df: pd.DataFrame) -> None:
    assert validate_schema(clean_df) is True


def test_validate_schema_rejects_missing_columns(clean_df: pd.DataFrame) -> None:
    broken = clean_df.drop(columns=["sqft"])
    with pytest.raises(ValueError):
        validate_schema(broken)
