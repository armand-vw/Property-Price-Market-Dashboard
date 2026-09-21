"""
conftest.py
===========
Pytest configuration and shared fixtures.

Placing this file at the project root also ensures the root package directory is
added to ``sys.path`` so tests can ``import config``, ``import model`` etc. when
run from the repository root (``pytest``).
"""

from __future__ import annotations

import pandas as pd
import pytest

from data_loader import clean_data, generate_synthetic_data


@pytest.fixture(scope="session")
def raw_df() -> pd.DataFrame:
    """A small raw (uncleaned) synthetic dataset shared across the test session."""
    return generate_synthetic_data(n_records=500, seed=7)


@pytest.fixture(scope="session")
def clean_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    """The cleaned version of :func:`raw_df`."""
    return clean_data(raw_df)


@pytest.fixture()
def single_property() -> dict:
    """A representative property feature dictionary for inference tests."""
    return {
        "neighborhood": "Riverside",
        "bedrooms": 3,
        "bathrooms": 2.0,
        "sqft": 2_000,
        "lot_size": 7_000,
        "year_built": 2005,
        "has_pool": False,
        "garage_spaces": 2,
    }
