"""Unit tests for the international market-data layer (offline)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from property_insights import config, international_data


def test_load_bis_index() -> None:
    bis = international_data.load_bis_index()
    assert set(config.COUNTRIES).issubset(set(bis["country_code"].unique()))
    assert {"period", "period_date", "index", "yoy_pct"}.issubset(bis.columns)
    assert (bis["index"] > 0).all()
    assert pd.api.types.is_datetime64_any_dtype(bis["period_date"])


def test_build_country_summary() -> None:
    bis = international_data.load_bis_index()
    summary = international_data.build_country_summary(bis)
    assert len(summary) == len(config.COUNTRIES)
    assert (summary["index"] > 0).all()
    assert np.isfinite(summary["yoy_pct"]).all()
    assert summary["country_code"].tolist() == config.COUNTRY_ORDER


def test_build_comparison_indexes_to_100() -> None:
    bis = international_data.load_bis_index()
    comparison = international_data.build_comparison(bis, quarters=20)
    assert not comparison.empty
    for _, group in comparison.groupby("country_code"):
        group = group.sort_values("period")
        np.testing.assert_allclose(group["indexed"].iloc[0], 100.0)
        assert len(group) <= 20


def test_load_uk_regions() -> None:
    uk = international_data.load_uk_regions()
    assert set(uk["region"]) == set(config.UK_NATIONS.values())
    assert (uk["avg_price_gbp"] > 0).all()
    assert uk["month"].max() <= "2099-12"


def test_build_uk_summary() -> None:
    uk = international_data.load_uk_regions()
    summary = international_data.build_uk_summary(uk)
    assert len(summary) == len(config.UK_NATIONS)
    assert summary["avg_price_gbp"].is_monotonic_decreasing


def test_get_country_data_falls_back_to_snapshot(monkeypatch) -> None:
    monkeypatch.setattr(international_data, "fetch_live_bis", lambda **_: None)
    data = international_data.get_country_data(force_refresh=True)
    assert data["source"] == "snapshot"
    assert len(data["summary"]) == len(config.COUNTRIES)
    assert data["fetched_at"] is not None
    assert not data["uk_summary"].empty


def test_country_name() -> None:
    assert international_data.country_name("US") == "United States"
    assert international_data.country_name("GB") == "United Kingdom"
