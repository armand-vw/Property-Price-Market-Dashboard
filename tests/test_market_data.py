"""Unit tests for the real market-data layer (offline; uses the snapshot)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from property_insights import config, market_data


def test_load_markets() -> None:
    markets = market_data.load_markets()
    assert len(markets) == config.TOP_N_MARKETS
    assert {"market_id", "market", "state", "size_rank"}.issubset(markets.columns)
    assert markets["size_rank"].is_monotonic_increasing


def test_market_history_covers_all_markets() -> None:
    history = market_data.load_market_history()
    markets = market_data.load_markets()
    assert set(history["market_id"]).issuperset(set(markets["market_id"]))
    assert (history["value"] > 0).all()


def test_build_market_summary() -> None:
    markets = market_data.load_markets()
    history = market_data.load_market_history()
    summary = market_data.build_market_summary(markets, history)

    assert len(summary) == len(markets)
    assert (summary["latest_value"] > 0).all()
    assert summary["latest_month"].notna().all()
    assert np.isfinite(summary["yoy_pct"]).all()
    assert summary["size_rank"].is_monotonic_increasing


def test_neighborhood_metadata() -> None:
    meta = market_data.load_neighborhood_meta()
    assert (meta["latest_value"] > 0).all()
    assert (meta["base_price_per_sqft"] > 0).all()
    per_market = meta.groupby("market_id").size()
    assert per_market.max() <= config.NEIGHBORHOODS_PER_MARKET
    assert per_market.min() >= 1


def test_list_neighborhoods_filters_by_market() -> None:
    meta = market_data.load_neighborhood_meta()
    market_id = int(meta["market_id"].iloc[0])
    subset = market_data.list_neighborhoods(meta, market_id)
    assert set(subset["market_id"]) == {market_id}
    assert subset["size_rank"].is_monotonic_increasing


def test_neighborhood_history_matches_meta() -> None:
    meta = market_data.load_neighborhood_meta()
    history = market_data.load_neighborhood_history()
    assert set(history["neighborhood_id"]).issubset(set(meta["neighborhood_id"]))
    assert (history["value"] > 0).all()


def test_get_synthetic_anchors() -> None:
    anchors = market_data.get_synthetic_anchors()
    assert {"market", "neighborhood", "base_price_per_sqft"}.issubset(anchors.columns)
    assert anchors["base_price_per_sqft"].between(20, 2_000).all()


def test_load_market_rents() -> None:
    rents = market_data.load_market_rents()
    assert {"market_id", "month", "value"}.issubset(rents.columns)
    assert (rents["value"] > 0).all()
    assert rents["market_id"].nunique() == config.TOP_N_MARKETS


def test_merge_rents_adds_yield() -> None:
    markets = market_data.load_markets()
    summary = market_data.build_market_summary(markets, market_data.load_market_history())
    merged = market_data.merge_rents(summary, market_data.load_market_rents())

    assert {"latest_rent", "rent_yoy_pct", "gross_yield_pct"}.issubset(merged.columns)
    assert (merged["latest_rent"] > 0).all()
    assert merged["gross_yield_pct"].between(0.5, 20).all()
    # Yield is annual rent over home value.
    expected = merged["latest_rent"] * 12 / merged["latest_value"] * 100
    np.testing.assert_allclose(merged["gross_yield_pct"], expected)


def test_merge_rents_handles_empty() -> None:
    markets = market_data.load_markets()
    summary = market_data.build_market_summary(markets, market_data.load_market_history())
    merged = market_data.merge_rents(summary, pd.DataFrame(columns=["market_id", "month", "value"]))
    assert merged["gross_yield_pct"].isna().all()


def test_get_market_data_falls_back_to_snapshot(monkeypatch) -> None:
    """When live fetches fail, the committed snapshot must be used."""
    monkeypatch.setattr(market_data, "fetch_live_metro_history", lambda **_: None)
    monkeypatch.setattr(market_data, "fetch_live_metro_rents", lambda **_: None)
    data = market_data.get_market_data(force_refresh=True)

    assert data["source"] == "snapshot"
    assert not data["summary"].empty
    assert data["fetched_at"] is not None
    assert "gross_yield_pct" in data["summary"].columns
    assert (data["summary"]["gross_yield_pct"] > 0).all()


def test_offline_mode_uses_snapshot(monkeypatch) -> None:
    """``RPE_OFFLINE`` must bypass the network entirely."""
    monkeypatch.setattr(market_data.config, "OFFLINE", True)
    data = market_data.get_market_data(force_refresh=True)
    assert data["source"] == "snapshot"
    assert not data["summary"].empty


def test_merge_health_adds_metrics() -> None:
    markets = market_data.load_markets()
    summary = market_data.build_market_summary(markets, market_data.load_market_history())
    merged = market_data.merge_health(summary, market_data.load_market_health())

    for metric in market_data.HEALTH_METRICS:
        assert metric in merged.columns
    assert merged["inventory"].notna().all()
    assert (merged["days_to_pending"] > 0).all()
    assert (merged["median_sale_price"] > 0).all()


def test_merge_health_handles_empty() -> None:
    markets = market_data.load_markets()
    summary = market_data.build_market_summary(markets, market_data.load_market_history())
    merged = market_data.merge_health(summary, pd.DataFrame(columns=["market_id", "metric", "month", "value"]))
    for metric in market_data.HEALTH_METRICS:
        assert merged[metric].isna().all()


def _national_summary() -> pd.DataFrame:
    markets = market_data.load_markets()
    summary = market_data.build_market_summary(markets, market_data.load_market_history())
    summary = market_data.merge_rents(summary, market_data.load_market_rents())
    return market_data.merge_health(summary, market_data.load_market_health())


def test_build_national_summary() -> None:
    national = market_data.build_national_summary(_national_summary())
    assert national["markets"] == config.TOP_N_MARKETS
    assert national["latest_value"] > 0
    assert national["latest_rent"] > 0
    assert national["days_to_pending"] > 0
    assert national["inventory"] > 0


def test_build_national_history() -> None:
    national = market_data.build_national_history(market_data.load_market_history())
    assert {"month", "value"}.issubset(national.columns)
    assert (national["value"] > 0).all()
    assert national["month"].is_monotonic_increasing


def test_build_national_health() -> None:
    national = market_data.build_national_health(market_data.load_market_health())
    assert set(national["metric"]) == set(market_data.HEALTH_METRICS)
    assert (national["value"] > 0).all()
