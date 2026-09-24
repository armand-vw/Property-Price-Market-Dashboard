"""Unit tests for the plain-English education layer."""

from __future__ import annotations

from property_insights import insights


def test_temperature_hot() -> None:
    result = insights.market_temperature(yoy=12.0, days_to_pending=15.0, inventory_yoy=-20.0)
    assert result["label"] == "Hot"
    assert result["score"] > 0
    assert result["basis"] == "composite"


def test_temperature_cool() -> None:
    result = insights.market_temperature(yoy=-12.0, days_to_pending=50.0, inventory_yoy=30.0)
    assert result["label"] == "Cool"
    assert result["score"] < 0


def test_temperature_warm_and_yoy_only() -> None:
    result = insights.market_temperature(0.5)
    assert result["label"] == "Warm"
    assert result["basis"] == "year-over-year"


def test_narrative_mentions_direction_and_read() -> None:
    narrative = insights.build_market_narrative(
        "Testland", yoy=5.0, change_5y=20.0, days_to_pending=20.0,
        temperature={"label": "Hot"},
    )
    assert "up 5.0%" in narrative
    assert "20 days" in narrative
    assert "hot market" in narrative.lower()


def test_narrative_without_data() -> None:
    assert "Select a market" in insights.build_market_narrative("Nowhere")


def test_glossary_covers_core_metrics() -> None:
    glossary = insights.glossary()
    for term in ("yoy", "yield", "inventory", "days_to_pending", "temperature"):
        assert term in glossary
