"""Lightweight UI smoke tests using Streamlit's AppTest (offline)."""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

import international_data
import market_data

APP_PATH = str(Path(__file__).resolve().parents[1] / "app.py")


def _offline(monkeypatch) -> None:
    """Force every data layer onto the committed snapshot (no network)."""
    monkeypatch.setattr(market_data, "fetch_live_metro_history", lambda *a, **k: None)
    monkeypatch.setattr(market_data, "fetch_live_metro_rents", lambda *a, **k: None)
    monkeypatch.setattr(international_data, "fetch_live_bis", lambda *a, **k: None)


def _selectbox(at: AppTest, label: str):
    return next(s for s in at.selectbox if s.label == label)


def test_app_runs_and_estimates(monkeypatch) -> None:
    _offline(monkeypatch)
    at = AppTest.from_file(APP_PATH, default_timeout=300)
    at.run()
    assert not at.exception, [e.value for e in at.exception]

    labels = [s.label for s in at.selectbox]
    assert "Select a country" in labels
    assert "Select a market" in labels

    submit = [b for b in at.button if b.label == "Estimate Value"]
    assert submit, "Estimate button not found"
    submit[0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert "prediction" in at.session_state
    assert at.session_state["prediction"]["point"] > 0


def test_app_seeds_from_query_params(monkeypatch) -> None:
    _offline(monkeypatch)
    at = AppTest.from_file(APP_PATH, default_timeout=300)
    at.query_params["market"] = "Miami, FL"
    at.query_params["sqft"] = "3000"
    at.run()

    assert not at.exception, [e.value for e in at.exception]
    assert _selectbox(at, "Select a market").value == "Miami, FL"
    assert at.session_state["sqft_input"] == 3000


def test_app_switches_country(monkeypatch) -> None:
    _offline(monkeypatch)
    at = AppTest.from_file(APP_PATH, default_timeout=300)
    at.run()

    # United States -> United Kingdom (adds the UK Regions tab).
    _selectbox(at, "Select a country").select("GB").run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.tabs) == 3

    # United Kingdom -> China (no valuation tool, 2 tabs).
    _selectbox(at, "Select a country").select("CN").run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.tabs) == 2
    assert not any(b.label == "Estimate Value" for b in at.button)
