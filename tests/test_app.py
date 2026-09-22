"""Lightweight UI smoke tests using Streamlit's AppTest (offline)."""

from __future__ import annotations

from pathlib import Path

import market_data
from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parents[1] / "app.py")


def test_app_runs_and_estimates(monkeypatch) -> None:
    monkeypatch.setattr(market_data, "fetch_live_metro_history", lambda *a, **k: None)
    monkeypatch.setattr(market_data, "fetch_live_metro_rents", lambda *a, **k: None)

    at = AppTest.from_file(APP_PATH, default_timeout=300)
    at.run()
    assert not at.exception, [e.value for e in at.exception]

    labels = [s.label for s in at.selectbox]
    assert "Select a market" in labels

    # Submit the estimator form (Estimate Value) and expect a prediction.
    submit = [b for b in at.button if b.label == "Estimate Value"]
    assert submit, "Estimate button not found"
    submit[0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert "prediction" in at.session_state
    assert at.session_state["prediction"]["point"] > 0


def test_app_seeds_from_query_params(monkeypatch) -> None:
    monkeypatch.setattr(market_data, "fetch_live_metro_history", lambda *a, **k: None)
    monkeypatch.setattr(market_data, "fetch_live_metro_rents", lambda *a, **k: None)

    at = AppTest.from_file(APP_PATH, default_timeout=300)
    at.query_params["market"] = "Miami, FL"
    at.query_params["sqft"] = "3000"
    at.run()

    assert not at.exception, [e.value for e in at.exception]
    assert at.selectbox[0].value == "Miami, FL"
    assert at.session_state["sqft_input"] == 3000
