"""Tests for the static GitHub Pages site builders (Explorer sidebar)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import config
import international_data
import market_data

BUILD_SITE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_site.py"


def _load_build_site():
    spec = importlib.util.spec_from_file_location("build_site", BUILD_SITE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sidebar_payload_has_countries_and_markets() -> None:
    build_site = _load_build_site()

    summary = market_data.build_market_summary(
        market_data.load_markets(), market_data.load_market_history()
    )
    history = market_data.load_market_history()
    bis = international_data.load_bis_index()
    country_summary = international_data.build_country_summary(bis)

    payload = build_site._build_sidebar_payload(summary, history, bis, country_summary)
    data = json.loads(payload["json"])

    assert len(data["countries"]) == len(config.COUNTRY_ORDER)
    assert len(data["markets"]) == config.TOP_N_MARKETS
    assert data["countries"][0]["code"] == "US"
    assert all(item["history"] for item in data["countries"])
    assert all(item["history"] for item in data["markets"])

    # Selector options rendered server-side.
    assert 'value="US"' in payload["country_options"]
    assert payload["country_options"].count("<option") == len(config.COUNTRY_ORDER)
    assert payload["market_options"].count("<option") == config.TOP_N_MARKETS
