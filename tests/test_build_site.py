"""Tests for the static GitHub Pages site builders (Plotly dropdowns)."""

from __future__ import annotations

import importlib.util
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


def test_market_explorer_has_dropdown() -> None:
    build_site = _load_build_site()
    html = build_site.build_market_explorer(
        market_data.load_market_history(), market_data.load_markets()
    )
    assert "updatemenus" in html
    # One button per market plus the "All markets" option.
    assert html.count('"label"') == config.TOP_N_MARKETS + 1


def test_country_explorer_has_dropdown() -> None:
    build_site = _load_build_site()
    html = build_site.build_country_explorer(international_data.load_bis_index())
    assert "updatemenus" in html
    assert html.count('"label"') == len(config.COUNTRY_ORDER) + 1
