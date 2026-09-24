"""
property_insights
=================
Core package for the Real Estate Price Estimator & Market Insights Dashboard.

Modules
-------
* :mod:`config`               - paths, schema, seeds, palette, constants.
* :mod:`data_loader`          - synthetic, market-anchored listings + cleaning.
* :mod:`market_data`          - US Zillow data (fetch, cache, fallback).
* :mod:`international_data`   - BIS + HM Land Registry data.
* :mod:`model`                - leak-free XGBoost pipeline + persistence.
* :mod:`insights`             - plain-English education layer.
* :mod:`real_data`            - real-data (Ames) benchmark helpers.

The Streamlit entrypoint lives at the repository root (``app.py``) and imports
from this package.
"""

__all__ = [
    "config",
    "data_loader",
    "insights",
    "international_data",
    "market_data",
    "model",
    "real_data",
]
