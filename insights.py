"""
insights.py
===========
Plain-English "education" layer that turns the raw analytics into something
everyday users can understand.

* :func:`market_temperature` - a Hot / Warm / Cool read from year-over-year
  change, selling speed and inventory direction.
* :func:`build_market_narrative` - a short sentence summarising the selected
  market and what it means for buyers or sellers.
* :func:`glossary` - one-line definitions for every metric in the dashboard.

Pure functions only (no I/O, no Streamlit), so both the Streamlit app and the
static GitHub Pages site share exactly the same logic.
"""

from __future__ import annotations

import math

import config


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _is_num(value) -> bool:
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def market_temperature(
    yoy: float | None,
    days_to_pending: float | None = None,
    inventory_yoy: float | None = None,
) -> dict:
    """Return a Hot / Warm / Cool market read.

    Components (each in ``[-1, +1]``):
    * year-over-year price change (stronger growth = hotter),
    * days to pending (faster sales = hotter) - US only,
    * inventory direction (falling inventory = hotter) - US only.

    Countries without market-health data fall back to a YoY-only rating, which
    is documented in the UI.
    """
    components: list[float] = []
    if _is_num(yoy):
        components.append(_clip(float(yoy) / 10.0, -1.0, 1.0))
    if _is_num(days_to_pending):
        components.append(_clip((30.0 - float(days_to_pending)) / 20.0, -1.0, 1.0))
    if _is_num(inventory_yoy):
        components.append(_clip(-float(inventory_yoy) / 20.0, -1.0, 1.0))

    score = sum(components) / len(components) if components else 0.0
    if score >= config.TEMPERATURE_HOT:
        label = "Hot"
    elif score <= config.TEMPERATURE_COOL:
        label = "Cool"
    else:
        label = "Warm"

    return {
        "label": label,
        "score": round(score, 3),
        "basis": "composite" if len(components) > 1 else "year-over-year",
    }


def build_market_narrative(
    place: str,
    yoy: float | None = None,
    change_5y: float | None = None,
    days_to_pending: float | None = None,
    temperature: dict | None = None,
) -> str:
    """Return a short, plain-English summary of a market for everyday users."""
    sentences: list[str] = []

    if _is_num(yoy):
        direction = "up" if float(yoy) >= 0 else "down"
        sentence = f"Home values in {place} are {direction} {abs(float(yoy)):.1f}% over the past year"
        if _is_num(change_5y):
            five = "higher" if float(change_5y) >= 0 else "lower"
            sentence += f" ({abs(float(change_5y)):.0f}% {five} than five years ago)"
        if _is_num(days_to_pending):
            sentence += f", and homes typically go under contract in {float(days_to_pending):.0f} days"
        sentences.append(sentence + ".")

    if temperature and temperature.get("label"):
        label = temperature["label"]
        if label == "Hot":
            read = "Sellers have the upper hand — expect competition and quick decisions."
        elif label == "Cool":
            read = "Buyers have more room to negotiate and more time to decide."
        else:
            read = "Conditions look fairly balanced between buyers and sellers."
        sentences.append(f"That points to a {label.lower()} market. {read}")

    if not sentences:
        return "Select a market to see a plain-English summary here."
    return " ".join(sentences)


def glossary() -> dict[str, str]:
    """Return the plain-English glossary of dashboard metrics."""
    return dict(config.GLOSSARY)
