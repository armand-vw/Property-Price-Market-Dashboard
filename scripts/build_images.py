"""
scripts/build_images.py
=======================
Generate the README hero image (``docs/hero.png``) from the committed market
snapshot. The same file is served by the GitHub Pages site for social previews.

Uses matplotlib (headless, no browser required) with the same light-corporate
palette as the dashboard. Reads only committed data (``market_data/`` and
``models/feature_importance.csv``), so it is deterministic and offline.

Usage
-----
    python scripts/build_images.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from property_insights import (  # noqa: E402
    config,  # noqa: E402
    market_data,  # noqa: E402
)

DOCS_DIR = PROJECT_ROOT / "docs"

INK = config.COLORS["ink"]
MUTED = config.COLORS["muted"]
PRIMARY = config.COLORS["primary"]
ACCENT = config.COLORS["accent"]
SUCCESS = config.COLORS["success"]
DANGER = config.COLORS["danger"]
BORDER = config.COLORS["border"]

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "text.color": INK,
        "axes.labelcolor": MUTED,
        "xtick.color": MUTED,
        "ytick.color": INK,
        "axes.edgecolor": BORDER,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


def _clean(ax) -> None:
    """Strip chart junk for a modern look."""
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(BORDER)
    ax.tick_params(length=0)
    ax.set_axisbelow(True)


def _snapshot() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the committed market summary, history and feature importances."""
    markets = market_data.load_markets()
    history = market_data.load_market_history()
    rents = market_data.load_market_rents()
    summary = market_data.merge_rents(
        market_data.build_market_summary(markets, history), rents
    )
    importance = pd.read_csv(config.IMPORTANCE_PATH)
    return summary, history, importance


def plot_market_values(summary: pd.DataFrame, ax) -> None:
    """Median home value by market."""
    frame = summary.sort_values("latest_value")
    ax.barh(frame["market"], frame["latest_value"], color=PRIMARY, height=0.7)
    for y, value in enumerate(frame["latest_value"]):
        ax.text(value, y, f"  ${value / 1e6:.2f}M", va="center", fontsize=9, color=INK)
    ax.set_title("Median Home Value by Market", fontsize=13, fontweight="bold", loc="left")
    ax.set_xlabel("Median home value ($)")
    ax.xaxis.set_major_formatter(lambda x, _: f"${x / 1e6:.1f}M")
    _clean(ax)


def plot_rental_yield(summary: pd.DataFrame, ax) -> None:
    """Gross rental yield by market."""
    frame = summary.dropna(subset=["gross_yield_pct"]).sort_values("gross_yield_pct")
    ax.barh(frame["market"], frame["gross_yield_pct"], color=ACCENT, height=0.7)
    for y, value in enumerate(frame["gross_yield_pct"]):
        ax.text(value, y, f"  {value:.1f}%", va="center", fontsize=9, color=INK)
    ax.set_title("Gross Rental Yield by Market", fontsize=13, fontweight="bold", loc="left")
    ax.set_xlabel("Annual rent ÷ home value (%)")
    _clean(ax)


def plot_market_growth(history: pd.DataFrame, summary: pd.DataFrame, ax) -> None:
    """Ten-year indexed home-value growth for the largest markets."""
    colors = [PRIMARY, ACCENT, "#14B8A6", "#F59E0B", "#EF4444", "#8B5CF6"]
    top = summary.sort_values("size_rank").head(6)
    for color, row in zip(colors, top.itertuples(), strict=False):
        series = history[history["market_id"] == row.market_id].sort_values("month")
        series = series[series["month"] >= series["month"].max() - pd.DateOffset(years=10)]
        if series.empty:
            continue
        indexed = series["value"] / series["value"].iloc[0] * 100.0
        ax.plot(series["month"], indexed, color=color, linewidth=2.2, label=row.market.split(",")[0])
    ax.set_title("10-Year Home Value Growth (indexed)", fontsize=13, fontweight="bold", loc="left")
    ax.set_ylabel("Index (=100)")
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper left")
    _clean(ax)


def plot_feature_importance(importance: pd.DataFrame, ax) -> None:
    """Model feature importances."""
    frame = importance.sort_values("importance")
    colors = [PRIMARY if f in {"market", "neighborhood"} else "#A5B4FC" for f in frame["feature"]]
    ax.barh(frame["feature"], frame["importance"], color=colors, height=0.7)
    for y, value in enumerate(frame["importance"]):
        ax.text(value, y, f"  {value:.3f}", va="center", fontsize=9, color=INK)
    ax.set_title("What Drives Home Prices", fontsize=13, fontweight="bold", loc="left")
    ax.set_xlabel("Relative importance")
    _clean(ax)


def make_hero(summary: pd.DataFrame, history: pd.DataFrame, importance: pd.DataFrame) -> None:
    """Compose a 2x2 hero montage for the README."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    plot_market_values(summary, axes[0, 0])
    plot_rental_yield(summary, axes[0, 1])
    plot_market_growth(history, summary, axes[1, 0])
    plot_feature_importance(importance, axes[1, 1])
    fig.suptitle(
        "Real Estate Price Estimator & Market Insights",
        fontsize=19,
        fontweight="bold",
        x=0.01,
        ha="left",
        y=0.99,
    )
    fig.text(
        0.01,
        0.955,
        "Live Zillow market data · XGBoost valuation · 15 US metros",
        fontsize=11,
        color=MUTED,
        ha="left",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(DOCS_DIR / "hero.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    summary, history, importance = _snapshot()

    make_hero(summary, history, importance)

    hero = DOCS_DIR / "hero.png"
    print("=" * 62)
    print("README hero image generated")
    print("=" * 62)
    print(f"  {hero.name:<26} {hero.stat().st_size / 1e3:7.1f} KB")


if __name__ == "__main__":
    main()
