"""
market_data.py
==============
Access to real US housing-market data from Zillow Research.

Two layers
----------
1. **Live metro layer** - the metro-level ZHVI file (~4 MB) is fetched at runtime
   from Zillow's public research CDN, cached on disk for
   :data:`config.MARKET_CACHE_TTL_HOURS`, and used to produce current median
   home values and trends. If the network is unavailable the committed snapshot
   is used instead, so the app never breaks.
2. **Snapshot neighbourhood layer** - the neighbourhood file (~100 MB) is too
   large to fetch at runtime, so ``scripts/build_market_snapshot.py`` reduces it
   to the chosen metros and commits compact CSVs under ``market_data/``.

All Zillow data is monthly aggregate home values (ZHVI), not listing-level data.
"""

from __future__ import annotations

import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

import config

#: Human-readable provenance of the most recent market data load.
LAST_SOURCE: str = "unknown"
#: UTC timestamp of the most recent market data load.
LAST_FETCHED: datetime | None = None


# --------------------------------------------------------------------------- #
# Snapshot loaders (committed, network-free)
# --------------------------------------------------------------------------- #
def load_markets() -> pd.DataFrame:
    """Load the committed market metadata (one row per metro)."""
    markets = pd.read_csv(config.MARKETS_PATH)
    markets["market_id"] = markets["market_id"].astype(int)
    return markets.sort_values("size_rank").reset_index(drop=True)


def load_market_history() -> pd.DataFrame:
    """Load committed monthly metro home values (long format)."""
    history = pd.read_csv(config.MARKET_HISTORY_PATH, parse_dates=["month"])
    history["market_id"] = history["market_id"].astype(int)
    return history


def load_neighborhood_meta() -> pd.DataFrame:
    """Load committed neighbourhood metadata and price anchors."""
    meta = pd.read_csv(config.NEIGHBORHOOD_META_PATH)
    for column in ("market_id", "neighborhood_id"):
        meta[column] = meta[column].astype(int)
    return meta


def load_neighborhood_history() -> pd.DataFrame:
    """Load committed monthly neighbourhood home values (long format)."""
    history = pd.read_csv(config.NEIGHBORHOOD_HISTORY_PATH, parse_dates=["month"])
    history["neighborhood_id"] = history["neighborhood_id"].astype(int)
    return history


def load_market_rents() -> pd.DataFrame:
    """Load committed monthly metro rents (ZORI, long format)."""
    rents = pd.read_csv(config.MARKET_RENTS_PATH, parse_dates=["month"])
    rents["market_id"] = rents["market_id"].astype(int)
    return rents


# --------------------------------------------------------------------------- #
# Live metro fetch (with cache + fallback)
# --------------------------------------------------------------------------- #
def _download_cached(url: str, filename: str, force_refresh: bool = False) -> Path | None:
    """Download ``url`` to the cache dir, honouring the TTL. ``None`` on failure."""
    config.MARKET_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    destination = config.MARKET_CACHE_DIR / filename

    fresh = (
        destination.exists()
        and (time.time() - destination.stat().st_mtime)
        < config.MARKET_CACHE_TTL_HOURS * 3600
    )
    if fresh and not force_refresh:
        return destination

    try:
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(
            request, timeout=config.MARKET_FETCH_TIMEOUT
        ) as response:
            payload = response.read()
        destination.write_bytes(payload)
        return destination
    except (urllib.error.URLError, TimeoutError, OSError):
        # Keep a stale cache if we have one; otherwise fall back to the snapshot.
        return destination if destination.exists() else None


def _parse_zillow_long(
    raw: pd.DataFrame,
    months_kept: int,
) -> pd.DataFrame:
    """Melt a Zillow wide file (metro-level) into long ``market_id, month, value``."""
    month_columns = sorted(
        column for column in raw.columns if len(column) == 10 and column[4] == "-"
    )[-months_kept:]
    if not month_columns:
        return pd.DataFrame(columns=["market_id", "month", "value"])

    markets = raw[raw["RegionType"] == "msa"].sort_values("SizeRank")
    markets = markets.head(config.TOP_N_MARKETS)
    long = markets.melt(
        id_vars=["RegionID"],
        value_vars=month_columns,
        var_name="month",
        value_name="value",
    ).dropna(subset=["value"])
    long = long.rename(columns={"RegionID": "market_id"})
    long["market_id"] = long["market_id"].astype(int)
    long["month"] = pd.to_datetime(long["month"])
    return long.sort_values(["market_id", "month"]).reset_index(drop=True)


def fetch_live_metro_history(force_refresh: bool = False) -> pd.DataFrame | None:
    """Fetch and parse live metro-level ZHVI into long format.

    Returns ``None`` if the network is unavailable, signalling the caller to use
    the committed snapshot.
    """
    raw_path = _download_cached(
        config.ZILLOW_METRO_ZHVI_URL, "metro_zhvi_live.csv", force_refresh
    )
    if raw_path is None:
        return None
    try:
        raw = pd.read_csv(raw_path, low_memory=False)
    except (OSError, pd.errors.ParserError):
        return None
    return _parse_zillow_long(raw, config.MARKET_HISTORY_MONTHS)


def fetch_live_metro_rents(force_refresh: bool = False) -> pd.DataFrame | None:
    """Fetch and parse live metro-level ZORI rents into long format.

    Returns ``None`` if the network is unavailable, signalling the caller to use
    the committed snapshot.
    """
    raw_path = _download_cached(
        config.ZILLOW_METRO_ZORI_URL, "metro_zori_live.csv", force_refresh
    )
    if raw_path is None:
        return None
    try:
        raw = pd.read_csv(raw_path, low_memory=False)
    except (OSError, pd.errors.ParserError):
        return None
    return _parse_zillow_long(raw, config.RENT_HISTORY_MONTHS)


# --------------------------------------------------------------------------- #
# Market overview
# --------------------------------------------------------------------------- #
def build_market_summary(
    markets: pd.DataFrame, history: pd.DataFrame
) -> pd.DataFrame:
    """Combine market metadata with the latest values and trend changes.

    Pure function (no I/O) so it can be unit-tested with the committed snapshot.
    Adds ``latest_value``, ``latest_month``, ``mom_pct``, ``yoy_pct``,
    ``change_5y_pct`` and ``value_1y_ago``.
    """
    summary_rows: list[dict] = []
    history = history.sort_values("month")

    for row in markets.itertuples():
        group = history[history["market_id"] == row.market_id]
        if group.empty:
            continue

        latest = group.iloc[-1]
        prior = group.iloc[-2] if len(group) > 1 else latest
        one_year_ago = group[group["month"] <= latest["month"] - pd.DateOffset(months=12)]
        five_years_ago = group[group["month"] <= latest["month"] - pd.DateOffset(months=60)]

        def _pct(new: float, old: float) -> float:
            return (new - old) / old * 100.0 if old else 0.0

        summary_rows.append(
            {
                "market_id": int(row.market_id),
                "market": row.market,
                "state": row.state,
                "size_rank": int(row.size_rank),
                "latest_value": float(latest["value"]),
                "latest_month": latest["month"],
                "mom_pct": _pct(latest["value"], prior["value"]),
                "yoy_pct": _pct(
                    latest["value"],
                    one_year_ago.iloc[-1]["value"] if not one_year_ago.empty else latest["value"],
                ),
                "change_5y_pct": _pct(
                    latest["value"],
                    five_years_ago.iloc[-1]["value"] if not five_years_ago.empty else latest["value"],
                ),
                "value_1y_ago": float(one_year_ago.iloc[-1]["value"])
                if not one_year_ago.empty
                else float(latest["value"]),
            }
        )

    return pd.DataFrame(summary_rows).sort_values("size_rank").reset_index(drop=True)


def merge_rents(summary: pd.DataFrame, rents: pd.DataFrame) -> pd.DataFrame:
    """Add ``latest_rent``, ``rent_yoy_pct`` and ``gross_yield_pct`` to a summary.

    Pure function (no I/O). ``gross_yield_pct`` is annual rent divided by the
    market's median home value.
    """
    if rents.empty:
        summary["latest_rent"] = float("nan")
        summary["rent_yoy_pct"] = float("nan")
        summary["gross_yield_pct"] = float("nan")
        return summary

    rents = rents.sort_values("month")
    rows: list[dict] = []
    for row in summary.itertuples():
        group = rents[rents["market_id"] == row.market_id]
        if group.empty:
            rows.append({"market_id": row.market_id, "latest_rent": float("nan"),
                         "rent_yoy_pct": float("nan")})
            continue
        latest = group.iloc[-1]
        one_year_ago = group[group["month"] <= latest["month"] - pd.DateOffset(months=12)]
        base = float(one_year_ago.iloc[-1]["value"]) if not one_year_ago.empty else float(latest["value"])
        rows.append(
            {
                "market_id": row.market_id,
                "latest_rent": float(latest["value"]),
                "rent_yoy_pct": (float(latest["value"]) - base) / base * 100.0 if base else 0.0,
            }
        )

    merged = summary.merge(pd.DataFrame(rows), on="market_id", how="left")
    merged["gross_yield_pct"] = merged["latest_rent"] * 12.0 / merged["latest_value"] * 100.0
    return merged


def get_market_data(force_refresh: bool = False) -> dict:
    """Return the current market overview, rents and history.

    Tries the live Zillow fetch first and transparently falls back to the
    committed snapshot. Returns a dict with ``summary`` (one row per market,
    including rent and gross yield), ``history``, ``rents``, ``source``
    (``"live"``/``"snapshot"``) and ``fetched_at``.
    """
    global LAST_SOURCE, LAST_FETCHED

    markets = load_markets()

    live = fetch_live_metro_history(force_refresh=force_refresh)
    if live is not None and not live.empty:
        history, source = live, "live"
    else:
        history, source = load_market_history(), "snapshot"

    live_rents = fetch_live_metro_rents(force_refresh=force_refresh)
    if live_rents is not None and not live_rents.empty:
        rents = live_rents
    else:
        rents = load_market_rents()

    summary = merge_rents(build_market_summary(markets, history), rents)
    LAST_SOURCE = source
    LAST_FETCHED = datetime.now(UTC)
    return {
        "summary": summary,
        "history": history,
        "rents": rents,
        "source": source,
        "fetched_at": LAST_FETCHED,
    }


# --------------------------------------------------------------------------- #
# Neighbourhood helpers
# --------------------------------------------------------------------------- #
def list_neighborhoods(meta: pd.DataFrame, market_id: int) -> pd.DataFrame:
    """Return the neighbourhoods belonging to a market, most prominent first."""
    subset = meta[meta["market_id"] == market_id].copy()
    return subset.sort_values("size_rank").reset_index(drop=True)


def neighborhood_market_lookup(
    meta: pd.DataFrame, markets: pd.DataFrame
) -> pd.DataFrame:
    """Join neighbourhood metadata to its market display name."""
    names = markets[["market_id", "market", "state"]]
    return meta.merge(names, on="market_id", how="left")


# --------------------------------------------------------------------------- #
# Synthetic anchors
# --------------------------------------------------------------------------- #
def get_synthetic_anchors() -> pd.DataFrame:
    """Neighbourhood anchors used to synthesise listings.

    Returns one row per neighbourhood with its real market/neighbourhood identity
    and a price-per-square-foot anchor derived from the real median home value.
    """
    meta = load_neighborhood_meta()
    markets = load_markets()
    return neighborhood_market_lookup(meta, markets)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    """Print a short market overview (live if available, snapshot otherwise)."""
    data = get_market_data(force_refresh=True)
    summary = data["summary"]
    print("=" * 74)
    print(f"US market overview  (source: {data['source']})")
    print("=" * 74)
    for row in summary.itertuples():
        print(
            f"{row.market:<20} ${row.latest_value:>11,.0f}   "
            f"MoM {row.mom_pct:+5.2f}%   YoY {row.yoy_pct:+6.2f}%   "
            f"5y {row.change_5y_pct:+6.1f}%   "
            f"Rent ${row.latest_rent:>8,.0f}   Yield {row.gross_yield_pct:4.1f}%"
        )


if __name__ == "__main__":
    main()
