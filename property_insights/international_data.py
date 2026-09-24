"""
international_data.py
=====================
Access to international housing-market data for the country switcher.

Two sources
-----------
* **BIS** *Selected Residential Property Prices* - a free bulk CSV covering six
  countries (nominal index ``2010 = 100`` + year-on-year change, quarterly,
  national). Small enough to fetch live with a 24h cache and a committed
  snapshot fallback.
* **HM Land Registry** *UK House Price Index* - monthly average price (GBP), HPI
  and annual change for the four UK nations, served from the committed snapshot
  (rebuilt monthly by the refresh workflow).

All values are national and index-based except UK prices, which are GBP.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from . import config

BIS_UNIT_INDEX = "628"
BIS_UNIT_YOY = "771"
UKHPI_BASE = "http://landregistry.data.gov.uk/data/ukhpi/region/{slug}.json"
QUARTER_PATTERN = re.compile(r"^\d{4}-Q[1-4]$")

#: Human-readable provenance of the most recent international load.
LAST_SOURCE: str = "unknown"
LAST_FETCHED: datetime | None = None


# --------------------------------------------------------------------------- #
# Snapshot loaders
# --------------------------------------------------------------------------- #
def load_bis_index() -> pd.DataFrame:
    """Load the committed BIS index snapshot."""
    frame = pd.read_csv(config.BIS_INDEX_PATH)
    frame["period_date"] = pd.PeriodIndex(frame["period"], freq="Q").to_timestamp()
    return frame.sort_values(["country_code", "period"]).reset_index(drop=True)


def load_uk_regions() -> pd.DataFrame:
    """Load the committed UK region snapshot."""
    frame = pd.read_csv(config.UK_REGIONS_PATH)
    frame["month_date"] = pd.to_datetime(frame["month"], format="%Y-%m", errors="coerce")
    return frame.sort_values(["region", "month"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# BIS parsing (shared with the snapshot builder)
# --------------------------------------------------------------------------- #
def parse_bis_zip(zip_path: Path) -> pd.DataFrame:
    """Parse the BIS bulk archive into a tidy international index frame."""
    raw = pd.read_csv(zip_path, compression="zip", low_memory=False)
    quarter_cols = [
        column for column in raw.columns if QUARTER_PATTERN.match(str(column))
    ][-config.INTERNATIONAL_HISTORY_QUARTERS :]
    wanted = set(config.COUNTRIES)

    def to_long(unit: str, name: str) -> pd.DataFrame:
        subset = raw[
            (raw["UNIT_MEASURE"].astype(str) == unit)
            & (raw["Series"].astype(str).str.contains(":N:"))
            & (raw["REF_AREA"].isin(wanted))
        ]
        long = subset.melt(
            id_vars=["REF_AREA", "Reference area"],
            value_vars=quarter_cols,
            var_name="period",
            value_name=name,
        ).dropna(subset=[name])
        return long.rename(
            columns={"REF_AREA": "country_code", "Reference area": "country"}
        )

    index = to_long(BIS_UNIT_INDEX, "index")
    yoy = to_long(BIS_UNIT_YOY, "yoy_pct")
    merged = index.merge(
        yoy[["country_code", "period", "yoy_pct"]],
        on=["country_code", "period"],
        how="left",
    )
    merged["period_date"] = pd.PeriodIndex(merged["period"], freq="Q").to_timestamp()
    return (
        merged[["country_code", "country", "period", "period_date", "index", "yoy_pct"]]
        .sort_values(["country_code", "period"])
        .reset_index(drop=True)
    )


# --------------------------------------------------------------------------- #
# Live BIS fetch (cache + fallback)
# --------------------------------------------------------------------------- #
def _download_cached(url: str, filename: str, force_refresh: bool = False) -> Path | None:
    """Download ``url`` to the cache dir with a TTL. ``None`` on failure."""
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
            destination.write_bytes(response.read())
        return destination
    except (urllib.error.URLError, TimeoutError, OSError):
        return destination if destination.exists() else None


def fetch_live_bis(force_refresh: bool = False) -> pd.DataFrame | None:
    """Fetch and parse live BIS data. ``None`` when offline or unreachable."""
    if config.OFFLINE:
        return None
    path = _download_cached(config.BIS_SPP_URL, "bis_spp_live.zip", force_refresh)
    if path is None:
        return None
    try:
        return parse_bis_zip(path)
    except (OSError, ValueError, pd.errors.ParserError):
        return None


def fetch_uk_regions(max_months: int = config.UK_HISTORY_MONTHS) -> pd.DataFrame:
    """Fetch the latest monthly UKHPI observations for the four UK nations."""
    rows: list[dict] = []
    for slug, name in config.UK_NATIONS.items():
        url = f"{UKHPI_BASE.format(slug=slug)}?_view=all&_pageSize=200"
        collected = 0
        while url and collected < max_months:
            request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(request, timeout=60) as response:
                result = json.loads(response.read().decode("utf-8"))["result"]
            for item in result.get("items", []):
                if not isinstance(item, dict):
                    continue
                rows.append(
                    {
                        "region": name,
                        "region_slug": slug,
                        "month": item.get("refMonth"),
                        "avg_price_gbp": item.get("averagePrice"),
                        "hpi": item.get("housePriceIndex"),
                        "yoy_pct": item.get("percentageAnnualChange"),
                    }
                )
                collected += 1
                if collected >= max_months:
                    break
            url = result.get("next")
            if not result.get("items"):
                break

    frame = pd.DataFrame(rows).dropna(subset=["month"])
    frame["month_date"] = pd.to_datetime(frame["month"], format="%Y-%m", errors="coerce")
    return frame.sort_values(["region", "month"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Aggregates
# --------------------------------------------------------------------------- #
def build_country_summary(bis: pd.DataFrame) -> pd.DataFrame:
    """Latest index, YoY and 5-year change per country (pure function)."""
    rows: list[dict] = []
    for code in config.COUNTRY_ORDER:
        group = bis[bis["country_code"] == code].sort_values("period")
        if group.empty:
            continue
        latest = group.iloc[-1]
        five_years_ago = group[group["period"] <= latest["period"]].iloc[:-20]
        base = five_years_ago.iloc[-1] if not five_years_ago.empty else group.iloc[0]
        rows.append(
            {
                "country_code": code,
                "country": config.COUNTRIES[code]["name"],
                "period": latest["period"],
                "index": float(latest["index"]),
                "yoy_pct": float(latest["yoy_pct"]) if pd.notna(latest["yoy_pct"]) else float("nan"),
                "change_5y_pct": (
                    (float(latest["index"]) - float(base["index"])) / float(base["index"]) * 100.0
                    if float(base["index"])
                    else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


def build_comparison(bis: pd.DataFrame, quarters: int = config.INTERNATIONAL_HISTORY_QUARTERS) -> pd.DataFrame:
    """Index each country to 100 at the start of a common window (pure function).

    Uses the curated :data:`config.COMPARISON_COUNTRIES` so the chart stays
    readable; every country remains available in the switcher.
    """
    frames: list[pd.DataFrame] = []
    for code in config.COMPARISON_COUNTRIES:
        group = bis[bis["country_code"] == code].sort_values("period").tail(quarters).copy()
        if group.empty:
            continue
        first = float(group["index"].iloc[0])
        group["indexed"] = group["index"] / first * 100.0
        group["country"] = config.COUNTRIES[code]["name"]
        frames.append(group)
    if not frames:
        return pd.DataFrame(columns=["country_code", "country", "period_date", "indexed"])
    return pd.concat(frames, ignore_index=True)


def build_uk_summary(uk: pd.DataFrame) -> pd.DataFrame:
    """Latest price, HPI and annual change per UK nation (pure function)."""
    if uk.empty:
        return uk
    latest = uk.sort_values("month").groupby("region", as_index=False).tail(1)
    return latest[
        ["region", "month", "avg_price_gbp", "hpi", "yoy_pct"]
    ].sort_values("avg_price_gbp", ascending=False).reset_index(drop=True)


def get_country_data(force_refresh: bool = False) -> dict:
    """Return the international overview, comparison and UK regions.

    Uses the live BIS fetch when possible and the committed snapshot otherwise.
    """
    global LAST_SOURCE, LAST_FETCHED

    live = fetch_live_bis(force_refresh=force_refresh)
    if live is not None and not live.empty:
        bis, source = live, "live"
    else:
        bis, source = load_bis_index(), "snapshot"

    uk = load_uk_regions()
    LAST_SOURCE = source
    LAST_FETCHED = datetime.now(UTC)
    return {
        "bis": bis,
        "summary": build_country_summary(bis),
        "comparison": build_comparison(bis),
        "uk_regions": uk,
        "uk_summary": build_uk_summary(uk),
        "source": source,
        "fetched_at": LAST_FETCHED,
    }


def country_name(code: str) -> str:
    """Human-readable country name for a code."""
    return config.COUNTRIES.get(code, {}).get("name", code)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    """Print a short international overview."""
    data = get_country_data(force_refresh=True)
    print("=" * 70)
    print(f"International overview  (source: {data['source']})")
    print("=" * 70)
    for row in data["summary"].itertuples():
        print(
            f"{row.country:<16} index {row.index:7.2f}   "
            f"YoY {row.yoy_pct:+6.2f}%   5y {row.change_5y_pct:+6.1f}%"
        )
    print("-" * 70)
    print("UK nations (latest):")
    for row in data["uk_summary"].itertuples():
        print(f"  {row.region:<18} £{row.avg_price_gbp:>9,.0f}   YoY {row.yoy_pct:+.1f}%")


if __name__ == "__main__":
    main()
