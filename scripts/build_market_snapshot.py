"""
scripts/build_market_snapshot.py
================================
Build the committed real-market snapshot from Zillow Research data.

Zillow's metro-level ZHVI file is small (~4 MB) and is fetched live by the app,
but the neighbourhood-level file is ~100 MB - too large to pull at runtime on the
free serverless tier. This script downloads it **once**, reduces it to the top
US metros and their most prominent neighbourhoods, and writes compact CSVs to
``market_data/`` that are committed to the repository.

Outputs
-------
market_data/markets.csv               # the selected metros
market_data/market_history.csv        # monthly metro home values
market_data/neighborhood_meta.csv     # neighbourhood metadata + price anchor
market_data/neighborhood_history.csv  # monthly neighbourhood home values

Usage
-----
    python scripts/build_market_snapshot.py            # reuse cached download
    python scripts/build_market_snapshot.py --redownload
"""

from __future__ import annotations

import argparse
import re
import sys
import urllib.request
from pathlib import Path

import pandas as pd

# Make the project root importable when run as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402

MONTH_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")
ID_COLUMNS = [
    "RegionID",
    "SizeRank",
    "RegionName",
    "City",
    "State",
    "Metro",
    "CountyName",
]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def metro_token(value: object) -> str | None:
    """Reduce a Zillow metro string to a comparable base token.

    ``"Phoenix-Mesa-Chandler, AZ"`` -> ``"phoenix"`` and ``"Phoenix, AZ"`` ->
    ``"phoenix"``, so metro- and neighbourhood-level files can be joined on a
    stable key despite their different naming conventions.
    """
    if not isinstance(value, str) or not value.strip():
        return None
    head = value.split(",")[0]
    return head.split("-")[0].strip().lower() or None


def download(url: str, destination: Path, redownload: bool = False) -> Path:
    """Download ``url`` to ``destination``, reusing the file if it exists."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not redownload:
        print(f"  using cached download: {destination.name}")
        return destination

    print(f"  downloading {url.split('/')[-1]} ...")
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=120) as response, open(
        destination, "wb"
    ) as handle:
        total = int(response.headers.get("Content-Length", 0))
        read = 0
        while True:
            chunk = response.read(1024 * 256)
            if not chunk:
                break
            handle.write(chunk)
            read += len(chunk)
            if total:
                pct = read / total * 100
                print(f"\r    {pct:5.1f}%  ({read / 1e6:,.1f} MB)", end="")
    print()
    return destination


def month_columns(columns: list[str]) -> list[str]:
    """Return the monthly value columns from a Zillow header, in order."""
    months = [column for column in columns if MONTH_PATTERN.match(column)]
    return sorted(months)


# --------------------------------------------------------------------------- #
# Metro level
# --------------------------------------------------------------------------- #
def select_markets(metro: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the top metros and their monthly history."""
    rows = metro[metro["RegionType"] == "msa"].sort_values("SizeRank")
    rows = rows.head(config.TOP_N_MARKETS).copy()
    rows["market_id"] = rows["RegionID"].astype(int)

    months = month_columns(list(metro.columns))[-config.MARKET_HISTORY_MONTHS :]
    history = rows.melt(
        id_vars=["market_id"],
        value_vars=months,
        var_name="month",
        value_name="value",
    ).dropna(subset=["value"])

    markets = rows[
        ["market_id", "RegionName", "StateName", "SizeRank"]
    ].rename(columns={"RegionName": "market", "StateName": "state", "SizeRank": "size_rank"})

    # Latest value + month per market.
    latest = (
        history.sort_values("month")
        .groupby("market_id")
        .tail(1)
        .rename(columns={"value": "latest_value", "month": "latest_month"})
    )
    markets = markets.merge(
        latest[["market_id", "latest_value", "latest_month"]], on="market_id", how="left"
    )
    markets = markets.sort_values("size_rank").reset_index(drop=True)
    return markets, history.sort_values(["market_id", "month"]).reset_index(drop=True)


def select_rents(zori: pd.DataFrame, markets: pd.DataFrame) -> pd.DataFrame:
    """Reduce the metro ZORI (rent) file to the chosen markets (long format)."""
    market_ids = set(markets["market_id"].tolist())
    months = month_columns(list(zori.columns))[-config.RENT_HISTORY_MONTHS :]

    ours = zori[zori["RegionID"].astype(int).isin(market_ids)].copy()
    ours["market_id"] = ours["RegionID"].astype(int)
    long = ours.melt(
        id_vars=["market_id"],
        value_vars=months,
        var_name="month",
        value_name="value",
    ).dropna(subset=["value"])
    return long.sort_values(["market_id", "month"]).reset_index(drop=True)


def select_neighborhoods(
    raw_path: Path,
    markets: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stream the large neighbourhood file and reduce it to the chosen markets."""
    token_to_market = {
        metro_token(row.market): int(row.market_id) for row in markets.itertuples()
    }

    # --- Pass 1: which neighbourhoods belong to our metros? --------------- #
    print("  pass 1/2: scanning neighbourhood metadata ...")
    candidates: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        raw_path,
        usecols=["RegionID", "SizeRank", "RegionName", "City", "Metro", "CountyName"],
        chunksize=20_000,
        low_memory=False,
    ):
        chunk["market_id"] = chunk["Metro"].map(
            lambda value: token_to_market.get(metro_token(value))
        )
        matched = chunk.dropna(subset=["market_id"])
        if not matched.empty:
            matched = matched.copy()
            matched["market_id"] = matched["market_id"].astype(int)
            candidates.append(matched)

    if not candidates:
        raise RuntimeError("No neighbourhoods matched the selected metros.")

    all_candidates = pd.concat(candidates, ignore_index=True)
    selected = (
        all_candidates.sort_values("SizeRank")
        .groupby("market_id", group_keys=False)
        .head(config.NEIGHBORHOODS_PER_MARKET)
        .copy()
    )
    selected["neighborhood_id"] = selected["RegionID"].astype(int)
    kept_ids = set(selected["neighborhood_id"].tolist())
    print(
        f"    matched {len(all_candidates):,} neighbourhoods, "
        f"keeping {len(kept_ids):,} across {selected['market_id'].nunique()} markets"
    )

    # --- Pass 2: pull only the recent history for the kept neighbourhoods - #
    print("  pass 2/2: extracting neighbourhood history ...")
    header = pd.read_csv(raw_path, nrows=0)
    months = month_columns(list(header.columns))[-config.NEIGHBORHOOD_HISTORY_MONTHS :]
    usecols = ["RegionID"] + months

    history_frames: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        raw_path,
        usecols=usecols,
        chunksize=20_000,
        low_memory=False,
    ):
        chunk["neighborhood_id"] = chunk["RegionID"].astype(int)
        matched = chunk[chunk["neighborhood_id"].isin(kept_ids)]
        if matched.empty:
            continue
        long = matched.melt(
            id_vars=["neighborhood_id"],
            value_vars=months,
            var_name="month",
            value_name="value",
        ).dropna(subset=["value"])
        history_frames.append(long[["neighborhood_id", "month", "value"]])

    history = pd.concat(history_frames, ignore_index=True)
    history = history.sort_values(["neighborhood_id", "month"]).reset_index(drop=True)

    # --- Neighbourhood metadata + synthetic price anchor ------------------ #
    meta = selected[
        ["neighborhood_id", "market_id", "RegionName", "City", "CountyName", "SizeRank"]
    ].rename(
        columns={
            "RegionName": "neighborhood",
            "City": "city",
            "CountyName": "county",
            "SizeRank": "size_rank",
        }
    )
    latest_values = (
        history.sort_values("month")
        .groupby("neighborhood_id")
        .tail(1)
        .rename(columns={"value": "latest_value", "month": "latest_month"})
    )
    meta = meta.merge(
        latest_values[["neighborhood_id", "latest_value", "latest_month"]],
        on="neighborhood_id",
        how="left",
    )
    meta["base_price_per_sqft"] = meta["latest_value"] / config.ASSUMED_MEDIAN_SQFT
    meta = meta.sort_values(["market_id", "size_rank"]).reset_index(drop=True)
    return meta, history


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Zillow market snapshot.")
    parser.add_argument(
        "--redownload",
        action="store_true",
        help="Ignore cached raw downloads and fetch again.",
    )
    args = parser.parse_args()

    config.MARKET_DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache_dir = config.MARKET_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 62)
    print("Building Zillow market snapshot")
    print("=" * 62)

    print("[1/4] Metro home values")
    metro_path = download(config.ZILLOW_METRO_ZHVI_URL, cache_dir / "metro_zhvi.csv", args.redownload)
    metro = pd.read_csv(metro_path, low_memory=False)
    markets, market_history = select_markets(metro)
    markets.to_csv(config.MARKETS_PATH, index=False)
    market_history.to_csv(config.MARKET_HISTORY_PATH, index=False)
    print(f"  selected {len(markets)} metros: {', '.join(markets['market'].head(5))}, ...")

    print("[2/4] Metro rents")
    rents_path = download(config.ZILLOW_METRO_ZORI_URL, cache_dir / "metro_zori.csv", args.redownload)
    rents = select_rents(pd.read_csv(rents_path, low_memory=False), markets)
    rents.to_csv(config.MARKET_RENTS_PATH, index=False)

    print("[3/4] Neighbourhood home values")
    hood_path = download(
        config.ZILLOW_NEIGHBORHOOD_ZHVI_URL,
        cache_dir / "neighborhood_zhvi.csv",
        args.redownload,
    )
    hood_meta, hood_history = select_neighborhoods(hood_path, markets)
    hood_meta.to_csv(config.NEIGHBORHOOD_META_PATH, index=False)
    hood_history.to_csv(config.NEIGHBORHOOD_HISTORY_PATH, index=False)

    print("[4/4] Done")
    print("-" * 62)
    print(f"Markets              : {len(markets)}")
    print(f"Market history rows  : {len(market_history):,}")
    print(f"Rent history rows    : {len(rents):,}")
    print(f"Neighborhoods        : {len(hood_meta)}")
    print(f"Hood history rows    : {len(hood_history):,}")
    print(f"Snapshot directory   : {config.MARKET_DATA_DIR}")
    for path in (
        config.MARKETS_PATH,
        config.MARKET_HISTORY_PATH,
        config.MARKET_RENTS_PATH,
        config.NEIGHBORHOOD_META_PATH,
        config.NEIGHBORHOOD_HISTORY_PATH,
    ):
        print(f"  {path.name:<28} {path.stat().st_size / 1e3:8.1f} KB")


if __name__ == "__main__":
    main()
