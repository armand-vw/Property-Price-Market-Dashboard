"""
scripts/build_international_snapshot.py
=======================================
Build the committed international-market snapshot.

Sources
-------
* **BIS** *Selected Residential Property Prices* - a free bulk CSV covering the
  six supported countries (nominal index ``2010 = 100`` and year-on-year change,
  quarterly, national).
* **HM Land Registry** *UK House Price Index* - monthly average price (GBP), HPI
  and annual change for the four UK nations.

Outputs
-------
market_data/international/bis_index.csv   # country_code, country, period, index, yoy_pct
market_data/international/uk_regions.csv  # region, region_slug, month, avg_price_gbp, hpi, yoy_pct

Usage
-----
    python scripts/build_international_snapshot.py
    python scripts/build_international_snapshot.py --redownload
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from international_data import fetch_uk_regions, parse_bis_zip  # noqa: E402


def download(url: str, destination: Path, redownload: bool = False) -> Path:
    """Download ``url`` to ``destination`` (reused unless ``--redownload``)."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not redownload:
        print(f"  using cached download: {destination.name}")
        return destination

    print(f"  downloading {url.split('/')[-1]} ...")
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=120) as response, open(
        destination, "wb"
    ) as handle:
        handle.write(response.read())
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the international snapshot.")
    parser.add_argument("--redownload", action="store_true")
    args = parser.parse_args()

    config.INTERNATIONAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache = config.MARKET_CACHE_DIR
    cache.mkdir(parents=True, exist_ok=True)

    print("=" * 62)
    print("Building international market snapshot")
    print("=" * 62)

    print("[1/2] BIS residential property prices")
    bis_zip = download(config.BIS_SPP_URL, cache / "bis_spp.zip", args.redownload)
    bis = parse_bis_zip(bis_zip).drop(columns=["period_date"])
    bis.to_csv(config.BIS_INDEX_PATH, index=False)
    print(f"  countries: {', '.join(sorted(bis['country_code'].unique()))}")

    print("[2/2] UK Land Registry nations")
    uk = fetch_uk_regions().drop(columns=["month_date"], errors="ignore")
    uk.to_csv(config.UK_REGIONS_PATH, index=False)

    print("-" * 62)
    print(f"BIS rows            : {len(bis):,} ({bis['country_code'].nunique()} countries)")
    print(f"BIS latest period   : {bis['period'].max()}")
    print(f"UK rows             : {len(uk):,} ({uk['region'].nunique()} nations)")
    for path in (config.BIS_INDEX_PATH, config.UK_REGIONS_PATH):
        print(f"  {path.name:<20} {path.stat().st_size / 1e3:8.1f} KB")


if __name__ == "__main__":
    main()
