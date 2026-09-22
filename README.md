# 🏙️ Real Estate Price Estimator & Market Insights Dashboard

[![CI](https://github.com/armand-vw/Property-Price-Market-Dashboard/actions/workflows/ci.yml/badge.svg)](https://github.com/armand-vw/Property-Price-Market-Dashboard/actions/workflows/ci.yml)
[![Project Page](https://img.shields.io/badge/Project%20Page-GitHub%20Pages-4F46E5?logo=githubpages&logoColor=white)](https://armand-vw.github.io/Property-Price-Market-Dashboard/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)](https://armand-vw.github.io/Property-Price-Market-Dashboard/)
![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e.svg)](LICENSE)

An end-to-end, production-style machine-learning application that estimates
property values and surfaces **live US market insights** across the 15 largest
metros. Built with Python, Streamlit, XGBoost, scikit-learn and Plotly.

> **Data:** market values and trends are **real monthly data from
> [Zillow Research](https://www.zillow.com/research/data/)** (ZHVI), fetched live
> from the public CDN with a committed snapshot fallback. Listing-level features
> are synthesised and calibrated to each neighbourhood's real median value.

---

## 🔗 Live Demo

| | Link |
| --- | --- |
| **Interactive project page** (GitHub Pages, no install) | **https://armand-vw.github.io/Property-Price-Market-Dashboard/** |
| **Live prediction app** (Streamlit Community Cloud) | *Link added after deployment* |

The GitHub Pages site is a **preview** — it renders real, interactive Plotly
charts from the market data and trained model at build time. The full app with
live estimation runs on Streamlit Community Cloud ([`app.py`](app.py)).

---

## ✨ Highlights

- **Live market data** for 15 major US metros (New York, Los Angeles, Chicago,
  Dallas, Houston, Washington DC, Philadelphia, Miami, Atlanta, Boston, Phoenix,
  San Francisco, Riverside, Detroit, Seattle) — real monthly median home values,
  MoM / YoY / 5-year changes and 10-year trends.
- **Real neighbourhood values** — each market carries its actual neighbourhoods
  and their median values from a compact committed snapshot (the raw file is
  ~100 MB; it is reduced at build time).
- **Resilient data layer** — live Zillow fetch with a 24-hour cache, disk
  caching and automatic fallback to the committed snapshot if the network fails.
- **Market-anchored valuation model** — synthetic listings calibrated to real
  neighbourhood price-per-sqft, so estimates track local price levels
  (San Francisco ≫ Detroit) while retaining rich features (beds, baths, size,
  age, pool, garage).
- **Leak-free scikit-learn pipeline** with a log-transformed target and
  quantified evaluation (MAE, RMSE, MAPE, median APE, R², baseline).
- **Empirical valuation ranges** from held-out residual quantiles rather than an
  over-confident symmetric `± MAPE` band.
- **Polished multi-tab dashboard** — Markets, Market Analytics, Model Insights
  and Data Explorer, plus a live valuation tool with neighbourhood comparison.

---

## 📁 Project Structure

```
Property-Price-Market-Dashboard/
├── app.py                       # Streamlit dashboard (entry point)
├── market_data.py               # Zillow fetch, cache, fallback, aggregates
├── data_loader.py               # Market-anchored synthesis + cleaning
├── model.py                     # Pipeline, training, metrics, persistence
├── config.py                    # Paths, schema, seeds, palette, constants
├── market_data/                 # Committed real-market snapshot (small)
│   ├── markets.csv              # 15 metros
│   ├── market_history.csv       # monthly metro home values
│   ├── neighborhood_meta.csv    # neighbourhood values + price anchors
│   └── neighborhood_history.csv # monthly neighbourhood home values
├── scripts/
│   ├── build_market_snapshot.py # Builds market_data/ from Zillow (~100 MB once)
│   └── build_site.py            # Renders the GitHub Pages site into docs/
├── docs/                        # GitHub Pages landing page (static)
├── tests/                       # pytest suite (offline)
├── .github/workflows/ci.yml     # CI: pytest on push / PR
├── requirements.txt             # Pinned dependencies
├── runtime.txt                  # Python version for Streamlit Cloud
├── data/                        # Generated CSV + fetch cache (git-ignored)
└── models/                      # Fitted pipeline + metrics (artifacts)
```

---

## 🚀 Quickstart

```bash
git clone git@github.com:armand-vw/Property-Price-Market-Dashboard.git
cd Property-Price-Market-Dashboard

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

streamlit run app.py               # opens http://localhost:8501
```

On first launch the app generates the synthetic dataset, loads market data
(live Zillow fetch, snapshot fallback) and trains the model, caching artifacts
under `data/` and `models/`.

> **Debian/Ubuntu users:** if `python3 -m venv` fails, install the venv package
> with `sudo apt install python3.12-venv` (or use `pip install virtualenv`).

### Optional CLI

```bash
python data_loader.py --force            # rebuild data/housing.csv
python model.py                          # retrain + print evaluation summary
python market_data.py                    # print the live market overview
python scripts/build_market_snapshot.py  # refresh market_data/ from Zillow
python scripts/build_site.py             # regenerate the GitHub Pages site
pytest -q                                # run the test suite
```

---

## 🧠 How It Works

### 1. Real market data (`market_data.py`)

Zillow Research publishes monthly **ZHVI** (smoothed, seasonally-adjusted typical
home value) by geography. The app:

1. fetches the **metro** file (~4 MB) at runtime, caches it for 24 hours on disk,
   and falls back to the committed snapshot on any network error;
2. reads **neighbourhood** values from the committed snapshot produced by
   `scripts/build_market_snapshot.py`, which downloads the ~100 MB neighbourhood
   file once and reduces it to the 15 markets × top 12 neighbourhoods.

For each market the app derives the latest median value, month-over-month,
year-over-year and 5-year changes, and a 10-year trend.

### 2. Market-anchored synthesis (`data_loader.py`)

Listing-level features are generated per neighbourhood and priced as:

```
price = neighbourhood_base_per_sqft × sqft × age × beds × baths × pool × garage × lot × noise
```

where `neighbourhood_base_per_sqft` is the **real** neighbourhood median value
divided by an assumed median home size. This yields 7,000+ realistic listings
across 15 markets and 180 locations. Missing values and duplicates are injected
so cleaning is exercised, then:

1. duplicates dropped, types coerced, impossible rows removed,
2. **neighbourhood-median imputation** (global median fallback),
3. IQR winsorisation of `price` and `sqft`,
4. derived `price_per_sqft` and `property_age`.

### 3. Model pipeline (`model.py`)

```
Raw features
    │
    ├── Numeric  → SimpleImputer(median) → StandardScaler ─┐
    │                                                       ├──► XGBRegressor
    └── Category → SimpleImputer(mode)   → OneHotEncoder ──┘   (market, neighborhood)
                            wrapped in  ──► TransformedTargetRegressor(log1p / expm1)
```

The full object is persisted with `joblib`, so a single artifact accepts raw
features and returns prices in dollars. If XGBoost is unavailable the pipeline
falls back to `RandomForestRegressor`.

### 4. Evaluation (`model.py`)

80/20 train/test split, scored in dollars: MAE, RMSE, MAPE, median APE, R², and
a baseline comparison against always predicting the median.

---

## 📊 Model Performance

Reproduced with the default configuration (`RANDOM_SEED=42`):

| Metric | Value |
| --- | --- |
| Estimator | `XGBRegressor` (700 trees, lr 0.03, depth 5) |
| Listings (train / test) | 5,617 / 1,405 |
| Markets / locations | 15 / 180 |
| MAE | **$89,171** |
| RMSE | **$131,178** |
| MAPE | **14.66%** |
| Median APE | **12.62%** |
| R² | **0.919** |
| Improvement vs. median baseline | **74.8% lower MAE** |

**Top price drivers:** market, neighbourhood, square footage, bedrooms, pool.

> Metrics are regenerated into `models/metrics.json` on every training run.

---

## 🖥️ Dashboard Tour

- **Executive KPI row** — listings, the selected market's **real** median value
  and YoY, model accuracy (100 − MAPE) and R².
- **🌎 Markets** — live median value, MoM/YoY/5-year changes, a 10-year value
  trend, latest-value and YoY comparison across all 15 markets, and a real
  neighbourhood value table.
- **📊 Market Analytics** — price-per-sqft scatter by location (colour-coded by
  age band), median listing price by location, and feature importances.
- **🤖 Model Insights** — metric cards, predicted-vs-actual parity and residual
  diagnostics.
- **🗂️ Data Explorer** — neighbourhood market summary, filterable listings and
  CSV export.
- **🎯 Live Valuation Tool (sidebar)** — pick a market and location, set the
  property details and press **Estimate Value** to get the estimate, an
  empirical valuation range, and a comparison against the **real** neighbourhood
  median.

---

## 🧪 Testing

```bash
pytest -q
```

The suite is **network-free**: it uses the committed market snapshot and small
synthetic fixtures, covering data reproducibility, cleaning invariants, schema
validation, market aggregation, live-to-snapshot fallback, pipeline fitting,
feature-importance aggregation and inference ranges.

---

## 🛠️ Engineering Notes & Design Decisions

- **No target leakage.** All preprocessing is fitted inside the pipeline on the
  training fold only.
- **Real where it matters.** Market levels and neighbourhood values are real;
  only listing-level attributes are synthesised (no free listing-level source
  exists). This is stated in the UI and README.
- **Resilient by design.** Live fetch → disk cache → committed snapshot, with an
  environment-tunable timeout. The app never depends on the network being up.
- **Honest uncertainty.** Valuation ranges come from empirical residual
  quantiles, not a symmetric `± MAPE` band.
- **Bounded threads.** `N_JOBS` is capped (`min(4, cpu)`) to avoid OpenMP
  oversubscription; override with `RPE_N_JOBS`.
- **Single source of truth.** Paths, schema, seeds and constants live in
  `config.py`.

---

## 🗺️ Roadmap

- [ ] Add rents (ZORI) and gross rental yield per market.
- [ ] Add inventory, days-on-market and sale-to-list metrics.
- [ ] Swap synthetic listings for a real listing-level dataset behind the same
      loader interface.
- [ ] SHAP values for per-prediction explainability.
- [ ] Hyper-parameter tuning with Optuna and cross-validated MAPE.
- [ ] Containerise with Docker.
- [x] Live real market data with snapshot fallback.
- [x] Continuous integration with GitHub Actions (`pytest`).
- [x] Static project page on GitHub Pages.

---

## 📄 License & Attribution

Released under the MIT License.

Market data © [Zillow](https://www.zillow.com/research/data/) (ZHVI), used under
their public research terms. Listing-level estimates are illustrative and do not
constitute financial advice.
