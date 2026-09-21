# 🏙️ Real Estate Price Estimator & Market Insights Dashboard

[![CI](https://github.com/armand-vw/Property-Price-Market-Dashboard/actions/workflows/ci.yml/badge.svg)](https://github.com/armand-vw/Property-Price-Market-Dashboard/actions/workflows/ci.yml)
[![Project Page](https://img.shields.io/badge/Project%20Page-GitHub%20Pages-4F46E5?logo=githubpages&logoColor=white)](https://armand-vw.github.io/Property-Price-Market-Dashboard/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)](https://armand-vw.github.io/Property-Price-Market-Dashboard/)
![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e.svg)](LICENSE)

An end-to-end, production-style machine-learning application that **estimates
property values** and **surfaces market insights** through an interactive
Streamlit dashboard. Built with Python, XGBoost, scikit-learn, Plotly and
Streamlit.

> **Portfolio project** — the dataset is synthesised from configurable
> neighbourhood price profiles, so the whole pipeline runs offline and is fully
> reproducible from a single command.

---

## 🔗 Live Demo

| | Link |
| --- | --- |
| **Interactive project page** (GitHub Pages, no install) | **https://armand-vw.github.io/Property-Price-Market-Dashboard/** |
| **Live prediction app** (Streamlit Community Cloud) | *Link added after deployment* |

The GitHub Pages site is a **preview** of the project — it renders the real,
interactive Plotly analytics from the dataset and trained model at build time.
The full live valuation tool (with model inference) runs on Streamlit Community
Cloud; the dashboard source lives in [`app.py`](app.py).

---

## ✨ Highlights

- **Realistic synthetic data engine** with correlated features (size, beds,
  baths, age, pool, garage, location) and deliberately injected missing values
  and duplicates to exercise the cleaning layer.
- **Leak-free scikit-learn pipeline** — median imputation, standard scaling and
  one-hot encoding are fitted *inside* the pipeline, never on the test set.
- **Log-transformed target** (`log1p` / `expm1`) to handle the right-skew of
  property prices, wrapped in a `TransformedTargetRegressor`.
- **Quantified evaluation** — MAE, RMSE, MAPE, median APE, R² and a
  baseline comparison against simply predicting the median.
- **Empirical valuation ranges** derived from held-out residual quantiles
  instead of an over-confident symmetric `± MAPE` band.
- **Polished, multi-tab dashboard** — executive KPIs, market analytics, model
  diagnostics and a live valuation tool with neighbourhood comparison.
- **Zero-setup demo** — the app trains and caches the model automatically the
  first time it runs; no manual step required.

---

## 📁 Project Structure

```
Property-Price-Market-Dashboard/
├── app.py                       # Streamlit dashboard (entry point)
├── model.py                     # Pipeline, training, metrics, persistence
├── data_loader.py               # Data synthesis, cleaning, aggregates
├── config.py                    # Paths, schema, seeds, palette, profiles
├── conftest.py                  # Pytest fixtures + root sys.path setup
├── requirements.txt             # Pinned dependencies
├── runtime.txt                  # Python version for Streamlit Cloud
├── LICENSE                      # MIT
├── .streamlit/
│   └── config.toml              # Light corporate theme
├── .github/workflows/ci.yml     # CI: pytest on push / PR
├── scripts/
│   └── build_site.py            # Renders the GitHub Pages site into docs/
├── docs/                        # GitHub Pages landing page (static)
│   ├── index.html
│   └── style.css
├── tests/
│   ├── test_data_loader.py
│   └── test_model.py
├── data/                        # Generated CSV (git-ignored)
└── models/                      # Fitted pipeline + metrics (artifacts)
```

---

## 🚀 Quickstart

### 1. Clone and enter the project

```bash
git clone git@github.com:armand-vw/Property-Price-Market-Dashboard.git
cd Property-Price-Market-Dashboard
```

### 2. Create a virtual environment and install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> **Debian/Ubuntu users:** if `python3 -m venv` fails, install the venv package
> with `sudo apt install python3.12-venv` (or use `pip install virtualenv`).

### 3. Run the dashboard

```bash
streamlit run app.py
```

The app opens at <http://localhost:8501>. On first launch it generates the
dataset, trains the model and caches both under `data/` and `models/`.

### Optional: regenerate data or retrain from the CLI

```bash
python data_loader.py --force   # rebuild data/housing.csv
python model.py                 # retrain + print an evaluation summary
pytest -q                       # run the test suite
```

---

## 🧠 How It Works

### 1. Data synthesis & cleaning (`data_loader.py`)

Each record is generated from a neighbourhood profile (base price-per-sqft plus
a location "prestige" premium). Square footage is drawn from a log-normal
distribution, and bedrooms/bathrooms/garages follow from size. Price is a
product of location, size, age depreciation, amenity and lot factors multiplied
by log-normal noise, producing the heavy right tail typical of real listings.

Cleaning then:

1. drops exact duplicates,
2. coerces and validates types,
3. removes physically impossible rows,
4. median-imputes missing values using the **neighbourhood median** (global
   median as fallback),
5. winsorises `price` and `sqft` at the IQR fences,
6. derives `price_per_sqft` and `property_age`.

### 2. Model pipeline (`model.py`)

```
Raw features
    │
    ├── Numeric  → SimpleImputer(median) → StandardScaler ─┐
    │                                                       ├──► XGBRegressor
    └── Category → SimpleImputer(mode)   → OneHotEncoder ──┘
                            wrapped in  ──► TransformedTargetRegressor(log1p / expm1)
```

The entire object is persisted with `joblib`, so a single artifact accepts raw
features and returns prices in dollars — imputer, scaler, encoder and model
included. If XGBoost is unavailable, the pipeline transparently falls back to a
scikit-learn `RandomForestRegressor`.

### 3. Evaluation (`model.py`)

80/20 train/test split, evaluated in dollars:

| Metric | Description |
| --- | --- |
| **MAE** | Mean absolute error — average size of a miss |
| **RMSE** | Root mean squared error — penalises large misses |
| **MAPE** | Mean absolute percentage error — scale-free accuracy |
| **Median APE** | Robust central percentage error |
| **R²** | Share of price variance explained |
| **Baseline MAE** | Error from always predicting the median |

---

## 📊 Model Performance

Reproduced with the default configuration (`N_RECORDS=2000`, `seed=42`):

| Metric | Value |
| --- | --- |
| Estimator | `XGBRegressor` (700 trees, lr 0.03, depth 5) |
| Train / Test rows | 1,560 / 390 |
| MAE | **$96,459** |
| RMSE | **$135,919** |
| MAPE | **10.06%** |
| Median APE | **8.76%** |
| R² | **0.942** |
| Improvement vs. median baseline | **77.6% lower MAE** |

**Top price drivers:** neighbourhood, square footage, bedrooms, pool, bathrooms.

> Metrics are regenerated into `models/metrics.json` on every training run, so
> this table can be refreshed automatically.

---

## 🖥️ Dashboard Tour

- **Executive KPI row** — total listings, median market price, model accuracy
  (100 − MAPE) and R².
- **📊 Market Analytics** — price-per-sqft scatter colour-coded by property age
  band, median price by neighbourhood, and an aggregated feature-importance
  chart.
- **🤖 Model Insights** — full metric cards, predicted-vs-actual parity plot and
  the distribution of percentage errors.
- **🗂️ Data Explorer** — neighbourhood market summary, filterable listing table
  and CSV export.
- **🎯 Live Valuation Tool (sidebar)** — pick a neighbourhood, drag the size
  slider, set beds/baths/age/pool/garage and press **Estimate Value** to get:
  - the estimated market price,
  - an empirical valuation range (10th–90th percentile of test errors),
  - a direct comparison against the selected neighbourhood's median.

---

## 🧪 Testing

```bash
pytest -q
```

The suite covers data reproducibility, cleaning invariants, type/schema
validation, metric correctness, pipeline fitting, feature-importance
aggregation and the inference range logic. No trained artifact is required.

---

## ☁️ Deployment

### Live app — Streamlit Community Cloud

1. Push this repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Select the repo, branch `main`, and main file `app.py`.
4. (Optional) Python `3.12` to match `runtime.txt`. Then **Deploy**.

No secrets are required. On the first cold start the app installs dependencies
and trains the model automatically (a few seconds).

### Project page — GitHub Pages

The `docs/` folder is published as a static project page. Regenerate it from the
real dataset and model at any time:

```bash
python scripts/build_site.py

# point the "Launch live app" buttons at your deployment:
LIVE_APP_URL=https://your-app.streamlit.app python scripts/build_site.py
```

Then publish via **Settings → Pages → Deploy from branch → `main` / `/docs`**.

---

## 🛠️ Engineering Notes & Design Decisions

- **No target leakage.** All preprocessing is fitted inside the pipeline on the
  training fold only, so one-hot encoding and imputation never see the test set.
- **Honest uncertainty.** A `± MAPE` band implies symmetry that price errors do
  not have; the app instead reports the empirical 10th/90th percentile residual
  range.
- **Reproducibility first.** Every random process is seeded via `config.py`, and
  `tests/test_data_loader.py::test_generate_is_reproducible` enforces it.
- **Graceful degradation.** XGBoost import failure falls back to a random
  forest; the app trains on demand when artifacts are missing.
- **Single source of truth.** Paths, schema, seeds, palette and neighbourhood
  profiles all live in `config.py`.

---

## 🗺️ Roadmap

- [ ] Swap the synthetic source for a real dataset (e.g. Kaggle Ames Housing)
      behind the same loader interface.
- [ ] Add SHAP values for per-prediction explainability.
- [ ] Hyper-parameter tuning with Optuna and cross-validated MAPE.
- [ ] Containerise with Docker.
- [x] Continuous integration with GitHub Actions (`pytest`).
- [x] Static project page on GitHub Pages.
- [x] Deploy to Streamlit Community Cloud.

---

## 📄 License

Released under the MIT License. The dataset is synthetic and provided for
demonstration only; estimates do not constitute financial advice.
