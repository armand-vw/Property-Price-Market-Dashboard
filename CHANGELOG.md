# Changelog

All notable changes to this project are documented here. The format is loosely
based on [Keep a Changelog](https://keepachangelog.com/), and the project aims
to follow [Semantic Versioning](https://semver.org/).

## [1.1.0] — 2026-09

### Added
- **International markets**: 26-country switcher (US + 25 countries) using the
  BIS Selected Residential Property Prices dataset.
- **US market health**: for-sale inventory, median days-to-pending and median
  sale price per metro (Zillow).
- **Plain-English education layer**: metric tooltips, a "what this means"
  summary, a Hot/Warm/Cool market temperature and a Market 101 glossary.
- **Per-prediction explainability**: SHAP contributions from XGBoost (no extra
  dependency) rendered as a "Why this estimate?" chart.
- **GitHub Pages demo**: pinned, country-only sidebar with client-side charts,
  URL state and a social-preview image.
- **Offline mode** (`RPE_OFFLINE=1`) and a self-contained **Dockerfile**.
- **CI**: ruff lint job, pytest job and a Docker build + health-check job.
- Model card (`MODEL_CARD.md`).

### Changed
- The US is presented as a **national** view (median across 15 metros); the
  per-metro selector was removed from both the app and the Pages sidebar.
- The valuation tool now uses a single **all-US location dropdown** and derives
  the metro internally.

## [1.0.0] — 2026-08

### Added
- Synthetic, market-anchored housing dataset generator and cleaning pipeline.
- Leak-free XGBoost pipeline with log-target and quantified evaluation
  (MAE, RMSE, MAPE, median APE, R², baseline).
- Streamlit dashboard (market analytics, model insights, data explorer) with a
  live valuation tool and shareable URLs.
- Streamlit Community–independent GitHub Pages project page, README and tests.
