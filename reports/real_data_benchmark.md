# Real-data benchmark — Ames Housing

Generated 2026-09-24 20:09 UTC · `scripts/benchmark_real_data.py`.

## What this is

The shipped dashboard is trained on **synthetic** listings calibrated to real
Zillow medians. To give an honest, real-data read on modelling quality, the
**same leak-free pipeline** is trained and evaluated here on the **real** Ames
Housing dataset (1,460 actual sales, 25
neighbourhoods), with 5-fold cross-validation.

- Target: `log1p(price)` (inverted to dollars for scoring).
- Preprocessing (imputation, scaling, one-hot) is fitted **inside** each fold.
- Metrics are mean ± std across 5 folds.

## Results

| Model | MAE | RMSE | MAPE | R² |
| --- | --- | --- | --- | --- |
| Baseline (predict median) | $55,656 ± $2,835 | $81,275 ± $5,303 | 31.8% ± 3.1% | -0.054 ± 0.021 |
| XGBoost (project defaults) | $20,367 ± $1,337 | $32,228 ± $6,920 | 12.1% ± 0.6% | 0.820 ± 0.099 |
| XGBoost (lightly tuned) | $20,192 ± $1,115 | $31,986 ± $5,424 | 11.9% ± 0.6% | 0.827 ± 0.076 |

Best hyper-parameters: `subsample=0.7, n_estimators=500, min_child_weight=4, max_depth=6, learning_rate=0.02, colsample_bytree=0.7`.

## Top drivers of price (real data)

| Feature | Importance |
| --- | --- |
| neighborhood | 0.504 |
| garage_spaces | 0.220 |
| sqft | 0.097 |
| year_built | 0.094 |
| bathrooms | 0.026 |
| bedrooms | 0.025 |
| lot_size | 0.024 |
| has_pool | 0.009 |
| market | 0.000 |

## Caveats

- Ames is a **single, static city** dataset (2006–2010 sales); it validates the
  modelling approach, not live US market coverage.
- The dashboard's synthetic listings are still used for the interactive product
  (documented in `MODEL_CARD.md`); this report is the real-data counterpart.
