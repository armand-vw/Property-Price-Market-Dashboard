# Model Card — Real Estate Price Estimator

This card documents the machine-learning model behind the **US valuation tool**
in the Real Estate Price Estimator & Market Insights Dashboard, following the
spirit of the [Model Cards](https://arxiv.org/abs/1810.03993) framework.

## Model details

| | |
| --- | --- |
| **Model type** | Gradient-boosted decision trees (`XGBRegressor`), with a `RandomForestRegressor` fallback |
| **Task** | Regression — estimate the sale price of a single-family home |
| **Target** | `log1p(price)` (inverse `expm1` applied at prediction time) |
| **Preprocessing** | scikit-learn `ColumnTransformer`: median imputation + standard scaling (numeric), mode imputation + one-hot encoding (categorical) — all fitted inside the pipeline |
| **Framework** | scikit-learn 1.9, XGBoost 3.4 |
| **Explainability** | Aggregated gain feature importances + per-prediction SHAP contributions via `Booster.predict(pred_contribs=True)` |
| **Artifact** | `models/price_model.joblib` (committed, ~1.6 MB) |
| **License** | MIT |

Hyper-parameters: 700 trees, learning rate 0.03, max depth 5, `min_child_weight`
2, subsample/colsample 0.85, L2 `reg_lambda` 1.0, `tree_method="hist"`.

### Input features

`bedrooms`, `bathrooms`, `sqft`, `lot_size`, `year_built`, `has_pool`,
`garage_spaces`, and the categoricals `market` (US metro) and `neighborhood`.

## Intended use

- **Intended:** illustrative, educational point estimates of home value for US
  locations in the supported metros; demonstrating an end-to-end ML workflow.
- **Not intended:** lending, appraisal, investment, tax or insurance decisions.
  Estimates are not financial advice.

## Training data

- **15 US metros × 12 neighbourhoods = 180 locations**, 7,022 listings.
- Listing-level records are **synthesised** (beds, baths, size, age, pool,
  garage, lot) and **anchored to real Zillow Research median home values** per
  neighbourhood, so price *levels* are realistic while the listing attributes
  are simulated.
- No personally identifiable information is used. Real market aggregates are
  sourced from Zillow Research (ZHVI, ZORI, inventory, days-to-pending, median
  sale price) and reprocessed into compact committed snapshots.

## Evaluation

80/20 hold-out split, scored in dollars on the test set (`RANDOM_SEED=42`,
`n_test=1,405`):

| Metric | Value |
| --- | --- |
| MAE | **$89,171** |
| RMSE | **$131,178** |
| MAPE | **14.66%** |
| Median APE | **12.62%** |
| R² | **0.919** |
| Improvement vs. median baseline | **74.8% lower MAE** |

Valuation ranges shown in the app come from the empirical 10th–90th percentile
of percentage errors on the hold-out set (≈ ±15–26%), not a symmetric `± MAPE`
band.

### Real-data benchmark (Ames Housing)

Because the dashboard's listings are synthetic, the **same pipeline** is also
cross-validated on the real **Ames Housing** dataset (1,460 actual sales,
25 neighbourhoods, 5-fold CV):

| Model | MAE | RMSE | MAPE | R² |
| --- | --- | --- | --- | --- |
| Baseline (predict median) | $55,656 | $81,275 | 31.8% | −0.054 |
| **XGBoost (project defaults)** | **$20,367** | **$32,228** | **12.1%** | **0.820** |
| XGBoost (lightly tuned) | $20,192 | $31,986 | 11.9% | 0.827 |

See [`reports/real_data_benchmark.md`](../reports/real_data_benchmark.md) and
[`notebooks/`](../notebooks). Ames is a single, static city, so this validates the
*approach* rather than live US coverage.

## Limitations

- **Synthetic listings.** Only geographic price levels are real; individual
  listing attributes are simulated, so the model has not seen real transaction
  micro-data.
- **US-only valuation.** Other countries expose national index analytics only.
- **Location dominance.** Because neighbourhood price levels span a wide range,
  location features dominate the model; within-neighbourhood size/amenity
  effects are comparatively small.
- **No temporal validation.** The model is trained on a single cross-section,
  not a time series, so it does not model market timing.
- **Coverage.** Only the 15 largest US metros are supported; unseen categories
  are handled by `handle_unknown="ignore"` but will fall back to broad effects.

## Ethical considerations

- Listing-level estimates are **illustrative** and must not be presented as
  appraisals or used for decisions affecting people.
- Synthetic training data cannot encode real-world discrimination; the model
  should not be used to infer anything about protected characteristics. The
  `neighborhood` feature can proxy for demographic composition, so real-world
  deployment would require a fairness review.

## Reproducing

```bash
python scripts/build_market_snapshot.py   # rebuild committed market snapshots
python data_loader.py --force             # rebuild synthetic listings
python model.py                           # retrain + metrics + feature importances
pytest -q                                 # 54 tests
```
