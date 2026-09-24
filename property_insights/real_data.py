"""
real_data.py
============
Real-data benchmark helpers for the **Ames Housing** dataset (OpenML).

This is deliberately separate from the shipped dashboard: the dashboard uses
synthetic listings anchored to real Zillow medians, while this module trains the
**same** leak-free pipeline on **real** property sales to give an honest
cross-validated read on model quality.

Public API
----------
* :func:`load_real_data`            - load the committed ``datasets/ames.csv``.
* :func:`cross_validate_metrics`    - 5-fold CV MAE / RMSE / MAPE / R².
* :func:`tune_model`                - light ``RandomizedSearchCV`` tuning.
* :func:`benchmark`                 - baseline vs. model vs. tuned comparison.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.metrics import (
    make_scorer,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)
from sklearn.model_selection import RandomizedSearchCV, cross_validate
from sklearn.pipeline import Pipeline

from . import config
from . import model as model_lib

AMES_PATH: Path = config.BASE_DIR / "datasets" / "ames.csv"

#: Metrics reported for every candidate, in display order.
METRIC_NAMES: list[str] = ["mae", "rmse", "mape", "r2"]


def load_real_data(path: Path = AMES_PATH) -> pd.DataFrame:
    """Load the committed Ames real-sales frame and validate its schema."""
    if not Path(path).exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python scripts/fetch_real_dataset.py` first."
        )
    frame = pd.read_csv(path)
    missing = set(config.RAW_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"Real dataset is missing columns: {sorted(missing)}")
    return frame


def _scoring() -> dict:
    """scikit-learn scorer dict for MAE / RMSE / MAPE / R²."""
    return {
        "mae": make_scorer(mean_absolute_error, greater_is_better=False),
        "rmse": make_scorer(
            lambda y_true, y_pred: float(np.sqrt(mean_squared_error(y_true, y_pred))),
            greater_is_better=False,
        ),
        "mape": make_scorer(mean_absolute_percentage_error, greater_is_better=False),
        "r2": "r2",
    }


def cross_validate_metrics(
    estimator,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 5,
    seed: int = config.RANDOM_SEED,
) -> pd.DataFrame:
    """Return mean and standard deviation of MAE / RMSE / MAPE / R² over folds."""
    from sklearn.model_selection import KFold

    kfold = KFold(n_splits=cv, shuffle=True, random_state=seed)
    raw = cross_validate(estimator, X, y, cv=kfold, scoring=_scoring(), n_jobs=1)

    rows: list[dict] = []
    for metric in METRIC_NAMES:
        values = -raw[f"test_{metric}"] if metric != "r2" else raw[f"test_{metric}"]
        if metric == "mape":
            values = values * 100.0
        rows.append({"metric": metric, "mean": float(np.mean(values)), "std": float(np.std(values))})
    return pd.DataFrame(rows).set_index("metric")


def tune_model(
    X: pd.DataFrame,
    y: pd.Series,
    n_iter: int = 24,
    cv: int = 4,
    seed: int = config.RANDOM_SEED,
    n_jobs: int = config.N_JOBS,
) -> tuple[object, dict]:
    """Light hyper-parameter search; returns ``(best_estimator, best_params)``."""
    search = RandomizedSearchCV(
        estimator=model_lib.build_pipeline(seed=seed),
        param_distributions={
            "regressor__model__n_estimators": [300, 500, 700, 900],
            "regressor__model__max_depth": [3, 4, 5, 6, 8],
            "regressor__model__learning_rate": [0.02, 0.03, 0.05, 0.08],
            "regressor__model__subsample": [0.7, 0.85, 1.0],
            "regressor__model__colsample_bytree": [0.7, 0.85, 1.0],
            "regressor__model__min_child_weight": [1, 2, 4],
        },
        n_iter=n_iter,
        scoring="neg_mean_absolute_error",
        cv=cv,
        random_state=seed,
        n_jobs=n_jobs,
        refit=True,
    )
    search.fit(X, y)
    return search.best_estimator_, dict(search.best_params_)


def benchmark(frame: pd.DataFrame, cv: int = 5, tune_iter: int = 24, seed: int = config.RANDOM_SEED) -> dict:
    """Run the full real-data benchmark (baseline vs. model vs. tuned)."""
    X = frame[config.FEATURE_COLUMNS]
    y = frame[config.TARGET_COLUMN]

    baseline = Pipeline([("model", DummyRegressor(strategy="median"))])
    baseline_metrics = cross_validate_metrics(baseline, X, y, cv=cv, seed=seed)

    current = model_lib.build_pipeline(seed=seed)
    current_metrics = cross_validate_metrics(current, X, y, cv=cv, seed=seed)

    tuned, best_params = tune_model(X, y, n_iter=tune_iter, cv=max(2, cv - 1), seed=seed)
    tuned_metrics = cross_validate_metrics(tuned, X, y, cv=cv, seed=seed)

    current.fit(X, y)
    importance = model_lib.get_feature_importance(current)

    return {
        "n_records": int(len(frame)),
        "n_neighborhoods": int(frame["neighborhood"].nunique()),
        "cv": cv,
        "baseline": baseline_metrics,
        "model": current_metrics,
        "tuned": tuned_metrics,
        "best_params": best_params,
        "importance": importance,
    }
