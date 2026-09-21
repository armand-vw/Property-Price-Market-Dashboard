"""
model.py
========
Machine-learning pipeline for the Real Estate Price Estimator.

Design
------
A single :class:`sklearn.compose.ColumnTransformer` handles preprocessing
(median imputation + standard scaling for numerics, mode imputation + one-hot
encoding for the neighbourhood) and is chained into an XGBoost gradient-boosted
tree regressor. The whole thing is wrapped in a
:class:`sklearn.compose.TransformedTargetRegressor` that trains on ``log1p`` of
the sale price and inverts the transform automatically at prediction time. This:

* tames the heavy right-skew of property prices, and
* means the persisted artifact is fully self-contained - it accepts raw features
  and returns prices in dollars, with the imputer, scaler and encoder baked in.

Artifacts written by :func:`train_pipeline`::

    models/price_model.joblib      # preprocessor + scaler + encoder + model
    models/metrics.json            # evaluation metrics + metadata
    models/feature_importance.csv  # ranked, human-readable feature importances

Run directly to train from the command line::

    python model.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config
from data_loader import load_or_create_data


# --------------------------------------------------------------------------- #
# Pipeline construction
# --------------------------------------------------------------------------- #
def build_preprocessor() -> ColumnTransformer:
    """Build the preprocessing transformer.

    * Numeric features: median imputation followed by standard scaling. Scaling
      is a no-op for tree splits but keeps the artifact model-agnostic (e.g. if
      the estimator is swapped for a linear model).
    * Categorical features: most-frequent imputation followed by one-hot
      encoding. ``handle_unknown="ignore"`` makes the pipeline robust to unseen
      neighbourhoods at inference time.
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, config.NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, config.CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def build_estimator(seed: int = config.RANDOM_SEED) -> Any:
    """Return the gradient-boosted regressor, falling back to a random forest.

    XGBoost is preferred; if the package is unavailable on the host the function
    degrades gracefully to scikit-learn's :class:`RandomForestRegressor` so the
    project always runs.
    """
    try:
        from xgboost import XGBRegressor

        return XGBRegressor(
            n_estimators=700,
            learning_rate=0.03,
            max_depth=5,
            min_child_weight=2,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            random_state=seed,
            n_jobs=config.N_JOBS,
            tree_method="hist",
        )
    except ImportError:  # pragma: no cover - depends on environment
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=config.N_JOBS,
        )


def build_pipeline(seed: int = config.RANDOM_SEED) -> TransformedTargetRegressor:
    """Assemble the full, fit-ready pipeline.

    The returned object maps raw feature rows -> dollar predictions, learning on
    the ``log1p`` target internally.
    """
    regressor = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", build_estimator(seed)),
        ]
    )
    return TransformedTargetRegressor(
        regressor=regressor,
        func=np.log1p,
        inverse_func=np.expm1,
    )


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute and return the evaluation metrics for a set of predictions.

    Metrics
    -------
    MAE:
        Mean Absolute Error, in dollars - average size of a miss.
    RMSE:
        Root Mean Squared Error, in dollars - penalises large misses.
    MAPE:
        Mean Absolute Percentage Error - scale-free headline accuracy.
    median_ape:
        Median absolute percentage error - robust to a handful of large misses.
    residual_p10 / residual_p90:
        Signed percentage-error quantiles; used to build an empirical valuation
        range that is more honest than a symmetric ``± MAPE`` band.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residuals = y_pred - y_true
    percentage_error = residuals / np.clip(np.abs(y_true), 1e-9, None) * 100.0

    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "mape": float(np.mean(np.abs(percentage_error))),
        "median_ape": float(np.median(np.abs(percentage_error))),
        "residual_p10": float(np.percentile(percentage_error, 10)),
        "residual_p90": float(np.percentile(percentage_error, 90)),
    }


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train_pipeline(
    df: pd.DataFrame,
    test_size: float = config.TEST_SIZE,
    seed: int = config.RANDOM_SEED,
    save: bool = True,
) -> dict[str, Any]:
    """Train, evaluate and (optionally) persist the price model.

    Parameters
    ----------
    df:
        Cleaned housing dataframe.
    test_size:
        Fraction of records held out for evaluation.
    seed:
        Random seed for the split and estimator.
    save:
        Persist the pipeline and artifact files to :data:`config.MODEL_DIR`.

    Returns
    -------
    dict
        ``{"model", "metrics", "importance", "feature_importance"}`` where
        ``metrics`` includes metadata (model name, timestamp, row counts).
    """
    if df.empty:
        raise ValueError("Cannot train on an empty dataframe.")

    X = df[config.FEATURE_COLUMNS]
    y = df[config.TARGET_COLUMN].to_numpy(dtype=float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    model = build_pipeline(seed=seed)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = compute_metrics(y_test, y_pred)

    # Context: how much better than simply guessing the training median.
    baseline_pred = np.full_like(y_test, np.median(y_train))
    metrics["baseline_mae"] = float(mean_absolute_error(y_test, baseline_pred))
    metrics["improvement_vs_baseline_pct"] = float(
        (1 - metrics["mae"] / metrics["baseline_mae"]) * 100.0
    )

    importance = get_feature_importance(model)
    metrics.update(
        {
            "model_name": type(
                model.regressor_.named_steps["model"]
            ).__name__,
            "n_records": int(len(df)),
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "test_size": float(test_size),
            "features": config.FEATURE_COLUMNS,
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "target_transform": "log1p / expm1",
        }
    )

    if save:
        save_artifacts(model, metrics, importance)

    return {
        "model": model,
        "metrics": metrics,
        "importance": importance,
        "feature_importance": importance,
    }


def save_artifacts(
    model: TransformedTargetRegressor,
    metrics: dict[str, Any],
    importance: pd.DataFrame,
) -> None:
    """Persist the fitted pipeline, metrics and feature importances to disk."""
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, config.MODEL_PATH)

    with open(config.METRICS_PATH, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    importance.to_csv(config.IMPORTANCE_PATH, index=False)


# --------------------------------------------------------------------------- #
# Feature importance
# --------------------------------------------------------------------------- #
def get_feature_importance(
    model: TransformedTargetRegressor,
    top_n: int | None = 15,
) -> pd.DataFrame:
    """Extract ranked feature importances from the fitted pipeline.

    One-hot encoded neighbourhood columns are aggregated back into a single
    ``neighborhood`` feature so the chart answers the business question "what
    moves price?" rather than "which dummy column moved price?".

    Returns
    -------
    pandas.DataFrame
        Columns ``feature`` and ``importance``, sorted descending.
    """
    pipeline: Pipeline = model.regressor_
    preprocessor: ColumnTransformer = pipeline.named_steps["preprocessor"]
    estimator = pipeline.named_steps["model"]

    raw_names = list(preprocessor.get_feature_names_out())
    importances = np.asarray(estimator.feature_importances_, dtype=float)

    if len(raw_names) != len(importances):
        raise ValueError("Mismatch between feature names and importances.")

    cleaned = [
        "neighborhood" if name.startswith("categorical__neighborhood") else name
        for name in raw_names
    ]
    cleaned = [name.split("__", 1)[1] if "__" in name else name for name in cleaned]

    frame = (
        pd.DataFrame({"feature": cleaned, "raw_importance": importances})
        .groupby("feature", as_index=False)["raw_importance"]
        .sum()
        .rename(columns={"raw_importance": "importance"})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    if top_n is not None:
        frame = frame.head(top_n).reset_index(drop=True)

    return frame


# --------------------------------------------------------------------------- #
# Persistence & inference helpers
# --------------------------------------------------------------------------- #
def load_pipeline(path=config.MODEL_PATH) -> TransformedTargetRegressor:
    """Load a persisted pipeline from disk."""
    if not path.exists():
        raise FileNotFoundError(
            f"No trained model at {path}. Run `python model.py` first."
        )
    return joblib.load(path)


def _load_metrics(path=config.METRICS_PATH) -> dict[str, Any]:
    """Load persisted metrics, returning an empty dict when absent."""
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def features_to_frame(features: dict[str, Any]) -> pd.DataFrame:
    """Convert a single property's feature dict into a one-row dataframe.

    Unknown keys are ignored and missing keys are filled with sensible defaults
    so the function is safe to call from the UI layer.
    """
    row: dict[str, Any] = {}
    for column in config.FEATURE_COLUMNS:
        if column in features:
            row[column] = features[column]
        elif column == "neighborhood":
            row[column] = next(iter(config.NEIGHBORHOODS))
        else:
            row[column] = np.nan
    return pd.DataFrame([row], columns=config.FEATURE_COLUMNS)


def predict_price(model: TransformedTargetRegressor, features: dict[str, Any]) -> float:
    """Predict a single property's price in dollars."""
    frame = features_to_frame(features)
    return float(model.predict(frame)[0])


def predict_with_range(
    model: TransformedTargetRegressor,
    features: dict[str, Any],
    metrics: dict[str, Any],
) -> dict[str, float]:
    """Predict a price and derive an empirical valuation range.

    The range comes from the 10th/90th percentile of the model's percentage
    errors on the held-out test set, i.e. an ~80% empirical interval rather than
    a naive symmetric band.

    Returns
    -------
    dict
        ``point``, ``low``, ``high``, ``half_width_pct`` and ``mape``.
    """
    point = predict_price(model, features)
    lower_pct = float(metrics.get("residual_p10", -metrics.get("mape", 8.0)))
    upper_pct = float(metrics.get("residual_p90", metrics.get("mape", 8.0)))
    mape = float(metrics.get("mape", 0.0))

    low = point * (1.0 + lower_pct / 100.0)
    high = point * (1.0 + upper_pct / 100.0)

    return {
        "point": point,
        "low": max(low, 0.0),
        "high": high,
        "half_width_pct": max(abs(lower_pct), abs(upper_pct)),
        "mape": mape,
    }


def ensure_model(
    data: pd.DataFrame | None = None,
    train_if_missing: bool = True,
) -> tuple[TransformedTargetRegressor, dict[str, Any], pd.DataFrame]:
    """Return ``(model, metrics, importance)``, training on demand.

    This is the entry point used by the dashboard: it makes ``streamlit run
    app.py`` work from a clean clone with no manual training step, while still
    reusing the cached artifacts on subsequent runs.
    """
    if config.MODEL_PATH.exists() and config.METRICS_PATH.exists():
        model = load_pipeline()
        metrics = _load_metrics()
        if config.IMPORTANCE_PATH.exists():
            importance = pd.read_csv(config.IMPORTANCE_PATH)
        else:
            importance = get_feature_importance(model)
        return model, metrics, importance

    if not train_if_missing:
        raise FileNotFoundError("No trained model available and training is disabled.")

    if data is None:
        data = load_or_create_data()

    result = train_pipeline(data, save=True)
    return result["model"], result["metrics"], result["importance"]


# --------------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    """Train from the command line and print an evaluation summary."""
    data = load_or_create_data()
    result = train_pipeline(data)
    metrics = result["metrics"]

    print("=" * 62)
    print("Model training complete")
    print("=" * 62)
    print(f"Estimator            : {metrics['model_name']}")
    print(f"Train / Test         : {metrics['n_train']:,} / {metrics['n_test']:,}")
    print(f"MAE                  : ${metrics['mae']:,.0f}")
    print(f"RMSE                 : ${metrics['rmse']:,.0f}")
    print(f"MAPE                 : {metrics['mape']:.2f}%")
    print(f"Median APE           : {metrics['median_ape']:.2f}%")
    print(f"R2                   : {metrics['r2']:.4f}")
    print(
        "Improvement vs median: "
        f"{metrics['improvement_vs_baseline_pct']:.1f}% lower MAE"
    )
    print("-" * 62)
    print("Top drivers of price:")
    for _, row in result["importance"].head(8).iterrows():
        print(f"  {row['feature']:<18} {row['importance']:.3f}")
    print("-" * 62)
    print(f"Saved pipeline       : {config.MODEL_PATH}")
    print(f"Saved metrics        : {config.METRICS_PATH}")


if __name__ == "__main__":
    main()
