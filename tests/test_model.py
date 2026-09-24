"""Unit tests for the machine-learning pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd

from property_insights import config
from property_insights.model import (
    build_pipeline,
    build_preprocessor,
    compute_metrics,
    ensure_model,
    explain_prediction,
    features_to_frame,
    get_feature_importance,
    predict_with_range,
    train_pipeline,
)


def test_compute_metrics_perfect_prediction() -> None:
    y = np.array([100.0, 200.0, 300.0])
    metrics = compute_metrics(y, y)
    assert metrics["mae"] == 0.0
    assert metrics["rmse"] == 0.0
    assert metrics["mape"] == 0.0
    assert metrics["r2"] == 1.0


def test_build_pipeline_fits_and_predicts(clean_df: pd.DataFrame) -> None:
    model = build_pipeline()
    X = clean_df[config.FEATURE_COLUMNS]
    y = clean_df[config.TARGET_COLUMN]
    model.fit(X, y)

    predictions = model.predict(X.head(10))
    assert len(predictions) == 10
    assert np.all(predictions > 0)


def test_train_pipeline_returns_metrics(clean_df: pd.DataFrame) -> None:
    result = train_pipeline(clean_df, save=False)
    metrics = result["metrics"]

    for key in ["mae", "rmse", "r2", "mape", "median_ape", "baseline_mae"]:
        assert key in metrics
        assert np.isfinite(metrics[key])

    assert metrics["r2"] > 0.5
    assert 0 < metrics["mape"] < 50
    assert metrics["n_records"] == len(clean_df)


def test_train_pipeline_does_not_touch_artifacts(clean_df: pd.DataFrame) -> None:
    before = config.MODEL_PATH.exists()
    train_pipeline(clean_df, save=False)
    after = config.MODEL_PATH.exists()
    assert before == after


def test_feature_importance_is_aggregated_and_sorted(clean_df: pd.DataFrame) -> None:
    result = train_pipeline(clean_df, save=False)
    importance = get_feature_importance(result["model"])

    assert set(importance.columns) == {"feature", "importance"}
    assert (importance["importance"] >= 0).all()
    assert "neighborhood" in set(importance["feature"])
    assert importance["importance"].is_monotonic_decreasing


def test_features_to_frame_uses_correct_schema(single_property: dict) -> None:
    frame = features_to_frame(single_property)
    assert list(frame.columns) == config.FEATURE_COLUMNS
    assert len(frame) == 1


def test_features_to_frame_fills_defaults() -> None:
    frame = features_to_frame({"sqft": 1_500})
    assert list(frame.columns) == config.FEATURE_COLUMNS
    assert pd.notna(frame.loc[0, "neighborhood"])


def test_predict_with_range_ordering(clean_df: pd.DataFrame, single_property: dict) -> None:
    result = train_pipeline(clean_df, save=False)
    estimate = predict_with_range(result["model"], single_property, result["metrics"])

    assert estimate["low"] <= estimate["point"] <= estimate["high"]
    assert estimate["point"] > 0
    assert estimate["half_width_pct"] > 0


def test_explain_prediction_returns_contributions(single_property: dict) -> None:
    model, _, _ = ensure_model()
    explanation = explain_prediction(model, single_property)

    # The committed artifact is XGBoost, so contributions are available.
    assert explanation is not None
    assert {"feature", "impact_pct", "log_contribution"}.issubset(explanation.columns)
    assert not explanation.empty
    assert explanation["feature"].is_unique
    assert explanation["impact_pct"].abs().max() > 0


def test_explain_prediction_none_without_booster(clean_df: pd.DataFrame, single_property: dict) -> None:
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import Pipeline

    pipeline = Pipeline([("preprocessor", build_preprocessor()), ("model", LinearRegression())])
    model = TransformedTargetRegressor(regressor=pipeline)
    model.fit(clean_df[config.FEATURE_COLUMNS], clean_df[config.TARGET_COLUMN])

    assert explain_prediction(model, single_property) is None
