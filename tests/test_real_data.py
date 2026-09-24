"""Tests for the real-data (Ames Housing) benchmark helpers."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline

from property_insights import config, real_data


def test_load_real_data_schema() -> None:
    frame = real_data.load_real_data()
    assert set(config.RAW_COLUMNS).issubset(frame.columns)
    assert len(frame) > 1_000
    assert (frame["price"] > 0).all()
    assert frame["neighborhood"].nunique() > 5


def test_load_real_data_missing_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        real_data.load_real_data(tmp_path / "does_not_exist.csv")


def test_cross_validate_metrics_runs() -> None:
    frame = real_data.load_real_data().sample(300, random_state=1)
    X = frame[config.FEATURE_COLUMNS]
    y = frame[config.TARGET_COLUMN]
    metrics = real_data.cross_validate_metrics(
        Pipeline([("model", DummyRegressor(strategy="median"))]), X, y, cv=3
    )
    assert set(metrics.index) == set(real_data.METRIC_NAMES)
    assert np.isfinite(metrics["mean"]).all()


def test_benchmark_returns_expected_keys() -> None:
    frame = real_data.load_real_data().sample(400, random_state=1)
    result = real_data.benchmark(frame, cv=3, tune_iter=2)
    for key in ["baseline", "model", "tuned", "best_params", "importance", "n_records"]:
        assert key in result
    assert result["n_records"] == len(frame)
    assert not result["importance"].empty
