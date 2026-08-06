"""Tests for the compact logistic regression training module."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from modeling.training.train_logistic_regression import (
    COMPACT_FEATURE_COLUMNS,
    model_artifact_path,
    predict_logistic_regression,
    train_logistic_regression,
    training_summary_path,
    write_logistic_regression_artifacts,
)


def make_training_frame(rows: int = 40) -> pd.DataFrame:
    """Build a synthetic training frame with the compact feature columns."""

    rng = np.random.default_rng(seed=0)

    win_pct_diff = rng.uniform(-0.4, 0.4, size=rows)
    rolling_win_pct_diff = rng.uniform(-0.4, 0.4, size=rows)
    point_diff_diff = rng.uniform(-10.0, 10.0, size=rows)
    rest_diff = rng.integers(-2, 3, size=rows).astype("float64")
    back_to_back_diff = rng.integers(-1, 2, size=rows).astype("float64")
    elo_diff = rng.uniform(-200.0, 200.0, size=rows)

    # A higher home win-pct differential should make a home win more likely,
    # which keeps the fitted coefficients sane without asserting exact values.
    home_win_probability = 1.0 / (1.0 + np.exp(-(win_pct_diff * 4.0)))
    home_win = rng.binomial(1, home_win_probability)

    return pd.DataFrame(
        {
            "SEASON_WIN_PCT_DIFF": win_pct_diff,
            "ROLLING_10_WIN_PCT_DIFF": rolling_win_pct_diff,
            "ROLLING_10_POINT_DIFFERENTIAL_DIFF": point_diff_diff,
            "DAYS_REST_DIFF": rest_diff,
            "IS_BACK_TO_BACK_DIFF": back_to_back_diff,
            "ELO_RATING_DIFF": elo_diff,
            "home_win": home_win,
        }
    )


def test_train_returns_fitted_pipeline_and_summary() -> None:
    train = make_training_frame()

    pipeline, summary = train_logistic_regression(train)

    assert summary.train_rows == len(train)
    assert summary.feature_columns == COMPACT_FEATURE_COLUMNS
    assert set(summary.coefficients) == set(COMPACT_FEATURE_COLUMNS)

    predictions = predict_logistic_regression(pipeline, train)

    assert len(predictions) == len(train)
    assert ((predictions >= 0.0) & (predictions <= 1.0)).all()


def test_train_rejects_missing_columns() -> None:
    incomplete = make_training_frame().drop(columns=["DAYS_REST_DIFF"])

    with pytest.raises(ValueError, match="missing required columns"):
        train_logistic_regression(incomplete)


def test_train_rejects_empty_data() -> None:
    empty = make_training_frame().iloc[0:0]

    with pytest.raises(ValueError, match="empty"):
        train_logistic_regression(empty)


def test_train_rejects_non_binary_target() -> None:
    invalid = make_training_frame()
    invalid.loc[0, "home_win"] = 2

    with pytest.raises(ValueError, match="0 and 1"):
        train_logistic_regression(invalid)


def test_predict_handles_missing_feature_values() -> None:
    train = make_training_frame()
    pipeline, _ = train_logistic_regression(train)

    dataset_with_gaps = make_training_frame(rows=5).drop(columns=["home_win"])
    dataset_with_gaps.loc[0, "SEASON_WIN_PCT_DIFF"] = np.nan

    predictions = predict_logistic_regression(pipeline, dataset_with_gaps)

    assert len(predictions) == 5
    assert np.isfinite(predictions).all()


def test_predict_rejects_missing_columns() -> None:
    train = make_training_frame()
    pipeline, _ = train_logistic_regression(train)

    with pytest.raises(ValueError, match="missing required columns"):
        predict_logistic_regression(pipeline, pd.DataFrame({"SEASON_WIN_PCT_DIFF": [0.1]}))


def test_predict_rejects_empty_dataset() -> None:
    train = make_training_frame()
    pipeline, _ = train_logistic_regression(train)

    empty = train.drop(columns=["home_win"]).iloc[0:0]

    with pytest.raises(ValueError, match="empty"):
        predict_logistic_regression(pipeline, empty)


def test_write_artifacts_persists_reloadable_model(tmp_path: Path) -> None:
    train = make_training_frame()
    pipeline, summary = train_logistic_regression(train)

    model_path, summary_path = write_logistic_regression_artifacts(
        pipeline=pipeline,
        summary=summary,
        project_root=tmp_path,
    )

    assert model_path == model_artifact_path(tmp_path)
    assert summary_path == training_summary_path(tmp_path)
    assert model_path.exists()
    assert summary_path.exists()

    reloaded_pipeline = joblib.load(model_path)
    original_predictions = predict_logistic_regression(pipeline, train)
    reloaded_predictions = predict_logistic_regression(reloaded_pipeline, train)

    np.testing.assert_array_equal(original_predictions, reloaded_predictions)
