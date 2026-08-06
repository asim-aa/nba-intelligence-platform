"""Tests for the CatBoost training module."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from catboost import CatBoostClassifier
from modeling.training.train_catboost import (
    FEATURE_COLUMNS,
    model_artifact_path,
    predict_catboost,
    train_catboost,
    training_summary_path,
    write_catboost_artifacts,
)
from pipelines.features.build_modeling_dataset import (
    CATEGORICAL_FEATURE_COLUMNS,
    NUMERIC_FEATURE_COLUMNS,
)

TEAM_IDS = tuple(1610612737 + offset for offset in range(30))


def make_frame(rows: int, seed: int) -> pd.DataFrame:
    """Build a synthetic frame covering the full feature manifest."""

    rng = np.random.default_rng(seed=seed)

    win_pct_diff = rng.uniform(-0.4, 0.4, size=rows)
    home_win = rng.binomial(1, 1.0 / (1.0 + np.exp(-(win_pct_diff * 4.0))))
    home_win[0] = 1
    home_win[1] = 0

    data: dict[str, object] = {"home_win": home_win}

    for column in NUMERIC_FEATURE_COLUMNS:
        data[column] = rng.uniform(-1.0, 1.0, size=rows)

    # Give the model at least one genuinely informative numeric feature.
    data["SEASON_WIN_PCT_DIFF"] = win_pct_diff

    for column in CATEGORICAL_FEATURE_COLUMNS:
        data[column] = rng.choice(TEAM_IDS, size=rows)

    return pd.DataFrame(data)


def test_train_returns_fitted_model_and_summary() -> None:
    train = make_frame(rows=60, seed=1)
    validation = make_frame(rows=20, seed=2)

    model, summary = train_catboost(train, validation)

    assert isinstance(model, CatBoostClassifier)
    assert summary.train_rows == len(train)
    assert summary.validation_rows == len(validation)
    assert summary.feature_columns == FEATURE_COLUMNS
    assert summary.best_iteration >= 0
    assert set(summary.feature_importances) == set(FEATURE_COLUMNS)

    predictions = predict_catboost(model, train)

    assert len(predictions) == len(train)
    assert ((predictions >= 0.0) & (predictions <= 1.0)).all()


def test_train_rejects_missing_columns() -> None:
    train = make_frame(rows=60, seed=1).drop(columns=["DAYS_REST_DIFF"])
    validation = make_frame(rows=20, seed=2)

    with pytest.raises(ValueError, match="missing required columns"):
        train_catboost(train, validation)


def test_train_rejects_empty_training_data() -> None:
    train = make_frame(rows=60, seed=1).iloc[0:0]
    validation = make_frame(rows=20, seed=2)

    with pytest.raises(ValueError, match="empty"):
        train_catboost(train, validation)


def test_train_rejects_non_binary_target() -> None:
    train = make_frame(rows=60, seed=1)
    train.loc[0, "home_win"] = 2
    validation = make_frame(rows=20, seed=2)

    with pytest.raises(ValueError, match="0 and 1"):
        train_catboost(train, validation)


def test_predict_handles_missing_numeric_values_natively() -> None:
    train = make_frame(rows=60, seed=1)
    validation = make_frame(rows=20, seed=2)
    model, _ = train_catboost(train, validation)

    dataset_with_gaps = make_frame(rows=5, seed=3).drop(columns=["home_win"])
    dataset_with_gaps.loc[0, "SEASON_WIN_PCT_DIFF"] = np.nan

    predictions = predict_catboost(model, dataset_with_gaps)

    assert len(predictions) == 5
    assert np.isfinite(predictions).all()


def test_predict_rejects_missing_columns() -> None:
    train = make_frame(rows=60, seed=1)
    validation = make_frame(rows=20, seed=2)
    model, _ = train_catboost(train, validation)

    with pytest.raises(ValueError, match="missing required columns"):
        predict_catboost(model, pd.DataFrame({"SEASON_WIN_PCT_DIFF": [0.1]}))


def test_predict_rejects_empty_dataset() -> None:
    train = make_frame(rows=60, seed=1)
    validation = make_frame(rows=20, seed=2)
    model, _ = train_catboost(train, validation)

    empty = train.drop(columns=["home_win"]).iloc[0:0]

    with pytest.raises(ValueError, match="empty"):
        predict_catboost(model, empty)


def test_write_artifacts_persists_reloadable_model(tmp_path: Path) -> None:
    train = make_frame(rows=60, seed=1)
    validation = make_frame(rows=20, seed=2)
    model, summary = train_catboost(train, validation)

    model_path, summary_path = write_catboost_artifacts(
        model=model,
        summary=summary,
        project_root=tmp_path,
    )

    assert model_path == model_artifact_path(tmp_path)
    assert summary_path == training_summary_path(tmp_path)
    assert model_path.exists()
    assert summary_path.exists()
    assert not (tmp_path / "catboost_info").exists()

    reloaded_model = CatBoostClassifier()
    reloaded_model.load_model(str(model_path))

    original_predictions = predict_catboost(model, train)
    reloaded_predictions = predict_catboost(reloaded_model, train)

    np.testing.assert_allclose(original_predictions, reloaded_predictions)
