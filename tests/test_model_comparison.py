"""Tests for the Phase 7 baseline vs. logistic regression vs. CatBoost comparison."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from modeling.data.split_dataset import train_output_path, validation_output_path
from modeling.evaluation.run_model_comparison import (
    comparison_summary_path,
    run_model_comparison,
)
from modeling.training.train_catboost import (
    model_artifact_path as catboost_model_artifact_path,
)
from modeling.training.train_catboost import (
    training_summary_path as catboost_training_summary_path,
)
from modeling.training.train_logistic_regression import (
    model_artifact_path as logistic_model_artifact_path,
)
from modeling.training.train_logistic_regression import (
    training_summary_path as logistic_training_summary_path,
)
from pipelines.features.build_modeling_dataset import (
    CATEGORICAL_FEATURE_COLUMNS,
    NUMERIC_FEATURE_COLUMNS,
)

TEAM_IDS = tuple(1610612737 + offset for offset in range(30))


def make_split_frame(rows: int, seed: int) -> pd.DataFrame:
    """Build a synthetic split covering the full feature manifest.

    HOME_SEASON_WIN_PCT/AWAY_SEASON_WIN_PCT and the compact logistic
    regression columns get realistic, informative values; the remaining
    numeric columns (needed only so CatBoost sees the full manifest) are
    generic noise.
    """

    rng = np.random.default_rng(seed=seed)

    win_pct_diff = rng.uniform(-0.4, 0.4, size=rows)
    home_win = rng.binomial(1, 1.0 / (1.0 + np.exp(-(win_pct_diff * 4.0))))
    # Guarantee both classes exist regardless of the random draw, since
    # ROC-AUC is undefined for a single-class target.
    home_win[0] = 1
    home_win[1] = 0

    data: dict[str, object] = {"home_win": home_win}

    for column in NUMERIC_FEATURE_COLUMNS:
        data[column] = rng.uniform(-1.0, 1.0, size=rows)

    data["HOME_SEASON_WIN_PCT"] = rng.uniform(0.2, 0.8, size=rows)
    data["AWAY_SEASON_WIN_PCT"] = rng.uniform(0.2, 0.8, size=rows)
    data["SEASON_WIN_PCT_DIFF"] = win_pct_diff
    data["ROLLING_10_WIN_PCT_DIFF"] = rng.uniform(-0.4, 0.4, size=rows)
    data["ROLLING_10_POINT_DIFFERENTIAL_DIFF"] = rng.uniform(-10.0, 10.0, size=rows)
    data["DAYS_REST_DIFF"] = rng.integers(-2, 3, size=rows).astype("float64")
    data["IS_BACK_TO_BACK_DIFF"] = rng.integers(-1, 2, size=rows).astype("float64")

    for column in CATEGORICAL_FEATURE_COLUMNS:
        data[column] = rng.choice(TEAM_IDS, size=rows)

    return pd.DataFrame(data)


def write_splits(project_root: Path) -> None:
    train_path = train_output_path(project_root)
    validation_path = validation_output_path(project_root)

    train_path.parent.mkdir(parents=True, exist_ok=True)

    make_split_frame(rows=60, seed=1).to_parquet(train_path, index=False)
    make_split_frame(rows=20, seed=2).to_parquet(validation_path, index=False)


def test_run_model_comparison_scores_four_models_on_two_splits(tmp_path: Path) -> None:
    write_splits(tmp_path)

    records = run_model_comparison(project_root=tmp_path)

    assert len(records) == 8

    model_names = {record["model_name"] for record in records}
    split_names = {record["split_name"] for record in records}

    assert model_names == {
        "always_home",
        "better_record",
        "logistic_regression",
        "catboost",
    }
    assert split_names == {"train", "validation"}

    for record in records:
        assert 0.0 <= record["accuracy"] <= 1.0
        assert record["log_loss"] >= 0.0
        assert record["brier_score"] >= 0.0


def test_run_model_comparison_never_creates_or_requires_a_test_split(tmp_path: Path) -> None:
    write_splits(tmp_path)
    test_path = tmp_path / "data" / "processed" / "nba" / "modeling" / "splits" / "test.parquet"

    run_model_comparison(project_root=tmp_path)

    assert not test_path.exists()


def test_run_model_comparison_writes_summary_and_model_artifacts(tmp_path: Path) -> None:
    write_splits(tmp_path)

    records = run_model_comparison(project_root=tmp_path)

    summary_path = comparison_summary_path(tmp_path)
    assert summary_path.exists()

    saved_records = pd.read_json(summary_path)
    assert len(saved_records) == len(records)

    assert logistic_model_artifact_path(tmp_path).exists()
    assert logistic_training_summary_path(tmp_path).exists()
    assert catboost_model_artifact_path(tmp_path).exists()
    assert catboost_training_summary_path(tmp_path).exists()


def test_run_model_comparison_requires_training_split(tmp_path: Path) -> None:
    validation_path = validation_output_path(tmp_path)
    validation_path.parent.mkdir(parents=True, exist_ok=True)
    make_split_frame(rows=20, seed=2).to_parquet(validation_path, index=False)

    with pytest.raises(FileNotFoundError, match="Training split"):
        run_model_comparison(project_root=tmp_path)


def test_run_model_comparison_requires_validation_split(tmp_path: Path) -> None:
    train_path = train_output_path(tmp_path)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    make_split_frame(rows=60, seed=1).to_parquet(train_path, index=False)

    with pytest.raises(FileNotFoundError, match="Validation split"):
        run_model_comparison(project_root=tmp_path)
