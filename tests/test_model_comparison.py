"""Tests for the Phase 7 baseline vs. logistic regression comparison."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from modeling.data.split_dataset import train_output_path, validation_output_path
from modeling.evaluation.run_model_comparison import (
    comparison_summary_path,
    run_model_comparison,
)
from modeling.training.train_logistic_regression import (
    model_artifact_path,
    training_summary_path,
)


def make_split_frame(rows: int, seed: int) -> pd.DataFrame:
    """Build a synthetic split with every column the comparison run needs."""

    rng = np.random.default_rng(seed=seed)

    win_pct_diff = rng.uniform(-0.4, 0.4, size=rows)
    home_win = rng.binomial(1, 1.0 / (1.0 + np.exp(-(win_pct_diff * 4.0))))
    # Guarantee both classes exist regardless of the random draw, since
    # ROC-AUC is undefined for a single-class target.
    home_win[0] = 1
    home_win[1] = 0

    return pd.DataFrame(
        {
            "home_win": home_win,
            "HOME_SEASON_WIN_PCT": rng.uniform(0.2, 0.8, size=rows),
            "AWAY_SEASON_WIN_PCT": rng.uniform(0.2, 0.8, size=rows),
            "SEASON_WIN_PCT_DIFF": win_pct_diff,
            "ROLLING_10_WIN_PCT_DIFF": rng.uniform(-0.4, 0.4, size=rows),
            "ROLLING_10_POINT_DIFFERENTIAL_DIFF": rng.uniform(-10.0, 10.0, size=rows),
            "DAYS_REST_DIFF": rng.integers(-2, 3, size=rows).astype("float64"),
            "IS_BACK_TO_BACK_DIFF": rng.integers(-1, 2, size=rows).astype("float64"),
        }
    )


def write_splits(project_root: Path) -> None:
    train_path = train_output_path(project_root)
    validation_path = validation_output_path(project_root)

    train_path.parent.mkdir(parents=True, exist_ok=True)

    make_split_frame(rows=30, seed=1).to_parquet(train_path, index=False)
    make_split_frame(rows=10, seed=2).to_parquet(validation_path, index=False)


def test_run_model_comparison_scores_three_models_on_two_splits(tmp_path: Path) -> None:
    write_splits(tmp_path)

    records = run_model_comparison(project_root=tmp_path)

    assert len(records) == 6

    model_names = {record["model_name"] for record in records}
    split_names = {record["split_name"] for record in records}

    assert model_names == {"always_home", "better_record", "logistic_regression"}
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

    assert model_artifact_path(tmp_path).exists()
    assert training_summary_path(tmp_path).exists()


def test_run_model_comparison_requires_training_split(tmp_path: Path) -> None:
    validation_path = validation_output_path(tmp_path)
    validation_path.parent.mkdir(parents=True, exist_ok=True)
    make_split_frame(rows=10, seed=2).to_parquet(validation_path, index=False)

    with pytest.raises(FileNotFoundError, match="Training split"):
        run_model_comparison(project_root=tmp_path)


def test_run_model_comparison_requires_validation_split(tmp_path: Path) -> None:
    train_path = train_output_path(tmp_path)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    make_split_frame(rows=30, seed=1).to_parquet(train_path, index=False)

    with pytest.raises(FileNotFoundError, match="Validation split"):
        run_model_comparison(project_root=tmp_path)
