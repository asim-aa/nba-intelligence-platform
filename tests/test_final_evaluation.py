"""Tests for the Phase 8 one-time held-out test evaluation."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from modeling.data.split_dataset import test_output_path as split_test_output_path
from modeling.data.split_dataset import train_output_path, validation_output_path
from modeling.evaluation.run_final_evaluation import (
    SELECTED_MODEL,
    calibration_table_path,
    final_evaluation_summary_path,
    reliability_diagram_path,
    run_final_evaluation,
)
from pipelines.features.build_modeling_dataset import (
    CATEGORICAL_FEATURE_COLUMNS,
    NUMERIC_FEATURE_COLUMNS,
)

TEAM_IDS = tuple(1610612737 + offset for offset in range(30))


def make_split_frame(rows: int, seed: int) -> pd.DataFrame:
    """Build a synthetic split covering the full feature manifest."""

    rng = np.random.default_rng(seed=seed)

    win_pct_diff = rng.uniform(-0.4, 0.4, size=rows)
    home_win = rng.binomial(1, 1.0 / (1.0 + np.exp(-(win_pct_diff * 4.0))))
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
    data["ELO_RATING_DIFF"] = rng.uniform(-200.0, 200.0, size=rows)

    for column in CATEGORICAL_FEATURE_COLUMNS:
        data[column] = rng.choice(TEAM_IDS, size=rows)

    return pd.DataFrame(data)


def write_splits(project_root: Path) -> None:
    train_path = train_output_path(project_root)
    validation_path = validation_output_path(project_root)
    test_path = split_test_output_path(project_root)

    train_path.parent.mkdir(parents=True, exist_ok=True)

    make_split_frame(rows=60, seed=1).to_parquet(train_path, index=False)
    make_split_frame(rows=20, seed=2).to_parquet(validation_path, index=False)
    make_split_frame(rows=20, seed=3).to_parquet(test_path, index=False)


def test_run_final_evaluation_scores_four_models_on_three_splits(tmp_path: Path) -> None:
    write_splits(tmp_path)

    summary = run_final_evaluation(project_root=tmp_path)

    records = summary["all_model_all_split_metrics"]
    assert len(records) == 12

    model_names = {record["model_name"] for record in records}
    split_names = {record["split_name"] for record in records}

    assert model_names == {
        "always_home",
        "better_record",
        "logistic_regression",
        "catboost",
    }
    assert split_names == {"train", "validation", "test"}


def test_run_final_evaluation_marks_logistic_regression_as_selected(tmp_path: Path) -> None:
    write_splits(tmp_path)

    summary = run_final_evaluation(project_root=tmp_path)

    assert summary["selected_model"] == "logistic_regression" == SELECTED_MODEL
    assert summary["selected_model_test_metrics"]["model_name"] == "logistic_regression"
    assert summary["selected_model_test_metrics"]["split_name"] == "test"
    assert isinstance(summary["selection_basis"], str)
    assert len(summary["selection_basis"]) > 0


def test_run_final_evaluation_writes_all_artifacts(tmp_path: Path) -> None:
    write_splits(tmp_path)

    run_final_evaluation(project_root=tmp_path)

    summary_path = final_evaluation_summary_path(tmp_path)
    calibration_path = calibration_table_path(tmp_path)
    diagram_path = reliability_diagram_path(tmp_path)

    assert summary_path.exists()
    assert calibration_path.exists()
    assert diagram_path.exists()
    assert diagram_path.stat().st_size > 0

    saved_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert saved_summary["selected_model"] == "logistic_regression"

    calibration_table = json.loads(calibration_path.read_text(encoding="utf-8"))
    assert len(calibration_table) > 0
    assert "mean_predicted_probability" in calibration_table[0]
    assert "actual_home_win_rate" in calibration_table[0]


def test_run_final_evaluation_requires_training_split(tmp_path: Path) -> None:
    validation_path = validation_output_path(tmp_path)
    test_path = split_test_output_path(tmp_path)
    validation_path.parent.mkdir(parents=True, exist_ok=True)
    make_split_frame(rows=20, seed=2).to_parquet(validation_path, index=False)
    make_split_frame(rows=20, seed=3).to_parquet(test_path, index=False)

    with pytest.raises(FileNotFoundError, match="Training split"):
        run_final_evaluation(project_root=tmp_path)


def test_run_final_evaluation_requires_validation_split(tmp_path: Path) -> None:
    train_path = train_output_path(tmp_path)
    test_path = split_test_output_path(tmp_path)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    make_split_frame(rows=60, seed=1).to_parquet(train_path, index=False)
    make_split_frame(rows=20, seed=3).to_parquet(test_path, index=False)

    with pytest.raises(FileNotFoundError, match="Validation split"):
        run_final_evaluation(project_root=tmp_path)


def test_run_final_evaluation_requires_test_split(tmp_path: Path) -> None:
    train_path = train_output_path(tmp_path)
    validation_path = validation_output_path(tmp_path)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    make_split_frame(rows=60, seed=1).to_parquet(train_path, index=False)
    make_split_frame(rows=20, seed=2).to_parquet(validation_path, index=False)

    with pytest.raises(FileNotFoundError, match="Test split"):
        run_final_evaluation(project_root=tmp_path)
