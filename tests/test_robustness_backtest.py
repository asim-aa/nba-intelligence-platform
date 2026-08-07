"""Tests for the walk-forward season-cutoff robustness backtest."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from modeling.data.split_dataset import modeling_dataset_input_path
from modeling.evaluation.run_robustness_backtest import (
    BacktestFoldResult,
    build_expanding_window_folds,
    ordered_seasons,
    run_robustness_backtest,
    summarize_metric_spread,
)
from pipelines.features.build_modeling_dataset import (
    CATEGORICAL_FEATURE_COLUMNS,
    NUMERIC_FEATURE_COLUMNS,
)

TEAM_IDS = tuple(1610612737 + offset for offset in range(30))

SEASONS = (
    "2019-20",
    "2020-21",
    "2021-22",
    "2022-23",
    "2023-24",
)


def make_season_rows(season: str, rows: int, seed: int) -> pd.DataFrame:
    """Build a synthetic single-season slice covering the full feature manifest."""

    rng = np.random.default_rng(seed=seed)

    win_pct_diff = rng.uniform(-0.4, 0.4, size=rows)
    home_win = rng.binomial(1, 1.0 / (1.0 + np.exp(-(win_pct_diff * 4.0))))
    home_win[0] = 1
    home_win[1] = 0

    data: dict[str, object] = {"SEASON": season, "home_win": home_win}

    for column in NUMERIC_FEATURE_COLUMNS:
        data[column] = rng.uniform(-1.0, 1.0, size=rows)

    data["SEASON_WIN_PCT_DIFF"] = win_pct_diff
    data["ROLLING_10_WIN_PCT_DIFF"] = rng.uniform(-0.4, 0.4, size=rows)
    data["ROLLING_10_POINT_DIFFERENTIAL_DIFF"] = rng.uniform(-10.0, 10.0, size=rows)
    data["DAYS_REST_DIFF"] = rng.integers(-2, 3, size=rows).astype("float64")
    data["IS_BACK_TO_BACK_DIFF"] = rng.integers(-1, 2, size=rows).astype("float64")
    data["ELO_RATING_DIFF"] = rng.uniform(-200.0, 200.0, size=rows)

    for column in CATEGORICAL_FEATURE_COLUMNS:
        data[column] = rng.choice(TEAM_IDS, size=rows)

    return pd.DataFrame(data)


def write_modeling_dataset(project_root: Path, seasons: tuple[str, ...] = SEASONS) -> None:
    frames = [make_season_rows(season, rows=40, seed=index) for index, season in enumerate(seasons)]
    dataset = pd.concat(frames, ignore_index=True)

    path = modeling_dataset_input_path(project_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(path, index=False)


# --- Fold construction -----------------------------------------------------


def test_ordered_seasons_sorts_chronologically() -> None:
    shuffled = pd.DataFrame({"SEASON": ["2021-22", "2019-20", "2020-21"]})

    assert ordered_seasons(shuffled) == ("2019-20", "2020-21", "2021-22")


def test_build_expanding_window_folds_respects_min_train_seasons() -> None:
    seasons = ("2019-20", "2020-21", "2021-22", "2022-23")

    folds = build_expanding_window_folds(seasons, min_train_seasons=2)

    assert folds == [
        (("2019-20", "2020-21"), "2021-22"),
        (("2019-20", "2020-21", "2021-22"), "2022-23"),
    ]


def test_build_expanding_window_folds_window_expands_each_step() -> None:
    seasons = ("2019-20", "2020-21", "2021-22", "2022-23")

    folds = build_expanding_window_folds(seasons, min_train_seasons=1)

    train_season_counts = [len(train_seasons) for train_seasons, _ in folds]

    assert train_season_counts == [1, 2, 3]


def test_build_expanding_window_folds_rejects_non_positive_minimum() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        build_expanding_window_folds(("2019-20", "2020-21"), min_train_seasons=0)


# --- Metric spread -----------------------------------------------------


def make_fold(log_loss: float) -> BacktestFoldResult:
    return BacktestFoldResult(
        test_season="2022-23",
        train_seasons=("2019-20",),
        train_rows=40,
        test_rows=40,
        metrics={
            "log_loss": log_loss,
            "brier_score": 0.2,
            "roc_auc": 0.7,
            "accuracy": 0.6,
            "expected_calibration_error": 0.02,
        },
    )


def test_summarize_metric_spread_computes_hand_computable_mean_and_std() -> None:
    folds = [make_fold(log_loss=0.4), make_fold(log_loss=0.6)]

    spread = summarize_metric_spread(folds)

    assert spread["log_loss"].mean == pytest.approx(0.5)
    assert spread["log_loss"].std == pytest.approx(0.1)
    assert spread["log_loss"].min == pytest.approx(0.4)
    assert spread["log_loss"].max == pytest.approx(0.6)


def test_summarize_metric_spread_zero_variance_for_identical_folds() -> None:
    folds = [make_fold(log_loss=0.5), make_fold(log_loss=0.5)]

    spread = summarize_metric_spread(folds)

    assert spread["log_loss"].std == pytest.approx(0.0)


# --- End-to-end -----------------------------------------------------


def test_run_robustness_backtest_produces_one_fold_per_valid_cutoff(tmp_path: Path) -> None:
    write_modeling_dataset(tmp_path)

    summary = run_robustness_backtest(tmp_path, min_train_seasons=3)

    # 5 seasons, min_train_seasons=3 -> cutoffs at index 3 and 4 -> 2 folds.
    assert len(summary.folds) == 2
    assert [fold.test_season for fold in summary.folds] == ["2022-23", "2023-24"]
    assert summary.folds[0].train_seasons == ("2019-20", "2020-21", "2021-22")
    assert summary.folds[1].train_seasons == ("2019-20", "2020-21", "2021-22", "2022-23")


def test_run_robustness_backtest_reports_spread_for_every_metric(tmp_path: Path) -> None:
    write_modeling_dataset(tmp_path)

    summary = run_robustness_backtest(tmp_path, min_train_seasons=3)

    assert set(summary.metric_spread) == {
        "log_loss",
        "brier_score",
        "roc_auc",
        "accuracy",
        "expected_calibration_error",
    }


def test_run_robustness_backtest_raises_when_too_few_seasons(tmp_path: Path) -> None:
    write_modeling_dataset(tmp_path, seasons=("2019-20", "2020-21"))

    with pytest.raises(ValueError, match="Not enough seasons"):
        run_robustness_backtest(tmp_path, min_train_seasons=3)


def test_run_robustness_backtest_raises_when_dataset_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        run_robustness_backtest(tmp_path)
