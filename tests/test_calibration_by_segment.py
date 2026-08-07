"""Tests for the frozen model's calibration-by-segment breakdown."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from modeling.data.split_dataset import test_output_path as split_test_output_path
from modeling.data.split_dataset import train_output_path
from modeling.evaluation.run_calibration_by_segment import (
    back_to_back_segment,
    run_calibration_by_segment,
    season_phase_segment,
)
from pipelines.features.build_modeling_dataset import (
    CATEGORICAL_FEATURE_COLUMNS,
    NUMERIC_FEATURE_COLUMNS,
)

TEAM_IDS = tuple(1610612737 + offset for offset in range(30))


def make_split_frame(
    rows: int,
    seed: int,
    home_prior_games: np.ndarray,
    away_prior_games: np.ndarray,
    home_b2b: np.ndarray,
    away_b2b: np.ndarray,
) -> pd.DataFrame:
    """Build a synthetic split covering the full feature manifest, with
    controllable segment-defining columns so segment membership is
    hand-verifiable.
    """

    rng = np.random.default_rng(seed=seed)

    win_pct_diff = rng.uniform(-0.4, 0.4, size=rows)
    home_win = rng.binomial(1, 1.0 / (1.0 + np.exp(-(win_pct_diff * 4.0))))
    home_win[0] = 1
    home_win[1] = 0

    data: dict[str, object] = {"home_win": home_win}

    for column in NUMERIC_FEATURE_COLUMNS:
        data[column] = rng.uniform(-1.0, 1.0, size=rows)

    data["SEASON_WIN_PCT_DIFF"] = win_pct_diff
    data["ROLLING_10_WIN_PCT_DIFF"] = rng.uniform(-0.4, 0.4, size=rows)
    data["ROLLING_10_POINT_DIFFERENTIAL_DIFF"] = rng.uniform(-10.0, 10.0, size=rows)
    data["DAYS_REST_DIFF"] = rng.integers(-2, 3, size=rows).astype("float64")
    data["IS_BACK_TO_BACK_DIFF"] = rng.integers(-1, 2, size=rows).astype("float64")
    data["ELO_RATING_DIFF"] = rng.uniform(-200.0, 200.0, size=rows)

    data["HOME_PRIOR_GAMES_PLAYED"] = home_prior_games
    data["AWAY_PRIOR_GAMES_PLAYED"] = away_prior_games
    data["HOME_IS_BACK_TO_BACK"] = home_b2b
    data["AWAY_IS_BACK_TO_BACK"] = away_b2b

    for column in CATEGORICAL_FEATURE_COLUMNS:
        data[column] = rng.choice(TEAM_IDS, size=rows)

    return pd.DataFrame(data)


def write_splits(project_root: Path, test: pd.DataFrame, train_rows: int = 60) -> None:
    train = make_split_frame(
        rows=train_rows,
        seed=1,
        home_prior_games=np.full(train_rows, 30.0),
        away_prior_games=np.full(train_rows, 30.0),
        home_b2b=np.zeros(train_rows, dtype="int64"),
        away_b2b=np.zeros(train_rows, dtype="int64"),
    )

    train_path = train_output_path(project_root)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    train.to_parquet(train_path, index=False)
    test.to_parquet(split_test_output_path(project_root), index=False)


# --- Segment definitions -----------------------------------------------------


def test_season_phase_segment_buckets_by_average_games_played() -> None:
    dataset = pd.DataFrame(
        {
            "HOME_PRIOR_GAMES_PLAYED": [5.0, 30.0, 70.0],
            "AWAY_PRIOR_GAMES_PLAYED": [5.0, 30.0, 70.0],
        }
    )

    labels = season_phase_segment(dataset)

    assert labels.tolist() == [
        "early (<=20 games played)",
        "mid (21-50 games played)",
        "late (51+ games played)",
    ]


def test_back_to_back_segment_flags_either_team() -> None:
    dataset = pd.DataFrame(
        {
            "HOME_IS_BACK_TO_BACK": [1, 0, 0],
            "AWAY_IS_BACK_TO_BACK": [0, 1, 0],
        }
    )

    labels = back_to_back_segment(dataset)

    assert labels.tolist() == [
        "back-to-back (either team)",
        "back-to-back (either team)",
        "rest (neither team)",
    ]


# --- End-to-end -----------------------------------------------------


def test_run_calibration_by_segment_covers_both_dimensions(tmp_path: Path) -> None:
    rows = 80
    test = make_split_frame(
        rows=rows,
        seed=2,
        home_prior_games=np.where(np.arange(rows) < 40, 10.0, 60.0),
        away_prior_games=np.where(np.arange(rows) < 40, 10.0, 60.0),
        home_b2b=np.where(np.arange(rows) % 2 == 0, 1, 0),
        away_b2b=np.zeros(rows, dtype="int64"),
    )
    write_splits(tmp_path, test)

    results = run_calibration_by_segment(tmp_path)

    dimensions = {result.segment_dimension for result in results}
    assert dimensions == {"season_phase", "back_to_back"}

    season_labels = {r.segment_label for r in results if r.segment_dimension == "season_phase"}
    assert season_labels == {"early (<=20 games played)", "late (51+ games played)"}

    b2b_labels = {r.segment_label for r in results if r.segment_dimension == "back_to_back"}
    assert b2b_labels == {"back-to-back (either team)", "rest (neither team)"}


def test_run_calibration_by_segment_row_counts_match_segment_membership(tmp_path: Path) -> None:
    rows = 80
    test = make_split_frame(
        rows=rows,
        seed=3,
        home_prior_games=np.where(np.arange(rows) < 40, 10.0, 60.0),
        away_prior_games=np.where(np.arange(rows) < 40, 10.0, 60.0),
        home_b2b=np.where(np.arange(rows) % 2 == 0, 1, 0),
        away_b2b=np.zeros(rows, dtype="int64"),
    )
    write_splits(tmp_path, test)

    results = run_calibration_by_segment(tmp_path)

    by_label = {(r.segment_dimension, r.segment_label): r.metrics["rows"] for r in results}

    assert by_label[("season_phase", "early (<=20 games played)")] == 40
    assert by_label[("season_phase", "late (51+ games played)")] == 40
    assert by_label[("back_to_back", "back-to-back (either team)")] == 40
    assert by_label[("back_to_back", "rest (neither team)")] == 40


def test_run_calibration_by_segment_raises_when_splits_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        run_calibration_by_segment(tmp_path)
