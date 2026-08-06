"""Tests for the Bradley-Terry better-record baseline."""

import numpy as np
import pandas as pd
import pytest
from modeling.baselines.predict_better_record import predict_better_record


def test_stronger_home_record_yields_higher_home_probability() -> None:
    dataset = pd.DataFrame(
        {
            "HOME_SEASON_WIN_PCT": [0.6],
            "AWAY_SEASON_WIN_PCT": [0.4],
        }
    )

    predictions = predict_better_record(dataset)

    assert predictions[0] == pytest.approx(0.6)


def test_equal_records_yield_a_coin_flip() -> None:
    dataset = pd.DataFrame(
        {
            "HOME_SEASON_WIN_PCT": [0.5],
            "AWAY_SEASON_WIN_PCT": [0.5],
        }
    )

    predictions = predict_better_record(dataset)

    assert predictions[0] == pytest.approx(0.5)


def test_missing_win_percentages_are_treated_as_average() -> None:
    dataset = pd.DataFrame(
        {
            "HOME_SEASON_WIN_PCT": [np.nan],
            "AWAY_SEASON_WIN_PCT": [np.nan],
        }
    )

    predictions = predict_better_record(dataset)

    assert predictions[0] == pytest.approx(0.5)


def test_both_teams_winless_falls_back_to_coin_flip() -> None:
    dataset = pd.DataFrame(
        {
            "HOME_SEASON_WIN_PCT": [0.0],
            "AWAY_SEASON_WIN_PCT": [0.0],
        }
    )

    predictions = predict_better_record(dataset)

    assert predictions[0] == pytest.approx(0.5)
    assert np.isfinite(predictions).all()


def test_rejects_missing_required_columns() -> None:
    with pytest.raises(ValueError, match="missing required columns"):
        predict_better_record(pd.DataFrame({"HOME_SEASON_WIN_PCT": [0.5]}))


def test_rejects_empty_dataset() -> None:
    dataset = pd.DataFrame(
        {
            "HOME_SEASON_WIN_PCT": pd.Series(dtype="float64"),
            "AWAY_SEASON_WIN_PCT": pd.Series(dtype="float64"),
        }
    )

    with pytest.raises(ValueError, match="empty"):
        predict_better_record(dataset)


def test_all_predictions_remain_valid_probabilities() -> None:
    dataset = pd.DataFrame(
        {
            "HOME_SEASON_WIN_PCT": [0.0, 1.0, 0.5, np.nan],
            "AWAY_SEASON_WIN_PCT": [0.0, 0.0, 0.5, 0.3],
        }
    )

    predictions = predict_better_record(dataset)

    assert ((predictions >= 0.0) & (predictions <= 1.0)).all()
    assert np.isfinite(predictions).all()
