"""Tests for the always-home baseline."""

import pandas as pd
import pytest
from modeling.baselines.predict_always_home import (
    fit_always_home_baseline,
    predict_always_home,
)


def test_fit_learns_training_home_win_rate() -> None:
    train = pd.DataFrame({"home_win": [1, 1, 1, 0]})

    baseline = fit_always_home_baseline(train)

    assert baseline.home_win_probability == pytest.approx(0.75)


def test_fit_rejects_missing_home_win_column() -> None:
    with pytest.raises(ValueError, match="home_win"):
        fit_always_home_baseline(pd.DataFrame({"other": [1, 2]}))


def test_fit_rejects_empty_training_data() -> None:
    with pytest.raises(ValueError, match="empty"):
        fit_always_home_baseline(pd.DataFrame({"home_win": []}))


def test_fit_rejects_non_binary_targets() -> None:
    with pytest.raises(ValueError, match="0 and 1"):
        fit_always_home_baseline(pd.DataFrame({"home_win": [1, 2]}))


def test_predict_returns_constant_probability_for_every_row() -> None:
    train = pd.DataFrame({"home_win": [1, 1, 0]})
    baseline = fit_always_home_baseline(train)
    dataset = pd.DataFrame({"home_win": [1, 0, 1, 0, 1]})

    predictions = predict_always_home(baseline, dataset)

    assert len(predictions) == len(dataset)
    assert (predictions == baseline.home_win_probability).all()


def test_predict_rejects_empty_dataset() -> None:
    baseline = fit_always_home_baseline(pd.DataFrame({"home_win": [1, 0]}))

    with pytest.raises(ValueError, match="empty"):
        predict_always_home(baseline, pd.DataFrame({"home_win": []}))
