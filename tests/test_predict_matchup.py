"""Tests for the frozen-model prediction wrapper."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from modeling.serving.predict_matchup import load_selected_model, predict_home_win_probability
from modeling.training.train_logistic_regression import (
    COMPACT_FEATURE_COLUMNS,
    train_logistic_regression,
    write_logistic_regression_artifacts,
)


def make_training_frame(rows: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(seed=0)
    win_pct_diff = rng.uniform(-0.4, 0.4, size=rows)
    home_win = rng.binomial(1, 1.0 / (1.0 + np.exp(-(win_pct_diff * 4.0))))

    return pd.DataFrame(
        {
            "SEASON_WIN_PCT_DIFF": win_pct_diff,
            "ROLLING_10_WIN_PCT_DIFF": rng.uniform(-0.4, 0.4, size=rows),
            "ROLLING_10_POINT_DIFFERENTIAL_DIFF": rng.uniform(-10.0, 10.0, size=rows),
            "DAYS_REST_DIFF": rng.integers(-2, 3, size=rows).astype("float64"),
            "IS_BACK_TO_BACK_DIFF": rng.integers(-1, 2, size=rows).astype("float64"),
            "ELO_RATING_DIFF": rng.uniform(-200.0, 200.0, size=rows),
            "home_win": home_win,
        }
    )


def test_load_selected_model_rejects_missing_artifact(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Selected model artifact"):
        load_selected_model(tmp_path)


def test_load_and_predict_round_trip(tmp_path: Path) -> None:
    train = make_training_frame()
    pipeline, summary = train_logistic_regression(train)
    write_logistic_regression_artifacts(pipeline=pipeline, summary=summary, project_root=tmp_path)

    reloaded = load_selected_model(tmp_path)
    matchup_row = train.drop(columns=["home_win"]).iloc[[0]]

    probability = predict_home_win_probability(reloaded, matchup_row)

    assert 0.0 <= probability <= 1.0


def test_predict_home_win_probability_rejects_multiple_rows(tmp_path: Path) -> None:
    train = make_training_frame()
    pipeline, _ = train_logistic_regression(train)

    with pytest.raises(ValueError, match="exactly one"):
        predict_home_win_probability(pipeline, train.drop(columns=["home_win"]))


def test_predict_home_win_probability_matches_direct_call(tmp_path: Path) -> None:
    train = make_training_frame()
    pipeline, _ = train_logistic_regression(train)
    matchup_row = train.drop(columns=["home_win"]).iloc[[0]]

    wrapped = predict_home_win_probability(pipeline, matchup_row)
    direct = pipeline.predict_proba(matchup_row.loc[:, list(COMPACT_FEATURE_COLUMNS)])[0, 1]

    assert wrapped == pytest.approx(direct)
