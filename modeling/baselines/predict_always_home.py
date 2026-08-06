"""Predict every game's home-win probability as the training-set base rate.

This is the probabilistic form of the simplest possible rule from
project_spec.md section 5: "always predicting the home team." Because NBA
home teams have historically won more than half their games, this constant
agrees with the classification rule "always predict the home team" at the
standard 0.50 threshold, while still producing a calibrated probability for
log loss, Brier score, and calibration scoring.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AlwaysHomeBaseline:
    """Store the single learned constant: the training-set home-win rate."""

    home_win_probability: float


def fit_always_home_baseline(train: pd.DataFrame) -> AlwaysHomeBaseline:
    """Learn the home-win rate from the training split only."""

    if "home_win" not in train.columns:
        raise ValueError("Training data must contain a home_win column")

    if train.empty:
        raise ValueError("Training data cannot be empty")

    if not train["home_win"].isin([0, 1]).all():
        raise ValueError("home_win must contain only 0 and 1")

    return AlwaysHomeBaseline(home_win_probability=float(train["home_win"].mean()))


def predict_always_home(
    baseline: AlwaysHomeBaseline,
    dataset: pd.DataFrame,
) -> np.ndarray:
    """Return the constant home-win probability for every row in dataset."""

    if dataset.empty:
        raise ValueError("Dataset cannot be empty")

    return np.full(
        shape=len(dataset),
        fill_value=baseline.home_win_probability,
        dtype="float64",
    )
