"""Predict home-win probability from each team's pregame season record.

This implements project_spec.md section 5's "rule based on the better
pregame season record" using the Bradley-Terry pairwise comparison formula:

    P(home win) = home_win_pct / (home_win_pct + away_win_pct)

It is a fixed formula, not a fitted model, so there is no training step.
Missing win percentages (a team's first game of a season) are treated as a
league-average 0.5, and the rare case where both teams are still winless
falls back to an uninformative 0.5 probability instead of dividing by zero.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd

REQUIRED_COLUMNS: Final[frozenset[str]] = frozenset({"HOME_SEASON_WIN_PCT", "AWAY_SEASON_WIN_PCT"})


def predict_better_record(dataset: pd.DataFrame) -> np.ndarray:
    """Return a Bradley-Terry home-win probability for every row."""

    missing_columns = REQUIRED_COLUMNS - set(dataset.columns)

    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing_columns)}")

    if dataset.empty:
        raise ValueError("Dataset cannot be empty")

    home_pct = dataset["HOME_SEASON_WIN_PCT"].fillna(0.5).to_numpy(dtype="float64")
    away_pct = dataset["AWAY_SEASON_WIN_PCT"].fillna(0.5).to_numpy(dtype="float64")

    combined_strength = home_pct + away_pct
    both_winless = combined_strength == 0.0

    # Substitute a safe denominator where both teams are winless so no
    # division-by-zero warning fires; the outer np.where discards that branch.
    safe_denominator = np.where(both_winless, 1.0, combined_strength)

    return np.where(both_winless, 0.5, home_pct / safe_denominator)
