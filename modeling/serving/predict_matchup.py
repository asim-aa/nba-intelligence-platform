"""Predict a home-win probability using the frozen, selected model.

The selected model is logistic regression (see
modeling/evaluation/run_final_evaluation.py and its SELECTED_MODEL
constant). This loads exactly the artifact that Phase 8 already
evaluated, not a freshly retrained copy, so a dashboard prediction is
provably the same model whose test performance is on record -- and reuses
predict_logistic_regression() rather than re-deriving how to call it.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from modeling.training.train_logistic_regression import (
    model_artifact_path,
    predict_logistic_regression,
)


def load_selected_model(project_root: Path):
    """Load the frozen logistic regression pipeline from disk."""

    path = model_artifact_path(project_root)

    if not path.exists():
        raise FileNotFoundError(
            f"Selected model artifact does not exist: {path}. "
            "Run `uv run python -m modeling.evaluation.run_model_comparison` first."
        )

    return joblib.load(path)


def predict_home_win_probability(model, feature_row: pd.DataFrame) -> float:
    """Return the frozen model's home-win probability for one matchup row."""

    if len(feature_row) != 1:
        raise ValueError(f"Expected exactly one matchup row, got {len(feature_row)}")

    probabilities = predict_logistic_regression(model, feature_row)

    return float(probabilities[0])
