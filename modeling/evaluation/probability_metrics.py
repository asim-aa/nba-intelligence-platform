"""Evaluate binary win-probability predictions consistently.
Every model in Phases 6-8 predicts the probability that the home team wins.
This module calculates the same classification, ranking, probability, and
calibration metrics for baselines, logistic regression, and CatBoost.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Final

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

DEFAULT_CALIBRATION_BINS: Final[int] = 10


@dataclass(frozen=True)
class ProbabilityMetrics:
    """Store the evaluation results for one model and dataset."""

    model_name: str
    split_name: str
    rows: int
    log_loss: float
    brier_score: float
    roc_auc: float
    accuracy: float
    expected_calibration_error: float
    mean_predicted_probability: float
    actual_home_win_rate: float


def validate_binary_targets(targets: pd.Series | np.ndarray) -> np.ndarray:
    """Return validated binary targets as a one-dimensional integer array."""

    values = np.asarray(targets)

    if values.ndim != 1:
        raise ValueError("Targets must be one-dimensional")

    if values.size == 0:
        raise ValueError("Targets cannot be empty")

    if not np.isin(values, [0, 1]).all():
        raise ValueError("Targets must contain only 0 and 1")

    return values.astype("int8")


def validate_probabilities(
    probabilities: pd.Series | np.ndarray,
    expected_rows: int,
) -> np.ndarray:
    """Return validated home-win probabilities as a float array."""

    values = np.asarray(probabilities, dtype="float64")

    if values.ndim != 1:
        raise ValueError("Probabilities must be one-dimensional")

    if len(values) != expected_rows:
        raise ValueError(f"Expected {expected_rows} probabilities, found {len(values)}")

    if not np.isfinite(values).all():
        raise ValueError("Probabilities must all be finite")

    if not ((values >= 0.0) & (values <= 1.0)).all():
        raise ValueError("Probabilities must remain within [0, 1]")

    return values


def build_calibration_table(
    targets: pd.Series | np.ndarray,
    probabilities: pd.Series | np.ndarray,
    bins: int = DEFAULT_CALIBRATION_BINS,
) -> pd.DataFrame:
    """Group predictions into probability bins for reliability analysis.

    Each row compares the average predicted probability with the observed
    home-win rate among games placed in that probability interval.
    """

    if bins < 2:
        raise ValueError("Calibration bins must be at least 2")

    y_true = validate_binary_targets(targets)
    y_prob = validate_probabilities(
        probabilities,
        expected_rows=len(y_true),
    )

    # Include probability 1.0 in the final bin instead of creating an
    # out-of-range index after multiplying by the number of bins.
    bin_indexes = np.minimum(
        (y_prob * bins).astype("int64"),
        bins - 1,
    )

    frame = pd.DataFrame(
        {
            "target": y_true,
            "probability": y_prob,
            "bin_index": bin_indexes,
        }
    )

    grouped = frame.groupby(
        "bin_index",
        sort=True,
        observed=True,
    )

    table = grouped.agg(
        rows=("target", "size"),
        mean_predicted_probability=("probability", "mean"),
        actual_home_win_rate=("target", "mean"),
    ).reset_index()

    table["bin_lower_bound"] = table["bin_index"] / bins
    table["bin_upper_bound"] = (table["bin_index"] + 1) / bins

    table["absolute_calibration_error"] = (
        table["mean_predicted_probability"] - table["actual_home_win_rate"]
    ).abs()

    return table[
        [
            "bin_index",
            "bin_lower_bound",
            "bin_upper_bound",
            "rows",
            "mean_predicted_probability",
            "actual_home_win_rate",
            "absolute_calibration_error",
        ]
    ]


def expected_calibration_error(
    targets: pd.Series | np.ndarray,
    probabilities: pd.Series | np.ndarray,
    bins: int = DEFAULT_CALIBRATION_BINS,
) -> float:
    """Calculate weighted expected calibration error.

    The error for each probability bin is weighted by that bin's share of all
    games. Lower values indicate that predicted probabilities more closely
    match observed outcome frequencies.
    """

    table = build_calibration_table(
        targets=targets,
        probabilities=probabilities,
        bins=bins,
    )

    weights = table["rows"] / table["rows"].sum()

    return float((weights * table["absolute_calibration_error"]).sum())


def evaluate_probabilities(
    *,
    model_name: str,
    split_name: str,
    targets: pd.Series | np.ndarray,
    probabilities: pd.Series | np.ndarray,
    threshold: float = 0.5,
    calibration_bins: int = DEFAULT_CALIBRATION_BINS,
) -> ProbabilityMetrics:
    """Calculate all project metrics for one probability vector."""

    if not model_name.strip():
        raise ValueError("model_name cannot be empty")

    if not split_name.strip():
        raise ValueError("split_name cannot be empty")

    if not 0.0 < threshold < 1.0:
        raise ValueError("Classification threshold must be within (0, 1)")

    y_true = validate_binary_targets(targets)
    y_prob = validate_probabilities(
        probabilities,
        expected_rows=len(y_true),
    )

    predicted_classes = (y_prob >= threshold).astype("int8")

    # Log loss is the primary ranking metric because incorrect confident
    # predictions should be penalized more heavily than uncertain mistakes.
    return ProbabilityMetrics(
        model_name=model_name,
        split_name=split_name,
        rows=len(y_true),
        log_loss=float(
            log_loss(
                y_true,
                y_prob,
                labels=[0, 1],
            )
        ),
        brier_score=float(
            brier_score_loss(
                y_true,
                y_prob,
            )
        ),
        roc_auc=float(
            roc_auc_score(
                y_true,
                y_prob,
            )
        ),
        accuracy=float(
            accuracy_score(
                y_true,
                predicted_classes,
            )
        ),
        expected_calibration_error=(
            expected_calibration_error(
                targets=y_true,
                probabilities=y_prob,
                bins=calibration_bins,
            )
        ),
        mean_predicted_probability=float(y_prob.mean()),
        actual_home_win_rate=float(y_true.mean()),
    )


def metrics_to_record(
    metrics: ProbabilityMetrics,
) -> dict[str, object]:
    """Convert metrics into a JSON-serializable dictionary."""

    return asdict(metrics)
