"""Tests for shared NBA probability-evaluation metrics."""

import numpy as np
import pandas as pd
import pytest
from modeling.evaluation.probability_metrics import (
    build_calibration_table,
    evaluate_probabilities,
    expected_calibration_error,
    metrics_to_record,
    validate_binary_targets,
    validate_probabilities,
)


def test_validate_binary_targets_accepts_binary_values() -> None:
    """Binary targets should be returned as an integer array."""

    result = validate_binary_targets(pd.Series([0, 1, 1, 0]))

    assert result.tolist() == [0, 1, 1, 0]
    assert result.dtype == np.int8


def test_validate_binary_targets_rejects_invalid_values() -> None:
    """Targets outside zero and one should fail."""

    with pytest.raises(
        ValueError,
        match="only 0 and 1",
    ):
        validate_binary_targets(pd.Series([0, 1, 2]))


def test_validate_probabilities_rejects_out_of_range_values() -> None:
    """Probabilities must remain between zero and one."""

    with pytest.raises(
        ValueError,
        match=r"within \[0, 1\]",
    ):
        validate_probabilities(
            probabilities=np.array([0.2, 1.1]),
            expected_rows=2,
        )


def test_calibration_table_preserves_all_rows() -> None:
    """Calibration bins should contain every evaluated game exactly once."""

    targets = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.2, 0.8, 0.9])

    table = build_calibration_table(
        targets=targets,
        probabilities=probabilities,
        bins=5,
    )

    assert table["rows"].sum() == 4
    assert table["bin_index"].tolist() == [0, 1, 4]


def test_expected_calibration_error_is_zero_when_bins_are_perfect() -> None:
    """Perfectly aligned bin frequencies should have zero calibration error."""

    targets = np.array([0, 0, 1, 1])
    probabilities = np.array([0.0, 0.0, 1.0, 1.0])

    result = expected_calibration_error(
        targets=targets,
        probabilities=probabilities,
        bins=2,
    )

    assert result == pytest.approx(0.0)


def test_evaluate_probabilities_returns_expected_perfect_metrics() -> None:
    """Perfect probability separation should give ideal classification metrics."""

    targets = np.array([0, 0, 1, 1])
    probabilities = np.array([0.0, 0.1, 0.9, 1.0])

    metrics = evaluate_probabilities(
        model_name="perfect-example",
        split_name="validation",
        targets=targets,
        probabilities=probabilities,
    )

    assert metrics.rows == 4
    assert metrics.roc_auc == pytest.approx(1.0)
    assert metrics.accuracy == pytest.approx(1.0)
    assert metrics.log_loss < 0.06
    assert metrics.brier_score < 0.01


def test_evaluate_probabilities_scores_constant_predictions() -> None:
    """A constant probability should still produce valid baseline metrics."""

    targets = np.array([0, 1, 1, 0])
    probabilities = np.full(
        shape=4,
        fill_value=0.5,
    )

    metrics = evaluate_probabilities(
        model_name="constant",
        split_name="validation",
        targets=targets,
        probabilities=probabilities,
    )

    assert metrics.mean_predicted_probability == pytest.approx(0.5)
    assert metrics.actual_home_win_rate == pytest.approx(0.5)
    assert metrics.log_loss == pytest.approx(np.log(2))
    assert metrics.brier_score == pytest.approx(0.25)
    assert metrics.roc_auc == pytest.approx(0.5)
    assert metrics.accuracy == pytest.approx(0.5)


def test_metrics_to_record_returns_serializable_values() -> None:
    """Metric dataclasses should convert into plain record dictionaries."""

    metrics = evaluate_probabilities(
        model_name="example",
        split_name="validation",
        targets=np.array([0, 1]),
        probabilities=np.array([0.25, 0.75]),
    )

    record = metrics_to_record(metrics)

    assert record["model_name"] == "example"
    assert record["split_name"] == "validation"
    assert record["rows"] == 2
    assert isinstance(record["log_loss"], float)
