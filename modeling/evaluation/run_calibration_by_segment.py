"""Break the frozen model's held-out test performance down by segment.

Phase 8 (run_final_evaluation.py) reports one calibration number for the
entire test set. That hides whether the model is equally trustworthy in
every situation it gets used in -- a bare "0.0213 expected calibration
error" says nothing about whether predictions are just as reliable in
December as in April, or on a back-to-back as on full rest.

This does not re-evaluate anything: it refits logistic regression on the
training split exactly the way run_final_evaluation.py does (same
deterministic train_logistic_regression() call, same training data, no
retuning), predicts on the same held-out test set Phase 8 already scored
once, and slices those predictions into segments. It is the same one-time
test result, viewed at finer granularity, not a new look at the test set
that could inform any decision -- project_spec.md section 7 (rule 8) is
about not using test performance to pick between models or hyperparameters,
and nothing here does that.

Run with: uv run python -m modeling.evaluation.run_calibration_by_segment
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from modeling.data.split_dataset import test_output_path, train_output_path
from modeling.evaluation.probability_metrics import evaluate_probabilities, metrics_to_record
from modeling.training.train_logistic_regression import (
    predict_logistic_regression,
    train_logistic_regression,
)

MODEL_NAME: Final[str] = "logistic_regression"

SEASON_PHASE_BIN_EDGES: Final[tuple[float, ...]] = (-np.inf, 20.0, 50.0, np.inf)
SEASON_PHASE_LABELS: Final[tuple[str, ...]] = (
    "early (<=20 games played)",
    "mid (21-50 games played)",
    "late (51+ games played)",
)


@dataclass(frozen=True)
class SegmentResult:
    """One segment's slice of the frozen test-set predictions."""

    segment_dimension: str
    segment_label: str
    metrics: dict[str, object]


def calibration_by_segment_output_path(project_root: Path) -> Path:
    """Return the segment-breakdown summary path."""

    return (
        project_root
        / "artifacts"
        / "nba"
        / "calibration_by_segment"
        / "calibration_by_segment_summary.json"
    )


def season_phase_segment(dataset: pd.DataFrame) -> pd.Series:
    """Bucket games by how far into the season both teams already were.

    Uses the average of both teams' prior-games-played so a game counts as
    "early" only when neither team has much history yet.
    """

    average_games_played = (
        dataset["HOME_PRIOR_GAMES_PLAYED"] + dataset["AWAY_PRIOR_GAMES_PLAYED"]
    ) / 2

    return pd.cut(
        average_games_played,
        bins=SEASON_PHASE_BIN_EDGES,
        labels=SEASON_PHASE_LABELS,
    ).astype(str)


def back_to_back_segment(dataset: pd.DataFrame) -> pd.Series:
    """Bucket games by whether either team is playing on zero days' rest."""

    either_team_b2b = dataset["HOME_IS_BACK_TO_BACK"].eq(1) | dataset["AWAY_IS_BACK_TO_BACK"].eq(1)

    return either_team_b2b.map({True: "back-to-back (either team)", False: "rest (neither team)"})


SEGMENT_DIMENSIONS: Final[dict[str, object]] = {
    "season_phase": season_phase_segment,
    "back_to_back": back_to_back_segment,
}


def evaluate_segments(
    test: pd.DataFrame,
    probabilities: np.ndarray,
    dimension_name: str,
    segment_labels: pd.Series,
) -> list[SegmentResult]:
    """Score each segment of one dimension with the shared metrics function."""

    results: list[SegmentResult] = []

    for label in sorted(segment_labels.dropna().unique()):
        mask = segment_labels.eq(label)

        metrics = evaluate_probabilities(
            model_name=MODEL_NAME,
            split_name=label,
            targets=test.loc[mask, "home_win"],
            probabilities=probabilities[mask.to_numpy()],
        )

        results.append(
            SegmentResult(
                segment_dimension=dimension_name,
                segment_label=label,
                metrics=metrics_to_record(metrics),
            )
        )

    return results


def run_calibration_by_segment(project_root: Path) -> list[SegmentResult]:
    """Slice the frozen model's test-set predictions by pregame segment."""

    train_path = train_output_path(project_root)
    test_path = test_output_path(project_root)

    if not train_path.exists():
        raise FileNotFoundError(f"Training split does not exist: {train_path}")

    if not test_path.exists():
        raise FileNotFoundError(f"Test split does not exist: {test_path}")

    train = pd.read_parquet(train_path)
    test = pd.read_parquet(test_path)

    pipeline, _ = train_logistic_regression(train)
    probabilities = predict_logistic_regression(pipeline, test)

    results: list[SegmentResult] = []

    for dimension_name, segment_fn in SEGMENT_DIMENSIONS.items():
        segment_labels = segment_fn(test)
        results.extend(evaluate_segments(test, probabilities, dimension_name, segment_labels))

    return results


def write_summary(results: list[SegmentResult], project_root: Path) -> Path:
    """Persist the segment breakdown as JSON."""

    output_path = calibration_by_segment_output_path(project_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "note": (
            "Same frozen test-set predictions Phase 8 already evaluated "
            "once, sliced by segment -- not a new test-set evaluation."
        ),
        "segments": [asdict(result) for result in results],
    }

    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    return output_path


def print_report(results: list[SegmentResult]) -> None:
    """Print a readable per-segment calibration report."""

    print(f"\nCalibration by segment ({MODEL_NAME}, frozen test-set predictions):")
    header = f"{'dimension':<16}{'segment':<28}{'rows':>6}{'log_loss':>10}{'brier':>8}{'ece':>8}"
    print(header)

    for result in results:
        metrics = result.metrics
        print(
            f"{result.segment_dimension:<16}{result.segment_label:<28}{metrics['rows']:>6}"
            f"{metrics['log_loss']:>10.4f}{metrics['brier_score']:>8.4f}"
            f"{metrics['expected_calibration_error']:>8.4f}"
        )


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the segment breakdown."""

    return argparse.ArgumentParser(description=__doc__).parse_args()


def main() -> None:
    """Run the calibration-by-segment breakdown from the command line."""

    parse_args()
    project_root = Path(__file__).resolve().parents[2]

    results = run_calibration_by_segment(project_root)
    output_path = write_summary(results, project_root)

    print_report(results)
    print(f"\nSummary: {output_path}")


if __name__ == "__main__":
    main()
