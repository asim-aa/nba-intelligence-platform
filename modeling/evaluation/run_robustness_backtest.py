"""Walk-forward backtest: how much do the frozen model's headline metrics
move under a different season cutoff?

Phase 8 (run_final_evaluation.py) reports one number for each metric, from
one specific train/validation/test boundary. That number is real, but it
answers a narrower question than it looks like it answers: it says nothing
about how much log loss, Brier score, ROC-AUC, accuracy, or calibration
error would have come out if the test seasons had been chosen differently.

This script answers that question without touching or reopening Phase 8.
It refits the exact same model -- logistic regression on the exact same
compact feature set, via the exact same train_logistic_regression(), with
no hyperparameter search of any kind -- across an expanding window of
season cutoffs, and reports the mean and standard deviation of each metric
across those cutoffs. Per project_spec.md section 7 (rule 8), the official
test seasons (2024-25, 2025-26) are evaluated only once to select a model;
nothing here informs any model or hyperparameter choice, so it does not
violate that rule -- it characterizes uncertainty around an already-final
result, the same way a confidence interval would, rather than searching
for a better one. The Phase 8 numbers in README.md remain the answer to
"how good is the selected model"; this script's numbers answer "how
sensitive is that answer to which seasons happened to be the test set."

Run with: uv run python -m modeling.evaluation.run_robustness_backtest
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from modeling.data.split_dataset import modeling_dataset_input_path
from modeling.evaluation.probability_metrics import evaluate_probabilities, metrics_to_record
from modeling.training.train_logistic_regression import (
    predict_logistic_regression,
    train_logistic_regression,
)

# Folds with too little training history behave erratically (an Elo/rolling
# feature set barely has anything to condition on yet), so the walk stops
# below this many training seasons rather than starting from the first
# season in the dataset.
MIN_TRAIN_SEASONS: Final[int] = 3

METRIC_NAMES: Final[tuple[str, ...]] = (
    "log_loss",
    "brier_score",
    "roc_auc",
    "accuracy",
    "expected_calibration_error",
)


@dataclass(frozen=True)
class BacktestFoldResult:
    """One expanding-window fold: train on everything before, test on one season."""

    test_season: str
    train_seasons: tuple[str, ...]
    train_rows: int
    test_rows: int
    metrics: dict[str, object]


@dataclass(frozen=True)
class MetricSpread:
    """Mean, standard deviation, min, and max of one metric across folds."""

    mean: float
    std: float
    min: float
    max: float


@dataclass(frozen=True)
class RobustnessBacktestSummary:
    """The full walk-forward backtest result."""

    folds: list[BacktestFoldResult]
    metric_spread: dict[str, MetricSpread]
    note: str


def robustness_backtest_output_path(project_root: Path) -> Path:
    """Return the backtest summary path."""

    return project_root / "artifacts" / "nba" / "robustness_backtest" / "backtest_summary.json"


def ordered_seasons(dataset: pd.DataFrame) -> tuple[str, ...]:
    """Return every season in the dataset, chronologically sorted.

    Season labels ("2015-16", "2016-17", ...) are fixed-width and
    zero-padded, so lexicographic sorting is already chronological.
    """

    return tuple(sorted(dataset["SEASON"].unique()))


def build_expanding_window_folds(
    seasons: tuple[str, ...],
    min_train_seasons: int = MIN_TRAIN_SEASONS,
) -> list[tuple[tuple[str, ...], str]]:
    """Return (train_seasons, test_season) pairs for each walk-forward cutoff.

    Each fold trains on every season strictly before the cutoff and tests
    on the cutoff season alone -- the same chronological discipline as
    split_dataset.py, just repeated at every valid cutoff instead of one.
    """

    if min_train_seasons < 1:
        raise ValueError("min_train_seasons must be at least 1")

    folds: list[tuple[tuple[str, ...], str]] = []

    for index in range(min_train_seasons, len(seasons)):
        train_seasons = seasons[:index]
        test_season = seasons[index]
        folds.append((train_seasons, test_season))

    return folds


def run_fold(
    dataset: pd.DataFrame, train_seasons: tuple[str, ...], test_season: str
) -> BacktestFoldResult:
    """Fit on train_seasons and evaluate on test_season, once."""

    train = dataset.loc[dataset["SEASON"].isin(train_seasons)].reset_index(drop=True)
    test = dataset.loc[dataset["SEASON"].eq(test_season)].reset_index(drop=True)

    if train.empty:
        raise ValueError(f"No training rows for seasons {train_seasons}")

    if test.empty:
        raise ValueError(f"No test rows for season {test_season}")

    pipeline, _ = train_logistic_regression(train)
    probabilities = predict_logistic_regression(pipeline, test)

    metrics = evaluate_probabilities(
        model_name="logistic_regression",
        split_name=test_season,
        targets=test["home_win"],
        probabilities=probabilities,
    )

    return BacktestFoldResult(
        test_season=test_season,
        train_seasons=train_seasons,
        train_rows=len(train),
        test_rows=len(test),
        metrics=metrics_to_record(metrics),
    )


def summarize_metric_spread(folds: list[BacktestFoldResult]) -> dict[str, MetricSpread]:
    """Compute mean/std/min/max for each metric across every fold."""

    spread: dict[str, MetricSpread] = {}

    for metric_name in METRIC_NAMES:
        values = np.array([fold.metrics[metric_name] for fold in folds], dtype="float64")

        spread[metric_name] = MetricSpread(
            mean=float(values.mean()),
            std=float(values.std(ddof=0)),
            min=float(values.min()),
            max=float(values.max()),
        )

    return spread


def run_robustness_backtest(
    project_root: Path,
    min_train_seasons: int = MIN_TRAIN_SEASONS,
) -> RobustnessBacktestSummary:
    """Walk logistic regression forward across every valid season cutoff."""

    input_path = modeling_dataset_input_path(project_root)

    if not input_path.exists():
        raise FileNotFoundError(f"Modeling dataset does not exist: {input_path}")

    dataset = pd.read_parquet(input_path)
    seasons = ordered_seasons(dataset)
    fold_configs = build_expanding_window_folds(seasons, min_train_seasons=min_train_seasons)

    if not fold_configs:
        raise ValueError(
            f"Not enough seasons ({len(seasons)}) for min_train_seasons={min_train_seasons}"
        )

    folds = [
        run_fold(dataset, train_seasons=train_seasons, test_season=test_season)
        for train_seasons, test_season in fold_configs
    ]

    summary = RobustnessBacktestSummary(
        folds=folds,
        metric_spread=summarize_metric_spread(folds),
        note=(
            "Supplementary diagnostic only. Does not alter, retune, or "
            "supersede the frozen Phase 8 selected model or its official "
            "one-time test result -- see run_final_evaluation.py."
        ),
    )

    return summary


def write_summary(summary: RobustnessBacktestSummary, project_root: Path) -> Path:
    """Persist the backtest summary as JSON."""

    output_path = robustness_backtest_output_path(project_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "note": summary.note,
        "metric_spread": {name: asdict(spread) for name, spread in summary.metric_spread.items()},
        "folds": [asdict(fold) for fold in summary.folds],
    }

    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    return output_path


def print_report(summary: RobustnessBacktestSummary) -> None:
    """Print a readable walk-forward report."""

    print(f"\nWalk-forward robustness backtest across {len(summary.folds)} season cutoffs:")
    print(
        f"{'test_season':<14}{'log_loss':>10}{'brier':>10}{'roc_auc':>10}{'accuracy':>10}{'ece':>8}"
    )

    for fold in summary.folds:
        metrics = fold.metrics
        print(
            f"{fold.test_season:<14}{metrics['log_loss']:>10.4f}{metrics['brier_score']:>10.4f}"
            f"{metrics['roc_auc']:>10.4f}{metrics['accuracy']:>10.4f}"
            f"{metrics['expected_calibration_error']:>8.4f}"
        )

    print("\nMetric spread across cutoffs (mean ± std, [min, max]):")
    for metric_name in METRIC_NAMES:
        spread = summary.metric_spread[metric_name]
        print(
            f"  {metric_name:<28} {spread.mean:.4f} ± {spread.std:.4f}  "
            f"[{spread.min:.4f}, {spread.max:.4f}]"
        )

    print(f"\n{summary.note}")


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the robustness backtest."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--min-train-seasons",
        type=int,
        default=MIN_TRAIN_SEASONS,
        help="Minimum training seasons before the walk-forward window starts.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the walk-forward robustness backtest from the command line."""

    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]

    summary = run_robustness_backtest(project_root, min_train_seasons=args.min_train_seasons)
    output_path = write_summary(summary, project_root)

    print_report(summary)
    print(f"\nSummary: {output_path}")


if __name__ == "__main__":
    main()
