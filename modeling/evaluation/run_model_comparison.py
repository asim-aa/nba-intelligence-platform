"""Compare the required Phase 7 baselines against logistic regression.

This orchestrator fits the always-home baseline and the logistic regression
model on the training split, then scores every model -- including the
fitting-free better-record rule -- on both the training and validation
splits using the shared probability_metrics module.

This script never reads test.parquet. Per project_spec.md section 7 (rule
8), the held-out test seasons are evaluated only after the modeling
approach is frozen, which is a later, deliberate step rather than part of
routine model comparison.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from modeling.baselines.predict_always_home import (
    fit_always_home_baseline,
    predict_always_home,
)
from modeling.baselines.predict_better_record import predict_better_record
from modeling.data.split_dataset import train_output_path, validation_output_path
from modeling.evaluation.probability_metrics import evaluate_probabilities, metrics_to_record
from modeling.training.train_logistic_regression import (
    predict_logistic_regression,
    train_logistic_regression,
    write_logistic_regression_artifacts,
)


def comparison_summary_path(project_root: Path) -> Path:
    """Return the model-comparison summary path."""

    return project_root / "artifacts" / "nba" / "model_comparison" / "comparison_summary.json"


def load_splits(project_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the training and validation splits. Test is never read here."""

    train_path = train_output_path(project_root)
    validation_path = validation_output_path(project_root)

    if not train_path.exists():
        raise FileNotFoundError(f"Training split does not exist: {train_path}")

    if not validation_path.exists():
        raise FileNotFoundError(f"Validation split does not exist: {validation_path}")

    return pd.read_parquet(train_path), pd.read_parquet(validation_path)


def print_comparison_table(records: list[dict[str, object]]) -> None:
    """Print a readable model x split metrics table."""

    header = (
        f"{'model':<22}{'split':<12}{'log_loss':>10}{'brier':>10}"
        f"{'roc_auc':>10}{'accuracy':>10}{'ece':>8}"
    )
    print(header)

    for record in records:
        print(
            f"{record['model_name']:<22}{record['split_name']:<12}"
            f"{record['log_loss']:>10.4f}{record['brier_score']:>10.4f}"
            f"{record['roc_auc']:>10.4f}{record['accuracy']:>10.4f}"
            f"{record['expected_calibration_error']:>8.4f}"
        )


def run_model_comparison(project_root: Path) -> list[dict[str, object]]:
    """Fit every model on train, then score all of them on train and validation."""

    train, validation = load_splits(project_root)

    always_home_baseline = fit_always_home_baseline(train)
    logistic_pipeline, logistic_summary = train_logistic_regression(train)

    write_logistic_regression_artifacts(
        pipeline=logistic_pipeline,
        summary=logistic_summary,
        project_root=project_root,
    )

    predictors: dict[str, Callable[[pd.DataFrame], np.ndarray]] = {
        "always_home": lambda df: predict_always_home(always_home_baseline, df),
        "better_record": predict_better_record,
        "logistic_regression": lambda df: predict_logistic_regression(logistic_pipeline, df),
    }

    splits = {"train": train, "validation": validation}

    records: list[dict[str, object]] = []

    for split_name, split_df in splits.items():
        for model_name, predict_fn in predictors.items():
            probabilities = predict_fn(split_df)

            metrics = evaluate_probabilities(
                model_name=model_name,
                split_name=split_name,
                targets=split_df["home_win"],
                probabilities=probabilities,
            )

            records.append(metrics_to_record(metrics))

    summary_path = comparison_summary_path(project_root)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(records, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nModel comparison complete (train and validation only; test untouched):")
    print_comparison_table(records)
    print(f"\nSummary: {summary_path}")

    return records


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the comparison run."""

    return argparse.ArgumentParser(description=__doc__).parse_args()


def main() -> None:
    """Run the Phase 7 baseline and logistic regression comparison."""

    parse_args()
    project_root = Path(__file__).resolve().parents[2]

    run_model_comparison(project_root=project_root)


if __name__ == "__main__":
    main()
