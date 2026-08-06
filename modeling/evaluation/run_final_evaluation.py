"""Run the Phase 8 final evaluation: the one-time held-out test scoring.

Per project_spec.md section 7 (rule 8), the held-out test seasons (2024-25,
2025-26) may be evaluated only after the modeling approach is frozen. Model
selection already happened on validation (see run_model_comparison.py and
its git history): logistic regression, trained on the compact feature set
including the Elo rating differential, beat both required baselines and
CatBoost on every validation metric (log loss, Brier score, ROC-AUC,
accuracy, calibration error), with no sign of overfitting. That decision
is final and does not change based on anything this script reports.

This script fits every Phase 7 candidate on the training split only --
exactly how each one was validated, so the model tested here is the model
that was actually selected, not a retrained variant -- and evaluates all
of them on train, validation, and test. Reporting every candidate's test
performance is transparent context, not a re-opened competition:
logistic_regression is the selected model, and only its result is the
answer to the prediction problem.

Run this once. After it runs, test performance is known and can no longer
inform any modeling decision without violating the evaluation design.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Final

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from modeling.baselines.predict_always_home import (
    fit_always_home_baseline,
    predict_always_home,
)
from modeling.baselines.predict_better_record import predict_better_record
from modeling.data.split_dataset import (
    test_output_path,
    train_output_path,
    validation_output_path,
)
from modeling.evaluation.probability_metrics import (
    build_calibration_table,
    evaluate_probabilities,
    metrics_to_record,
)
from modeling.training.train_catboost import predict_catboost, train_catboost
from modeling.training.train_logistic_regression import (
    predict_logistic_regression,
    train_logistic_regression,
)

SELECTED_MODEL: Final[str] = "logistic_regression"

SELECTION_BASIS: Final[str] = (
    "Selected on 2023-24 validation performance, before the test seasons "
    "were read. Beat both required baselines and CatBoost on every metric "
    "(log loss, Brier score, ROC-AUC, accuracy, calibration error) with no "
    "sign of overfitting. See run_model_comparison.py and its git history "
    "for the full validation comparison that produced this decision."
)


def final_evaluation_directory(project_root: Path) -> Path:
    """Return the directory holding every Phase 8 artifact."""

    return project_root / "artifacts" / "nba" / "final_evaluation"


def final_evaluation_summary_path(project_root: Path) -> Path:
    """Return the Phase 8 summary path."""

    return final_evaluation_directory(project_root) / "final_evaluation_summary.json"


def calibration_table_path(project_root: Path) -> Path:
    """Return the selected model's test-set calibration table path."""

    return final_evaluation_directory(project_root) / "selected_model_test_calibration.json"


def reliability_diagram_path(project_root: Path) -> Path:
    """Return the selected model's test-set reliability diagram path."""

    return final_evaluation_directory(project_root) / "selected_model_test_reliability.png"


def load_splits(project_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train, validation, and test.

    This is the only script in the project that reads the test split. It
    should stay that way.
    """

    train_path = train_output_path(project_root)
    validation_path = validation_output_path(project_root)
    test_path = test_output_path(project_root)

    if not train_path.exists():
        raise FileNotFoundError(f"Training split does not exist: {train_path}")

    if not validation_path.exists():
        raise FileNotFoundError(f"Validation split does not exist: {validation_path}")

    if not test_path.exists():
        raise FileNotFoundError(f"Test split does not exist: {test_path}")

    return (
        pd.read_parquet(train_path),
        pd.read_parquet(validation_path),
        pd.read_parquet(test_path),
    )


def render_reliability_diagram(
    calibration_table: pd.DataFrame,
    output_path: Path,
) -> None:
    """Save a reliability diagram comparing predicted vs. actual win rates."""

    figure, axis = plt.subplots(figsize=(6, 6))

    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    axis.plot(
        calibration_table["mean_predicted_probability"],
        calibration_table["actual_home_win_rate"],
        marker="o",
        color="tab:blue",
        label=f"{SELECTED_MODEL} (test)",
    )

    axis.set_xlabel("Mean predicted home-win probability")
    axis.set_ylabel("Actual home-win rate")
    axis.set_title("Reliability diagram: frozen model on the held-out test set")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.legend()
    figure.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


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


def run_final_evaluation(project_root: Path) -> dict[str, object]:
    """Fit every Phase 7 candidate on train, then score all of them on
    train, validation, and the held-out test set -- once.
    """

    train, validation, test = load_splits(project_root)

    always_home_baseline = fit_always_home_baseline(train)
    logistic_pipeline, _ = train_logistic_regression(train)
    catboost_model, _ = train_catboost(train, validation)

    predictors: dict[str, Callable[[pd.DataFrame], np.ndarray]] = {
        "always_home": lambda df: predict_always_home(always_home_baseline, df),
        "better_record": predict_better_record,
        "logistic_regression": lambda df: predict_logistic_regression(logistic_pipeline, df),
        "catboost": lambda df: predict_catboost(catboost_model, df),
    }

    splits = {"train": train, "validation": validation, "test": test}

    records: list[dict[str, object]] = []
    test_probabilities_by_model: dict[str, np.ndarray] = {}

    for split_name, split_df in splits.items():
        for model_name, predict_fn in predictors.items():
            probabilities = predict_fn(split_df)

            if split_name == "test":
                test_probabilities_by_model[model_name] = probabilities

            metrics = evaluate_probabilities(
                model_name=model_name,
                split_name=split_name,
                targets=split_df["home_win"],
                probabilities=probabilities,
            )
            records.append(metrics_to_record(metrics))

    calibration_table = build_calibration_table(
        targets=test["home_win"],
        probabilities=test_probabilities_by_model[SELECTED_MODEL],
    )

    render_reliability_diagram(
        calibration_table=calibration_table,
        output_path=reliability_diagram_path(project_root),
    )

    calibration_path = calibration_table_path(project_root)
    calibration_path.parent.mkdir(parents=True, exist_ok=True)
    calibration_path.write_text(
        calibration_table.to_json(orient="records", indent=2) + "\n",
        encoding="utf-8",
    )

    selected_model_test_metrics = next(
        record
        for record in records
        if record["model_name"] == SELECTED_MODEL and record["split_name"] == "test"
    )

    summary: dict[str, object] = {
        "selected_model": SELECTED_MODEL,
        "selection_basis": SELECTION_BASIS,
        "selected_model_test_metrics": selected_model_test_metrics,
        "all_model_all_split_metrics": records,
    }

    summary_path = final_evaluation_summary_path(project_root)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nPhase 8 final evaluation complete.")
    print(f"Selected model (decided on validation, before test was read): {SELECTED_MODEL}")
    print(f"\n{SELECTED_MODEL} test metrics:")
    print(json.dumps(selected_model_test_metrics, indent=2))
    print("\nFull comparison across every model and split (context only):")
    print_comparison_table(records)
    print(f"\nSummary: {summary_path}")
    print(f"Calibration table: {calibration_path}")
    print(f"Reliability diagram: {reliability_diagram_path(project_root)}")

    return summary


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the final evaluation run."""

    return argparse.ArgumentParser(description=__doc__).parse_args()


def main() -> None:
    """Run the Phase 8 final evaluation."""

    parse_args()
    project_root = Path(__file__).resolve().parents[2]

    run_final_evaluation(project_root=project_root)


if __name__ == "__main__":
    main()
