"""Train the "more complex model" from project_spec.md section 5.

Unlike the compact logistic regression, CatBoost uses the full feature
manifest: all 41 numeric features plus the two team-ID columns as native
categoricals (no one-hot encoding needed). CatBoost also handles missing
numeric values natively -- a team's first games of a season keep their
NaN history features instead of being imputed. The validation split is
used only as an early-stopping eval_set (selecting the number of trees is
model/hyperparameter selection, which project_spec.md section 6 explicitly
allows on the validation season).

Per section 5, this model is only worth adopting over logistic regression
if it improves out-of-time probability performance while staying
calibrated -- that comparison happens in run_model_comparison, not here.

The first version of this model used CatBoost's untuned defaults (depth 6,
l2_leaf_reg 3) and showed a much wider train/validation log-loss gap than
logistic regression, a sign of overfitting even with early stopping. The
hyperparameters below (shallower trees, stronger L2, a slower learning
rate, and row/column subsampling) were chosen via a validation-only search
across several regularization strengths -- selecting on validation log
loss and Brier score, per project_spec.md section 6 -- and both narrowed
that gap by roughly a third and improved every primary validation metric
over the untuned defaults.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from pipelines.features.build_modeling_dataset import (
    CATEGORICAL_FEATURE_COLUMNS,
    NUMERIC_FEATURE_COLUMNS,
)

FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    *CATEGORICAL_FEATURE_COLUMNS,
    *NUMERIC_FEATURE_COLUMNS,
)

TARGET_COLUMN: Final[str] = "home_win"

RANDOM_SEED: Final[int] = 0
ITERATIONS: Final[int] = 2000
EARLY_STOPPING_ROUNDS: Final[int] = 50

# Regularization settings chosen by a validation-only search over depth,
# l2_leaf_reg, learning_rate, and subsampling strength (see module
# docstring). Shallower trees and stronger L2 curb overfitting directly;
# rsm/bagging_temperature add row and column subsampling for the same
# reason random forests subsample.
DEPTH: Final[int] = 4
L2_LEAF_REG: Final[float] = 15.0
LEARNING_RATE: Final[float] = 0.03
RSM: Final[float] = 0.8
BAGGING_TEMPERATURE: Final[float] = 1.0


@dataclass(frozen=True)
class CatBoostSummary:
    """Describe one fitted CatBoost run."""

    train_rows: int
    validation_rows: int
    feature_columns: tuple[str, ...]
    best_iteration: int
    best_validation_log_loss: float
    feature_importances: dict[str, float]


def model_artifact_path(project_root: Path) -> Path:
    """Return the persisted CatBoost model path."""

    return project_root / "artifacts" / "nba" / "catboost" / "model.cbm"


def training_summary_path(project_root: Path) -> Path:
    """Return the training summary path."""

    return project_root / "artifacts" / "nba" / "catboost" / "training_summary.json"


def build_model() -> CatBoostClassifier:
    """Construct the CatBoost classifier with early-stopping enabled."""

    return CatBoostClassifier(
        iterations=ITERATIONS,
        depth=DEPTH,
        l2_leaf_reg=L2_LEAF_REG,
        learning_rate=LEARNING_RATE,
        rsm=RSM,
        bagging_temperature=BAGGING_TEMPERATURE,
        loss_function="Logloss",
        eval_metric="Logloss",
        random_seed=RANDOM_SEED,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        use_best_model=True,
        allow_writing_files=False,
        verbose=False,
    )


def validate_columns(dataset: pd.DataFrame, *, require_target: bool) -> None:
    """Verify a dataset has the columns CatBoost training or scoring needs."""

    required = set(FEATURE_COLUMNS)

    if require_target:
        required = required | {TARGET_COLUMN}

    missing_columns = required - set(dataset.columns)

    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing_columns)}")

    if dataset.empty:
        raise ValueError("Dataset cannot be empty")


def train_catboost(
    train: pd.DataFrame,
    validation: pd.DataFrame,
) -> tuple[CatBoostClassifier, CatBoostSummary]:
    """Fit CatBoost on train, early-stopping against the validation split."""

    validate_columns(train, require_target=True)
    validate_columns(validation, require_target=True)

    if not train[TARGET_COLUMN].isin([0, 1]).all():
        raise ValueError("home_win must contain only 0 and 1")

    train_features = train.loc[:, list(FEATURE_COLUMNS)]
    train_target = train[TARGET_COLUMN]

    validation_features = validation.loc[:, list(FEATURE_COLUMNS)]
    validation_target = validation[TARGET_COLUMN]

    model = build_model()

    model.fit(
        train_features,
        train_target,
        cat_features=list(CATEGORICAL_FEATURE_COLUMNS),
        eval_set=(validation_features, validation_target),
    )

    best_scores = model.get_best_score()

    summary = CatBoostSummary(
        train_rows=len(train),
        validation_rows=len(validation),
        feature_columns=FEATURE_COLUMNS,
        best_iteration=int(model.get_best_iteration()),
        best_validation_log_loss=float(best_scores["validation"]["Logloss"]),
        feature_importances=dict(
            zip(
                model.feature_names_,
                np.asarray(model.get_feature_importance()).tolist(),
                strict=True,
            )
        ),
    )

    return model, summary


def predict_catboost(
    model: CatBoostClassifier,
    dataset: pd.DataFrame,
) -> np.ndarray:
    """Return home-win probabilities for dataset using a fitted model."""

    validate_columns(dataset, require_target=False)

    features = dataset.loc[:, list(FEATURE_COLUMNS)]

    probabilities = model.predict_proba(features)[:, 1]

    return np.asarray(probabilities, dtype="float64")


def write_catboost_artifacts(
    model: CatBoostClassifier,
    summary: CatBoostSummary,
    project_root: Path,
) -> tuple[Path, Path]:
    """Persist the fitted model and its training summary."""

    model_path = model_artifact_path(project_root)
    summary_path = training_summary_path(project_root)

    model_path.parent.mkdir(parents=True, exist_ok=True)

    model.save_model(str(model_path))

    summary_path.write_text(
        json.dumps(asdict(summary), indent=2) + "\n",
        encoding="utf-8",
    )

    return model_path, summary_path
