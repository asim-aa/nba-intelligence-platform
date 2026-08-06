"""Train a compact, interpretable logistic regression win-probability model.

Per project_spec.md section 5, every more complex model must be compared
against this one. The feature set intentionally excludes the categorical
team-ID columns and keeps only the 10-game rolling window to stay compact:
season-long form, recent form, recent point differential, rest advantage,
back-to-back status, and the opponent-adjusted Elo rating gap (see
build_team_elo_ratings.py). Missing values (a team's first games of a
season) are imputed with the training median inside the pipeline, so the
same fitted transform can be reapplied to validation and test data without
ever fitting on their distributions (project_spec.md section 7, rule 6).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

COMPACT_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "SEASON_WIN_PCT_DIFF",
    "ROLLING_10_WIN_PCT_DIFF",
    "ROLLING_10_POINT_DIFFERENTIAL_DIFF",
    "DAYS_REST_DIFF",
    "IS_BACK_TO_BACK_DIFF",
    "ELO_RATING_DIFF",
)

TARGET_COLUMN: Final[str] = "home_win"

RANDOM_STATE: Final[int] = 0


@dataclass(frozen=True)
class LogisticRegressionSummary:
    """Describe one fitted logistic regression run."""

    train_rows: int
    feature_columns: tuple[str, ...]
    coefficients: dict[str, float]
    intercept: float


def model_artifact_path(project_root: Path) -> Path:
    """Return the persisted pipeline path."""

    return project_root / "artifacts" / "nba" / "logistic_regression" / "model.joblib"


def training_summary_path(project_root: Path) -> Path:
    """Return the training summary path."""

    return project_root / "artifacts" / "nba" / "logistic_regression" / "training_summary.json"


def build_pipeline() -> Pipeline:
    """Construct the impute -> scale -> classify pipeline."""

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def train_logistic_regression(
    train: pd.DataFrame,
) -> tuple[Pipeline, LogisticRegressionSummary]:
    """Fit the pipeline on the training split only."""

    missing_columns = (set(COMPACT_FEATURE_COLUMNS) | {TARGET_COLUMN}) - set(train.columns)

    if missing_columns:
        raise ValueError(f"Training data is missing required columns: {sorted(missing_columns)}")

    if train.empty:
        raise ValueError("Training data cannot be empty")

    if not train[TARGET_COLUMN].isin([0, 1]).all():
        raise ValueError("home_win must contain only 0 and 1")

    features = train.loc[:, list(COMPACT_FEATURE_COLUMNS)]
    target = train[TARGET_COLUMN]

    pipeline = build_pipeline()
    pipeline.fit(features, target)

    classifier: LogisticRegression = pipeline.named_steps["classifier"]

    summary = LogisticRegressionSummary(
        train_rows=len(train),
        feature_columns=COMPACT_FEATURE_COLUMNS,
        coefficients=dict(
            zip(
                COMPACT_FEATURE_COLUMNS,
                classifier.coef_[0].tolist(),
                strict=True,
            )
        ),
        intercept=float(classifier.intercept_[0]),
    )

    return pipeline, summary


def predict_logistic_regression(
    pipeline: Pipeline,
    dataset: pd.DataFrame,
) -> np.ndarray:
    """Return home-win probabilities for dataset using a fitted pipeline."""

    missing_columns = set(COMPACT_FEATURE_COLUMNS) - set(dataset.columns)

    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing_columns)}")

    if dataset.empty:
        raise ValueError("Dataset cannot be empty")

    features = dataset.loc[:, list(COMPACT_FEATURE_COLUMNS)]

    probabilities = pipeline.predict_proba(features)[:, 1]

    return np.asarray(probabilities, dtype="float64")


def write_logistic_regression_artifacts(
    pipeline: Pipeline,
    summary: LogisticRegressionSummary,
    project_root: Path,
) -> tuple[Path, Path]:
    """Persist the fitted pipeline and its training summary."""

    model_path = model_artifact_path(project_root)
    summary_path = training_summary_path(project_root)

    model_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, model_path)

    summary_path.write_text(
        json.dumps(asdict(summary), indent=2) + "\n",
        encoding="utf-8",
    )

    return model_path, summary_path
