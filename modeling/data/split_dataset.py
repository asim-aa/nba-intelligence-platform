"""Create chronological train, validation, and test NBA datasets.

The modeling dataset spans the full target acquisition window from
docs/project_spec.md §3. We split by complete season instead of randomly
shuffling games, matching the provisional split from project_spec.md §6:

    Train:      2015-16 through 2022-23
    Validation: 2023-24
    Test:       2024-25 through 2025-26

The validation season is used for model and hyperparameter selection. The
test seasons remain untouched until Phase 8's final evaluation.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

import pandas as pd

TRAIN_SEASONS: Final[tuple[str, ...]] = (
    "2015-16",
    "2016-17",
    "2017-18",
    "2018-19",
    "2019-20",
    "2020-21",
    "2021-22",
    "2022-23",
)

VALIDATION_SEASONS: Final[tuple[str, ...]] = ("2023-24",)

TEST_SEASONS: Final[tuple[str, ...]] = ("2024-25", "2025-26")

REQUIRED_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "SEASON",
        "GAME_ID",
        "GAME_DATE",
        "HOME_TEAM_ID",
        "AWAY_TEAM_ID",
        "home_win",
    }
)


@dataclass(frozen=True)
class DatasetPartitionSummary:
    """Describe one chronological modeling partition."""

    name: str
    seasons: tuple[str, ...]
    rows: int
    first_game_date: str
    last_game_date: str
    home_wins: int
    away_wins: int
    home_win_rate: float


@dataclass(frozen=True)
class ChronologicalSplitSummary:
    """Describe the complete train/validation/test split."""

    source_rows: int
    train: DatasetPartitionSummary
    validation: DatasetPartitionSummary
    test: DatasetPartitionSummary


def modeling_dataset_input_path(project_root: Path) -> Path:
    """Return the Phase 5 modeling-dataset path."""

    return (
        project_root
        / "data"
        / "processed"
        / "nba"
        / "modeling"
        / "pregame_modeling_dataset.parquet"
    )


def split_output_directory(project_root: Path) -> Path:
    """Return the directory containing chronological split artifacts."""

    return project_root / "data" / "processed" / "nba" / "modeling" / "splits"


def train_output_path(project_root: Path) -> Path:
    """Return the training-set Parquet path."""

    return split_output_directory(project_root) / "train.parquet"


def validation_output_path(project_root: Path) -> Path:
    """Return the validation-set Parquet path."""

    return split_output_directory(project_root) / "validation.parquet"


def test_output_path(project_root: Path) -> Path:
    """Return the held-out test-set Parquet path."""

    return split_output_directory(project_root) / "test.parquet"


def split_summary_output_path(project_root: Path) -> Path:
    """Return the split metadata path."""

    return split_output_directory(project_root) / "split_summary.json"


def validate_source_dataset(dataset: pd.DataFrame) -> None:
    """Validate the Phase 5 dataset before chronological splitting."""

    missing_columns = REQUIRED_COLUMNS - set(dataset.columns)

    if missing_columns:
        raise ValueError(f"Modeling dataset is missing required columns: {sorted(missing_columns)}")

    if dataset.empty:
        raise ValueError("Modeling dataset cannot be empty")

    duplicate_games = int(
        dataset.duplicated(
            subset=["SEASON", "GAME_ID"],
        ).sum()
    )

    if duplicate_games:
        raise ValueError(f"Modeling dataset contains {duplicate_games} duplicate games")

    if dataset["GAME_DATE"].isna().any():
        raise ValueError("Modeling dataset contains missing GAME_DATE values")

    if not dataset["home_win"].isin([0, 1]).all():
        raise ValueError("Modeling dataset contains home_win values outside {0, 1}")


def validate_split_configuration() -> None:
    """Ensure no season belongs to more than one partition."""

    train_set = set(TRAIN_SEASONS)
    validation_set = set(VALIDATION_SEASONS)
    test_set = set(TEST_SEASONS)

    if train_set & validation_set:
        raise ValueError("Train and validation seasons overlap")

    if train_set & test_set:
        raise ValueError("Train and test seasons overlap")

    if validation_set & test_set:
        raise ValueError("Validation and test seasons overlap")


def build_partition_summary(
    name: str,
    seasons: tuple[str, ...],
    partition: pd.DataFrame,
) -> DatasetPartitionSummary:
    """Summarize one chronological dataset partition."""

    home_wins = int(partition["home_win"].sum())
    away_wins = int(len(partition) - home_wins)

    return DatasetPartitionSummary(
        name=name,
        seasons=seasons,
        rows=len(partition),
        first_game_date=(partition["GAME_DATE"].min().date().isoformat()),
        last_game_date=(partition["GAME_DATE"].max().date().isoformat()),
        home_wins=home_wins,
        away_wins=away_wins,
        home_win_rate=float(partition["home_win"].mean()),
    )


def validate_partition(
    partition: pd.DataFrame,
    expected_seasons: tuple[str, ...],
    name: str,
) -> None:
    """Validate one partition's rows and season membership."""

    if partition.empty:
        raise ValueError(f"{name} partition cannot be empty")

    actual_seasons = set(partition["SEASON"].unique())
    expected_season_set = set(expected_seasons)

    if actual_seasons != expected_season_set:
        raise ValueError(
            f"{name} seasons do not match configuration: "
            f"expected={sorted(expected_season_set)}, "
            f"actual={sorted(actual_seasons)}"
        )

    if not partition["GAME_DATE"].is_monotonic_increasing:
        raise ValueError(f"{name} partition is not sorted chronologically")


def split_dataset(
    dataset: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    ChronologicalSplitSummary,
]:
    """Split the modeling dataset into complete chronological seasons."""

    validate_source_dataset(dataset)
    validate_split_configuration()

    working = dataset.copy()

    working["GAME_DATE"] = pd.to_datetime(
        working["GAME_DATE"],
        errors="raise",
    )

    working = working.sort_values(
        ["GAME_DATE", "GAME_ID"],
        kind="stable",
    ).reset_index(drop=True)

    configured_seasons = {
        *TRAIN_SEASONS,
        *VALIDATION_SEASONS,
        *TEST_SEASONS,
    }

    actual_seasons = set(working["SEASON"].unique())

    missing_seasons = configured_seasons - actual_seasons
    unexpected_seasons = actual_seasons - configured_seasons

    if missing_seasons or unexpected_seasons:
        raise ValueError(
            "Dataset seasons do not match the split configuration. "
            f"Missing={sorted(missing_seasons)}, "
            f"unexpected={sorted(unexpected_seasons)}"
        )

    train = working.loc[working["SEASON"].isin(TRAIN_SEASONS)].reset_index(drop=True)

    validation = working.loc[working["SEASON"].isin(VALIDATION_SEASONS)].reset_index(drop=True)

    test = working.loc[working["SEASON"].isin(TEST_SEASONS)].reset_index(drop=True)

    validate_partition(
        partition=train,
        expected_seasons=TRAIN_SEASONS,
        name="train",
    )

    validate_partition(
        partition=validation,
        expected_seasons=VALIDATION_SEASONS,
        name="validation",
    )

    validate_partition(
        partition=test,
        expected_seasons=TEST_SEASONS,
        name="test",
    )

    # This is the central temporal invariant: every training game must occur
    # before validation, and every validation game before the final test set.
    if train["GAME_DATE"].max() >= validation["GAME_DATE"].min():
        raise ValueError("Training dates overlap or follow validation dates")

    if validation["GAME_DATE"].max() >= test["GAME_DATE"].min():
        raise ValueError("Validation dates overlap or follow test dates")

    if len(train) + len(validation) + len(test) != len(working):
        raise ValueError("Train, validation, and test rows do not preserve source count")

    summary = ChronologicalSplitSummary(
        source_rows=len(working),
        train=build_partition_summary(
            name="train",
            seasons=TRAIN_SEASONS,
            partition=train,
        ),
        validation=build_partition_summary(
            name="validation",
            seasons=VALIDATION_SEASONS,
            partition=validation,
        ),
        test=build_partition_summary(
            name="test",
            seasons=TEST_SEASONS,
            partition=test,
        ),
    )

    return train, validation, test, summary


def write_split_outputs(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    summary: ChronologicalSplitSummary,
    project_root: Path,
) -> tuple[Path, Path, Path, Path]:
    """Write all chronological partitions and their metadata."""

    train_path = train_output_path(project_root)
    validation_path = validation_output_path(project_root)
    test_path = test_output_path(project_root)
    summary_path = split_summary_output_path(project_root)

    train_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    train.to_parquet(
        train_path,
        index=False,
    )

    validation.to_parquet(
        validation_path,
        index=False,
    )

    test.to_parquet(
        test_path,
        index=False,
    )

    summary_path.write_text(
        json.dumps(
            asdict(summary),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return train_path, validation_path, test_path, summary_path


def build_chronological_splits(
    project_root: Path,
) -> ChronologicalSplitSummary:
    """Read the modeling data and persist chronological partitions."""

    input_path = modeling_dataset_input_path(project_root)

    if not input_path.exists():
        raise FileNotFoundError(f"Modeling dataset does not exist: {input_path}")

    dataset = pd.read_parquet(input_path)

    train, validation, test, summary = split_dataset(dataset)

    train_path, validation_path, test_path, summary_path = write_split_outputs(
        train=train,
        validation=validation,
        test=test,
        summary=summary,
        project_root=project_root,
    )

    print("\nChronological split complete:")
    print(json.dumps(asdict(summary), indent=2))
    print(f"Train: {train_path}")
    print(f"Validation: {validation_path}")
    print(f"Test: {test_path}")
    print(f"Summary: {summary_path}")

    return summary


def parse_args() -> argparse.Namespace:
    """Parse command-line options for chronological splitting."""

    return argparse.ArgumentParser(
        description=__doc__,
    ).parse_args()


def main() -> None:
    """Create and persist the season-based modeling splits."""

    parse_args()
    project_root = Path(__file__).resolve().parents[2]

    build_chronological_splits(
        project_root=project_root,
    )


if __name__ == "__main__":
    main()
