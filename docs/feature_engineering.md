# Leakage-Safe Feature Engineering

## Objective

Phase 5 converts the completed one-row-per-game NBA dataset into a
pregame modeling dataset for predicting `home_win`.

The final row grain is:

> One row per NBA game, using only information available before tipoff.

## Pipeline

```text
all_seasons.parquet
        ↓
team_history.parquet
        ↓
team_pregame_features.parquet
        ↓
pregame_modeling_dataset.parquet

We’ll finish Phase 5 with four pieces:

```text
1. Build one modeling row per game
2. Test the modeling-dataset logic
3. Add a one-command feature pipeline
4. Test the feature-pipeline orchestration
```

---

# 1. Build the final modeling dataset

Create:

```bash
touch pipelines/features/build_modeling_dataset.py
```

Paste the entire file:

```python
"""Build the final leakage-safe pregame NBA modeling dataset.

The pregame team-feature table contains one row for each team's perspective
in a game. A classification model needs one row per physical game.

This module:

1. Separates home-team and away-team pregame feature rows.
2. Joins the two perspectives into one matchup row.
3. Attaches the home_win target from the completed game dataset.
4. Creates home-minus-away comparison features.
5. Writes a stable feature manifest for downstream modeling.

No same-game score, win/loss result, or point differential is included as a
predictor. The only current-game outcome retained is the target, home_win.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd


REQUIRED_GAME_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "SEASON",
        "SEASON_ID",
        "GAME_ID",
        "GAME_DATE",
        "HOME_TEAM_ID",
        "HOME_TEAM_ABBREVIATION",
        "AWAY_TEAM_ID",
        "AWAY_TEAM_ABBREVIATION",
        "home_win",
    }
)

# Columns required from team_pregame_features.parquet.
REQUIRED_TEAM_FEATURE_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "SEASON",
        "SEASON_ID",
        "GAME_ID",
        "GAME_DATE",
        "TEAM_ID",
        "TEAM_ABBREVIATION",
        "OPPONENT_TEAM_ID",
        "OPPONENT_TEAM_ABBREVIATION",
        "IS_HOME",
        "TEAM_GAME_NUMBER",
        "PRIOR_GAMES_PLAYED",
        "DAYS_REST",
        "IS_BACK_TO_BACK",
        "SEASON_WIN_PCT",
        "ROLLING_5_WIN_PCT",
        "ROLLING_10_WIN_PCT",
        "ROLLING_5_POINTS_SCORED",
        "ROLLING_10_POINTS_SCORED",
        "ROLLING_5_POINTS_ALLOWED",
        "ROLLING_10_POINTS_ALLOWED",
        "ROLLING_5_POINT_DIFFERENTIAL",
        "ROLLING_10_POINT_DIFFERENTIAL",
    }
)

# These are the actual pregame measurements copied for both teams.
SIDE_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "PRIOR_GAMES_PLAYED",
    "DAYS_REST",
    "IS_BACK_TO_BACK",
    "SEASON_WIN_PCT",
    "ROLLING_5_WIN_PCT",
    "ROLLING_10_WIN_PCT",
    "ROLLING_5_POINTS_SCORED",
    "ROLLING_10_POINTS_SCORED",
    "ROLLING_5_POINTS_ALLOWED",
    "ROLLING_10_POINTS_ALLOWED",
    "ROLLING_5_POINT_DIFFERENTIAL",
    "ROLLING_10_POINT_DIFFERENTIAL",
)

# Every feature in this collection receives a home-minus-away version.
DIFFERENCE_SOURCE_COLUMNS: Final[tuple[str, ...]] = SIDE_FEATURE_COLUMNS

# These features should be unavailable before a team's first game because
# they depend on completed same-season history.
HISTORICAL_VALUE_COLUMNS: Final[tuple[str, ...]] = (
    "DAYS_REST",
    "SEASON_WIN_PCT",
    "ROLLING_5_WIN_PCT",
    "ROLLING_10_WIN_PCT",
    "ROLLING_5_POINTS_SCORED",
    "ROLLING_10_POINTS_SCORED",
    "ROLLING_5_POINTS_ALLOWED",
    "ROLLING_10_POINTS_ALLOWED",
    "ROLLING_5_POINT_DIFFERENTIAL",
    "ROLLING_10_POINT_DIFFERENTIAL",
)

# These same-game outcome fields are forbidden from the final predictors.
LEAKED_OUTCOME_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "HOME_PTS",
        "AWAY_PTS",
        "HOME_WL",
        "AWAY_WL",
        "TEAM_WIN",
        "TEAM_WL",
        "TEAM_PTS",
        "OPPONENT_PTS",
        "POINT_DIFFERENTIAL",
    }
)

IDENTIFIER_COLUMNS: Final[tuple[str, ...]] = (
    "SEASON",
    "SEASON_ID",
    "GAME_ID",
    "GAME_DATE",
    "HOME_TEAM_ID",
    "HOME_TEAM_ABBREVIATION",
    "AWAY_TEAM_ID",
    "AWAY_TEAM_ABBREVIATION",
)

CATEGORICAL_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "HOME_TEAM_ID",
    "AWAY_TEAM_ID",
)

HOME_FEATURE_COLUMNS: Final[tuple[str, ...]] = tuple(
    f"HOME_{column}" for column in SIDE_FEATURE_COLUMNS
)

AWAY_FEATURE_COLUMNS: Final[tuple[str, ...]] = tuple(
    f"AWAY_{column}" for column in SIDE_FEATURE_COLUMNS
)

DIFFERENCE_FEATURE_COLUMNS: Final[tuple[str, ...]] = tuple(
    f"{column}_DIFF" for column in DIFFERENCE_SOURCE_COLUMNS
)

HISTORY_FLAG_COLUMNS: Final[tuple[str, ...]] = (
    "HOME_HAS_HISTORY",
    "AWAY_HAS_HISTORY",
    "BOTH_TEAMS_HAVE_HISTORY",
    "BOTH_TEAMS_HAVE_5_GAMES",
    "BOTH_TEAMS_HAVE_10_GAMES",
)

NUMERIC_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    *HOME_FEATURE_COLUMNS,
    *AWAY_FEATURE_COLUMNS,
    *DIFFERENCE_FEATURE_COLUMNS,
    *HISTORY_FLAG_COLUMNS,
)

MODELING_OUTPUT_COLUMNS: Final[tuple[str, ...]] = (
    *IDENTIFIER_COLUMNS,
    *HOME_FEATURE_COLUMNS,
    *AWAY_FEATURE_COLUMNS,
    *DIFFERENCE_FEATURE_COLUMNS,
    *HISTORY_FLAG_COLUMNS,
    "home_win",
)


@dataclass(frozen=True)
class ModelingDatasetSummary:
    """Describe the completed one-row-per-game modeling dataset."""

    source_game_rows: int
    source_team_feature_rows: int
    output_model_rows: int
    unique_games: int
    seasons: int
    numeric_feature_count: int
    categorical_feature_count: int
    rows_with_both_teams_history: int
    rows_with_both_teams_5_games: int
    rows_with_both_teams_10_games: int
    rows_with_any_missing_numeric_features: int
    first_game_date: str
    last_game_date: str


def normalize_game_id(value: object) -> str:
    """Normalize GAME_ID values while preserving leading zeroes."""

    if value is None:
        raise ValueError("GAME_ID cannot be None")

    value_string = str(value).strip()

    if value_string.endswith(".0"):
        value_string = value_string[:-2]

    return value_string.zfill(10)


def games_input_path(project_root: Path) -> Path:
    """Return the completed Phase 4 game dataset path."""

    return project_root / "data" / "processed" / "nba" / "games" / "all_seasons.parquet"


def team_features_input_path(project_root: Path) -> Path:
    """Return the leakage-safe team pregame feature path."""

    return (
        project_root / "data" / "processed" / "nba" / "features" / "team_pregame_features.parquet"
    )


def modeling_dataset_output_path(project_root: Path) -> Path:
    """Return the final pregame modeling-dataset path."""

    return (
        project_root
        / "data"
        / "processed"
        / "nba"
        / "modeling"
        / "pregame_modeling_dataset.parquet"
    )


def modeling_dataset_summary_path(project_root: Path) -> Path:
    """Return the final modeling-dataset summary path."""

    return (
        project_root
        / "data"
        / "processed"
        / "nba"
        / "modeling"
        / "pregame_modeling_dataset_summary.json"
    )


def feature_manifest_output_path(project_root: Path) -> Path:
    """Return the machine-readable model feature manifest path."""

    return project_root / "data" / "processed" / "nba" / "modeling" / "feature_manifest.json"


def prepare_source_frames(
    games: pd.DataFrame,
    team_features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normalize identifiers and date types before validation and joining."""

    prepared_games = games.copy()
    prepared_features = team_features.copy()

    prepared_games["GAME_ID"] = prepared_games["GAME_ID"].map(normalize_game_id)
    prepared_features["GAME_ID"] = prepared_features["GAME_ID"].map(normalize_game_id)

    prepared_games["GAME_DATE"] = pd.to_datetime(
        prepared_games["GAME_DATE"],
        errors="raise",
    )
    prepared_features["GAME_DATE"] = pd.to_datetime(
        prepared_features["GAME_DATE"],
        errors="raise",
    )

    # SEASON_ID occasionally arrives as an integer from Parquet inference.
    # Converting both sources to strings ensures stable merge behavior.
    prepared_games["SEASON_ID"] = prepared_games["SEASON_ID"].astype(str)
    prepared_features["SEASON_ID"] = prepared_features["SEASON_ID"].astype(str)

    team_id_columns = (
        "HOME_TEAM_ID",
        "AWAY_TEAM_ID",
    )

    for column in team_id_columns:
        prepared_games[column] = prepared_games[column].astype("int64")

    feature_team_id_columns = (
        "TEAM_ID",
        "OPPONENT_TEAM_ID",
    )

    for column in feature_team_id_columns:
        prepared_features[column] = prepared_features[column].astype("int64")

    return prepared_games, prepared_features


def validate_source_datasets(
    games: pd.DataFrame,
    team_features: pd.DataFrame,
) -> None:
    """Validate game and team-feature inputs before joining them."""

    missing_game_columns = REQUIRED_GAME_COLUMNS - set(games.columns)

    if missing_game_columns:
        raise ValueError(
            f"Game dataset is missing required columns: {sorted(missing_game_columns)}"
        )

    missing_feature_columns = REQUIRED_TEAM_FEATURE_COLUMNS - set(team_features.columns)

    if missing_feature_columns:
        raise ValueError(
            f"Team-feature dataset is missing required columns: {sorted(missing_feature_columns)}"
        )

    if games.empty:
        raise ValueError("Game dataset cannot be empty")

    if team_features.empty:
        raise ValueError("Team-feature dataset cannot be empty")

    duplicate_games = int(
        games.duplicated(
            subset=["SEASON", "GAME_ID"],
        ).sum()
    )

    if duplicate_games:
        raise ValueError(f"Game dataset contains {duplicate_games} duplicate games")

    duplicate_team_features = int(
        team_features.duplicated(
            subset=["SEASON", "GAME_ID", "TEAM_ID"],
        ).sum()
    )

    if duplicate_team_features:
        raise ValueError(
            f"Team-feature dataset contains {duplicate_team_features} duplicate team-game rows"
        )

    leaked_feature_columns = LEAKED_OUTCOME_COLUMNS & set(team_features.columns)

    if leaked_feature_columns:
        raise ValueError(
            f"Team-feature dataset contains current-game outcomes: {sorted(leaked_feature_columns)}"
        )

    invalid_home_values = ~team_features["IS_HOME"].isin([0, 1])

    if invalid_home_values.any():
        raise ValueError("Team-feature dataset contains IS_HOME values outside {0, 1}")

    expected_game_numbers = team_features["PRIOR_GAMES_PLAYED"].astype("int64") + 1

    actual_game_numbers = team_features["TEAM_GAME_NUMBER"].astype("int64")

    if not expected_game_numbers.equals(actual_game_numbers):
        raise ValueError("TEAM_GAME_NUMBER is inconsistent with PRIOR_GAMES_PLAYED")

    grouped_features = team_features.groupby(
        ["SEASON", "GAME_ID"],
        sort=False,
    )

    row_counts = grouped_features.size()

    if not row_counts.eq(2).all():
        raise ValueError("Every game must contain exactly two team-feature rows")

    home_counts = grouped_features["IS_HOME"].sum()

    if not home_counts.eq(1).all():
        raise ValueError("Every game must contain exactly one home feature row")

    if len(team_features) != len(games) * 2:
        raise ValueError(f"Expected {len(games) * 2} team-feature rows, found {len(team_features)}")

    game_keys = set(
        games[["SEASON", "GAME_ID"]].itertuples(
            index=False,
            name=None,
        )
    )

    feature_keys = set(
        team_features[["SEASON", "GAME_ID"]]
        .drop_duplicates()
        .itertuples(
            index=False,
            name=None,
        )
    )

    if game_keys != feature_keys:
        missing_feature_games = sorted(game_keys - feature_keys)[:10]
        unexpected_feature_games = sorted(feature_keys - game_keys)[:10]

        raise ValueError(
            "Game and team-feature GAME_ID sets do not match. "
            f"Missing features for: {missing_feature_games}; "
            f"unexpected features for: {unexpected_feature_games}"
        )


def build_side_feature_rows(
    team_features: pd.DataFrame,
    is_home: int,
    prefix: str,
) -> pd.DataFrame:
    """Create one consistently prefixed side of the final matchup."""

    rows = team_features.loc[
        team_features["IS_HOME"] == is_home,
        [
            "SEASON",
            "SEASON_ID",
            "GAME_ID",
            "GAME_DATE",
            "TEAM_ID",
            "TEAM_ABBREVIATION",
            "OPPONENT_TEAM_ID",
            "OPPONENT_TEAM_ABBREVIATION",
            *SIDE_FEATURE_COLUMNS,
        ],
    ].copy()

    rename_mapping = {
        "TEAM_ID": f"{prefix}_TEAM_ID",
        "TEAM_ABBREVIATION": f"{prefix}_TEAM_ABBREVIATION",
        "OPPONENT_TEAM_ID": f"{prefix}_OPPONENT_TEAM_ID",
        "OPPONENT_TEAM_ABBREVIATION": (f"{prefix}_OPPONENT_TEAM_ABBREVIATION"),
    }

    rename_mapping.update({column: f"{prefix}_{column}" for column in SIDE_FEATURE_COLUMNS})

    return rows.rename(columns=rename_mapping)


def validate_perspective_join(perspectives: pd.DataFrame) -> None:
    """Verify that the two team perspectives describe the same matchup."""

    home_opponent_matches = perspectives["HOME_OPPONENT_TEAM_ID"].astype("int64") == perspectives[
        "AWAY_TEAM_ID"
    ].astype("int64")

    away_opponent_matches = perspectives["AWAY_OPPONENT_TEAM_ID"].astype("int64") == perspectives[
        "HOME_TEAM_ID"
    ].astype("int64")

    if not home_opponent_matches.all() or not away_opponent_matches.all():
        raise ValueError("Home and away opponent IDs do not cross-match")

    home_abbreviation_matches = (
        perspectives["HOME_OPPONENT_TEAM_ABBREVIATION"] == perspectives["AWAY_TEAM_ABBREVIATION"]
    )

    away_abbreviation_matches = (
        perspectives["AWAY_OPPONENT_TEAM_ABBREVIATION"] == perspectives["HOME_TEAM_ABBREVIATION"]
    )

    if not home_abbreviation_matches.all() or not away_abbreviation_matches.all():
        raise ValueError("Home and away opponent abbreviations do not cross-match")


def add_matchup_comparison_features(
    modeling: pd.DataFrame,
) -> pd.DataFrame:
    """Add home-minus-away features and history-availability flags."""

    result = modeling.copy()

    for feature_name in DIFFERENCE_SOURCE_COLUMNS:
        result[f"{feature_name}_DIFF"] = (
            result[f"HOME_{feature_name}"] - result[f"AWAY_{feature_name}"]
        )

    result["HOME_HAS_HISTORY"] = result["HOME_PRIOR_GAMES_PLAYED"].gt(0).astype("int8")

    result["AWAY_HAS_HISTORY"] = result["AWAY_PRIOR_GAMES_PLAYED"].gt(0).astype("int8")

    result["BOTH_TEAMS_HAVE_HISTORY"] = (
        result["HOME_PRIOR_GAMES_PLAYED"].gt(0) & result["AWAY_PRIOR_GAMES_PLAYED"].gt(0)
    ).astype("int8")

    result["BOTH_TEAMS_HAVE_5_GAMES"] = (
        result["HOME_PRIOR_GAMES_PLAYED"].ge(5) & result["AWAY_PRIOR_GAMES_PLAYED"].ge(5)
    ).astype("int8")

    result["BOTH_TEAMS_HAVE_10_GAMES"] = (
        result["HOME_PRIOR_GAMES_PLAYED"].ge(10) & result["AWAY_PRIOR_GAMES_PLAYED"].ge(10)
    ).astype("int8")

    return result


def validate_modeling_dataset(
    modeling: pd.DataFrame,
    expected_row_count: int,
) -> None:
    """Enforce structural and leakage invariants on the final dataset."""

    if len(modeling) != expected_row_count:
        raise ValueError(f"Expected {expected_row_count} modeling rows, found {len(modeling)}")

    missing_columns = set(MODELING_OUTPUT_COLUMNS) - set(modeling.columns)

    if missing_columns:
        raise ValueError(f"Modeling dataset is missing columns: {sorted(missing_columns)}")

    leaked_columns = LEAKED_OUTCOME_COLUMNS & set(modeling.columns)

    if leaked_columns:
        raise ValueError(
            f"Modeling dataset contains leaked outcome columns: {sorted(leaked_columns)}"
        )

    duplicate_games = int(
        modeling.duplicated(
            subset=["SEASON", "GAME_ID"],
        ).sum()
    )

    if duplicate_games:
        raise ValueError(f"Modeling dataset contains {duplicate_games} duplicate games")

    if not modeling["home_win"].isin([0, 1]).all():
        raise ValueError("Modeling dataset contains home_win values outside {0, 1}")

    same_team_mask = modeling["HOME_TEAM_ID"].astype("int64") == modeling["AWAY_TEAM_ID"].astype(
        "int64"
    )

    if same_team_mask.any():
        raise ValueError("Modeling dataset contains identical home and away teams")

    for feature_name in DIFFERENCE_SOURCE_COLUMNS:
        expected_difference = modeling[f"HOME_{feature_name}"] - modeling[f"AWAY_{feature_name}"]

        actual_difference = modeling[f"{feature_name}_DIFF"]

        if not np.allclose(
            expected_difference.to_numpy(dtype="float64"),
            actual_difference.to_numpy(dtype="float64"),
            equal_nan=True,
        ):
            raise ValueError(f"{feature_name}_DIFF is inconsistent with side features")

    expected_home_history = modeling["HOME_PRIOR_GAMES_PLAYED"].gt(0).astype("int8")

    expected_away_history = modeling["AWAY_PRIOR_GAMES_PLAYED"].gt(0).astype("int8")

    if not expected_home_history.equals(modeling["HOME_HAS_HISTORY"].astype("int8")):
        raise ValueError("HOME_HAS_HISTORY is inconsistent with prior games")

    if not expected_away_history.equals(modeling["AWAY_HAS_HISTORY"].astype("int8")):
        raise ValueError("AWAY_HAS_HISTORY is inconsistent with prior games")

    expected_both_history = (expected_home_history.eq(1) & expected_away_history.eq(1)).astype(
        "int8"
    )

    if not expected_both_history.equals(modeling["BOTH_TEAMS_HAVE_HISTORY"].astype("int8")):
        raise ValueError("BOTH_TEAMS_HAVE_HISTORY is inconsistent")

    for side in ("HOME", "AWAY"):
        prior_games = modeling[f"{side}_PRIOR_GAMES_PLAYED"]
        first_game_mask = prior_games.eq(0)
        experienced_mask = prior_games.gt(0)

        first_game_columns = [f"{side}_{column}" for column in HISTORICAL_VALUE_COLUMNS]

        if (
            not modeling.loc[
                first_game_mask,
                first_game_columns,
            ]
            .isna()
            .all()
            .all()
        ):
            raise ValueError(f"{side} first-game rows contain historical values")

        if (
            not modeling.loc[
                first_game_mask,
                f"{side}_IS_BACK_TO_BACK",
            ]
            .eq(0)
            .all()
        ):
            raise ValueError(f"{side} first-game rows must not be back-to-back")

        if (
            modeling.loc[
                experienced_mask,
                first_game_columns,
            ]
            .isna()
            .any()
            .any()
        ):
            raise ValueError(f"{side} experienced-team rows contain missing history")


def build_feature_manifest() -> dict[str, object]:
    """Create the stable feature contract consumed by modeling code."""

    return {
        "schema_version": 1,
        "row_grain": "one row per NBA game",
        "prediction_time": "pregame",
        "identifier_columns": list(IDENTIFIER_COLUMNS),
        "categorical_feature_columns": list(CATEGORICAL_FEATURE_COLUMNS),
        "numeric_feature_columns": list(NUMERIC_FEATURE_COLUMNS),
        "feature_columns": [
            *CATEGORICAL_FEATURE_COLUMNS,
            *NUMERIC_FEATURE_COLUMNS,
        ],
        "target_column": "home_win",
        "cold_start_policy": (
            "Retain early-season games. Historical numeric features remain "
            "missing when a team has no prior same-season games."
        ),
        "leakage_policy": (
            "All rolling and expanding statistics are shifted by one team "
            "appearance. Same-game scores and results are excluded."
        ),
    }


def build_modeling_dataset(
    games: pd.DataFrame,
    team_features: pd.DataFrame,
) -> tuple[pd.DataFrame, ModelingDatasetSummary]:
    """Join two pregame team perspectives into one modeling row per game."""

    prepared_games, prepared_features = prepare_source_frames(
        games=games,
        team_features=team_features,
    )

    validate_source_datasets(
        games=prepared_games,
        team_features=prepared_features,
    )

    home_rows = build_side_feature_rows(
        team_features=prepared_features,
        is_home=1,
        prefix="HOME",
    )

    away_rows = build_side_feature_rows(
        team_features=prepared_features,
        is_home=0,
        prefix="AWAY",
    )

    matchup_keys = [
        "SEASON",
        "SEASON_ID",
        "GAME_ID",
        "GAME_DATE",
    ]

    perspectives = home_rows.merge(
        away_rows,
        on=matchup_keys,
        how="inner",
        validate="one_to_one",
    )

    validate_perspective_join(perspectives)

    perspectives = perspectives.drop(
        columns=[
            "HOME_OPPONENT_TEAM_ID",
            "HOME_OPPONENT_TEAM_ABBREVIATION",
            "AWAY_OPPONENT_TEAM_ID",
            "AWAY_OPPONENT_TEAM_ABBREVIATION",
        ]
    )

    safe_game_columns = [
        *IDENTIFIER_COLUMNS,
        "home_win",
    ]

    game_targets = prepared_games.loc[
        :,
        safe_game_columns,
    ].copy()

    # Joining on all matchup identifiers verifies that the pregame rows agree
    # with the completed game table's official home and away assignment.
    modeling = perspectives.merge(
        game_targets,
        on=list(IDENTIFIER_COLUMNS),
        how="inner",
        validate="one_to_one",
    )

    modeling = add_matchup_comparison_features(modeling)

    modeling = modeling.sort_values(
        ["GAME_DATE", "GAME_ID"],
        kind="stable",
    ).reset_index(drop=True)

    modeling = modeling.loc[
        :,
        list(MODELING_OUTPUT_COLUMNS),
    ].copy()

    validate_modeling_dataset(
        modeling=modeling,
        expected_row_count=len(prepared_games),
    )

    summary = ModelingDatasetSummary(
        source_game_rows=len(prepared_games),
        source_team_feature_rows=len(prepared_features),
        output_model_rows=len(modeling),
        unique_games=int(modeling["GAME_ID"].nunique()),
        seasons=int(modeling["SEASON"].nunique()),
        numeric_feature_count=len(NUMERIC_FEATURE_COLUMNS),
        categorical_feature_count=len(CATEGORICAL_FEATURE_COLUMNS),
        rows_with_both_teams_history=int(modeling["BOTH_TEAMS_HAVE_HISTORY"].sum()),
        rows_with_both_teams_5_games=int(modeling["BOTH_TEAMS_HAVE_5_GAMES"].sum()),
        rows_with_both_teams_10_games=int(modeling["BOTH_TEAMS_HAVE_10_GAMES"].sum()),
        rows_with_any_missing_numeric_features=int(
            modeling.loc[
                :,
                list(NUMERIC_FEATURE_COLUMNS),
            ]
            .isna()
            .any(axis=1)
            .sum()
        ),
        first_game_date=(modeling["GAME_DATE"].min().date().isoformat()),
        last_game_date=(modeling["GAME_DATE"].max().date().isoformat()),
    )

    return modeling, summary


def write_modeling_dataset_outputs(
    modeling: pd.DataFrame,
    summary: ModelingDatasetSummary,
    project_root: Path,
) -> tuple[Path, Path, Path]:
    """Write the modeling dataset, summary, and feature manifest."""

    dataset_path = modeling_dataset_output_path(project_root)
    summary_path = modeling_dataset_summary_path(project_root)
    manifest_path = feature_manifest_output_path(project_root)

    dataset_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    modeling.to_parquet(
        dataset_path,
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

    manifest_path.write_text(
        json.dumps(
            build_feature_manifest(),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return dataset_path, summary_path, manifest_path


def build_modeling_dataset_from_files(
    project_root: Path,
) -> ModelingDatasetSummary:
    """Read Phase 4 and pregame feature outputs and build model data."""

    game_path = games_input_path(project_root)
    team_feature_path = team_features_input_path(project_root)

    if not game_path.exists():
        raise FileNotFoundError(f"Completed game dataset does not exist: {game_path}")

    if not team_feature_path.exists():
        raise FileNotFoundError(f"Team pregame feature dataset does not exist: {team_feature_path}")

    games = pd.read_parquet(game_path)
    team_features = pd.read_parquet(team_feature_path)

    modeling, summary = build_modeling_dataset(
        games=games,
        team_features=team_features,
    )

    dataset_path, summary_path, manifest_path = write_modeling_dataset_outputs(
        modeling=modeling,
        summary=summary,
        project_root=project_root,
    )

    print("\nPregame modeling-dataset build complete:")
    print(json.dumps(asdict(summary), indent=2))
    print(f"Modeling dataset: {dataset_path}")
    print(f"Summary: {summary_path}")
    print(f"Feature manifest: {manifest_path}")

    return summary


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the modeling-dataset build."""

    return argparse.ArgumentParser(
        description=__doc__,
    ).parse_args()


def main() -> None:
    """Build the final leakage-safe modeling dataset."""

    parse_args()
    project_root = Path(__file__).resolve().parents[2]

    build_modeling_dataset_from_files(
        project_root=project_root,
    )


if __name__ == "__main__":
    main()
```

## What this file does

`build_modeling_dataset.py` joins the home and away pregame perspectives into one row per game, attaches `home_win`, creates home-minus-away comparisons, and writes the feature contract used by Phase 6.

It connects:

```text
all_seasons.parquet
            +
team_pregame_features.parquet
            ↓
build_modeling_dataset.py
            ↓
pregame_modeling_dataset.parquet
```

The resulting dataset contains **41 numeric features**, **2 categorical features**, identifiers, and the target.

---

# 2. Test the modeling dataset

Create:

```bash
touch tests/test_modeling_dataset.py
```

Paste:

```python
"""Tests for the final leakage-safe NBA modeling dataset.

These tests verify the home/away join, matchup differences, history flags,
target preservation, leakage prevention, validation, and output persistence
without reading real project data.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from pipelines.features.build_modeling_dataset import (
    DIFFERENCE_SOURCE_COLUMNS,
    LEAKED_OUTCOME_COLUMNS,
    build_feature_manifest,
    build_modeling_dataset,
    feature_manifest_output_path,
    modeling_dataset_output_path,
    modeling_dataset_summary_path,
    validate_modeling_dataset,
    validate_source_datasets,
    write_modeling_dataset_outputs,
)


def make_games() -> pd.DataFrame:
    """Create two completed chronological game rows."""

    return pd.DataFrame(
        [
            {
                "SEASON": "2024-25",
                "SEASON_ID": "22024",
                "GAME_ID": "0022400001",
                "GAME_DATE": "2024-10-22",
                "HOME_TEAM_ID": 1610612747,
                "HOME_TEAM_ABBREVIATION": "LAL",
                "AWAY_TEAM_ID": 1610612738,
                "AWAY_TEAM_ABBREVIATION": "BOS",
                "home_win": 1,
            },
            {
                "SEASON": "2024-25",
                "SEASON_ID": "22024",
                "GAME_ID": "0022400002",
                "GAME_DATE": "2024-10-24",
                "HOME_TEAM_ID": 1610612738,
                "HOME_TEAM_ABBREVIATION": "BOS",
                "AWAY_TEAM_ID": 1610612747,
                "AWAY_TEAM_ABBREVIATION": "LAL",
                "home_win": 0,
            },
        ]
    )


def historical_values(
    *,
    prior_games: int,
    win_pct: float | None,
    points_scored: float | None,
    points_allowed: float | None,
    point_differential: float | None,
    days_rest: float | None,
) -> dict[str, object]:
    """Create the repeated rolling feature values for one team row."""

    return {
        "TEAM_GAME_NUMBER": prior_games + 1,
        "PRIOR_GAMES_PLAYED": prior_games,
        "DAYS_REST": days_rest,
        "IS_BACK_TO_BACK": 0,
        "SEASON_WIN_PCT": win_pct,
        "ROLLING_5_WIN_PCT": win_pct,
        "ROLLING_10_WIN_PCT": win_pct,
        "ROLLING_5_POINTS_SCORED": points_scored,
        "ROLLING_10_POINTS_SCORED": points_scored,
        "ROLLING_5_POINTS_ALLOWED": points_allowed,
        "ROLLING_10_POINTS_ALLOWED": points_allowed,
        "ROLLING_5_POINT_DIFFERENTIAL": point_differential,
        "ROLLING_10_POINT_DIFFERENTIAL": point_differential,
    }


def make_team_features() -> pd.DataFrame:
    """Create home and away pregame perspectives for two games."""

    return pd.DataFrame(
        [
            {
                "SEASON": "2024-25",
                "SEASON_ID": "22024",
                "GAME_ID": "0022400001",
                "GAME_DATE": "2024-10-22",
                "TEAM_ID": 1610612747,
                "TEAM_ABBREVIATION": "LAL",
                "OPPONENT_TEAM_ID": 1610612738,
                "OPPONENT_TEAM_ABBREVIATION": "BOS",
                "IS_HOME": 1,
                **historical_values(
                    prior_games=0,
                    win_pct=None,
                    points_scored=None,
                    points_allowed=None,
                    point_differential=None,
                    days_rest=None,
                ),
            },
            {
                "SEASON": "2024-25",
                "SEASON_ID": "22024",
                "GAME_ID": "0022400001",
                "GAME_DATE": "2024-10-22",
                "TEAM_ID": 1610612738,
                "TEAM_ABBREVIATION": "BOS",
                "OPPONENT_TEAM_ID": 1610612747,
                "OPPONENT_TEAM_ABBREVIATION": "LAL",
                "IS_HOME": 0,
                **historical_values(
                    prior_games=0,
                    win_pct=None,
                    points_scored=None,
                    points_allowed=None,
                    point_differential=None,
                    days_rest=None,
                ),
            },
            {
                "SEASON": "2024-25",
                "SEASON_ID": "22024",
                "GAME_ID": "0022400002",
                "GAME_DATE": "2024-10-24",
                "TEAM_ID": 1610612738,
                "TEAM_ABBREVIATION": "BOS",
                "OPPONENT_TEAM_ID": 1610612747,
                "OPPONENT_TEAM_ABBREVIATION": "LAL",
                "IS_HOME": 1,
                **historical_values(
                    prior_games=1,
                    win_pct=0.0,
                    points_scored=100.0,
                    points_allowed=110.0,
                    point_differential=-10.0,
                    days_rest=1.0,
                ),
            },
            {
                "SEASON": "2024-25",
                "SEASON_ID": "22024",
                "GAME_ID": "0022400002",
                "GAME_DATE": "2024-10-24",
                "TEAM_ID": 1610612747,
                "TEAM_ABBREVIATION": "LAL",
                "OPPONENT_TEAM_ID": 1610612738,
                "OPPONENT_TEAM_ABBREVIATION": "BOS",
                "IS_HOME": 0,
                **historical_values(
                    prior_games=1,
                    win_pct=1.0,
                    points_scored=110.0,
                    points_allowed=100.0,
                    point_differential=10.0,
                    days_rest=1.0,
                ),
            },
        ]
    )


def test_validate_source_datasets_rejects_missing_columns() -> None:
    """The join should fail when a required feature column is absent."""

    games = make_games()
    features = make_team_features().drop(columns=["ROLLING_5_WIN_PCT"])

    with pytest.raises(
        ValueError,
        match="missing required columns",
    ):
        validate_source_datasets(
            games=games,
            team_features=features,
        )


def test_validate_source_datasets_rejects_duplicate_team_rows() -> None:
    """The same team should not have two feature rows for one game."""

    games = make_games()
    features = make_team_features()

    features = pd.concat(
        [
            features,
            features.iloc[[0]],
        ],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match="duplicate team-game rows",
    ):
        validate_source_datasets(
            games=games,
            team_features=features,
        )


def test_build_modeling_dataset_creates_one_row_per_game() -> None:
    """Two team perspectives should become one matchup row."""

    modeling, summary = build_modeling_dataset(
        games=make_games(),
        team_features=make_team_features(),
    )

    assert len(modeling) == 2
    assert modeling["GAME_ID"].nunique() == 2
    assert modeling["home_win"].tolist() == [1, 0]

    assert summary.source_game_rows == 2
    assert summary.source_team_feature_rows == 4
    assert summary.output_model_rows == 2
    assert summary.unique_games == 2
    assert summary.seasons == 1


def test_build_modeling_dataset_assigns_home_and_away_features() -> None:
    """Each team's feature history should appear on the correct side."""

    modeling, _ = build_modeling_dataset(
        games=make_games(),
        team_features=make_team_features(),
    )

    second_game = modeling.loc[modeling["GAME_ID"] == "0022400002"].iloc[0]

    assert second_game["HOME_TEAM_ID"] == 1610612738
    assert second_game["HOME_TEAM_ABBREVIATION"] == "BOS"
    assert second_game["HOME_SEASON_WIN_PCT"] == pytest.approx(0.0)
    assert second_game["HOME_ROLLING_5_POINT_DIFFERENTIAL"] == (pytest.approx(-10.0))

    assert second_game["AWAY_TEAM_ID"] == 1610612747
    assert second_game["AWAY_TEAM_ABBREVIATION"] == "LAL"
    assert second_game["AWAY_SEASON_WIN_PCT"] == pytest.approx(1.0)
    assert second_game["AWAY_ROLLING_5_POINT_DIFFERENTIAL"] == (pytest.approx(10.0))


def test_build_modeling_dataset_calculates_differences() -> None:
    """Difference features should always be home minus away."""

    modeling, _ = build_modeling_dataset(
        games=make_games(),
        team_features=make_team_features(),
    )

    second_game = modeling.loc[modeling["GAME_ID"] == "0022400002"].iloc[0]

    assert second_game["SEASON_WIN_PCT_DIFF"] == pytest.approx(-1.0)
    assert second_game["ROLLING_5_POINTS_SCORED_DIFF"] == (pytest.approx(-10.0))
    assert second_game["ROLLING_5_POINTS_ALLOWED_DIFF"] == (pytest.approx(10.0))
    assert second_game["ROLLING_5_POINT_DIFFERENTIAL_DIFF"] == (pytest.approx(-20.0))
    assert second_game["DAYS_REST_DIFF"] == pytest.approx(0.0)


def test_build_modeling_dataset_creates_history_flags() -> None:
    """Cold-start flags should reflect available prior games."""

    modeling, summary = build_modeling_dataset(
        games=make_games(),
        team_features=make_team_features(),
    )

    first_game = modeling.loc[modeling["GAME_ID"] == "0022400001"].iloc[0]

    second_game = modeling.loc[modeling["GAME_ID"] == "0022400002"].iloc[0]

    assert first_game["HOME_HAS_HISTORY"] == 0
    assert first_game["AWAY_HAS_HISTORY"] == 0
    assert first_game["BOTH_TEAMS_HAVE_HISTORY"] == 0

    assert second_game["HOME_HAS_HISTORY"] == 1
    assert second_game["AWAY_HAS_HISTORY"] == 1
    assert second_game["BOTH_TEAMS_HAVE_HISTORY"] == 1
    assert second_game["BOTH_TEAMS_HAVE_5_GAMES"] == 0
    assert second_game["BOTH_TEAMS_HAVE_10_GAMES"] == 0

    assert summary.rows_with_both_teams_history == 1
    assert summary.rows_with_both_teams_5_games == 0
    assert summary.rows_with_both_teams_10_games == 0
    assert summary.rows_with_any_missing_numeric_features == 1


def test_build_modeling_dataset_rejects_opponent_mismatch() -> None:
    """The two team rows must identify each other as opponents."""

    features = make_team_features()
    features.loc[0, "OPPONENT_TEAM_ID"] = 999

    with pytest.raises(
        ValueError,
        match="opponent IDs do not cross-match",
    ):
        build_modeling_dataset(
            games=make_games(),
            team_features=features,
        )


def test_modeling_dataset_contains_no_same_game_outcomes() -> None:
    """Scores and team-level current results must not enter predictors."""

    modeling, _ = build_modeling_dataset(
        games=make_games(),
        team_features=make_team_features(),
    )

    assert LEAKED_OUTCOME_COLUMNS.isdisjoint(modeling.columns)
    assert "home_win" in modeling.columns


def test_validate_modeling_dataset_rejects_tampered_difference() -> None:
    """Validation should detect an incorrect comparison feature."""

    modeling, _ = build_modeling_dataset(
        games=make_games(),
        team_features=make_team_features(),
    )

    modeling.loc[1, "SEASON_WIN_PCT_DIFF"] = 99.0

    with pytest.raises(
        ValueError,
        match="SEASON_WIN_PCT_DIFF is inconsistent",
    ):
        validate_modeling_dataset(
            modeling=modeling,
            expected_row_count=2,
        )


def test_modeling_output_paths_use_modeling_directory(
    tmp_path: Path,
) -> None:
    """Dataset, metadata, and manifest should share one output folder."""

    expected_directory = tmp_path / "data" / "processed" / "nba" / "modeling"

    assert modeling_dataset_output_path(tmp_path) == (
        expected_directory / "pregame_modeling_dataset.parquet"
    )

    assert modeling_dataset_summary_path(tmp_path) == (
        expected_directory / "pregame_modeling_dataset_summary.json"
    )

    assert feature_manifest_output_path(tmp_path) == (expected_directory / "feature_manifest.json")


def test_write_modeling_dataset_outputs_creates_all_files(
    tmp_path: Path,
) -> None:
    """The dataset, summary, and feature contract should be persisted."""

    modeling, summary = build_modeling_dataset(
        games=make_games(),
        team_features=make_team_features(),
    )

    dataset_path, summary_path, manifest_path = write_modeling_dataset_outputs(
        modeling=modeling,
        summary=summary,
        project_root=tmp_path,
    )

    assert dataset_path.exists()
    assert summary_path.exists()
    assert manifest_path.exists()

    saved_modeling = pd.read_parquet(dataset_path)

    assert len(saved_modeling) == 2
    assert saved_modeling["home_win"].tolist() == [1, 0]

    summary_text = summary_path.read_text(encoding="utf-8")

    assert '"output_model_rows": 2' in summary_text
    assert '"numeric_feature_count": 41' in summary_text

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["target_column"] == "home_win"
    assert manifest["row_grain"] == "one row per NBA game"
    assert "HOME_TEAM_ID" in manifest["categorical_feature_columns"]
    assert "SEASON_WIN_PCT_DIFF" in manifest["numeric_feature_columns"]


def test_feature_manifest_matches_difference_configuration() -> None:
    """Every configured difference feature should appear in the manifest."""

    manifest = build_feature_manifest()
    numeric_features = set(manifest["numeric_feature_columns"])

    for feature_name in DIFFERENCE_SOURCE_COLUMNS:
        assert f"{feature_name}_DIFF" in numeric_features
```

---

# 3. Add the one-command Phase 5 pipeline

Create:

```bash
touch pipelines/features/run_feature_pipeline.py
```

Paste:

```python
"""Run the complete leakage-safe NBA feature engineering pipeline.

This orchestration module rebuilds every Phase 5 artifact in dependency order:

1. Convert game rows into chronological team history.
2. Build shifted, leakage-safe pregame team features.
3. Join home and away features into one modeling row per game.

The final output is ready for chronological splitting and baseline modeling.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from pipelines.features.build_modeling_dataset import (
    ModelingDatasetSummary,
    build_modeling_dataset_from_files,
    modeling_dataset_output_path,
)
from pipelines.features.build_pregame_team_features import (
    PregameFeatureSummary,
    build_pregame_team_feature_dataset,
)
from pipelines.features.build_team_history import (
    TeamHistorySummary,
    build_team_history_dataset,
)


@dataclass(frozen=True)
class FeaturePipelineSummary:
    """Describe one complete Phase 5 pipeline execution."""

    source_game_rows: int
    team_history_rows: int
    pregame_team_feature_rows: int
    modeling_rows: int
    seasons: int
    numeric_feature_count: int
    categorical_feature_count: int
    final_output_path: str


def feature_pipeline_summary_path(project_root: Path) -> Path:
    """Return the Phase 5 orchestration summary path."""

    return (
        project_root / "data" / "processed" / "nba" / "features" / "feature_pipeline_summary.json"
    )


def validate_stage_counts(
    team_history_summary: TeamHistorySummary,
    pregame_summary: PregameFeatureSummary,
    modeling_summary: ModelingDatasetSummary,
) -> None:
    """Verify that every feature stage preserved the expected row grain."""

    expected_team_rows = modeling_summary.source_game_rows * 2

    if team_history_summary.output_team_rows != expected_team_rows:
        raise ValueError("Team-history row count is inconsistent with source games")

    if pregame_summary.source_team_rows != expected_team_rows:
        raise ValueError("Pregame feature input count is inconsistent")

    if pregame_summary.output_feature_rows != expected_team_rows:
        raise ValueError("Pregame feature output count is inconsistent")

    if modeling_summary.source_team_feature_rows != expected_team_rows:
        raise ValueError("Modeling source feature count is inconsistent")

    if modeling_summary.output_model_rows != (modeling_summary.source_game_rows):
        raise ValueError("Modeling output must contain one row per source game")

    if team_history_summary.seasons != pregame_summary.seasons:
        raise ValueError("Team-history and pregame feature season counts differ")

    if pregame_summary.seasons != modeling_summary.seasons:
        raise ValueError("Pregame and modeling season counts differ")


def run_feature_pipeline(
    project_root: Path,
) -> FeaturePipelineSummary:
    """Execute all Phase 5 transformations and write final metadata."""

    print("Building chronological team history...")

    team_history_summary = build_team_history_dataset(
        project_root=project_root,
    )

    print("\nBuilding leakage-safe pregame team features...")

    pregame_summary = build_pregame_team_feature_dataset(
        project_root=project_root,
    )

    print("\nBuilding final one-row-per-game modeling dataset...")

    modeling_summary = build_modeling_dataset_from_files(
        project_root=project_root,
    )

    validate_stage_counts(
        team_history_summary=team_history_summary,
        pregame_summary=pregame_summary,
        modeling_summary=modeling_summary,
    )

    final_output_path = modeling_dataset_output_path(project_root)

    if not final_output_path.exists():
        raise FileNotFoundError(f"Final modeling dataset was not created: {final_output_path}")

    summary = FeaturePipelineSummary(
        source_game_rows=modeling_summary.source_game_rows,
        team_history_rows=team_history_summary.output_team_rows,
        pregame_team_feature_rows=(pregame_summary.output_feature_rows),
        modeling_rows=modeling_summary.output_model_rows,
        seasons=modeling_summary.seasons,
        numeric_feature_count=(modeling_summary.numeric_feature_count),
        categorical_feature_count=(modeling_summary.categorical_feature_count),
        final_output_path=str(final_output_path),
    )

    summary_path = feature_pipeline_summary_path(project_root)

    summary_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # The orchestration summary is written only after all three stages have
    # completed and their row-count invariants agree.
    summary_path.write_text(
        json.dumps(
            asdict(summary),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print("\nPhase 5 feature pipeline complete:")
    print(json.dumps(asdict(summary), indent=2))
    print(f"Pipeline summary: {summary_path}")

    return summary


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the Phase 5 pipeline."""

    return argparse.ArgumentParser(
        description=__doc__,
    ).parse_args()


def main() -> None:
    """Run all Phase 5 feature transformations."""

    parse_args()
    project_root = Path(__file__).resolve().parents[2]

    run_feature_pipeline(
        project_root=project_root,
    )


if __name__ == "__main__":
    main()
```

## What this file does

`run_feature_pipeline.py` is the Phase 5 orchestration layer. It rebuilds the three feature stages in the correct order and verifies that row counts remain consistent.

```text
all_seasons.parquet
        ↓
team_history.parquet
        ↓
team_pregame_features.parquet
        ↓
pregame_modeling_dataset.parquet
```

---

# 4. Test the feature-pipeline runner

Create:

```bash
touch tests/test_feature_pipeline.py
```

Paste:

```python
"""Tests for the complete Phase 5 feature-pipeline orchestration."""

from pathlib import Path
from types import SimpleNamespace

import pytest

import pipelines.features.run_feature_pipeline as pipeline_module
from pipelines.features.build_modeling_dataset import (
    modeling_dataset_output_path,
)
from pipelines.features.run_feature_pipeline import (
    feature_pipeline_summary_path,
    run_feature_pipeline,
)


def test_feature_pipeline_summary_path_uses_features_directory(
    tmp_path: Path,
) -> None:
    """The orchestration summary should live beside Phase 5 metadata."""

    assert feature_pipeline_summary_path(tmp_path) == (
        tmp_path / "data" / "processed" / "nba" / "features" / "feature_pipeline_summary.json"
    )


def test_run_feature_pipeline_writes_summary_after_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Successful stages should produce one final pipeline summary."""

    team_history_summary = SimpleNamespace(
        output_team_rows=4,
        seasons=1,
    )

    pregame_summary = SimpleNamespace(
        source_team_rows=4,
        output_feature_rows=4,
        seasons=1,
    )

    modeling_summary = SimpleNamespace(
        source_game_rows=2,
        source_team_feature_rows=4,
        output_model_rows=2,
        seasons=1,
        numeric_feature_count=41,
        categorical_feature_count=2,
    )

    monkeypatch.setattr(
        pipeline_module,
        "build_team_history_dataset",
        lambda project_root: team_history_summary,
    )

    monkeypatch.setattr(
        pipeline_module,
        "build_pregame_team_feature_dataset",
        lambda project_root: pregame_summary,
    )

    monkeypatch.setattr(
        pipeline_module,
        "build_modeling_dataset_from_files",
        lambda project_root: modeling_summary,
    )

    final_path = modeling_dataset_output_path(tmp_path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    final_path.write_bytes(b"test-output")

    summary = run_feature_pipeline(project_root=tmp_path)

    assert summary.source_game_rows == 2
    assert summary.team_history_rows == 4
    assert summary.pregame_team_feature_rows == 4
    assert summary.modeling_rows == 2
    assert summary.numeric_feature_count == 41
    assert summary.categorical_feature_count == 2

    summary_path = feature_pipeline_summary_path(tmp_path)

    assert summary_path.exists()
    assert '"modeling_rows": 2' in summary_path.read_text(encoding="utf-8")


def test_run_feature_pipeline_does_not_write_summary_after_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed feature stage must not create success metadata."""

    team_history_summary = SimpleNamespace(
        output_team_rows=4,
        seasons=1,
    )

    pregame_summary = SimpleNamespace(
        source_team_rows=4,
        output_feature_rows=4,
        seasons=1,
    )

    monkeypatch.setattr(
        pipeline_module,
        "build_team_history_dataset",
        lambda project_root: team_history_summary,
    )

    monkeypatch.setattr(
        pipeline_module,
        "build_pregame_team_feature_dataset",
        lambda project_root: pregame_summary,
    )

    def fail_modeling_build(project_root: Path) -> None:
        """Simulate a failure in the final feature stage."""

        del project_root
        raise RuntimeError("Simulated modeling-dataset failure")

    monkeypatch.setattr(
        pipeline_module,
        "build_modeling_dataset_from_files",
        fail_modeling_build,
    )

    with pytest.raises(
        RuntimeError,
        match="Simulated modeling-dataset failure",
    ):
        run_feature_pipeline(project_root=tmp_path)

    assert not feature_pipeline_summary_path(tmp_path).exists()
```

---

# 5. Add Phase 5 documentation

Create:

```bash
touch docs/feature_engineering.md
```

Paste:

````markdown
# Leakage-Safe Feature Engineering

## Objective

Phase 5 converts the completed one-row-per-game NBA dataset into a
pregame modeling dataset for predicting `home_win`.

The final row grain is:

> One row per NBA game, using only information available before tipoff.

## Pipeline

```text
all_seasons.parquet
        ↓
team_history.parquet
        ↓
team_pregame_features.parquet
        ↓
pregame_modeling_dataset.parquet
````

## Team history

Each game becomes two chronological team perspectives:

```text
Home team perspective
Away team perspective
```

Completed-game outcomes are retained in this intermediate table so they can
become historical inputs for later games.

## Leakage prevention

All outcome-derived statistics are shifted by one team appearance before
expanding or rolling calculations are performed.

For a team's Game 10:

```text
Games 1–9 may contribute to features.
Game 10 may not contribute to its own features.
```

Current-game fields excluded from the model predictors include:

```text
TEAM_WIN
TEAM_PTS
OPPONENT_PTS
POINT_DIFFERENTIAL
HOME_PTS
AWAY_PTS
HOME_WL
AWAY_WL
```

The only current-game outcome retained is the target:

```text
home_win
```

## Pregame feature families

### Season history

* Prior games played
* Season-to-date win percentage

### Recent form

* Rolling 5-game and 10-game win percentage
* Rolling 5-game and 10-game points scored
* Rolling 5-game and 10-game points allowed
* Rolling 5-game and 10-game point differential

### Schedule context

* Days of rest
* Back-to-back indicator

### Matchup comparisons

Every side-level measurement receives a home-minus-away comparison.

Example:

```text
SEASON_WIN_PCT_DIFF
    = HOME_SEASON_WIN_PCT - AWAY_SEASON_WIN_PCT
```

## Cold-start policy

The first game of each team's season has no same-season history.

These rows are retained with missing historical values instead of being
dropped or assigned invented statistics.

History-availability flags identify early-season records:

```text
HOME_HAS_HISTORY
AWAY_HAS_HISTORY
BOTH_TEAMS_HAVE_HISTORY
BOTH_TEAMS_HAVE_5_GAMES
BOTH_TEAMS_HAVE_10_GAMES
```

The modeling phase will define preprocessing and imputation behavior.

## Final feature contract

The feature contract is stored in:

```text
data/processed/nba/modeling/feature_manifest.json
```

It identifies:

* Identifier columns
* Categorical features
* Numeric features
* Target column
* Cold-start policy
* Leakage policy

````

---

# 6. Run everything

First run formatting and all tests:

```bash
uv run ruff check \
  pipelines/features/build_modeling_dataset.py \
  pipelines/features/run_feature_pipeline.py \
  tests/test_modeling_dataset.py \
  tests/test_feature_pipeline.py \
  --fix

uv run ruff format \
  pipelines/features/build_modeling_dataset.py \
  pipelines/features/run_feature_pipeline.py \
  tests/test_modeling_dataset.py \
  tests/test_feature_pipeline.py

uv run ruff check .
uv run ruff format --check .
uv run pytest
````

The expected total is:

```text
78 passed
```

Then run the complete Phase 5 pipeline:

```bash
uv run python -m pipelines.features.run_feature_pipeline
```

The core expected result is:

```json
{
  "source_game_rows": 1230,
  "team_history_rows": 2460,
  "pregame_team_feature_rows": 2460,
  "modeling_rows": 1230,
  "seasons": 1,
  "numeric_feature_count": 41,
  "categorical_feature_count": 2
}
```

Final outputs:

```text
data/processed/nba/features/team_history.parquet
data/processed/nba/features/team_pregame_features.parquet
data/processed/nba/features/feature_pipeline_summary.json

data/processed/nba/modeling/pregame_modeling_dataset.parquet
data/processed/nba/modeling/pregame_modeling_dataset_summary.json
data/processed/nba/modeling/feature_manifest.json
```

## Phase 5 completion criteria

Once the command succeeds:

```text
✓ One modeling row per NBA game
✓ All statistics use only previous games
✓ Current-game outcomes are excluded from predictors
✓ Home and away perspectives cross-validate
✓ Home-minus-away comparison features exist
✓ Cold-start games remain explicitly represented
✓ Feature columns are documented in a manifest
✓ Complete pipeline runs with one command
✓ Offline tests protect every stage
```

Then commit:

```bash
git add .
git commit -m "Complete leakage-safe NBA feature engineering pipeline"
git push origin main
```
