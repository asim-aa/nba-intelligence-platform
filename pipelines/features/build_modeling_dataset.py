"""Build the final leakage-safe pregame NBA modeling dataset.

The pregame team-feature table contains one row for each team's perspective
in a game. A classification model needs one row per physical game.

This module:

1. Separates home-team and away-team pregame feature rows.
2. Merges in each team's pregame Elo rating (a separate, cross-season
   source -- see build_team_elo_ratings.py -- since it is computed by a
   sequential simulation rather than the groupby/shift window functions
   used for the other pregame features).
3. Joins the two perspectives into one matchup row.
4. Attaches the home_win target from the completed game dataset.
5. Creates home-minus-away comparison features.
6. Writes a stable feature manifest for downstream modeling.

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

# Columns required from team_elo_ratings.parquet (see build_team_elo_ratings.py).
REQUIRED_ELO_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "SEASON",
        "SEASON_ID",
        "GAME_ID",
        "GAME_DATE",
        "TEAM_ID",
        "ELO_RATING",
    }
)

ELO_MERGE_KEYS: Final[tuple[str, ...]] = (
    "SEASON",
    "SEASON_ID",
    "GAME_ID",
    "GAME_DATE",
    "TEAM_ID",
)

# These are the actual pregame measurements copied for both teams. ELO_RATING
# is never missing (see build_team_elo_ratings.py), unlike the rest of these
# columns, which are NaN for a team's first game of a season.
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
    "ELO_RATING",
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


def elo_ratings_input_path(project_root: Path) -> Path:
    """Return the cross-season Elo ratings path (see build_team_elo_ratings.py)."""

    return project_root / "data" / "processed" / "nba" / "features" / "team_elo_ratings.parquet"


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
    elo_ratings: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Normalize identifiers and date types before validation and joining."""

    prepared_games = games.copy()
    prepared_features = team_features.copy()
    prepared_elo = elo_ratings.copy()

    prepared_games["GAME_ID"] = prepared_games["GAME_ID"].map(normalize_game_id)
    prepared_features["GAME_ID"] = prepared_features["GAME_ID"].map(normalize_game_id)
    prepared_elo["GAME_ID"] = prepared_elo["GAME_ID"].map(normalize_game_id)

    prepared_games["GAME_DATE"] = pd.to_datetime(
        prepared_games["GAME_DATE"],
        errors="raise",
    )
    prepared_features["GAME_DATE"] = pd.to_datetime(
        prepared_features["GAME_DATE"],
        errors="raise",
    )
    prepared_elo["GAME_DATE"] = pd.to_datetime(
        prepared_elo["GAME_DATE"],
        errors="raise",
    )

    # SEASON_ID occasionally arrives as an integer from Parquet inference.
    # Converting all sources to strings ensures stable merge behavior.
    prepared_games["SEASON_ID"] = prepared_games["SEASON_ID"].astype(str)
    prepared_features["SEASON_ID"] = prepared_features["SEASON_ID"].astype(str)
    prepared_elo["SEASON_ID"] = prepared_elo["SEASON_ID"].astype(str)

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

    prepared_elo["TEAM_ID"] = prepared_elo["TEAM_ID"].astype("int64")

    return prepared_games, prepared_features, prepared_elo


def validate_elo_input(
    elo_ratings: pd.DataFrame,
    team_features: pd.DataFrame,
) -> None:
    """Validate Elo ratings before merging them into the pregame features.

    The key set must exactly match team_features' so the upcoming
    one-to-one merge cannot silently drop or duplicate rows.
    """

    missing_columns = REQUIRED_ELO_COLUMNS - set(elo_ratings.columns)

    if missing_columns:
        raise ValueError(f"Elo ratings are missing required columns: {sorted(missing_columns)}")

    if elo_ratings.empty:
        raise ValueError("Elo ratings cannot be empty")

    duplicate_rows = int(elo_ratings.duplicated(subset=list(ELO_MERGE_KEYS)).sum())

    if duplicate_rows:
        raise ValueError(f"Elo ratings contain {duplicate_rows} duplicate team-game rows")

    if elo_ratings["ELO_RATING"].isna().any():
        raise ValueError("Elo ratings contain missing ELO_RATING values")

    elo_keys = set(elo_ratings[list(ELO_MERGE_KEYS)].itertuples(index=False, name=None))
    feature_keys = set(team_features[list(ELO_MERGE_KEYS)].itertuples(index=False, name=None))

    if elo_keys != feature_keys:
        missing_elo = sorted(feature_keys - elo_keys)[:10]
        unexpected_elo = sorted(elo_keys - feature_keys)[:10]

        raise ValueError(
            "Elo ratings and team-feature keys do not match. "
            f"Missing Elo for: {missing_elo}; unexpected Elo for: {unexpected_elo}"
        )


def merge_elo_into_team_features(
    team_features: pd.DataFrame,
    elo_ratings: pd.DataFrame,
) -> pd.DataFrame:
    """Attach each team-game row's pregame Elo rating."""

    return team_features.merge(
        elo_ratings[[*ELO_MERGE_KEYS, "ELO_RATING"]],
        on=list(ELO_MERGE_KEYS),
        how="inner",
        validate="one_to_one",
    )


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
    elo_ratings: pd.DataFrame,
) -> tuple[pd.DataFrame, ModelingDatasetSummary]:
    """Join two pregame team perspectives, plus Elo, into one row per game."""

    prepared_games, prepared_features, prepared_elo = prepare_source_frames(
        games=games,
        team_features=team_features,
        elo_ratings=elo_ratings,
    )

    validate_elo_input(
        elo_ratings=prepared_elo,
        team_features=prepared_features,
    )

    prepared_features = merge_elo_into_team_features(
        team_features=prepared_features,
        elo_ratings=prepared_elo,
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
    elo_path = elo_ratings_input_path(project_root)

    if not game_path.exists():
        raise FileNotFoundError(f"Completed game dataset does not exist: {game_path}")

    if not team_feature_path.exists():
        raise FileNotFoundError(f"Team pregame feature dataset does not exist: {team_feature_path}")

    if not elo_path.exists():
        raise FileNotFoundError(f"Elo rating dataset does not exist: {elo_path}")

    games = pd.read_parquet(game_path)
    team_features = pd.read_parquet(team_feature_path)
    elo_ratings = pd.read_parquet(elo_path)

    modeling, summary = build_modeling_dataset(
        games=games,
        team_features=team_features,
        elo_ratings=elo_ratings,
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
