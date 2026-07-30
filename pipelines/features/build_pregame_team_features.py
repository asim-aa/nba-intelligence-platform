"""Build leakage-safe pregame features for each NBA team-game row.

The team-history dataset contains completed-game outcomes such as TEAM_WIN,
TEAM_PTS, OPPONENT_PTS, and POINT_DIFFERENTIAL. Those values are not known
before their own game begins and therefore cannot directly become predictors.

This module groups each team's games chronologically and shifts outcomes by
one game before calculating expanding and rolling statistics. The resulting
output contains only identifiers and information available before tipoff.

Example:

    Game 1 outcome ┐
    Game 2 outcome ├── used to create Game 3 pregame features
    Game 3 outcome ┘   never included in Game 3 features
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

import pandas as pd

# Columns required from build_team_history.py.
REQUIRED_HISTORY_COLUMNS: Final[frozenset[str]] = frozenset(
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
        "TEAM_WIN",
        "TEAM_PTS",
        "OPPONENT_PTS",
        "POINT_DIFFERENTIAL",
        "TEAM_GAME_NUMBER",
    }
)

# These columns describe the current game's result. They are allowed in the
# source history table but must never appear in the final pregame feature
# table because they would leak information about the prediction target.
CURRENT_GAME_OUTCOME_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "TEAM_WIN",
        "TEAM_WL",
        "TEAM_PTS",
        "OPPONENT_PTS",
        "POINT_DIFFERENTIAL",
    }
)

# Rolling windows used in the first feature version.
ROLLING_WINDOWS: Final[tuple[int, ...]] = (
    5,
    10,
)

# Outcome-derived feature columns should be missing for a team's first game
# because no same-season history exists before that game.
OUTCOME_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
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

# Exact final column order makes downstream joins and model inputs stable.
PREGAME_OUTPUT_COLUMNS: Final[tuple[str, ...]] = (
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
)


@dataclass(frozen=True)
class PregameFeatureSummary:
    """Describe one pregame team-feature build."""

    source_team_rows: int
    output_feature_rows: int
    unique_teams: int
    seasons: int
    feature_columns: tuple[str, ...]
    rows_without_prior_history: int
    back_to_back_rows: int
    first_game_date: str
    last_game_date: str


def team_history_input_path(project_root: Path) -> Path:
    """Return the team-history path produced by build_team_history.py."""

    return project_root / "data" / "processed" / "nba" / "features" / "team_history.parquet"


def pregame_team_features_output_path(project_root: Path) -> Path:
    """Return the leakage-safe team-feature output path."""

    return (
        project_root / "data" / "processed" / "nba" / "features" / "team_pregame_features.parquet"
    )


def pregame_team_features_summary_path(project_root: Path) -> Path:
    """Return the metadata path for the pregame feature build."""

    return (
        project_root
        / "data"
        / "processed"
        / "nba"
        / "features"
        / "team_pregame_features_summary.json"
    )


def validate_team_history_input(team_history: pd.DataFrame) -> None:
    """Validate the completed-game history before calculating features."""

    missing_columns = REQUIRED_HISTORY_COLUMNS - set(team_history.columns)

    if missing_columns:
        raise ValueError(
            f"Team-history dataset is missing required columns: {sorted(missing_columns)}"
        )

    if team_history.empty:
        raise ValueError("Team-history dataset cannot be empty")

    duplicate_rows = int(
        team_history.duplicated(
            subset=[
                "SEASON",
                "GAME_ID",
                "TEAM_ID",
            ]
        ).sum()
    )

    if duplicate_rows:
        raise ValueError(f"Team history contains {duplicate_rows} duplicate team-game rows")

    required_value_columns = (
        "SEASON",
        "GAME_ID",
        "GAME_DATE",
        "TEAM_ID",
        "TEAM_ABBREVIATION",
        "OPPONENT_TEAM_ID",
        "OPPONENT_TEAM_ABBREVIATION",
        "IS_HOME",
        "TEAM_WIN",
        "TEAM_PTS",
        "OPPONENT_PTS",
        "POINT_DIFFERENTIAL",
        "TEAM_GAME_NUMBER",
    )

    columns_with_missing_values = [
        column for column in required_value_columns if team_history[column].isna().any()
    ]

    if columns_with_missing_values:
        raise ValueError(f"Team history contains missing values in: {columns_with_missing_values}")

    invalid_home_values = ~team_history["IS_HOME"].isin([0, 1])

    if invalid_home_values.any():
        raise ValueError("Team history contains IS_HOME values outside {0, 1}")

    invalid_win_values = ~team_history["TEAM_WIN"].isin([0, 1])

    if invalid_win_values.any():
        raise ValueError("Team history contains TEAM_WIN values outside {0, 1}")

    same_team_mask = team_history["TEAM_ID"].astype("int64") == team_history[
        "OPPONENT_TEAM_ID"
    ].astype("int64")

    if same_team_mask.any():
        raise ValueError("Team history contains a team playing itself")

    expected_point_differential = team_history["TEAM_PTS"] - team_history["OPPONENT_PTS"]

    if not expected_point_differential.equals(team_history["POINT_DIFFERENTIAL"]):
        raise ValueError("POINT_DIFFERENTIAL is inconsistent with team and opponent points")


def shifted_expanding_mean(series: pd.Series) -> pd.Series:
    """Calculate an expanding mean using only earlier values.

    shift(1) removes the current game's value before expanding() calculates
    the season-to-date statistic.
    """

    return (
        series.shift(1)
        .expanding(
            min_periods=1,
        )
        .mean()
    )


def shifted_rolling_mean(
    series: pd.Series,
    window: int,
) -> pd.Series:
    """Calculate a rolling mean over only the preceding games."""

    return (
        series.shift(1)
        .rolling(
            window=window,
            min_periods=1,
        )
        .mean()
    )


def add_outcome_history_features(
    working: pd.DataFrame,
) -> pd.DataFrame:
    """Add expanding and rolling features from shifted game outcomes."""

    result = working.copy()

    grouped = result.groupby(
        [
            "SEASON",
            "TEAM_ID",
        ],
        sort=False,
        group_keys=False,
    )

    # Season win percentage includes every prior same-season game while
    # excluding the current game through shift(1).
    result["SEASON_WIN_PCT"] = grouped["TEAM_WIN"].transform(shifted_expanding_mean)

    for window in ROLLING_WINDOWS:
        result[f"ROLLING_{window}_WIN_PCT"] = grouped["TEAM_WIN"].transform(
            lambda series, window=window: shifted_rolling_mean(
                series,
                window,
            )
        )

        result[f"ROLLING_{window}_POINTS_SCORED"] = grouped["TEAM_PTS"].transform(
            lambda series, window=window: shifted_rolling_mean(
                series,
                window,
            )
        )

        result[f"ROLLING_{window}_POINTS_ALLOWED"] = grouped["OPPONENT_PTS"].transform(
            lambda series, window=window: shifted_rolling_mean(
                series,
                window,
            )
        )

        result[f"ROLLING_{window}_POINT_DIFFERENTIAL"] = grouped["POINT_DIFFERENTIAL"].transform(
            lambda series, window=window: shifted_rolling_mean(
                series,
                window,
            )
        )

    return result


def add_schedule_features(
    working: pd.DataFrame,
) -> pd.DataFrame:
    """Add pregame rest and schedule-density features."""

    result = working.copy()

    grouped = result.groupby(
        [
            "SEASON",
            "TEAM_ID",
        ],
        sort=False,
    )

    previous_game_date = grouped["GAME_DATE"].shift(1)

    calendar_day_gap = (result["GAME_DATE"] - previous_game_date).dt.days

    # A team cannot play two distinct regular-season games in reverse
    # chronological order or on the same calendar date.
    invalid_gaps = calendar_day_gap.dropna().le(0)

    if invalid_gaps.any():
        raise ValueError("Team history contains non-positive gaps between games")

    # Consecutive calendar dates have a gap of one day but zero full rest
    # days between them. Therefore:
    #
    #     Oct 22 → Oct 23 = 0 rest days, back-to-back
    #     Oct 22 → Oct 24 = 1 rest day
    result["DAYS_REST"] = (calendar_day_gap - 1).astype("float64")

    result["IS_BACK_TO_BACK"] = calendar_day_gap.eq(1).fillna(False).astype("int8")

    return result


def validate_pregame_features(
    features: pd.DataFrame,
    expected_row_count: int,
) -> None:
    """Enforce leakage and structural invariants on pregame features."""

    if len(features) != expected_row_count:
        raise ValueError(
            f"Expected {expected_row_count} pregame feature rows, found {len(features)}"
        )

    missing_columns = set(PREGAME_OUTPUT_COLUMNS) - set(features.columns)

    if missing_columns:
        raise ValueError(f"Pregame feature dataset is missing columns: {sorted(missing_columns)}")

    leaked_columns = CURRENT_GAME_OUTCOME_COLUMNS & set(features.columns)

    if leaked_columns:
        raise ValueError(
            "Pregame feature dataset contains current-game outcome columns: "
            f"{sorted(leaked_columns)}"
        )

    duplicate_rows = int(
        features.duplicated(
            subset=[
                "SEASON",
                "GAME_ID",
                "TEAM_ID",
            ]
        ).sum()
    )

    if duplicate_rows:
        raise ValueError(f"Pregame features contain {duplicate_rows} duplicate team-game rows")

    expected_prior_games = features["TEAM_GAME_NUMBER"].astype("int64") - 1

    actual_prior_games = features["PRIOR_GAMES_PLAYED"].astype("int64")

    if not expected_prior_games.equals(actual_prior_games):
        raise ValueError("PRIOR_GAMES_PLAYED is inconsistent with TEAM_GAME_NUMBER")

    first_game_mask = features["PRIOR_GAMES_PLAYED"].eq(0)

    # First games must not have outcome-derived statistics because no earlier
    # same-season games exist.
    first_game_outcomes = features.loc[
        first_game_mask,
        list(OUTCOME_FEATURE_COLUMNS),
    ]

    if not first_game_outcomes.isna().all().all():
        raise ValueError("First team games contain outcome-derived pregame features")

    if (
        not features.loc[
            first_game_mask,
            "DAYS_REST",
        ]
        .isna()
        .all()
    ):
        raise ValueError("First team games must have missing DAYS_REST")

    experienced_team_mask = features["PRIOR_GAMES_PLAYED"].gt(0)

    experienced_features = features.loc[
        experienced_team_mask,
        list(OUTCOME_FEATURE_COLUMNS),
    ]

    if experienced_features.isna().any().any():
        raise ValueError("Rows with prior games contain missing historical features")

    percentage_columns = (
        "SEASON_WIN_PCT",
        "ROLLING_5_WIN_PCT",
        "ROLLING_10_WIN_PCT",
    )

    for column in percentage_columns:
        non_missing_values = features[column].dropna()

        if not non_missing_values.between(0, 1).all():
            raise ValueError(f"{column} contains values outside [0, 1]")

    non_missing_rest = features["DAYS_REST"].dropna()

    if non_missing_rest.lt(0).any():
        raise ValueError("DAYS_REST contains negative values")

    if not features["IS_BACK_TO_BACK"].isin([0, 1]).all():
        raise ValueError("IS_BACK_TO_BACK contains values outside {0, 1}")


def build_pregame_team_features(
    team_history: pd.DataFrame,
) -> tuple[pd.DataFrame, PregameFeatureSummary]:
    """Create one leakage-safe feature row per team appearance."""

    validate_team_history_input(team_history)

    working = team_history.copy()

    working["GAME_DATE"] = pd.to_datetime(
        working["GAME_DATE"],
        errors="raise",
    )

    # Sorting happens before every shift or rolling operation. Otherwise,
    # pandas could treat a later game as though it happened earlier.
    working = working.sort_values(
        [
            "SEASON",
            "TEAM_ID",
            "GAME_DATE",
            "GAME_ID",
        ],
        kind="stable",
    ).reset_index(drop=True)

    grouped = working.groupby(
        [
            "SEASON",
            "TEAM_ID",
        ],
        sort=False,
    )

    working["PRIOR_GAMES_PLAYED"] = grouped.cumcount().astype("int16")

    working = add_outcome_history_features(working)
    working = add_schedule_features(working)

    # Selecting an explicit safe-column list prevents same-game scores and
    # results from accidentally entering downstream model matrices.
    features = working.loc[
        :,
        list(PREGAME_OUTPUT_COLUMNS),
    ].copy()

    validate_pregame_features(
        features=features,
        expected_row_count=len(team_history),
    )

    summary = PregameFeatureSummary(
        source_team_rows=len(team_history),
        output_feature_rows=len(features),
        unique_teams=int(features["TEAM_ID"].nunique()),
        seasons=int(features["SEASON"].nunique()),
        feature_columns=tuple(
            column
            for column in PREGAME_OUTPUT_COLUMNS
            if column
            not in {
                "SEASON",
                "SEASON_ID",
                "GAME_ID",
                "GAME_DATE",
                "TEAM_ID",
                "TEAM_ABBREVIATION",
                "OPPONENT_TEAM_ID",
                "OPPONENT_TEAM_ABBREVIATION",
            }
        ),
        rows_without_prior_history=int(features["PRIOR_GAMES_PLAYED"].eq(0).sum()),
        back_to_back_rows=int(features["IS_BACK_TO_BACK"].sum()),
        first_game_date=(features["GAME_DATE"].min().date().isoformat()),
        last_game_date=(features["GAME_DATE"].max().date().isoformat()),
    )

    return features, summary


def write_pregame_team_feature_outputs(
    features: pd.DataFrame,
    summary: PregameFeatureSummary,
    project_root: Path,
) -> tuple[Path, Path]:
    """Write the pregame feature Parquet and metadata summary."""

    feature_path = pregame_team_features_output_path(project_root)
    summary_path = pregame_team_features_summary_path(project_root)

    feature_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    features.to_parquet(
        feature_path,
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

    return feature_path, summary_path


def build_pregame_team_feature_dataset(
    project_root: Path,
) -> PregameFeatureSummary:
    """Read team history and write leakage-safe pregame team features."""

    input_path = team_history_input_path(project_root)

    if not input_path.exists():
        raise FileNotFoundError(f"Team-history dataset does not exist: {input_path}")

    team_history = pd.read_parquet(input_path)

    features, summary = build_pregame_team_features(team_history)

    feature_path, summary_path = write_pregame_team_feature_outputs(
        features=features,
        summary=summary,
        project_root=project_root,
    )

    print("\nPregame team-feature build complete:")
    print(json.dumps(asdict(summary), indent=2))
    print(f"Pregame team features: {feature_path}")
    print(f"Summary: {summary_path}")

    return summary


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the feature build."""

    return argparse.ArgumentParser(
        description=__doc__,
    ).parse_args()


def main() -> None:
    """Run the leakage-safe team-feature build."""

    parse_args()
    project_root = Path(__file__).resolve().parents[2]

    build_pregame_team_feature_dataset(
        project_root=project_root,
    )


if __name__ == "__main__":
    main()
