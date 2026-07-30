"""Build a chronological team-game history from the game-level dataset.

The completed Phase 4 dataset contains one row per NBA game, with separate
home-team and away-team columns. Feature engineering is easier and safer in
a long format where each game contributes one row for each team's
perspective.

This module converts:

    one game row
        ↓
    one home-team history row
    one away-team history row

The output still contains same-game results such as points and wins. Those
columns are historical source values, not final model features. Later feature
modules must shift them before calculating pregame rolling statistics.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

import pandas as pd

# These fields must exist in the completed Phase 4 dataset.
REQUIRED_GAME_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "SEASON",
        "SEASON_ID",
        "GAME_ID",
        "GAME_DATE",
        "HOME_TEAM_ID",
        "HOME_TEAM_ABBREVIATION",
        "HOME_WL",
        "HOME_PTS",
        "AWAY_TEAM_ID",
        "AWAY_TEAM_ABBREVIATION",
        "AWAY_WL",
        "AWAY_PTS",
        "home_win",
    }
)


@dataclass(frozen=True)
class TeamHistorySummary:
    """Describe one game-level to team-history transformation."""

    source_game_rows: int
    source_games: int
    output_team_rows: int
    unique_teams: int
    seasons: int
    first_game_date: str
    last_game_date: str


def combined_games_input_path(project_root: Path) -> Path:
    """Return the combined Phase 4 game-dataset path."""

    return project_root / "data" / "processed" / "nba" / "games" / "all_seasons.parquet"


def team_history_output_path(project_root: Path) -> Path:
    """Return the long-format team-history output path."""

    return project_root / "data" / "processed" / "nba" / "features" / "team_history.parquet"


def team_history_summary_path(project_root: Path) -> Path:
    """Return the metadata path for the team-history transformation."""

    return project_root / "data" / "processed" / "nba" / "features" / "team_history_summary.json"


def validate_game_dataset_input(games: pd.DataFrame) -> None:
    """Validate the Phase 4 dataset before reshaping it.

    These checks protect the feature pipeline from malformed game rows.
    Feature engineering should never silently compensate for invalid
    upstream data.
    """

    missing_columns = REQUIRED_GAME_COLUMNS - set(games.columns)

    if missing_columns:
        raise ValueError(f"Game dataset is missing required columns: {sorted(missing_columns)}")

    if games.empty:
        raise ValueError("Game dataset cannot be empty")

    duplicate_games = int(
        games.duplicated(
            subset=["SEASON", "GAME_ID"],
        ).sum()
    )

    if duplicate_games:
        raise ValueError(f"Game dataset contains {duplicate_games} duplicate SEASON/GAME_ID rows")

    required_value_columns = [
        "SEASON",
        "GAME_ID",
        "GAME_DATE",
        "HOME_TEAM_ID",
        "HOME_TEAM_ABBREVIATION",
        "HOME_WL",
        "HOME_PTS",
        "AWAY_TEAM_ID",
        "AWAY_TEAM_ABBREVIATION",
        "AWAY_WL",
        "AWAY_PTS",
        "home_win",
    ]

    columns_with_missing_values = [
        column for column in required_value_columns if games[column].isna().any()
    ]

    if columns_with_missing_values:
        raise ValueError(f"Game dataset contains missing values in: {columns_with_missing_values}")

    same_team_mask = games["HOME_TEAM_ID"].astype("int64") == games["AWAY_TEAM_ID"].astype("int64")

    if same_team_mask.any():
        raise ValueError("Game dataset contains identical home and away teams")

    valid_results = ((games["HOME_WL"] == "W") & (games["AWAY_WL"] == "L")) | (
        (games["HOME_WL"] == "L") & (games["AWAY_WL"] == "W")
    )

    if not valid_results.all():
        raise ValueError("Game dataset contains invalid home and away results")

    expected_target = (games["HOME_WL"] == "W").astype("int8")

    actual_target = games["home_win"].astype("int8")

    if not expected_target.equals(actual_target):
        raise ValueError("Game dataset contains home_win values inconsistent with HOME_WL")


def build_home_team_rows(games: pd.DataFrame) -> pd.DataFrame:
    """Create one team-history row from each game's home perspective."""

    home_rows = games[
        [
            "SEASON",
            "SEASON_ID",
            "GAME_ID",
            "GAME_DATE",
            "HOME_TEAM_ID",
            "HOME_TEAM_ABBREVIATION",
            "HOME_WL",
            "HOME_PTS",
            "AWAY_TEAM_ID",
            "AWAY_TEAM_ABBREVIATION",
            "AWAY_PTS",
        ]
    ].copy()

    home_rows = home_rows.rename(
        columns={
            "HOME_TEAM_ID": "TEAM_ID",
            "HOME_TEAM_ABBREVIATION": "TEAM_ABBREVIATION",
            "HOME_WL": "TEAM_WL",
            "HOME_PTS": "TEAM_PTS",
            "AWAY_TEAM_ID": "OPPONENT_TEAM_ID",
            "AWAY_TEAM_ABBREVIATION": ("OPPONENT_TEAM_ABBREVIATION"),
            "AWAY_PTS": "OPPONENT_PTS",
        }
    )

    # IS_HOME is known before tipoff and may later be used as a model feature.
    home_rows["IS_HOME"] = 1

    # TEAM_WIN is an outcome from this game. It is retained only so future
    # rolling calculations can use shifted historical values.
    home_rows["TEAM_WIN"] = (home_rows["TEAM_WL"] == "W").astype("int8")

    return home_rows


def build_away_team_rows(games: pd.DataFrame) -> pd.DataFrame:
    """Create one team-history row from each game's away perspective."""

    away_rows = games[
        [
            "SEASON",
            "SEASON_ID",
            "GAME_ID",
            "GAME_DATE",
            "AWAY_TEAM_ID",
            "AWAY_TEAM_ABBREVIATION",
            "AWAY_WL",
            "AWAY_PTS",
            "HOME_TEAM_ID",
            "HOME_TEAM_ABBREVIATION",
            "HOME_PTS",
        ]
    ].copy()

    away_rows = away_rows.rename(
        columns={
            "AWAY_TEAM_ID": "TEAM_ID",
            "AWAY_TEAM_ABBREVIATION": "TEAM_ABBREVIATION",
            "AWAY_WL": "TEAM_WL",
            "AWAY_PTS": "TEAM_PTS",
            "HOME_TEAM_ID": "OPPONENT_TEAM_ID",
            "HOME_TEAM_ABBREVIATION": ("OPPONENT_TEAM_ABBREVIATION"),
            "HOME_PTS": "OPPONENT_PTS",
        }
    )

    away_rows["IS_HOME"] = 0
    away_rows["TEAM_WIN"] = (away_rows["TEAM_WL"] == "W").astype("int8")

    return away_rows


def validate_team_history(
    team_history: pd.DataFrame,
    expected_game_count: int,
) -> None:
    """Enforce invariants for the long-format team-history dataset."""

    expected_team_rows = expected_game_count * 2

    if len(team_history) != expected_team_rows:
        raise ValueError(
            f"Expected {expected_team_rows} team-history rows, found {len(team_history)}"
        )

    duplicate_team_rows = int(
        team_history.duplicated(
            subset=["SEASON", "GAME_ID", "TEAM_ID"],
        ).sum()
    )

    if duplicate_team_rows:
        raise ValueError(f"Team history contains {duplicate_team_rows} duplicate team-game rows")

    same_team_mask = team_history["TEAM_ID"].astype("int64") == team_history[
        "OPPONENT_TEAM_ID"
    ].astype("int64")

    if same_team_mask.any():
        raise ValueError("Team history contains a team playing itself")

    grouped_games = team_history.groupby(
        ["SEASON", "GAME_ID"],
        sort=False,
    )

    row_counts = grouped_games.size()

    if not row_counts.eq(2).all():
        raise ValueError("Every game must produce exactly two team-history rows")

    home_counts = grouped_games["IS_HOME"].sum()

    if not home_counts.eq(1).all():
        raise ValueError("Every game must contain exactly one home-team row")

    win_counts = grouped_games["TEAM_WIN"].sum()

    if not win_counts.eq(1).all():
        raise ValueError("Every game must contain exactly one winning-team row")

    point_differential_sums = grouped_games["POINT_DIFFERENTIAL"].sum()

    if not point_differential_sums.eq(0).all():
        raise ValueError("Team point differentials must cancel within each game")

    # Within each season, each team's sequence should start at game one and
    # increase by exactly one for every appearance.
    expected_numbers = (
        team_history.groupby(
            ["SEASON", "TEAM_ID"],
            sort=False,
        )
        .cumcount()
        .add(1)
    )

    actual_numbers = team_history["TEAM_GAME_NUMBER"].astype("int64")

    if not expected_numbers.equals(actual_numbers):
        raise ValueError("TEAM_GAME_NUMBER is inconsistent with chronological order")


def build_team_history(
    games: pd.DataFrame,
) -> tuple[pd.DataFrame, TeamHistorySummary]:
    """Convert one-row-per-game data into chronological team perspectives."""

    validate_game_dataset_input(games)

    working = games.copy()

    working["GAME_DATE"] = pd.to_datetime(
        working["GAME_DATE"],
        errors="raise",
    )

    home_rows = build_home_team_rows(working)
    away_rows = build_away_team_rows(working)

    team_history = pd.concat(
        [
            home_rows,
            away_rows,
        ],
        ignore_index=True,
    )

    # These outcome values describe the current game. They become safe
    # pregame features only after a later module shifts them backward.
    team_history["POINT_DIFFERENTIAL"] = team_history["TEAM_PTS"] - team_history["OPPONENT_PTS"]

    team_history = team_history.sort_values(
        [
            "SEASON",
            "TEAM_ID",
            "GAME_DATE",
            "GAME_ID",
        ],
        kind="stable",
    ).reset_index(drop=True)

    # The sequence resets each season. In the first feature version, teams
    # begin each season with no prior same-season information.
    team_history["TEAM_GAME_NUMBER"] = (
        team_history.groupby(
            ["SEASON", "TEAM_ID"],
            sort=False,
        )
        .cumcount()
        .add(1)
        .astype("int16")
    )

    validate_team_history(
        team_history=team_history,
        expected_game_count=len(working),
    )

    summary = TeamHistorySummary(
        source_game_rows=len(working),
        source_games=int(working[["SEASON", "GAME_ID"]].drop_duplicates().shape[0]),
        output_team_rows=len(team_history),
        unique_teams=int(team_history["TEAM_ID"].nunique()),
        seasons=int(team_history["SEASON"].nunique()),
        first_game_date=(team_history["GAME_DATE"].min().date().isoformat()),
        last_game_date=(team_history["GAME_DATE"].max().date().isoformat()),
    )

    return team_history, summary


def write_team_history_outputs(
    team_history: pd.DataFrame,
    summary: TeamHistorySummary,
    project_root: Path,
) -> tuple[Path, Path]:
    """Write the team-history Parquet and transformation metadata."""

    history_path = team_history_output_path(project_root)
    summary_path = team_history_summary_path(project_root)

    history_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    team_history.to_parquet(
        history_path,
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

    return history_path, summary_path


def build_team_history_dataset(
    project_root: Path,
) -> TeamHistorySummary:
    """Read the Phase 4 dataset and write the long team-history dataset."""

    input_path = combined_games_input_path(project_root)

    if not input_path.exists():
        raise FileNotFoundError(f"Combined game dataset does not exist: {input_path}")

    games = pd.read_parquet(input_path)

    team_history, summary = build_team_history(games)

    history_path, summary_path = write_team_history_outputs(
        team_history=team_history,
        summary=summary,
        project_root=project_root,
    )

    print("\nTeam-history build complete:")
    print(json.dumps(asdict(summary), indent=2))
    print(f"Team history: {history_path}")
    print(f"Summary: {summary_path}")

    return summary


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the team-history build."""

    return argparse.ArgumentParser(
        description=__doc__,
    ).parse_args()


def main() -> None:
    """Run the team-history transformation from the command line."""

    parse_args()
    project_root = Path(__file__).resolve().parents[2]

    build_team_history_dataset(
        project_root=project_root,
    )


if __name__ == "__main__":
    main()
