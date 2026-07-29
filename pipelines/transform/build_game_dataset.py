"""Convert NBA team-game rows into one game-level modeling dataset.

The raw NBA LeagueGameLog source contains two rows per game: one row from
each team's perspective. This module pairs those rows, identifies the home
and away teams when the source is unambiguous, creates the ``home_win``
target, and writes one processed row per game.

Games whose two source rows both use ``@`` or both use ``vs.`` are separated
for later schedule-based home/away resolution rather than guessed.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

import pandas as pd

from pipelines.ingestion.audit_nba_source import (
    classify_location,
    validate_schema,
)

# Columns retained from each team's raw row.
#
# These are useful for validating the transformation and building the first
# modeling dataset. Pregame features will later be derived from historical
# results rather than using same-game box-score statistics as model inputs.
TEAM_RESULT_COLUMNS: Final[tuple[str, ...]] = (
    "TEAM_ID",
    "TEAM_ABBREVIATION",
    "WL",
    "PTS",
)


@dataclass(frozen=True)
class GameDatasetSummary:
    """Describe the result of converting team rows into game rows."""

    season: str
    source_team_rows: int
    source_games: int
    resolved_games: int
    unresolved_games: int
    output_game_rows: int


def processed_season_directory(project_root: Path, season: str) -> Path:
    """Return the processed-data directory for one NBA season."""

    return project_root / "data" / "processed" / "nba" / "games" / f"season={season}"


def games_output_path(project_root: Path, season: str) -> Path:
    """Return the output path for resolved one-row-per-game records."""

    return processed_season_directory(project_root, season) / "games.parquet"


def unresolved_output_path(project_root: Path, season: str) -> Path:
    """Return the output path for games needing home/away resolution."""

    return processed_season_directory(project_root, season) / "unresolved_team_game_rows.parquet"


def summary_output_path(project_root: Path, season: str) -> Path:
    """Return the metadata path describing a transformation run."""

    return processed_season_directory(project_root, season) / "transformation_summary.json"


def identify_standard_games(frame: pd.DataFrame) -> pd.Series:
    """Return a Boolean mask marking games with one ``vs.`` and one ``@`` row.

    The returned Series is indexed by ``GAME_ID`` rather than by individual
    DataFrame rows. A standard game can be transformed without guessing
    because the ``vs.`` row identifies the home team.
    """

    locations = frame.assign(LOCATION_MARKER=frame["MATCHUP"].map(classify_location))

    grouped = locations.groupby("GAME_ID", sort=False)
    row_counts = grouped.size()
    home_marker_counts = grouped["LOCATION_MARKER"].apply(
        lambda values: int((values == "vs").sum())
    )
    away_marker_counts = grouped["LOCATION_MARKER"].apply(
        lambda values: int((values == "away").sum())
    )

    return (row_counts == 2) & (home_marker_counts == 1) & (away_marker_counts == 1)


def build_resolved_game_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert standard two-team games into one row per game.

    The function expects only games containing exactly one ``vs.`` row and
    one ``@`` row. It creates explicit home and away columns and derives the
    binary target ``home_win`` from the home team's result.
    """

    if frame.empty:
        return pd.DataFrame()

    working = frame.copy()
    working["LOCATION_MARKER"] = working["MATCHUP"].map(classify_location)

    home_rows = working.loc[
        working["LOCATION_MARKER"] == "vs",
        ["GAME_ID", "GAME_DATE", "SEASON_ID", *TEAM_RESULT_COLUMNS],
    ].copy()

    away_rows = working.loc[
        working["LOCATION_MARKER"] == "away",
        ["GAME_ID", *TEAM_RESULT_COLUMNS],
    ].copy()

    # Prefix the team-specific columns before joining the two perspectives.
    home_rows = home_rows.rename(
        columns={
            "TEAM_ID": "HOME_TEAM_ID",
            "TEAM_ABBREVIATION": "HOME_TEAM_ABBREVIATION",
            "WL": "HOME_WL",
            "PTS": "HOME_PTS",
        }
    )

    away_rows = away_rows.rename(
        columns={
            "TEAM_ID": "AWAY_TEAM_ID",
            "TEAM_ABBREVIATION": "AWAY_TEAM_ABBREVIATION",
            "WL": "AWAY_WL",
            "PTS": "AWAY_PTS",
        }
    )

    games = home_rows.merge(
        away_rows,
        on="GAME_ID",
        how="inner",
        validate="one_to_one",
    )

    # The model target is 1 when the officially designated home team won.
    games["home_win"] = (games["HOME_WL"] == "W").astype("int8")

    # Dates are normalized now so chronological sorting and feature
    # engineering behave consistently in later phases.
    games["GAME_DATE"] = pd.to_datetime(
        games["GAME_DATE"],
        errors="raise",
    )

    return games.sort_values(
        ["GAME_DATE", "GAME_ID"],
        kind="stable",
    ).reset_index(drop=True)


def transform_team_game_rows(
    frame: pd.DataFrame,
    season: str,
) -> tuple[pd.DataFrame, pd.DataFrame, GameDatasetSummary]:
    """Split raw team-game rows into resolved and unresolved game records.

    Standard games are transformed into one row per game. Special-event
    games with ambiguous location markers remain in their original two-row
    form so a later schedule-enrichment step can assign official home/away.
    """

    validate_schema(frame)

    duplicate_count = int(frame.duplicated(subset=["GAME_ID", "TEAM_ID"]).sum())
    if duplicate_count:
        raise ValueError(f"Found {duplicate_count} duplicate GAME_ID/TEAM_ID rows")

    standard_games = identify_standard_games(frame)
    resolved_game_ids = standard_games[standard_games].index
    unresolved_game_ids = standard_games[~standard_games].index

    resolved_source_rows = frame.loc[frame["GAME_ID"].isin(resolved_game_ids)].copy()

    unresolved_rows = frame.loc[frame["GAME_ID"].isin(unresolved_game_ids)].copy()

    games = build_resolved_game_rows(resolved_source_rows)

    summary = GameDatasetSummary(
        season=season,
        source_team_rows=len(frame),
        source_games=int(frame["GAME_ID"].nunique()),
        resolved_games=len(resolved_game_ids),
        unresolved_games=len(unresolved_game_ids),
        output_game_rows=len(games),
    )

    # A one-row-per-game transformation must preserve the resolved game
    # count exactly. This catches accidental row duplication or row loss.
    if summary.output_game_rows != summary.resolved_games:
        raise ValueError(
            "Resolved game count does not match output game-row count: "
            f"{summary.resolved_games} resolved versus "
            f"{summary.output_game_rows} output rows"
        )

    return games, unresolved_rows, summary


def write_transformation_outputs(
    games: pd.DataFrame,
    unresolved_rows: pd.DataFrame,
    summary: GameDatasetSummary,
    project_root: Path,
) -> tuple[Path, Path, Path]:
    """Write resolved games, unresolved rows, and transformation metadata."""

    games_path = games_output_path(project_root, summary.season)
    unresolved_path = unresolved_output_path(
        project_root,
        summary.season,
    )
    summary_path = summary_output_path(project_root, summary.season)

    games_path.parent.mkdir(parents=True, exist_ok=True)

    games.to_parquet(games_path, index=False)
    unresolved_rows.to_parquet(unresolved_path, index=False)
    summary_path.write_text(
        json.dumps(asdict(summary), indent=2) + "\n",
        encoding="utf-8",
    )

    return games_path, unresolved_path, summary_path


def raw_input_path(project_root: Path, season: str) -> Path:
    """Return the raw team-game Parquet path created by ingestion."""

    return (
        project_root
        / "data"
        / "raw"
        / "nba"
        / "league_game_log"
        / f"season={season}"
        / "team_game_log.parquet"
    )


def build_season_game_dataset(
    season: str,
    project_root: Path,
) -> GameDatasetSummary:
    """Read one ingested season, transform it, and write processed outputs."""

    source_path = raw_input_path(project_root, season)

    if not source_path.exists():
        raise FileNotFoundError(f"Raw season file does not exist: {source_path}")

    frame = pd.read_parquet(source_path)

    games, unresolved_rows, summary = transform_team_game_rows(
        frame=frame,
        season=season,
    )

    games_path, unresolved_path, summary_path = write_transformation_outputs(
        games=games,
        unresolved_rows=unresolved_rows,
        summary=summary,
        project_root=project_root,
    )

    print(json.dumps(asdict(summary), indent=2))
    print(f"Resolved games: {games_path}")
    print(f"Unresolved rows: {unresolved_path}")
    print(f"Summary: {summary_path}")

    return summary


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the game-level transformation."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--season",
        required=True,
        help="NBA season to transform, such as 2024-25",
    )
    return parser.parse_args()


def main() -> None:
    """Run the game-level transformation from the command line."""

    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]

    build_season_game_dataset(
        season=args.season,
        project_root=project_root,
    )


if __name__ == "__main__":
    main()
