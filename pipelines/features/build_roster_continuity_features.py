"""Build a leakage-safe "roster continuity" feature per team-game.

This is a proxy for pregame injury/availability status, not the real
thing: the NBA's official injury report only exists as PDFs with no clean
historical API (see docs/roadmap.md), so this is built entirely from data
already in this project's pipeline -- player game logs -- using the same
shift(1) discipline as every other leakage-safe feature here.

The idea in three steps:

1. For each player, on each team, compute a shifted rolling average of
   their minutes played over their trailing games (shifted_rolling_mean,
   reused from build_pregame_team_features.py) -- "how many minutes this
   player was expected to play, based only on games before this one."

2. For each team-game, identify the "core" as the top-3 players by that
   expected-minutes value, and compare their actual combined minutes in
   THIS game to their combined expected minutes. A healthy, full-strength
   game scores near 1.0; a missing star scores well below it. This step
   uses the current game's own box score, so on its own it is a postgame
   value -- exactly like TEAM_WIN or TEAM_PTS in build_team_history.py,
   not yet safe to use as a predictor for that same game.

3. Shift that ratio by one game and take a rolling average over the
   trailing games (shifted_rolling_mean again) -- "how intact has this
   team's core been over its last few games." This is the leakage-safe
   pregame feature: it never uses the game it is predicting, only games
   that already happened before it.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from pipelines.features.build_pregame_team_features import shifted_rolling_mean

# Trailing games used to estimate a player's expected minutes.
MINUTES_ROLLING_WINDOW: Final[int] = 10

# Top-N players by expected minutes counted as a team's "core" rotation.
CORE_SIZE: Final[int] = 3

# Trailing games averaged into the final pregame continuity feature.
CONTINUITY_ROLLING_WINDOW: Final[int] = 5

DEFAULT_SEASONS: Final[tuple[str, ...]] = (
    "2015-16",
    "2016-17",
    "2017-18",
    "2018-19",
    "2019-20",
    "2020-21",
    "2021-22",
    "2022-23",
    "2023-24",
    "2024-25",
    "2025-26",
)

REQUIRED_PLAYER_LOG_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "SEASON_ID",
        "PLAYER_ID",
        "PLAYER_NAME",
        "TEAM_ID",
        "TEAM_ABBREVIATION",
        "GAME_ID",
        "GAME_DATE",
        "MIN",
    }
)

OUTPUT_COLUMNS: Final[tuple[str, ...]] = (
    "SEASON",
    "SEASON_ID",
    "GAME_ID",
    "GAME_DATE",
    "TEAM_ID",
    "ROLLING_CORE_CONTINUITY",
    # Kept alongside the final feature (not just an internal intermediate)
    # so modeling/serving/matchup_features.py can extend a team's history
    # with a placeholder row and re-run compute_rolling_continuity for an
    # upcoming game, the same placeholder technique used everywhere else
    # for a "computed" (not-yet-played) matchup.
    "CORE_MINUTES_REALIZED_RATIO",
)


@dataclass(frozen=True)
class RosterContinuitySummary:
    """Describe one roster-continuity feature build."""

    source_player_rows: int
    output_team_game_rows: int
    unique_teams: int
    unique_players: int
    seasons: int
    rows_with_missing_continuity: int
    first_game_date: str
    last_game_date: str


def player_log_input_path(project_root: Path, season: str) -> Path:
    """Return one season's raw player game log path."""

    return (
        project_root
        / "data"
        / "raw"
        / "nba"
        / "league_game_log"
        / f"season={season}"
        / "player_game_log.parquet"
    )


def roster_continuity_output_path(project_root: Path) -> Path:
    """Return the roster-continuity feature output path."""

    return (
        project_root / "data" / "processed" / "nba" / "features" / "team_roster_continuity.parquet"
    )


def roster_continuity_summary_path(project_root: Path) -> Path:
    """Return the roster-continuity build summary path."""

    return (
        project_root
        / "data"
        / "processed"
        / "nba"
        / "features"
        / "team_roster_continuity_summary.json"
    )


def load_player_game_logs(project_root: Path, seasons: tuple[str, ...]) -> pd.DataFrame:
    """Load and concatenate every season's raw player game log."""

    frames: list[pd.DataFrame] = []

    for season in seasons:
        path = player_log_input_path(project_root, season)

        if not path.exists():
            raise FileNotFoundError(f"Player game log does not exist for {season}: {path}")

        frame = pd.read_parquet(path)

        missing_columns = REQUIRED_PLAYER_LOG_COLUMNS - set(frame.columns)

        if missing_columns:
            raise ValueError(
                f"Player game log for {season} is missing columns: {sorted(missing_columns)}"
            )

        frame = frame.copy()
        frame["SEASON"] = season
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)

    combined["GAME_DATE"] = pd.to_datetime(combined["GAME_DATE"], errors="raise")
    combined["MIN"] = pd.to_numeric(combined["MIN"], errors="raise")
    combined["TEAM_ID"] = combined["TEAM_ID"].astype("int64")
    combined["SEASON_ID"] = combined["SEASON_ID"].astype(str)

    return combined


def compute_player_rolling_minutes(player_rows: pd.DataFrame) -> pd.DataFrame:
    """Add each player's shift(1)-safe rolling average minutes.

    Grouped by (SEASON, TEAM_ID, PLAYER_ID) rather than just PLAYER_ID, so
    a mid-season trade starts a fresh window on the new team instead of
    carrying over a role from a different roster.
    """

    working = player_rows.sort_values(["GAME_DATE", "GAME_ID"], kind="stable").reset_index(
        drop=True
    )

    grouped = working.groupby(
        ["SEASON", "TEAM_ID", "PLAYER_ID"],
        sort=False,
        group_keys=False,
    )

    working["PLAYER_ROLLING_MIN"] = grouped["MIN"].transform(
        lambda series: shifted_rolling_mean(series, MINUTES_ROLLING_WINDOW)
    )

    return working


def compute_core_realized_ratio(player_rows_with_rolling: pd.DataFrame) -> pd.DataFrame:
    """One row per team-game: the realized-vs-expected minutes ratio of
    the team's top-CORE_SIZE players by expected (rolling) minutes.

    The core must be identified from each player's own established role --
    their rolling-minutes estimate carried forward through any games they
    sit out -- not just from whoever happens to have a row in this game.
    Ranking only among players present in the game would make a missing
    star undetectable by construction: they would simply be excluded from
    the ranking pool instead of showing up as an absence.

    Every team-game the team actually played appears in the result, even
    when a ratio cannot be computed yet (NaN) -- so the chronological
    sequence downstream has no silent gaps for shifted_rolling_mean to
    skip over.
    """

    team_game_index = (
        player_rows_with_rolling[["SEASON", "TEAM_ID", "GAME_ID", "GAME_DATE"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    player_universe = player_rows_with_rolling[["SEASON", "TEAM_ID", "PLAYER_ID"]].drop_duplicates()

    # Cross-join every player who has appeared for a team this season with
    # every game that team played, then attach the rows where they actually
    # played -- everything else stays NaN until forward-filled below.
    panel = player_universe.merge(team_game_index, on=["SEASON", "TEAM_ID"], how="left")

    panel = panel.merge(
        player_rows_with_rolling[
            ["SEASON", "TEAM_ID", "PLAYER_ID", "GAME_ID", "GAME_DATE", "MIN", "PLAYER_ROLLING_MIN"]
        ],
        on=["SEASON", "TEAM_ID", "PLAYER_ID", "GAME_ID", "GAME_DATE"],
        how="left",
    )

    panel = panel.sort_values(
        ["SEASON", "TEAM_ID", "PLAYER_ID", "GAME_DATE", "GAME_ID"],
        kind="stable",
    )

    # A player's own PLAYER_ROLLING_MIN is already shift(1)-safe (excludes
    # the game it was computed for). Forward-filling it through games they
    # did not play carries forward only already-past information, so this
    # stays leakage-safe -- it never looks ahead.
    panel["EXPECTED_MIN"] = panel.groupby(
        ["SEASON", "TEAM_ID", "PLAYER_ID"],
        sort=False,
    )["PLAYER_ROLLING_MIN"].transform(lambda series: series.ffill())

    panel["ACTUAL_MIN"] = panel["MIN"].fillna(0.0)

    panel = panel.dropna(subset=["EXPECTED_MIN"])

    panel["CORE_RANK"] = panel.groupby(
        ["SEASON", "TEAM_ID", "GAME_ID"],
        sort=False,
    )["EXPECTED_MIN"].rank(method="first", ascending=False)

    core_rows = panel.loc[panel["CORE_RANK"] <= CORE_SIZE]

    grouped = core_rows.groupby(
        ["SEASON", "TEAM_ID", "GAME_ID", "GAME_DATE"],
        sort=False,
    )

    core_summary = grouped.agg(
        CORE_PLAYER_COUNT=("PLAYER_ID", "size"),
        CORE_EXPECTED_MINUTES=("EXPECTED_MIN", "sum"),
        CORE_REALIZED_MINUTES=("ACTUAL_MIN", "sum"),
    ).reset_index()

    ratio_table = team_game_index.merge(
        core_summary,
        on=["SEASON", "TEAM_ID", "GAME_ID", "GAME_DATE"],
        how="left",
    )

    has_full_core = ratio_table["CORE_PLAYER_COUNT"].ge(CORE_SIZE)
    has_positive_expected = ratio_table["CORE_EXPECTED_MINUTES"].gt(0)

    ratio_table["CORE_MINUTES_REALIZED_RATIO"] = np.where(
        has_full_core.fillna(False) & has_positive_expected.fillna(False),
        ratio_table["CORE_REALIZED_MINUTES"] / ratio_table["CORE_EXPECTED_MINUTES"],
        np.nan,
    )

    return ratio_table[["SEASON", "TEAM_ID", "GAME_ID", "GAME_DATE", "CORE_MINUTES_REALIZED_RATIO"]]


def compute_rolling_continuity(ratio_table: pd.DataFrame) -> pd.DataFrame:
    """Shift and roll the per-game ratio into the final pregame feature."""

    working = ratio_table.sort_values(["GAME_DATE", "GAME_ID"], kind="stable").reset_index(
        drop=True
    )

    grouped = working.groupby(
        ["SEASON", "TEAM_ID"],
        sort=False,
        group_keys=False,
    )

    working["ROLLING_CORE_CONTINUITY"] = grouped["CORE_MINUTES_REALIZED_RATIO"].transform(
        lambda series: shifted_rolling_mean(series, CONTINUITY_ROLLING_WINDOW)
    )

    return working


def build_roster_continuity_features(
    player_rows: pd.DataFrame,
) -> tuple[pd.DataFrame, RosterContinuitySummary]:
    """Run the full three-stage roster-continuity computation."""

    missing_columns = REQUIRED_PLAYER_LOG_COLUMNS - set(player_rows.columns)

    if missing_columns:
        raise ValueError(f"Player rows are missing columns: {sorted(missing_columns)}")

    if player_rows.empty:
        raise ValueError("Player rows cannot be empty")

    with_rolling_minutes = compute_player_rolling_minutes(player_rows)
    ratio_table = compute_core_realized_ratio(with_rolling_minutes)
    continuity = compute_rolling_continuity(ratio_table)

    season_ids = (
        player_rows[["SEASON", "SEASON_ID"]].drop_duplicates().set_index("SEASON")["SEASON_ID"]
    )

    continuity["SEASON_ID"] = continuity["SEASON"].map(season_ids)

    output = continuity.loc[:, list(OUTPUT_COLUMNS)].reset_index(drop=True)

    duplicate_rows = int(output.duplicated(subset=["SEASON", "TEAM_ID", "GAME_ID"]).sum())

    if duplicate_rows:
        raise ValueError(
            f"Roster continuity output contains {duplicate_rows} duplicate team-game rows"
        )

    summary = RosterContinuitySummary(
        source_player_rows=len(player_rows),
        output_team_game_rows=len(output),
        unique_teams=int(output["TEAM_ID"].nunique()),
        unique_players=int(player_rows["PLAYER_ID"].nunique()),
        seasons=int(output["SEASON"].nunique()),
        rows_with_missing_continuity=int(output["ROLLING_CORE_CONTINUITY"].isna().sum()),
        first_game_date=(output["GAME_DATE"].min().date().isoformat()),
        last_game_date=(output["GAME_DATE"].max().date().isoformat()),
    )

    return output, summary


def write_roster_continuity_outputs(
    continuity: pd.DataFrame,
    summary: RosterContinuitySummary,
    project_root: Path,
) -> tuple[Path, Path]:
    """Write the roster-continuity feature table and its summary."""

    output_path = roster_continuity_output_path(project_root)
    summary_path = roster_continuity_summary_path(project_root)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    continuity.to_parquet(output_path, index=False)

    summary_path.write_text(
        json.dumps(asdict(summary), indent=2) + "\n",
        encoding="utf-8",
    )

    return output_path, summary_path


def build_roster_continuity_dataset(
    project_root: Path,
    seasons: tuple[str, ...] = DEFAULT_SEASONS,
) -> RosterContinuitySummary:
    """Read raw player logs and write the roster-continuity feature table."""

    player_rows = load_player_game_logs(project_root, seasons)

    continuity, summary = build_roster_continuity_features(player_rows)

    output_path, summary_path = write_roster_continuity_outputs(
        continuity=continuity,
        summary=summary,
        project_root=project_root,
    )

    print("\nRoster-continuity feature build complete:")
    print(json.dumps(asdict(summary), indent=2))
    print(f"Output: {output_path}")
    print(f"Summary: {summary_path}")

    return summary


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the roster-continuity build."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seasons",
        nargs="+",
        default=list(DEFAULT_SEASONS),
        help="NBA seasons to include, such as 2022-23 2023-24 2024-25",
    )
    return parser.parse_args()


def main() -> None:
    """Build the roster-continuity feature dataset from the command line."""

    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]

    build_roster_continuity_dataset(
        project_root=project_root,
        seasons=tuple(args.seasons),
    )


if __name__ == "__main__":
    main()
