"""Resolve ambiguous NBA home/away assignments using official NBA data.

LeagueGameLog occasionally contains games where both team rows use the same
location marker, making home/away impossible to determine from MATCHUP alone.

This module resolves those games using this order:

1. Query ScheduleLeagueV2 once and build a season-wide GAME_ID lookup.
2. Fall back to the NBA live box-score endpoint for any game not found.
3. Refuse to overwrite games.parquet unless every ambiguous game resolves.

The unresolved source rows remain on disk as an audit trail.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

import pandas as pd
from nba_api.live.nba.endpoints import boxscore
from nba_api.stats.endpoints import scheduleleaguev2

from pipelines.transform.build_game_dataset import (
    games_output_path,
    unresolved_output_path,
)

TEAM_RESULT_COLUMNS: Final[tuple[str, ...]] = (
    "TEAM_ID",
    "TEAM_ABBREVIATION",
    "WL",
    "PTS",
)

# A resolver accepts a GAME_ID and returns:
#
#     (official_home_team_id, official_away_team_id)
#
# Keeping this dependency injectable allows tests to use a fake resolver
# without making external requests.
TeamResolver = Callable[[str], tuple[int, int]]


@dataclass(frozen=True)
class ReconciliationSummary:
    """Describe the result of reconciling one NBA season."""

    season: str
    previously_resolved_games: int
    ambiguous_games_found: int
    reconciled_games: int
    remaining_unresolved_games: int
    final_game_rows: int


def reconciliation_summary_path(
    project_root: Path,
    season: str,
) -> Path:
    """Return the metadata path for one reconciliation run."""

    return (
        project_root
        / "data"
        / "processed"
        / "nba"
        / "games"
        / f"season={season}"
        / "reconciliation_summary.json"
    )


def normalize_game_id(value: object) -> str:
    """Normalize a GAME_ID while preserving its leading zeroes."""

    if value is None:
        raise ValueError("GAME_ID cannot be None")

    value_string = str(value).strip()

    # Some tabular parsers may represent numeric IDs with a trailing .0.
    if value_string.endswith(".0"):
        value_string = value_string[:-2]

    return value_string.zfill(10)


def normalize_team_id(value: Any) -> int:
    """Convert one team ID into a validated integer."""

    if value is None:
        raise ValueError("Team ID cannot be None")

    team_id = int(value)

    if team_id <= 0:
        raise ValueError(f"Team ID must be positive, received {team_id}")

    return team_id


def iter_nested_objects(value: Any) -> Iterator[Mapping[str, Any]]:
    """Yield every dictionary contained within a nested JSON-like object."""

    if isinstance(value, Mapping):
        yield value

        for nested_value in value.values():
            yield from iter_nested_objects(nested_value)

    elif isinstance(value, list):
        for item in value:
            yield from iter_nested_objects(item)


def first_present_value(
    record: Mapping[str, Any],
    keys: tuple[str, ...],
) -> Any:
    """Return the first available non-null value from several possible keys."""

    for key in keys:
        value = record.get(key)

        if value is not None:
            return value

    return None


def extract_team_id_from_object(
    team: Any,
    role: str,
) -> int | None:
    """Extract a team ID from either a nested team object or scalar value."""

    if isinstance(team, Mapping):
        value = first_present_value(
            team,
            (
                "teamId",
                "teamID",
                "TEAM_ID",
                "id",
            ),
        )
    else:
        value = team

    if value is None:
        return None

    try:
        return normalize_team_id(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid {role} team ID value: {value!r}") from error


def extract_schedule_game(
    record: Mapping[str, Any],
) -> tuple[str, int, int] | None:
    """Extract GAME_ID, home team ID, and away team ID from one object.

    ScheduleLeagueV2 has used nested and flattened response structures across
    nba_api versions. This extractor supports both forms rather than relying
    on one exact DataFrame column layout.
    """

    game_id_value = first_present_value(
        record,
        (
            "gameId",
            "gameID",
            "GAME_ID",
            "Game_ID",
        ),
    )

    if game_id_value is None:
        return None

    home_team_value = first_present_value(
        record,
        (
            "homeTeam",
            "home_team",
            "HOME_TEAM",
            "homeTeam_teamId",
            "homeTeamTeamId",
            "homeTeamId",
            "HOME_TEAM_ID",
        ),
    )

    away_team_value = first_present_value(
        record,
        (
            "awayTeam",
            "away_team",
            "AWAY_TEAM",
            "awayTeam_teamId",
            "awayTeamTeamId",
            "awayTeamId",
            "AWAY_TEAM_ID",
        ),
    )

    home_team_id = extract_team_id_from_object(
        home_team_value,
        role="home",
    )
    away_team_id = extract_team_id_from_object(
        away_team_value,
        role="away",
    )

    if home_team_id is None or away_team_id is None:
        return None

    if home_team_id == away_team_id:
        raise ValueError(f"Schedule game {game_id_value} has identical team IDs")

    return (
        normalize_game_id(game_id_value),
        home_team_id,
        away_team_id,
    )


def build_schedule_lookup(
    payload: Mapping[str, Any],
) -> dict[str, tuple[int, int]]:
    """Build a GAME_ID-to-home/away lookup from a schedule response."""

    lookup: dict[str, tuple[int, int]] = {}

    for record in iter_nested_objects(payload):
        extracted = extract_schedule_game(record)

        if extracted is None:
            continue

        game_id, home_team_id, away_team_id = extracted
        assignment = (home_team_id, away_team_id)

        existing_assignment = lookup.get(game_id)

        if existing_assignment is not None and existing_assignment != assignment:
            raise ValueError(
                f"Schedule contains conflicting assignments for {game_id}: "
                f"{existing_assignment} versus {assignment}"
            )

        lookup[game_id] = assignment

    if not lookup:
        raise ValueError("Schedule response contained no usable game assignments")

    return lookup


def fetch_schedule_lookup(
    season: str,
    timeout: int = 30,
    max_attempts: int = 3,
    retry_delay_seconds: float = 2.0,
) -> dict[str, tuple[int, int]]:
    """Download one season's schedule and construct a home/away lookup."""

    last_error: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            response = scheduleleaguev2.ScheduleLeagueV2(
                league_id="00",
                season=season,
                timeout=timeout,
            )

            payload = response.get_dict()

            if not isinstance(payload, Mapping):
                raise ValueError(f"Schedule response for {season} is not a dictionary")

            lookup = build_schedule_lookup(payload)

            print(f"Loaded {len(lookup)} schedule assignments for {season}")

            return lookup

        except Exception as error:
            last_error = error

            print(
                f"Schedule {season}: attempt "
                f"{attempt}/{max_attempts} failed: "
                f"{type(error).__name__}: {error}"
            )

            if attempt < max_attempts:
                time.sleep(retry_delay_seconds)

    raise RuntimeError(
        f"Failed to load schedule for {season} after {max_attempts} attempts"
    ) from last_error


def fetch_official_team_ids_from_boxscore(
    game_id: str,
    timeout: int = 30,
    max_attempts: int = 3,
    retry_delay_seconds: float = 2.0,
) -> tuple[int, int]:
    """Fetch official teams from the NBA live box-score endpoint."""

    normalized_game_id = normalize_game_id(game_id)
    last_error: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            response = boxscore.BoxScore(
                game_id=normalized_game_id,
                timeout=timeout,
            )

            payload = response.get_dict()

            if not isinstance(payload, Mapping):
                raise ValueError(f"Box-score response for {normalized_game_id} is not a dictionary")

            game = payload.get("game")

            if not isinstance(game, Mapping):
                raise ValueError(f"Box-score response for {normalized_game_id} has no game object")

            home_team = game.get("homeTeam")
            away_team = game.get("awayTeam")

            home_team_id = extract_team_id_from_object(
                home_team,
                role="home",
            )
            away_team_id = extract_team_id_from_object(
                away_team,
                role="away",
            )

            if home_team_id is None or away_team_id is None:
                raise ValueError(f"Box-score response for {normalized_game_id} is missing team IDs")

            if home_team_id == away_team_id:
                raise ValueError(f"Game {normalized_game_id} has identical team IDs")

            return home_team_id, away_team_id

        except Exception as error:
            last_error = error

            print(
                f"{normalized_game_id}: box-score attempt "
                f"{attempt}/{max_attempts} failed: "
                f"{type(error).__name__}: {error}"
            )

            if attempt < max_attempts:
                time.sleep(retry_delay_seconds)

    raise RuntimeError(
        f"Failed to resolve box score for {normalized_game_id} after {max_attempts} attempts"
    ) from last_error


def create_layered_resolver(
    season: str,
    timeout: int,
    max_attempts: int,
) -> TeamResolver:
    """Create a resolver using the schedule first and box score second."""

    schedule_lookup: dict[str, tuple[int, int]] = {}

    try:
        schedule_lookup = fetch_schedule_lookup(
            season=season,
            timeout=timeout,
            max_attempts=max_attempts,
        )
    except Exception as error:
        # A schedule failure does not immediately stop reconciliation because
        # individual live box-score lookups may still succeed.
        print(
            "Season schedule unavailable; box-score fallback will be used: "
            f"{type(error).__name__}: {error}"
        )

    def resolver(game_id: str) -> tuple[int, int]:
        normalized_game_id = normalize_game_id(game_id)

        schedule_assignment = schedule_lookup.get(normalized_game_id)

        if schedule_assignment is not None:
            print(f"{normalized_game_id}: resolved from season schedule")
            return schedule_assignment

        print(f"{normalized_game_id}: not found in schedule; trying live box score")

        return fetch_official_team_ids_from_boxscore(
            game_id=normalized_game_id,
            timeout=timeout,
            max_attempts=max_attempts,
        )

    return resolver


def select_team_row(
    game_rows: pd.DataFrame,
    team_id: int,
    role: str,
) -> pd.Series:
    """Select exactly one source row for an official team assignment."""

    matching_rows = game_rows.loc[game_rows["TEAM_ID"].astype("int64") == team_id]

    if len(matching_rows) != 1:
        raise ValueError(f"Expected one {role} row for team {team_id}, found {len(matching_rows)}")

    return matching_rows.iloc[0]


def build_reconciled_game_row(
    game_rows: pd.DataFrame,
    home_team_id: int,
    away_team_id: int,
) -> dict[str, Any]:
    """Build one game row using official home and away team assignments."""

    if len(game_rows) != 2:
        game_id = (
            normalize_game_id(game_rows["GAME_ID"].iloc[0]) if not game_rows.empty else "unknown"
        )

        raise ValueError(
            f"Game {game_id} must contain exactly two team rows; found {len(game_rows)}"
        )

    game_ids = {normalize_game_id(value) for value in game_rows["GAME_ID"].tolist()}

    if len(game_ids) != 1:
        raise ValueError("A reconciliation group must contain exactly one GAME_ID")

    game_id = next(iter(game_ids))

    source_team_ids = set(game_rows["TEAM_ID"].astype("int64").tolist())
    official_team_ids = {
        normalize_team_id(home_team_id),
        normalize_team_id(away_team_id),
    }

    # The secondary NBA source must reference exactly the same teams as the
    # original LeagueGameLog rows.
    if source_team_ids != official_team_ids:
        raise ValueError(
            f"Official team IDs for {game_id} do not match source rows: "
            f"official={sorted(official_team_ids)}, "
            f"source={sorted(source_team_ids)}"
        )

    home_row = select_team_row(
        game_rows=game_rows,
        team_id=home_team_id,
        role="home",
    )
    away_row = select_team_row(
        game_rows=game_rows,
        team_id=away_team_id,
        role="away",
    )

    if home_row["WL"] not in {"W", "L"}:
        raise ValueError(f"Game {game_id} has invalid home result {home_row['WL']!r}")

    if away_row["WL"] not in {"W", "L"}:
        raise ValueError(f"Game {game_id} has invalid away result {away_row['WL']!r}")

    if home_row["WL"] == away_row["WL"]:
        raise ValueError(f"Game {game_id} does not contain opposite results")

    return {
        "GAME_ID": game_id,
        "GAME_DATE": pd.to_datetime(
            home_row["GAME_DATE"],
            errors="raise",
        ),
        "SEASON_ID": home_row["SEASON_ID"],
        "HOME_TEAM_ID": int(home_row["TEAM_ID"]),
        "HOME_TEAM_ABBREVIATION": home_row["TEAM_ABBREVIATION"],
        "HOME_WL": home_row["WL"],
        "HOME_PTS": home_row["PTS"],
        "AWAY_TEAM_ID": int(away_row["TEAM_ID"]),
        "AWAY_TEAM_ABBREVIATION": away_row["TEAM_ABBREVIATION"],
        "AWAY_WL": away_row["WL"],
        "AWAY_PTS": away_row["PTS"],
        "home_win": int(home_row["WL"] == "W"),
    }


def reconcile_ambiguous_games(
    unresolved_rows: pd.DataFrame,
    resolver: TeamResolver,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Resolve ambiguous games and retain rows for failed assignments."""

    if unresolved_rows.empty:
        return pd.DataFrame(), unresolved_rows.copy()

    reconciled_records: list[dict[str, Any]] = []
    failed_game_ids: list[str] = []

    grouped_games = unresolved_rows.groupby(
        "GAME_ID",
        sort=False,
    )

    for game_id_value, game_rows in grouped_games:
        game_id = normalize_game_id(game_id_value)

        print(f"Resolving game {game_id}...")

        try:
            home_team_id, away_team_id = resolver(game_id)

            record = build_reconciled_game_row(
                game_rows=game_rows,
                home_team_id=home_team_id,
                away_team_id=away_team_id,
            )

            reconciled_records.append(record)

            print(f"{game_id}: home={home_team_id}, away={away_team_id}")

        except Exception as error:
            failed_game_ids.append(game_id)

            print(f"{game_id}: unresolved ({type(error).__name__}: {error})")

    reconciled_games = pd.DataFrame(reconciled_records)

    if not reconciled_games.empty:
        reconciled_games = reconciled_games.sort_values(
            ["GAME_DATE", "GAME_ID"],
            kind="stable",
        ).reset_index(drop=True)

    normalized_source_ids = unresolved_rows["GAME_ID"].map(normalize_game_id)

    remaining_rows = unresolved_rows.loc[normalized_source_ids.isin(failed_game_ids)].copy()

    return reconciled_games, remaining_rows


def validate_complete_game_dataset(
    games: pd.DataFrame,
    expected_game_count: int,
) -> None:
    """Enforce invariants for a complete one-row-per-game dataset."""

    required_columns = {
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

    missing_columns = required_columns - set(games.columns)

    if missing_columns:
        raise ValueError(f"Final game dataset is missing columns: {sorted(missing_columns)}")

    if len(games) != expected_game_count:
        raise ValueError(f"Expected {expected_game_count} final games, found {len(games)}")

    normalized_game_ids = games["GAME_ID"].map(normalize_game_id)

    duplicate_games = int(normalized_game_ids.duplicated().sum())

    if duplicate_games:
        raise ValueError(f"Final dataset contains {duplicate_games} duplicate GAME_IDs")

    columns_requiring_values = (
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
    )

    missing_value_columns = [
        column for column in columns_requiring_values if games[column].isna().any()
    ]

    if missing_value_columns:
        raise ValueError(f"Final dataset contains missing values in: {missing_value_columns}")

    same_team_mask = games["HOME_TEAM_ID"].astype("int64") == games["AWAY_TEAM_ID"].astype("int64")

    if same_team_mask.any():
        raise ValueError("Final dataset contains identical home and away teams")

    valid_results = ((games["HOME_WL"] == "W") & (games["AWAY_WL"] == "L")) | (
        (games["HOME_WL"] == "L") & (games["AWAY_WL"] == "W")
    )

    if not valid_results.all():
        raise ValueError("Final dataset contains invalid home/away results")

    expected_target = (games["HOME_WL"] == "W").astype("int8")

    actual_target = games["home_win"].astype("int8")

    if not expected_target.equals(actual_target):
        raise ValueError("Final dataset contains home_win values inconsistent with HOME_WL")


def reconcile_season(
    season: str,
    project_root: Path,
    resolver: TeamResolver,
) -> ReconciliationSummary:
    """Reconcile one season and write the completed game dataset."""

    games_path = games_output_path(project_root, season)
    unresolved_path = unresolved_output_path(
        project_root,
        season,
    )
    summary_path = reconciliation_summary_path(
        project_root,
        season,
    )

    if not games_path.exists():
        raise FileNotFoundError(f"Resolved game file does not exist: {games_path}")

    if not unresolved_path.exists():
        raise FileNotFoundError(f"Unresolved team-row file does not exist: {unresolved_path}")

    existing_games = pd.read_parquet(games_path)
    unresolved_rows = pd.read_parquet(unresolved_path)

    ambiguous_game_count = int(unresolved_rows["GAME_ID"].map(normalize_game_id).nunique())

    reconciled_games, remaining_rows = reconcile_ambiguous_games(
        unresolved_rows=unresolved_rows,
        resolver=resolver,
    )

    final_games = pd.concat(
        [existing_games, reconciled_games],
        ignore_index=True,
    )

    final_games["GAME_ID"] = final_games["GAME_ID"].map(normalize_game_id)
    final_games["GAME_DATE"] = pd.to_datetime(
        final_games["GAME_DATE"],
        errors="raise",
    )

    final_games = final_games.sort_values(
        ["GAME_DATE", "GAME_ID"],
        kind="stable",
    ).reset_index(drop=True)

    remaining_unresolved_games = int(
        remaining_rows["GAME_ID"].map(normalize_game_id).nunique()
        if not remaining_rows.empty
        else 0
    )

    summary = ReconciliationSummary(
        season=season,
        previously_resolved_games=len(existing_games),
        ambiguous_games_found=ambiguous_game_count,
        reconciled_games=len(reconciled_games),
        remaining_unresolved_games=remaining_unresolved_games,
        final_game_rows=len(final_games),
    )

    # This check must occur before complete-dataset validation. A reduced row
    # count is expected whenever an external lookup has failed.
    if summary.remaining_unresolved_games:
        raise ValueError(
            f"{summary.remaining_unresolved_games} games remain "
            f"unresolved; {summary.reconciled_games} of "
            f"{summary.ambiguous_games_found} were reconciled"
        )

    if summary.reconciled_games != summary.ambiguous_games_found:
        raise ValueError("Not every ambiguous game produced one reconciled row")

    expected_game_count = summary.previously_resolved_games + summary.ambiguous_games_found

    validate_complete_game_dataset(
        games=final_games,
        expected_game_count=expected_game_count,
    )

    # No project output is overwritten until all lookups and invariants have
    # succeeded. This makes reconciliation effectively transactional.
    final_games.to_parquet(
        games_path,
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

    print("\nReconciliation complete:")
    print(json.dumps(asdict(summary), indent=2))
    print(f"Completed games: {games_path}")
    print(f"Summary: {summary_path}")

    return summary


def parse_args() -> argparse.Namespace:
    """Parse command-line reconciliation options."""

    parser = argparse.ArgumentParser(
        description=__doc__,
    )

    parser.add_argument(
        "--season",
        required=True,
        help="NBA season to reconcile, such as 2024-25",
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Seconds allowed for each NBA request",
    )

    parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="Maximum attempts for each NBA data source",
    )

    return parser.parse_args()


def main() -> None:
    """Run season reconciliation from the command line."""

    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]

    resolver = create_layered_resolver(
        season=args.season,
        timeout=args.timeout,
        max_attempts=args.max_attempts,
    )

    reconcile_season(
        season=args.season,
        project_root=project_root,
        resolver=resolver,
    )


if __name__ == "__main__":
    main()
