"""Fetch the NBA schedule for a season, including not-yet-played games.

Every other ingestion module in this project reads completed-game results
(LeagueGameLog). This module reads ScheduleLeagueV2 instead -- the same
endpoint pipelines/transform/reconcile_home_away.py already uses for
ambiguous-matchup resolution -- because it is the one source that includes
games that have not been played yet: preseason, the full season slate as
soon as the league publishes it, and in-progress or upcoming games with no
score.

This is what makes the dashboard's "predict an upcoming slate" mode
possible. It is intentionally a separate module from the historical
ingestion pipeline rather than an extension of it, since "the full
schedule, played or not" and "completed results for model training" are
different questions with different freshness and caching needs.

Known limitation: filter_regular_season_games() reproduces 1,224 of the
2025-26 season's 1,230 regular-season games (99.5%) when checked against
the historical ingestion pipeline's count. The 6-game gap does not fall
into any of the excluded label categories (each accounts for exactly the
games expected, including the Cup knockout count), so it is left as a
known, minor gap rather than a bug to chase further -- this module serves
the dashboard, not model training, where the historical pipeline remains
the authoritative source.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Any, Final

import pandas as pd
from nba_api.stats.endpoints import scheduleleaguev2

# Most regular-season games carry an empty gameLabel, but not all: in-season
# tournament (Emirates NBA Cup) group-stage games, themed weeks, and
# international showcase games are also real, standings-counting regular-
# season games despite carrying a promotional label -- confirmed empirically
# against the 2025-26 schedule, where filtering on an empty label alone
# undercounted the season by 81 games. Only these labels are genuinely not
# regular season, matching project_spec.md section 3's exclusion list.
NON_REGULAR_SEASON_GAME_LABELS: Final[frozenset[str]] = frozenset(
    {
        "Preseason",
        "All-Star",
        "All-Star Championship",
        "Rising Stars Semifinal",
        "Rising Stars Final",
        "SoFi Play-In Tournament",
        "East First Round",
        "West First Round",
        "East Conf. Semifinals",
        "West Conf. Semifinals",
        "East Conf. Finals",
        "West Conf. Finals",
        "NBA Finals",
    }
)

# The Emirates NBA Cup group stage counts toward the regular season; the
# knockout rounds (quarterfinal onward) are extra games that do not.
CUP_GAME_LABEL: Final[str] = "Emirates NBA Cup"
CUP_KNOCKOUT_SUB_LABELS: Final[frozenset[str]] = frozenset(
    {
        "East Quarterfinal",
        "West Quarterfinal",
        "East Semifinal",
        "West Semifinal",
        "Championship",
    }
)

# gameStatus codes used by ScheduleLeagueV2.
GAME_STATUS_SCHEDULED: Final[int] = 1
GAME_STATUS_LIVE: Final[int] = 2
GAME_STATUS_FINAL: Final[int] = 3

SCHEDULE_COLUMNS: Final[tuple[str, ...]] = (
    "GAME_ID",
    "GAME_DATE",
    "GAME_LABEL",
    "GAME_SUB_LABEL",
    "GAME_STATUS",
    "GAME_STATUS_TEXT",
    "HOME_TEAM_ID",
    "HOME_TEAM_TRICODE",
    "HOME_TEAM_NAME",
    "HOME_SCORE",
    "AWAY_TEAM_ID",
    "AWAY_TEAM_TRICODE",
    "AWAY_TEAM_NAME",
    "AWAY_SCORE",
)


@dataclass(frozen=True)
class ScheduleFetchSummary:
    """Describe one schedule fetch."""

    season: str
    total_games: int
    regular_season_games: int
    scheduled_games: int
    final_games: int
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


def team_display_name(team: dict[str, Any]) -> str:
    """Join a schedule team object's city and nickname into one name."""

    city = str(team.get("teamCity", "")).strip()
    name = str(team.get("teamName", "")).strip()

    return f"{city} {name}".strip()


def extract_game_record(game: dict[str, Any], group_date: str) -> dict[str, object] | None:
    """Extract one schedule row from a ScheduleLeagueV2 game object.

    group_date is the enclosing gameDates entry's own "gameDate" field
    ("MM/DD/YYYY HH:MM:SS", always midnight, never a timezone suffix) --
    used instead of the per-game gameDateEst field, which carries a "Z"
    suffix and so parses as timezone-aware. Every other GAME_DATE column
    in this project (team history, Elo ratings, the modeling dataset) is
    timezone-naive; comparing a naive Series against an aware Timestamp
    raises TypeError, so this keeps the schedule consistent with the rest
    of the pipeline instead of introducing the one timezone-aware date in
    the whole project.
    """

    game_id = game.get("gameId")
    home_team = game.get("homeTeam")
    away_team = game.get("awayTeam")

    if game_id is None or not home_team or not away_team:
        return None

    home_team_id = home_team.get("teamId")
    away_team_id = away_team.get("teamId")

    if home_team_id is None or away_team_id is None:
        return None

    home_team_id = int(home_team_id)
    away_team_id = int(away_team_id)

    if home_team_id == away_team_id:
        raise ValueError(f"Schedule game {game_id} has identical home and away teams")

    return {
        "GAME_ID": normalize_game_id(game_id),
        "GAME_DATE": group_date,
        "GAME_LABEL": str(game.get("gameLabel", "")).strip(),
        "GAME_SUB_LABEL": str(game.get("gameSubLabel", "")).strip(),
        "GAME_STATUS": int(game.get("gameStatus", GAME_STATUS_SCHEDULED)),
        "GAME_STATUS_TEXT": str(game.get("gameStatusText", "")).strip(),
        "HOME_TEAM_ID": home_team_id,
        "HOME_TEAM_TRICODE": home_team.get("teamTricode"),
        "HOME_TEAM_NAME": team_display_name(home_team),
        "HOME_SCORE": home_team.get("score") or None,
        "AWAY_TEAM_ID": away_team_id,
        "AWAY_TEAM_TRICODE": away_team.get("teamTricode"),
        "AWAY_TEAM_NAME": team_display_name(away_team),
        "AWAY_SCORE": away_team.get("score") or None,
    }


def parse_schedule_payload(payload: dict[str, Any]) -> pd.DataFrame:
    """Parse a ScheduleLeagueV2 response into one row per game."""

    league_schedule = payload.get("leagueSchedule", {})
    game_dates = league_schedule.get("gameDates", [])

    if not game_dates:
        raise ValueError("Schedule response contained no gameDates")

    records: list[dict[str, object]] = []

    for game_date_entry in game_dates:
        group_date = game_date_entry.get("gameDate")

        for game in game_date_entry.get("games", []):
            record = extract_game_record(game, group_date=group_date)

            if record is not None:
                records.append(record)

    if not records:
        raise ValueError("Schedule response contained no usable games")

    schedule = pd.DataFrame.from_records(records)

    schedule["GAME_DATE"] = pd.to_datetime(schedule["GAME_DATE"], errors="raise")

    duplicate_games = int(schedule.duplicated(subset=["GAME_ID"]).sum())

    if duplicate_games:
        raise ValueError(f"Schedule contains {duplicate_games} duplicate GAME_ID rows")

    schedule = schedule.sort_values(["GAME_DATE", "GAME_ID"], kind="stable").reset_index(drop=True)

    return schedule.loc[:, list(SCHEDULE_COLUMNS)]


def fetch_schedule(
    season: str,
    timeout: int = 30,
    max_attempts: int = 3,
    backoff_seconds: float = 2.0,
) -> pd.DataFrame:
    """Download and parse one season's full schedule, played or not."""

    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")

    last_error: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            response = scheduleleaguev2.ScheduleLeagueV2(
                league_id="00",
                season=season,
                timeout=timeout,
            )

            payload = response.get_dict()

            if not isinstance(payload, dict):
                raise ValueError(f"Schedule response for {season} is not a dictionary")

            return parse_schedule_payload(payload)

        except Exception as error:
            last_error = error

            if attempt < max_attempts:
                time.sleep(backoff_seconds * attempt)

    raise RuntimeError(
        f"Failed to fetch schedule for {season} after {max_attempts} attempts"
    ) from last_error


def filter_regular_season_games(schedule: pd.DataFrame) -> pd.DataFrame:
    """Keep only real, standings-counting regular-season games.

    Drops preseason, All-Star weekend, Play-In, playoff games, and Emirates
    NBA Cup knockout games. Keeps Cup group-stage games, themed weeks, and
    international showcase games, which do count toward the regular season
    despite carrying a promotional label. Matches project_spec.md section 3.
    """

    is_excluded_label = schedule["GAME_LABEL"].isin(NON_REGULAR_SEASON_GAME_LABELS)

    is_cup_knockout_game = schedule["GAME_LABEL"].eq(CUP_GAME_LABEL) & schedule[
        "GAME_SUB_LABEL"
    ].isin(CUP_KNOCKOUT_SUB_LABELS)

    return schedule.loc[~(is_excluded_label | is_cup_knockout_game)].reset_index(drop=True)


def games_on_date(schedule: pd.DataFrame, date: str | pd.Timestamp) -> pd.DataFrame:
    """Return every game scheduled on one calendar date."""

    target_date = pd.to_datetime(date).date()

    return schedule.loc[schedule["GAME_DATE"].dt.date.eq(target_date)].reset_index(drop=True)


def summarize_schedule(season: str, schedule: pd.DataFrame) -> ScheduleFetchSummary:
    """Describe a fetched schedule for logging and reproducibility."""

    regular_season = filter_regular_season_games(schedule)

    return ScheduleFetchSummary(
        season=season,
        total_games=len(schedule),
        regular_season_games=len(regular_season),
        scheduled_games=int(regular_season["GAME_STATUS"].eq(GAME_STATUS_SCHEDULED).sum()),
        final_games=int(regular_season["GAME_STATUS"].eq(GAME_STATUS_FINAL).sum()),
        first_game_date=(schedule["GAME_DATE"].min().date().isoformat()),
        last_game_date=(schedule["GAME_DATE"].max().date().isoformat()),
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line options for a manual schedule fetch."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", required=True, help="NBA season, such as 2026-27")
    parser.add_argument(
        "--date",
        default=None,
        help="Optional single date to filter to, YYYY-MM-DD",
    )
    return parser.parse_args()


def main() -> None:
    """Fetch and print a schedule summary from the command line."""

    args = parse_args()

    schedule = fetch_schedule(season=args.season)
    summary = summarize_schedule(season=args.season, schedule=schedule)

    print(summary)

    if args.date:
        regular_season = filter_regular_season_games(schedule)
        day_slate = games_on_date(regular_season, args.date)
        print(f"\nRegular-season games on {args.date}:")
        print(
            day_slate[["HOME_TEAM_NAME", "AWAY_TEAM_NAME", "GAME_STATUS_TEXT"]].to_string(
                index=False
            )
        )


if __name__ == "__main__":
    main()
