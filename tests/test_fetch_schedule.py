"""Tests for the NBA schedule fetcher and regular-season filter."""

import pandas as pd
import pytest
from pipelines.ingestion.fetch_schedule import (
    GAME_STATUS_FINAL,
    GAME_STATUS_SCHEDULED,
    filter_regular_season_games,
    games_on_date,
    parse_schedule_payload,
    season_for_date,
    summarize_schedule,
)


def make_game(
    *,
    game_id: str,
    home_id: int = 1610612747,
    away_id: int = 1610612738,
    status: int = GAME_STATUS_FINAL,
    status_text: str = "Final",
    label: str = "",
    sub_label: str = "",
    home_score: str = "110",
    away_score: str = "100",
) -> dict:
    return {
        "gameId": game_id,
        "gameStatus": status,
        "gameStatusText": status_text,
        "gameLabel": label,
        "gameSubLabel": sub_label,
        "homeTeam": {
            "teamId": home_id,
            "teamCity": "Los Angeles",
            "teamName": "Lakers",
            "teamTricode": "LAL",
            "score": home_score,
        },
        "awayTeam": {
            "teamId": away_id,
            "teamCity": "Boston",
            "teamName": "Celtics",
            "teamTricode": "BOS",
            "score": away_score,
        },
    }


def make_payload(games_by_date: dict[str, list[dict]]) -> dict:
    """Build a ScheduleLeagueV2-shaped payload from {date: [games]}.

    Every game within one date shares that date's own group-level
    "gameDate" field, matching how the real endpoint nests games under a
    date entry rather than dating each game individually.
    """

    return {
        "leagueSchedule": {
            "gameDates": [
                {"gameDate": f"{date} 00:00:00", "games": games}
                for date, games in games_by_date.items()
            ]
        }
    }


def test_parse_schedule_payload_extracts_expected_fields() -> None:
    payload = make_payload({"11/05/2025": [make_game(game_id="0022500001")]})

    schedule = parse_schedule_payload(payload)

    assert len(schedule) == 1
    row = schedule.iloc[0]
    assert row["GAME_ID"] == "0022500001"
    assert row["HOME_TEAM_ID"] == 1610612747
    assert row["HOME_TEAM_NAME"] == "Los Angeles Lakers"
    assert row["AWAY_TEAM_NAME"] == "Boston Celtics"
    assert row["HOME_SCORE"] == "110"
    assert row["GAME_DATE"] == pd.Timestamp("2025-11-05")


def test_parse_schedule_payload_rejects_identical_teams() -> None:
    payload = make_payload({"11/05/2025": [make_game(game_id="0022500001", home_id=1, away_id=1)]})

    with pytest.raises(ValueError, match="identical"):
        parse_schedule_payload(payload)


def test_parse_schedule_payload_rejects_duplicate_game_ids() -> None:
    payload = make_payload(
        {
            "11/05/2025": [make_game(game_id="0022500001")],
            "11/06/2025": [make_game(game_id="0022500001")],
        }
    )

    with pytest.raises(ValueError, match="duplicate"):
        parse_schedule_payload(payload)


def test_parse_schedule_payload_rejects_empty_gamedates() -> None:
    with pytest.raises(ValueError, match="no gameDates"):
        parse_schedule_payload({"leagueSchedule": {"gameDates": []}})


def test_parse_schedule_payload_dates_are_timezone_naive() -> None:
    payload = make_payload({"11/05/2025": [make_game(game_id="0022500001")]})

    schedule = parse_schedule_payload(payload)

    assert schedule["GAME_DATE"].dt.tz is None


def test_filter_regular_season_games_keeps_unlabeled_games() -> None:
    schedule = parse_schedule_payload(
        make_payload({"11/05/2025": [make_game(game_id="0022500001", label="")]})
    )

    filtered = filter_regular_season_games(schedule)

    assert len(filtered) == 1


def test_filter_regular_season_games_drops_preseason() -> None:
    schedule = parse_schedule_payload(
        make_payload({"10/02/2025": [make_game(game_id="0012500001", label="Preseason")]})
    )

    filtered = filter_regular_season_games(schedule)

    assert filtered.empty


def test_filter_regular_season_games_drops_playoffs_and_play_in() -> None:
    schedule = parse_schedule_payload(
        make_payload(
            {
                "04/20/2026": [
                    make_game(game_id="0042500001", label="West First Round"),
                ],
                "04/18/2026": [
                    make_game(game_id="0052500001", label="SoFi Play-In Tournament"),
                ],
            }
        )
    )

    filtered = filter_regular_season_games(schedule)

    assert filtered.empty


def test_filter_regular_season_games_keeps_cup_group_stage() -> None:
    schedule = parse_schedule_payload(
        make_payload(
            {
                "11/10/2025": [
                    make_game(
                        game_id="0022500100",
                        label="Emirates NBA Cup",
                        sub_label="East Group A",
                    )
                ]
            }
        )
    )

    filtered = filter_regular_season_games(schedule)

    assert len(filtered) == 1


def test_filter_regular_season_games_drops_cup_knockout_rounds() -> None:
    schedule = parse_schedule_payload(
        make_payload(
            {
                "12/16/2025": [
                    make_game(
                        game_id="0022500200",
                        label="Emirates NBA Cup",
                        sub_label="Championship",
                    ),
                ],
                "12/09/2025": [
                    make_game(
                        game_id="0022500201",
                        label="Emirates NBA Cup",
                        sub_label="East Quarterfinal",
                    ),
                ],
            }
        )
    )

    filtered = filter_regular_season_games(schedule)

    assert filtered.empty


def test_filter_regular_season_games_keeps_international_showcases() -> None:
    schedule = parse_schedule_payload(
        make_payload({"01/15/2026": [make_game(game_id="0022500300", label="NBA London Game")]})
    )

    filtered = filter_regular_season_games(schedule)

    assert len(filtered) == 1


def test_games_on_date_filters_to_one_calendar_day() -> None:
    schedule = parse_schedule_payload(
        make_payload(
            {
                "11/05/2025": [make_game(game_id="0022500001")],
                "11/06/2025": [make_game(game_id="0022500002")],
            }
        )
    )

    day_slate = games_on_date(schedule, "2025-11-05")

    assert len(day_slate) == 1
    assert day_slate.iloc[0]["GAME_ID"] == "0022500001"


def test_summarize_schedule_counts_scheduled_and_final_games() -> None:
    schedule = parse_schedule_payload(
        make_payload(
            {
                "11/05/2025": [
                    make_game(
                        game_id="0022500001",
                        status=GAME_STATUS_FINAL,
                        status_text="Final",
                    )
                ],
                "01/01/2026": [
                    make_game(
                        game_id="0022500002",
                        status=GAME_STATUS_SCHEDULED,
                        status_text="7:00 pm ET",
                        home_score=None,
                        away_score=None,
                    )
                ],
            }
        )
    )

    summary = summarize_schedule("2025-26", schedule)

    assert summary.total_games == 2
    assert summary.regular_season_games == 2
    assert summary.final_games == 1
    assert summary.scheduled_games == 1


def test_season_for_date_regular_season_month() -> None:
    assert season_for_date(pd.Timestamp("2025-11-05")) == "2025-26"


def test_season_for_date_spring_still_belongs_to_prior_fall() -> None:
    assert season_for_date(pd.Timestamp("2026-03-15")) == "2025-26"


def test_season_for_date_new_season_start() -> None:
    assert season_for_date(pd.Timestamp("2026-10-15")) == "2026-27"


def test_season_for_date_august_cutover() -> None:
    assert season_for_date(pd.Timestamp("2026-07-31")) == "2025-26"
    assert season_for_date(pd.Timestamp("2026-08-01")) == "2026-27"
