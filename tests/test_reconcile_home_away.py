"""Tests for resolving ambiguous NBA home and away assignments.

These tests use small in-memory DataFrames, fake resolvers, and temporary
directories. No test calls the live NBA API or ScheduleLeagueV2 endpoint.
"""

from pathlib import Path

import pandas as pd
import pytest
from pipelines.transform.build_game_dataset import (
    games_output_path,
    unresolved_output_path,
)
from pipelines.transform.reconcile_home_away import (
    build_reconciled_game_row,
    build_schedule_lookup,
    normalize_game_id,
    reconcile_ambiguous_games,
    reconcile_season,
    reconciliation_summary_path,
    validate_complete_game_dataset,
)


def make_existing_game(
    game_id: str = "0022400001",
) -> pd.DataFrame:
    """Create one already-resolved game-level record."""

    return pd.DataFrame(
        [
            {
                "GAME_ID": game_id,
                "GAME_DATE": "2024-10-22",
                "SEASON_ID": "22024",
                "HOME_TEAM_ID": 1610612747,
                "HOME_TEAM_ABBREVIATION": "LAL",
                "HOME_WL": "W",
                "HOME_PTS": 112,
                "AWAY_TEAM_ID": 1610612738,
                "AWAY_TEAM_ABBREVIATION": "BOS",
                "AWAY_WL": "L",
                "AWAY_PTS": 108,
                "home_win": 1,
            }
        ]
    )


def make_ambiguous_game(
    game_id: str = "0022400633",
    first_team_id: int = 1610612759,
    second_team_id: int = 1610612754,
    first_abbreviation: str = "SAS",
    second_abbreviation: str = "IND",
) -> pd.DataFrame:
    """Create two source rows whose matchup markers do not identify home."""

    return pd.DataFrame(
        [
            {
                "GAME_ID": game_id,
                "GAME_DATE": "2025-01-23",
                "SEASON_ID": "22024",
                "TEAM_ID": first_team_id,
                "TEAM_ABBREVIATION": first_abbreviation,
                "MATCHUP": (f"{first_abbreviation} @ {second_abbreviation}"),
                "WL": "W",
                "PTS": 140,
            },
            {
                "GAME_ID": game_id,
                "GAME_DATE": "2025-01-23",
                "SEASON_ID": "22024",
                "TEAM_ID": second_team_id,
                "TEAM_ABBREVIATION": second_abbreviation,
                "MATCHUP": (f"{second_abbreviation} @ {first_abbreviation}"),
                "WL": "L",
                "PTS": 110,
            },
        ]
    )


def test_normalize_game_id_preserves_leading_zeroes() -> None:
    """Numeric-looking game IDs should become ten-character strings."""

    assert normalize_game_id("0022400633") == "0022400633"
    assert normalize_game_id("22400633") == "0022400633"
    assert normalize_game_id(22400633) == "0022400633"
    assert normalize_game_id(22400633.0) == "0022400633"


def test_build_schedule_lookup_reads_nested_schedule_payload() -> None:
    """Nested schedule responses should produce official team assignments."""

    payload = {
        "leagueSchedule": {
            "gameDates": [
                {
                    "games": [
                        {
                            "gameId": "0022400633",
                            "homeTeam": {
                                "teamId": 1610612759,
                            },
                            "awayTeam": {
                                "teamId": 1610612754,
                            },
                        }
                    ]
                }
            ]
        }
    }

    lookup = build_schedule_lookup(payload)

    assert lookup == {
        "0022400633": (
            1610612759,
            1610612754,
        )
    }


def test_build_schedule_lookup_reads_flattened_schedule_payload() -> None:
    """Flattened schedule fields should also produce valid assignments."""

    payload = {
        "scheduleRows": [
            {
                "GAME_ID": 22400621.0,
                "HOME_TEAM_ID": "1610612754",
                "AWAY_TEAM_ID": "1610612759",
            }
        ]
    }

    lookup = build_schedule_lookup(payload)

    assert lookup == {
        "0022400621": (
            1610612754,
            1610612759,
        )
    }


def test_build_reconciled_game_row_assigns_official_teams() -> None:
    """Official team IDs should control the final home and away assignment."""

    game_rows = make_ambiguous_game()

    record = build_reconciled_game_row(
        game_rows=game_rows,
        home_team_id=1610612759,
        away_team_id=1610612754,
    )

    assert record["GAME_ID"] == "0022400633"
    assert record["HOME_TEAM_ID"] == 1610612759
    assert record["HOME_TEAM_ABBREVIATION"] == "SAS"
    assert record["HOME_PTS"] == 140
    assert record["AWAY_TEAM_ID"] == 1610612754
    assert record["AWAY_TEAM_ABBREVIATION"] == "IND"
    assert record["AWAY_PTS"] == 110
    assert record["home_win"] == 1
    assert isinstance(record["GAME_DATE"], pd.Timestamp)


def test_reconcile_ambiguous_games_retains_failed_game_rows() -> None:
    """Successful games should resolve while failed games remain available."""

    first_game = make_ambiguous_game()

    second_game = make_ambiguous_game(
        game_id="0022400621",
        first_team_id=1610612754,
        second_team_id=1610612759,
        first_abbreviation="IND",
        second_abbreviation="SAS",
    )

    unresolved_rows = pd.concat(
        [
            first_game,
            second_game,
        ],
        ignore_index=True,
    )

    def fake_resolver(game_id: str) -> tuple[int, int]:
        """Resolve one game and simulate a failure for the other."""

        if game_id == "0022400633":
            return 1610612759, 1610612754

        raise RuntimeError("Simulated lookup failure")

    reconciled_games, remaining_rows = reconcile_ambiguous_games(
        unresolved_rows=unresolved_rows,
        resolver=fake_resolver,
    )

    assert len(reconciled_games) == 1
    assert reconciled_games.loc[0, "GAME_ID"] == "0022400633"

    assert len(remaining_rows) == 2
    assert remaining_rows["GAME_ID"].nunique() == 1
    assert remaining_rows["GAME_ID"].iloc[0] == "0022400621"


def test_validate_complete_game_dataset_accepts_valid_data() -> None:
    """A valid one-row-per-game dataset should pass every invariant."""

    games = make_existing_game()

    validate_complete_game_dataset(
        games=games,
        expected_game_count=1,
    )


def test_validate_complete_game_dataset_rejects_duplicate_games() -> None:
    """Two rows with the same GAME_ID should fail final validation."""

    game = make_existing_game()

    games = pd.concat(
        [
            game,
            game,
        ],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match="duplicate GAME_ID",
    ):
        validate_complete_game_dataset(
            games=games,
            expected_game_count=2,
        )


def test_validate_complete_game_dataset_rejects_wrong_target() -> None:
    """home_win must always agree with the home team's result."""

    games = make_existing_game()
    games.loc[0, "home_win"] = 0

    with pytest.raises(
        ValueError,
        match="inconsistent with HOME_WL",
    ):
        validate_complete_game_dataset(
            games=games,
            expected_game_count=1,
        )


def test_reconcile_season_writes_completed_dataset(
    tmp_path: Path,
) -> None:
    """A successful reconciliation should append and persist the new game."""

    season = "2024-25"

    games_path = games_output_path(
        tmp_path,
        season,
    )
    unresolved_path = unresolved_output_path(
        tmp_path,
        season,
    )
    summary_path = reconciliation_summary_path(
        tmp_path,
        season,
    )

    games_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    make_existing_game().to_parquet(
        games_path,
        index=False,
    )

    make_ambiguous_game().to_parquet(
        unresolved_path,
        index=False,
    )

    def fake_resolver(game_id: str) -> tuple[int, int]:
        """Return the official assignment without making a network call."""

        assert game_id == "0022400633"
        return 1610612759, 1610612754

    summary = reconcile_season(
        season=season,
        project_root=tmp_path,
        resolver=fake_resolver,
    )

    assert summary.previously_resolved_games == 1
    assert summary.ambiguous_games_found == 1
    assert summary.reconciled_games == 1
    assert summary.remaining_unresolved_games == 0
    assert summary.final_game_rows == 2

    saved_games = pd.read_parquet(games_path)

    assert len(saved_games) == 2
    assert saved_games["GAME_ID"].nunique() == 2
    assert summary_path.exists()
    assert '"final_game_rows": 2' in summary_path.read_text(encoding="utf-8")


def test_reconcile_season_does_not_overwrite_on_failure(
    tmp_path: Path,
) -> None:
    """Failed reconciliation must leave the existing dataset unchanged."""

    season = "2024-25"

    games_path = games_output_path(
        tmp_path,
        season,
    )
    unresolved_path = unresolved_output_path(
        tmp_path,
        season,
    )
    summary_path = reconciliation_summary_path(
        tmp_path,
        season,
    )

    games_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    original_games = make_existing_game()

    original_games.to_parquet(
        games_path,
        index=False,
    )

    make_ambiguous_game().to_parquet(
        unresolved_path,
        index=False,
    )

    def failing_resolver(game_id: str) -> tuple[int, int]:
        """Simulate an external schedule and box-score failure."""

        raise RuntimeError(f"No assignment available for {game_id}")

    with pytest.raises(
        ValueError,
        match="1 games remain unresolved",
    ):
        reconcile_season(
            season=season,
            project_root=tmp_path,
            resolver=failing_resolver,
        )

    # The original file must still contain only its initial resolved game.
    saved_games = pd.read_parquet(games_path)

    assert len(saved_games) == 1
    assert saved_games.loc[0, "GAME_ID"] == "0022400001"

    # Failed runs must not create misleading success metadata.
    assert not summary_path.exists()
