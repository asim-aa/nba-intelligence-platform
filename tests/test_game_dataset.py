"""Tests for converting raw NBA team-game rows into one row per game.

These tests use small in-memory DataFrames and temporary directories. They
verify the transformation logic without reading the project's real NBA data.
"""

from pathlib import Path

import pandas as pd
import pytest
from pipelines.transform.build_game_dataset import (
    build_resolved_game_rows,
    games_output_path,
    identify_standard_games,
    summary_output_path,
    transform_team_game_rows,
    unresolved_output_path,
    write_transformation_outputs,
)


def make_standard_game(
    game_id: str = "0022400001",
    home_won: bool = True,
) -> pd.DataFrame:
    """Create one valid game with one home row and one away row."""

    home_points = 110 if home_won else 101
    away_points = 101 if home_won else 110

    return pd.DataFrame(
        [
            {
                "GAME_ID": game_id,
                "GAME_DATE": "2024-10-22",
                "SEASON_ID": "22024",
                "TEAM_ID": 1,
                "TEAM_ABBREVIATION": "AAA",
                "MATCHUP": "AAA vs. BBB",
                "WL": "W" if home_won else "L",
                "PTS": home_points,
                "FG_PCT": 0.50,
                "FG3_PCT": 0.40,
                "FT_PCT": 0.80,
                "REB": 45,
                "AST": 25,
                "TOV": 12,
                "STL": 8,
                "BLK": 5,
            },
            {
                "GAME_ID": game_id,
                "GAME_DATE": "2024-10-22",
                "SEASON_ID": "22024",
                "TEAM_ID": 2,
                "TEAM_ABBREVIATION": "BBB",
                "MATCHUP": "BBB @ AAA",
                "WL": "L" if home_won else "W",
                "PTS": away_points,
                "FG_PCT": 0.46,
                "FG3_PCT": 0.35,
                "FT_PCT": 0.75,
                "REB": 40,
                "AST": 22,
                "TOV": 14,
                "STL": 6,
                "BLK": 4,
            },
        ]
    )


def make_ambiguous_game(
    game_id: str = "0022400002",
) -> pd.DataFrame:
    """Create a game where both source rows incorrectly use an away marker."""

    return pd.DataFrame(
        [
            {
                "GAME_ID": game_id,
                "GAME_DATE": "2024-11-01",
                "SEASON_ID": "22024",
                "TEAM_ID": 3,
                "TEAM_ABBREVIATION": "CCC",
                "MATCHUP": "CCC @ DDD",
                "WL": "W",
                "PTS": 115,
                "FG_PCT": 0.51,
                "FG3_PCT": 0.39,
                "FT_PCT": 0.82,
                "REB": 43,
                "AST": 27,
                "TOV": 10,
                "STL": 7,
                "BLK": 6,
            },
            {
                "GAME_ID": game_id,
                "GAME_DATE": "2024-11-01",
                "SEASON_ID": "22024",
                "TEAM_ID": 4,
                "TEAM_ABBREVIATION": "DDD",
                "MATCHUP": "DDD @ CCC",
                "WL": "L",
                "PTS": 107,
                "FG_PCT": 0.47,
                "FG3_PCT": 0.34,
                "FT_PCT": 0.77,
                "REB": 39,
                "AST": 21,
                "TOV": 15,
                "STL": 5,
                "BLK": 3,
            },
        ]
    )


def test_identify_standard_games_marks_valid_pair() -> None:
    """A game with one vs. row and one @ row should be resolvable."""

    frame = make_standard_game()

    result = identify_standard_games(frame)

    assert bool(result.loc["0022400001"]) is True


def test_identify_standard_games_marks_ambiguous_pair() -> None:
    """A game with two @ rows should not be treated as standard."""

    frame = make_ambiguous_game()

    result = identify_standard_games(frame)

    assert bool(result.loc["0022400002"]) is False


def test_build_resolved_game_rows_assigns_home_and_away() -> None:
    """The vs. row should become home and the @ row should become away."""

    frame = make_standard_game()

    games = build_resolved_game_rows(frame)

    assert len(games) == 1

    game = games.iloc[0]

    assert game["HOME_TEAM_ID"] == 1
    assert game["HOME_TEAM_ABBREVIATION"] == "AAA"
    assert game["AWAY_TEAM_ID"] == 2
    assert game["AWAY_TEAM_ABBREVIATION"] == "BBB"
    assert game["HOME_PTS"] == 110
    assert game["AWAY_PTS"] == 101


@pytest.mark.parametrize(
    ("home_won", "expected_target"),
    [
        (True, 1),
        (False, 0),
    ],
)
def test_build_resolved_game_rows_creates_home_win_target(
    home_won: bool,
    expected_target: int,
) -> None:
    """The target should reflect whether the designated home team won."""

    frame = make_standard_game(home_won=home_won)

    games = build_resolved_game_rows(frame)

    assert games.loc[0, "home_win"] == expected_target


def test_transform_team_game_rows_separates_ambiguous_games() -> None:
    """Resolved games should become game rows while ambiguous rows remain."""

    frame = pd.concat(
        [
            make_standard_game(),
            make_ambiguous_game(),
        ],
        ignore_index=True,
    )

    games, unresolved_rows, summary = transform_team_game_rows(
        frame=frame,
        season="2024-25",
    )

    assert len(games) == 1
    assert len(unresolved_rows) == 2
    assert unresolved_rows["GAME_ID"].nunique() == 1

    assert summary.source_team_rows == 4
    assert summary.source_games == 2
    assert summary.resolved_games == 1
    assert summary.unresolved_games == 1
    assert summary.output_game_rows == 1


def test_transform_team_game_rows_rejects_duplicate_team_rows() -> None:
    """Duplicate GAME_ID and TEAM_ID pairs should fail before transformation."""

    frame = make_standard_game()
    duplicate_row = frame.iloc[[0]].copy()
    frame = pd.concat([frame, duplicate_row], ignore_index=True)

    with pytest.raises(
        ValueError,
        match="duplicate GAME_ID/TEAM_ID",
    ):
        transform_team_game_rows(
            frame=frame,
            season="2024-25",
        )


def test_output_paths_use_processed_season_partition(
    tmp_path: Path,
) -> None:
    """All transformation outputs should share the same season directory."""

    expected_directory = tmp_path / "data" / "processed" / "nba" / "games" / "season=2024-25"

    assert games_output_path(tmp_path, "2024-25") == (expected_directory / "games.parquet")

    assert unresolved_output_path(tmp_path, "2024-25") == (
        expected_directory / "unresolved_team_game_rows.parquet"
    )

    assert summary_output_path(tmp_path, "2024-25") == (
        expected_directory / "transformation_summary.json"
    )


def test_write_transformation_outputs_creates_all_files(
    tmp_path: Path,
) -> None:
    """Resolved data, unresolved data, and metadata should all be persisted."""

    frame = pd.concat(
        [
            make_standard_game(),
            make_ambiguous_game(),
        ],
        ignore_index=True,
    )

    games, unresolved_rows, summary = transform_team_game_rows(
        frame=frame,
        season="2024-25",
    )

    games_path, unresolved_path, summary_path = write_transformation_outputs(
        games=games,
        unresolved_rows=unresolved_rows,
        summary=summary,
        project_root=tmp_path,
    )

    assert games_path.exists()
    assert unresolved_path.exists()
    assert summary_path.exists()

    saved_games = pd.read_parquet(games_path)
    saved_unresolved = pd.read_parquet(unresolved_path)

    assert len(saved_games) == 1
    assert len(saved_unresolved) == 2
    assert '"resolved_games": 1' in summary_path.read_text(encoding="utf-8")
