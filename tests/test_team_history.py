"""Tests for the chronological NBA team-history transformation.

These tests use small in-memory game datasets and temporary directories.
They verify perspective mapping, chronological ordering, team-game numbering,
validation, and output persistence without reading the real project data.
"""

from pathlib import Path

import pandas as pd
import pytest
from pipelines.features.build_team_history import (
    build_away_team_rows,
    build_home_team_rows,
    build_team_history,
    team_history_output_path,
    team_history_summary_path,
    validate_game_dataset_input,
    write_team_history_outputs,
)


def make_game(
    *,
    season: str = "2024-25",
    game_id: str = "0022400001",
    game_date: str = "2024-10-22",
    home_team_id: int = 1610612747,
    home_abbreviation: str = "LAL",
    away_team_id: int = 1610612738,
    away_abbreviation: str = "BOS",
    home_won: bool = True,
) -> pd.DataFrame:
    """Create one valid completed NBA game row."""

    home_points = 112 if home_won else 101
    away_points = 101 if home_won else 112

    return pd.DataFrame(
        [
            {
                "SEASON": season,
                "SEASON_ID": f"2{season[:4]}",
                "GAME_ID": game_id,
                "GAME_DATE": game_date,
                "HOME_TEAM_ID": home_team_id,
                "HOME_TEAM_ABBREVIATION": home_abbreviation,
                "HOME_WL": "W" if home_won else "L",
                "HOME_PTS": home_points,
                "AWAY_TEAM_ID": away_team_id,
                "AWAY_TEAM_ABBREVIATION": away_abbreviation,
                "AWAY_WL": "L" if home_won else "W",
                "AWAY_PTS": away_points,
                "home_win": int(home_won),
            }
        ]
    )


def test_validate_game_dataset_input_rejects_missing_columns() -> None:
    """Feature construction should fail when upstream columns are absent."""

    games = make_game().drop(columns=["AWAY_PTS"])

    with pytest.raises(
        ValueError,
        match="missing required columns",
    ):
        validate_game_dataset_input(games)


def test_validate_game_dataset_input_rejects_duplicate_games() -> None:
    """The same season and GAME_ID should not appear more than once."""

    game = make_game()

    games = pd.concat(
        [
            game,
            game,
        ],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match="duplicate SEASON/GAME_ID",
    ):
        validate_game_dataset_input(games)


def test_build_home_team_rows_maps_home_perspective() -> None:
    """Home columns should become generic team-history columns."""

    games = make_game()

    home_rows = build_home_team_rows(games)

    assert len(home_rows) == 1

    row = home_rows.iloc[0]

    assert row["TEAM_ID"] == 1610612747
    assert row["TEAM_ABBREVIATION"] == "LAL"
    assert row["OPPONENT_TEAM_ID"] == 1610612738
    assert row["OPPONENT_TEAM_ABBREVIATION"] == "BOS"
    assert row["TEAM_PTS"] == 112
    assert row["OPPONENT_PTS"] == 101
    assert row["IS_HOME"] == 1
    assert row["TEAM_WIN"] == 1


def test_build_away_team_rows_maps_away_perspective() -> None:
    """Away columns should become generic team-history columns."""

    games = make_game()

    away_rows = build_away_team_rows(games)

    assert len(away_rows) == 1

    row = away_rows.iloc[0]

    assert row["TEAM_ID"] == 1610612738
    assert row["TEAM_ABBREVIATION"] == "BOS"
    assert row["OPPONENT_TEAM_ID"] == 1610612747
    assert row["OPPONENT_TEAM_ABBREVIATION"] == "LAL"
    assert row["TEAM_PTS"] == 101
    assert row["OPPONENT_PTS"] == 112
    assert row["IS_HOME"] == 0
    assert row["TEAM_WIN"] == 0


def test_build_team_history_creates_two_rows_per_game() -> None:
    """Every completed game should produce two opposing team perspectives."""

    games = make_game()

    team_history, summary = build_team_history(games)

    assert len(team_history) == 2
    assert team_history["GAME_ID"].nunique() == 1
    assert team_history["IS_HOME"].sum() == 1
    assert team_history["TEAM_WIN"].sum() == 1
    assert team_history["POINT_DIFFERENTIAL"].sum() == 0

    assert summary.source_game_rows == 1
    assert summary.source_games == 1
    assert summary.output_team_rows == 2
    assert summary.unique_teams == 2
    assert summary.seasons == 1


def test_build_team_history_sorts_and_numbers_each_team() -> None:
    """Team-game numbers should follow chronological order, not input order."""

    later_game = make_game(
        game_id="0022400002",
        game_date="2024-10-24",
        home_team_id=1610612738,
        home_abbreviation="BOS",
        away_team_id=1610612747,
        away_abbreviation="LAL",
        home_won=False,
    )

    earlier_game = make_game(
        game_id="0022400001",
        game_date="2024-10-22",
        home_team_id=1610612747,
        home_abbreviation="LAL",
        away_team_id=1610612744,
        away_abbreviation="GSW",
        home_won=True,
    )

    # Deliberately provide the later game first.
    games = pd.concat(
        [
            later_game,
            earlier_game,
        ],
        ignore_index=True,
    )

    team_history, _ = build_team_history(games)

    lakers_history = team_history.loc[team_history["TEAM_ID"] == 1610612747].reset_index(drop=True)

    assert lakers_history["GAME_ID"].tolist() == [
        "0022400001",
        "0022400002",
    ]

    assert lakers_history["TEAM_GAME_NUMBER"].tolist() == [
        1,
        2,
    ]

    assert lakers_history["GAME_DATE"].tolist() == [
        pd.Timestamp("2024-10-22"),
        pd.Timestamp("2024-10-24"),
    ]


def test_team_game_number_resets_for_each_season() -> None:
    """A team's sequence should restart at one at the beginning of a season."""

    first_season_game = make_game(
        season="2023-24",
        game_id="0022300001",
        game_date="2023-10-24",
    )

    second_season_game = make_game(
        season="2024-25",
        game_id="0022400001",
        game_date="2024-10-22",
    )

    games = pd.concat(
        [
            first_season_game,
            second_season_game,
        ],
        ignore_index=True,
    )

    team_history, _ = build_team_history(games)

    lakers_history = team_history.loc[team_history["TEAM_ID"] == 1610612747].reset_index(drop=True)

    assert lakers_history["SEASON"].tolist() == [
        "2023-24",
        "2024-25",
    ]

    assert lakers_history["TEAM_GAME_NUMBER"].tolist() == [
        1,
        1,
    ]


def test_team_history_paths_use_processed_features_directory(
    tmp_path: Path,
) -> None:
    """Team history and metadata should share the features directory."""

    expected_directory = tmp_path / "data" / "processed" / "nba" / "features"

    assert team_history_output_path(tmp_path) == (expected_directory / "team_history.parquet")

    assert team_history_summary_path(tmp_path) == (expected_directory / "team_history_summary.json")


def test_write_team_history_outputs_creates_files(
    tmp_path: Path,
) -> None:
    """The team-history Parquet and summary metadata should be persisted."""

    games = make_game()

    team_history, summary = build_team_history(games)

    history_path, summary_path = write_team_history_outputs(
        team_history=team_history,
        summary=summary,
        project_root=tmp_path,
    )

    assert history_path.exists()
    assert summary_path.exists()

    saved_history = pd.read_parquet(history_path)

    assert len(saved_history) == 2
    assert saved_history["TEAM_ID"].nunique() == 2
    assert saved_history["POINT_DIFFERENTIAL"].sum() == 0

    metadata = summary_path.read_text(
        encoding="utf-8",
    )

    assert '"source_games": 1' in metadata
    assert '"output_team_rows": 2' in metadata
