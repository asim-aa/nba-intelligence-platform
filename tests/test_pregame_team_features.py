"""Tests for leakage-safe NBA pregame team features.

These tests use small chronological team histories to verify shifting,
rolling calculations, rest features, season resets, leakage prevention,
validation, and output persistence without reading the real NBA dataset.
"""

from pathlib import Path

import pandas as pd
import pytest
from pipelines.features.build_pregame_team_features import (
    build_pregame_team_features,
    pregame_team_features_output_path,
    pregame_team_features_summary_path,
    shifted_expanding_mean,
    shifted_rolling_mean,
    validate_team_history_input,
    write_pregame_team_feature_outputs,
)


def make_team_history_sequence() -> pd.DataFrame:
    """Create four chronological completed games for one team."""

    return pd.DataFrame(
        [
            {
                "SEASON": "2024-25",
                "SEASON_ID": "22024",
                "GAME_ID": "0022400001",
                "GAME_DATE": "2024-10-22",
                "TEAM_ID": 1610612747,
                "TEAM_ABBREVIATION": "LAL",
                "OPPONENT_TEAM_ID": 1610612738,
                "OPPONENT_TEAM_ABBREVIATION": "BOS",
                "IS_HOME": 1,
                "TEAM_WIN": 1,
                "TEAM_PTS": 110,
                "OPPONENT_PTS": 100,
                "POINT_DIFFERENTIAL": 10,
                "TEAM_GAME_NUMBER": 1,
            },
            {
                "SEASON": "2024-25",
                "SEASON_ID": "22024",
                "GAME_ID": "0022400002",
                "GAME_DATE": "2024-10-23",
                "TEAM_ID": 1610612747,
                "TEAM_ABBREVIATION": "LAL",
                "OPPONENT_TEAM_ID": 1610612744,
                "OPPONENT_TEAM_ABBREVIATION": "GSW",
                "IS_HOME": 0,
                "TEAM_WIN": 0,
                "TEAM_PTS": 100,
                "OPPONENT_PTS": 105,
                "POINT_DIFFERENTIAL": -5,
                "TEAM_GAME_NUMBER": 2,
            },
            {
                "SEASON": "2024-25",
                "SEASON_ID": "22024",
                "GAME_ID": "0022400003",
                "GAME_DATE": "2024-10-25",
                "TEAM_ID": 1610612747,
                "TEAM_ABBREVIATION": "LAL",
                "OPPONENT_TEAM_ID": 1610612756,
                "OPPONENT_TEAM_ABBREVIATION": "PHX",
                "IS_HOME": 1,
                "TEAM_WIN": 1,
                "TEAM_PTS": 120,
                "OPPONENT_PTS": 115,
                "POINT_DIFFERENTIAL": 5,
                "TEAM_GAME_NUMBER": 3,
            },
            {
                "SEASON": "2024-25",
                "SEASON_ID": "22024",
                "GAME_ID": "0022400004",
                "GAME_DATE": "2024-10-30",
                "TEAM_ID": 1610612747,
                "TEAM_ABBREVIATION": "LAL",
                "OPPONENT_TEAM_ID": 1610612746,
                "OPPONENT_TEAM_ABBREVIATION": "LAC",
                "IS_HOME": 0,
                "TEAM_WIN": 1,
                "TEAM_PTS": 130,
                "OPPONENT_PTS": 125,
                "POINT_DIFFERENTIAL": 5,
                "TEAM_GAME_NUMBER": 4,
            },
        ]
    )


def test_validate_team_history_input_rejects_missing_columns() -> None:
    """Feature construction should fail when a required column is absent."""

    history = make_team_history_sequence().drop(columns=["TEAM_PTS"])

    with pytest.raises(
        ValueError,
        match="missing required columns",
    ):
        validate_team_history_input(history)


def test_validate_team_history_input_rejects_wrong_point_differential() -> None:
    """Point differential must equal team points minus opponent points."""

    history = make_team_history_sequence()
    history.loc[0, "POINT_DIFFERENTIAL"] = 99

    with pytest.raises(
        ValueError,
        match="POINT_DIFFERENTIAL is inconsistent",
    ):
        validate_team_history_input(history)


def test_shifted_expanding_mean_excludes_current_value() -> None:
    """The expanding statistic should use only earlier game results."""

    wins = pd.Series(
        [
            1,
            0,
            1,
            1,
        ],
        dtype="int8",
    )

    result = shifted_expanding_mean(wins)

    assert pd.isna(result.iloc[0])
    assert result.iloc[1] == pytest.approx(1.0)
    assert result.iloc[2] == pytest.approx(0.5)
    assert result.iloc[3] == pytest.approx(2 / 3)


def test_shifted_rolling_mean_excludes_current_and_limits_window() -> None:
    """The rolling statistic should use only the preceding window."""

    wins = pd.Series(
        [
            1,
            0,
            1,
            0,
        ],
        dtype="int8",
    )

    result = shifted_rolling_mean(
        series=wins,
        window=2,
    )

    assert pd.isna(result.iloc[0])
    assert result.iloc[1] == pytest.approx(1.0)
    assert result.iloc[2] == pytest.approx(0.5)

    # Before Game 4, the previous two results are 0 and 1.
    assert result.iloc[3] == pytest.approx(0.5)


def test_build_features_excludes_current_game_outcomes() -> None:
    """The final feature table must not expose same-game results or scores."""

    history = make_team_history_sequence()

    features, summary = build_pregame_team_features(history)

    leaked_columns = {
        "TEAM_WIN",
        "TEAM_WL",
        "TEAM_PTS",
        "OPPONENT_PTS",
        "POINT_DIFFERENTIAL",
    }

    assert leaked_columns.isdisjoint(features.columns)

    first_game = features.loc[features["GAME_ID"] == "0022400001"].iloc[0]

    assert first_game["PRIOR_GAMES_PLAYED"] == 0
    assert pd.isna(first_game["SEASON_WIN_PCT"])
    assert pd.isna(first_game["ROLLING_5_POINTS_SCORED"])
    assert pd.isna(first_game["DAYS_REST"])
    assert first_game["IS_BACK_TO_BACK"] == 0

    assert summary.source_team_rows == 4
    assert summary.output_feature_rows == 4
    assert summary.rows_without_prior_history == 1


def test_build_features_calculates_prior_game_statistics() -> None:
    """Game 4 features should summarize Games 1 through 3 only."""

    history = make_team_history_sequence()

    features, _ = build_pregame_team_features(history)

    fourth_game = features.loc[features["GAME_ID"] == "0022400004"].iloc[0]

    assert fourth_game["PRIOR_GAMES_PLAYED"] == 3

    # Previous results were win, loss, win.
    assert fourth_game["SEASON_WIN_PCT"] == pytest.approx(2 / 3)
    assert fourth_game["ROLLING_5_WIN_PCT"] == pytest.approx(2 / 3)
    assert fourth_game["ROLLING_10_WIN_PCT"] == pytest.approx(2 / 3)

    # Previous points scored were 110, 100, and 120.
    assert fourth_game["ROLLING_5_POINTS_SCORED"] == pytest.approx(110.0)

    assert fourth_game["ROLLING_10_POINTS_SCORED"] == pytest.approx(110.0)

    # Previous points allowed were 100, 105, and 115.
    assert fourth_game["ROLLING_5_POINTS_ALLOWED"] == pytest.approx(320 / 3)

    # Previous point differentials were 10, -5, and 5.
    assert fourth_game["ROLLING_5_POINT_DIFFERENTIAL"] == pytest.approx(10 / 3)


def test_build_features_calculates_rest_and_back_to_back() -> None:
    """Calendar gaps should produce correct rest and back-to-back values."""

    history = make_team_history_sequence()

    features, summary = build_pregame_team_features(history)

    features_by_game = features.set_index("GAME_ID")

    # Oct. 22 to Oct. 23 is a back-to-back with zero full rest days.
    assert features_by_game.loc[
        "0022400002",
        "DAYS_REST",
    ] == pytest.approx(0.0)

    assert (
        features_by_game.loc[
            "0022400002",
            "IS_BACK_TO_BACK",
        ]
        == 1
    )

    # Oct. 23 to Oct. 25 leaves one full day between games.
    assert features_by_game.loc[
        "0022400003",
        "DAYS_REST",
    ] == pytest.approx(1.0)

    assert (
        features_by_game.loc[
            "0022400003",
            "IS_BACK_TO_BACK",
        ]
        == 0
    )

    # Oct. 25 to Oct. 30 leaves four full rest days.
    assert features_by_game.loc[
        "0022400004",
        "DAYS_REST",
    ] == pytest.approx(4.0)

    assert summary.back_to_back_rows == 1


def test_feature_history_resets_at_new_season() -> None:
    """The first appearance in each season should have no prior history."""

    first_season = make_team_history_sequence().iloc[[0]].copy()

    second_season = first_season.copy()
    second_season["SEASON"] = "2025-26"
    second_season["SEASON_ID"] = "22025"
    second_season["GAME_ID"] = "0022500001"
    second_season["GAME_DATE"] = "2025-10-21"
    second_season["TEAM_GAME_NUMBER"] = 1

    history = pd.concat(
        [
            first_season,
            second_season,
        ],
        ignore_index=True,
    )

    features, summary = build_pregame_team_features(history)

    assert features["PRIOR_GAMES_PLAYED"].tolist() == [
        0,
        0,
    ]

    assert features["SEASON_WIN_PCT"].isna().all()
    assert features["DAYS_REST"].isna().all()
    assert summary.rows_without_prior_history == 2
    assert summary.seasons == 2


def test_feature_build_rejects_same_day_team_games() -> None:
    """One team should not have two regular-season games on the same date."""

    history = make_team_history_sequence().iloc[:2].copy()
    history.loc[1, "GAME_DATE"] = history.loc[0, "GAME_DATE"]

    with pytest.raises(
        ValueError,
        match="non-positive gaps",
    ):
        build_pregame_team_features(history)


def test_pregame_feature_paths_use_features_directory(
    tmp_path: Path,
) -> None:
    """Feature data and metadata should share the processed feature folder."""

    expected_directory = tmp_path / "data" / "processed" / "nba" / "features"

    assert pregame_team_features_output_path(tmp_path) == (
        expected_directory / "team_pregame_features.parquet"
    )

    assert pregame_team_features_summary_path(tmp_path) == (
        expected_directory / "team_pregame_features_summary.json"
    )


def test_write_pregame_feature_outputs_creates_files(
    tmp_path: Path,
) -> None:
    """Pregame feature data and summary metadata should be persisted."""

    history = make_team_history_sequence()

    features, summary = build_pregame_team_features(history)

    feature_path, summary_path = write_pregame_team_feature_outputs(
        features=features,
        summary=summary,
        project_root=tmp_path,
    )

    assert feature_path.exists()
    assert summary_path.exists()

    saved_features = pd.read_parquet(feature_path)

    assert len(saved_features) == 4
    assert saved_features["PRIOR_GAMES_PLAYED"].tolist() == [
        0,
        1,
        2,
        3,
    ]

    assert "TEAM_WIN" not in saved_features.columns
    assert "TEAM_PTS" not in saved_features.columns

    metadata = summary_path.read_text(
        encoding="utf-8",
    )

    assert '"source_team_rows": 4' in metadata
    assert '"output_feature_rows": 4' in metadata
    assert '"rows_without_prior_history": 1' in metadata
