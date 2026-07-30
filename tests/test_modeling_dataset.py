"""Tests for the final leakage-safe NBA modeling dataset.

These tests verify the home/away join, matchup differences, history flags,
target preservation, leakage prevention, validation, and output persistence
without reading real project data.
"""

import json
from pathlib import Path
from typing import cast

import pandas as pd
import pytest
from pipelines.features.build_modeling_dataset import (
    DIFFERENCE_SOURCE_COLUMNS,
    LEAKED_OUTCOME_COLUMNS,
    build_feature_manifest,
    build_modeling_dataset,
    feature_manifest_output_path,
    modeling_dataset_output_path,
    modeling_dataset_summary_path,
    validate_modeling_dataset,
    validate_source_datasets,
    write_modeling_dataset_outputs,
)


def make_games() -> pd.DataFrame:
    """Create two completed chronological game rows."""

    return pd.DataFrame(
        [
            {
                "SEASON": "2024-25",
                "SEASON_ID": "22024",
                "GAME_ID": "0022400001",
                "GAME_DATE": "2024-10-22",
                "HOME_TEAM_ID": 1610612747,
                "HOME_TEAM_ABBREVIATION": "LAL",
                "AWAY_TEAM_ID": 1610612738,
                "AWAY_TEAM_ABBREVIATION": "BOS",
                "home_win": 1,
            },
            {
                "SEASON": "2024-25",
                "SEASON_ID": "22024",
                "GAME_ID": "0022400002",
                "GAME_DATE": "2024-10-24",
                "HOME_TEAM_ID": 1610612738,
                "HOME_TEAM_ABBREVIATION": "BOS",
                "AWAY_TEAM_ID": 1610612747,
                "AWAY_TEAM_ABBREVIATION": "LAL",
                "home_win": 0,
            },
        ]
    )


def historical_values(
    *,
    prior_games: int,
    win_pct: float | None,
    points_scored: float | None,
    points_allowed: float | None,
    point_differential: float | None,
    days_rest: float | None,
) -> dict[str, object]:
    """Create the repeated rolling feature values for one team row."""

    return {
        "TEAM_GAME_NUMBER": prior_games + 1,
        "PRIOR_GAMES_PLAYED": prior_games,
        "DAYS_REST": days_rest,
        "IS_BACK_TO_BACK": 0,
        "SEASON_WIN_PCT": win_pct,
        "ROLLING_5_WIN_PCT": win_pct,
        "ROLLING_10_WIN_PCT": win_pct,
        "ROLLING_5_POINTS_SCORED": points_scored,
        "ROLLING_10_POINTS_SCORED": points_scored,
        "ROLLING_5_POINTS_ALLOWED": points_allowed,
        "ROLLING_10_POINTS_ALLOWED": points_allowed,
        "ROLLING_5_POINT_DIFFERENTIAL": point_differential,
        "ROLLING_10_POINT_DIFFERENTIAL": point_differential,
    }


def make_team_features() -> pd.DataFrame:
    """Create home and away pregame perspectives for two games."""

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
                **historical_values(
                    prior_games=0,
                    win_pct=None,
                    points_scored=None,
                    points_allowed=None,
                    point_differential=None,
                    days_rest=None,
                ),
            },
            {
                "SEASON": "2024-25",
                "SEASON_ID": "22024",
                "GAME_ID": "0022400001",
                "GAME_DATE": "2024-10-22",
                "TEAM_ID": 1610612738,
                "TEAM_ABBREVIATION": "BOS",
                "OPPONENT_TEAM_ID": 1610612747,
                "OPPONENT_TEAM_ABBREVIATION": "LAL",
                "IS_HOME": 0,
                **historical_values(
                    prior_games=0,
                    win_pct=None,
                    points_scored=None,
                    points_allowed=None,
                    point_differential=None,
                    days_rest=None,
                ),
            },
            {
                "SEASON": "2024-25",
                "SEASON_ID": "22024",
                "GAME_ID": "0022400002",
                "GAME_DATE": "2024-10-24",
                "TEAM_ID": 1610612738,
                "TEAM_ABBREVIATION": "BOS",
                "OPPONENT_TEAM_ID": 1610612747,
                "OPPONENT_TEAM_ABBREVIATION": "LAL",
                "IS_HOME": 1,
                **historical_values(
                    prior_games=1,
                    win_pct=0.0,
                    points_scored=100.0,
                    points_allowed=110.0,
                    point_differential=-10.0,
                    days_rest=1.0,
                ),
            },
            {
                "SEASON": "2024-25",
                "SEASON_ID": "22024",
                "GAME_ID": "0022400002",
                "GAME_DATE": "2024-10-24",
                "TEAM_ID": 1610612747,
                "TEAM_ABBREVIATION": "LAL",
                "OPPONENT_TEAM_ID": 1610612738,
                "OPPONENT_TEAM_ABBREVIATION": "BOS",
                "IS_HOME": 0,
                **historical_values(
                    prior_games=1,
                    win_pct=1.0,
                    points_scored=110.0,
                    points_allowed=100.0,
                    point_differential=10.0,
                    days_rest=1.0,
                ),
            },
        ]
    )


def test_validate_source_datasets_rejects_missing_columns() -> None:
    """The join should fail when a required feature column is absent."""

    games = make_games()
    features = make_team_features().drop(columns=["ROLLING_5_WIN_PCT"])

    with pytest.raises(
        ValueError,
        match="missing required columns",
    ):
        validate_source_datasets(
            games=games,
            team_features=features,
        )


def test_validate_source_datasets_rejects_duplicate_team_rows() -> None:
    """The same team should not have two feature rows for one game."""

    games = make_games()
    features = make_team_features()

    features = pd.concat(
        [
            features,
            features.iloc[[0]],
        ],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match="duplicate team-game rows",
    ):
        validate_source_datasets(
            games=games,
            team_features=features,
        )


def test_build_modeling_dataset_creates_one_row_per_game() -> None:
    """Two team perspectives should become one matchup row."""

    modeling, summary = build_modeling_dataset(
        games=make_games(),
        team_features=make_team_features(),
    )

    assert len(modeling) == 2
    assert modeling["GAME_ID"].nunique() == 2
    assert modeling["home_win"].tolist() == [1, 0]

    assert summary.source_game_rows == 2
    assert summary.source_team_feature_rows == 4
    assert summary.output_model_rows == 2
    assert summary.unique_games == 2
    assert summary.seasons == 1


def test_build_modeling_dataset_assigns_home_and_away_features() -> None:
    """Each team's feature history should appear on the correct side."""

    modeling, _ = build_modeling_dataset(
        games=make_games(),
        team_features=make_team_features(),
    )

    second_game = modeling.loc[modeling["GAME_ID"] == "0022400002"].iloc[0]

    assert second_game["HOME_TEAM_ID"] == 1610612738
    assert second_game["HOME_TEAM_ABBREVIATION"] == "BOS"
    assert second_game["HOME_SEASON_WIN_PCT"] == pytest.approx(0.0)
    assert second_game["HOME_ROLLING_5_POINT_DIFFERENTIAL"] == (pytest.approx(-10.0))

    assert second_game["AWAY_TEAM_ID"] == 1610612747
    assert second_game["AWAY_TEAM_ABBREVIATION"] == "LAL"
    assert second_game["AWAY_SEASON_WIN_PCT"] == pytest.approx(1.0)
    assert second_game["AWAY_ROLLING_5_POINT_DIFFERENTIAL"] == (pytest.approx(10.0))


def test_build_modeling_dataset_calculates_differences() -> None:
    """Difference features should always be home minus away."""

    modeling, _ = build_modeling_dataset(
        games=make_games(),
        team_features=make_team_features(),
    )

    second_game = modeling.loc[modeling["GAME_ID"] == "0022400002"].iloc[0]

    assert second_game["SEASON_WIN_PCT_DIFF"] == pytest.approx(-1.0)
    assert second_game["ROLLING_5_POINTS_SCORED_DIFF"] == (pytest.approx(-10.0))
    assert second_game["ROLLING_5_POINTS_ALLOWED_DIFF"] == (pytest.approx(10.0))
    assert second_game["ROLLING_5_POINT_DIFFERENTIAL_DIFF"] == (pytest.approx(-20.0))
    assert second_game["DAYS_REST_DIFF"] == pytest.approx(0.0)


def test_build_modeling_dataset_creates_history_flags() -> None:
    """Cold-start flags should reflect available prior games."""

    modeling, summary = build_modeling_dataset(
        games=make_games(),
        team_features=make_team_features(),
    )

    first_game = modeling.loc[modeling["GAME_ID"] == "0022400001"].iloc[0]

    second_game = modeling.loc[modeling["GAME_ID"] == "0022400002"].iloc[0]

    assert first_game["HOME_HAS_HISTORY"] == 0
    assert first_game["AWAY_HAS_HISTORY"] == 0
    assert first_game["BOTH_TEAMS_HAVE_HISTORY"] == 0

    assert second_game["HOME_HAS_HISTORY"] == 1
    assert second_game["AWAY_HAS_HISTORY"] == 1
    assert second_game["BOTH_TEAMS_HAVE_HISTORY"] == 1
    assert second_game["BOTH_TEAMS_HAVE_5_GAMES"] == 0
    assert second_game["BOTH_TEAMS_HAVE_10_GAMES"] == 0

    assert summary.rows_with_both_teams_history == 1
    assert summary.rows_with_both_teams_5_games == 0
    assert summary.rows_with_both_teams_10_games == 0
    assert summary.rows_with_any_missing_numeric_features == 1


def test_build_modeling_dataset_rejects_opponent_mismatch() -> None:
    """The two team rows must identify each other as opponents."""

    features = make_team_features()
    features.loc[0, "OPPONENT_TEAM_ID"] = 999

    with pytest.raises(
        ValueError,
        match="opponent IDs do not cross-match",
    ):
        build_modeling_dataset(
            games=make_games(),
            team_features=features,
        )


def test_modeling_dataset_contains_no_same_game_outcomes() -> None:
    """Scores and team-level current results must not enter predictors."""

    modeling, _ = build_modeling_dataset(
        games=make_games(),
        team_features=make_team_features(),
    )

    assert LEAKED_OUTCOME_COLUMNS.isdisjoint(modeling.columns)
    assert "home_win" in modeling.columns


def test_validate_modeling_dataset_rejects_tampered_difference() -> None:
    """Validation should detect an incorrect comparison feature."""

    modeling, _ = build_modeling_dataset(
        games=make_games(),
        team_features=make_team_features(),
    )

    modeling.loc[1, "SEASON_WIN_PCT_DIFF"] = 99.0

    with pytest.raises(
        ValueError,
        match="SEASON_WIN_PCT_DIFF is inconsistent",
    ):
        validate_modeling_dataset(
            modeling=modeling,
            expected_row_count=2,
        )


def test_modeling_output_paths_use_modeling_directory(
    tmp_path: Path,
) -> None:
    """Dataset, metadata, and manifest should share one output folder."""

    expected_directory = tmp_path / "data" / "processed" / "nba" / "modeling"

    assert modeling_dataset_output_path(tmp_path) == (
        expected_directory / "pregame_modeling_dataset.parquet"
    )

    assert modeling_dataset_summary_path(tmp_path) == (
        expected_directory / "pregame_modeling_dataset_summary.json"
    )

    assert feature_manifest_output_path(tmp_path) == (expected_directory / "feature_manifest.json")


def test_write_modeling_dataset_outputs_creates_all_files(
    tmp_path: Path,
) -> None:
    """The dataset, summary, and feature contract should be persisted."""

    modeling, summary = build_modeling_dataset(
        games=make_games(),
        team_features=make_team_features(),
    )

    dataset_path, summary_path, manifest_path = write_modeling_dataset_outputs(
        modeling=modeling,
        summary=summary,
        project_root=tmp_path,
    )

    assert dataset_path.exists()
    assert summary_path.exists()
    assert manifest_path.exists()

    saved_modeling = pd.read_parquet(dataset_path)

    assert len(saved_modeling) == 2
    assert saved_modeling["home_win"].tolist() == [1, 0]

    summary_text = summary_path.read_text(encoding="utf-8")

    assert '"output_model_rows": 2' in summary_text
    assert '"numeric_feature_count": 41' in summary_text

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["target_column"] == "home_win"
    assert manifest["row_grain"] == "one row per NBA game"
    assert "HOME_TEAM_ID" in manifest["categorical_feature_columns"]
    assert "SEASON_WIN_PCT_DIFF" in manifest["numeric_feature_columns"]


def test_feature_manifest_matches_difference_configuration() -> None:
    """Every configured difference feature should appear in the manifest."""

    manifest = build_feature_manifest()

    numeric_features = set(
        cast(
            list[str],
            manifest["numeric_feature_columns"],
        )
    )

    for feature_name in DIFFERENCE_SOURCE_COLUMNS:
        assert f"{feature_name}_DIFF" in numeric_features
