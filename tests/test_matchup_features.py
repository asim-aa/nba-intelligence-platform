"""Tests for the as-of-date matchup feature computer."""

from pathlib import Path

import pandas as pd
import pytest
from modeling.serving.matchup_features import (
    build_computed_matchup_row,
    compute_matchup_features,
    compute_team_current_elo,
    compute_team_rolling_state,
    find_historical_matchup,
)
from pipelines.features.build_modeling_dataset import (
    NUMERIC_FEATURE_COLUMNS,
    modeling_dataset_output_path,
)
from pipelines.features.build_team_elo_ratings import (
    INITIAL_RATING,
    carried_rating,
    elo_ratings_output_path,
)
from pipelines.features.build_team_history import team_history_output_path

TEAM_A = 1610612747
TEAM_B = 1610612738


def make_team_history_rows(
    team_id: int, season: str, dates: list[str], wins: list[int]
) -> list[dict]:
    rows = []
    for date, win in zip(dates, wins, strict=True):
        rows.append(
            {
                "SEASON": season,
                "TEAM_ID": team_id,
                "GAME_DATE": pd.Timestamp(date),
                "TEAM_WIN": win,
                "TEAM_PTS": 110 if win else 100,
                "OPPONENT_PTS": 100 if win else 110,
                "POINT_DIFFERENTIAL": 10 if win else -10,
            }
        )
    return rows


def test_compute_team_rolling_state_uses_full_real_history() -> None:
    team_history = pd.DataFrame(
        make_team_history_rows(
            TEAM_A,
            "2025-26",
            ["2025-10-22", "2025-10-24", "2025-10-26"],
            [1, 1, 0],
        )
    )

    state = compute_team_rolling_state(
        team_history=team_history,
        team_id=TEAM_A,
        season="2025-26",
        as_of_date=pd.Timestamp("2025-10-29"),
    )

    assert state["PRIOR_GAMES_PLAYED"] == 3
    assert state["SEASON_WIN_PCT"] == pytest.approx(2 / 3)
    assert state["ROLLING_5_WIN_PCT"] == pytest.approx(2 / 3)
    assert state["ROLLING_5_POINTS_SCORED"] == pytest.approx((110 + 110 + 100) / 3)
    # Oct 26 -> Oct 29 is a 3-day gap = 2 rest days.
    assert state["DAYS_REST"] == pytest.approx(2.0)
    assert state["IS_BACK_TO_BACK"] == 0


def test_compute_team_rolling_state_back_to_back() -> None:
    team_history = pd.DataFrame(make_team_history_rows(TEAM_A, "2025-26", ["2025-10-22"], [1]))

    state = compute_team_rolling_state(
        team_history=team_history,
        team_id=TEAM_A,
        season="2025-26",
        as_of_date=pd.Timestamp("2025-10-23"),
    )

    assert state["DAYS_REST"] == pytest.approx(0.0)
    assert state["IS_BACK_TO_BACK"] == 1


def test_compute_team_rolling_state_first_game_of_season() -> None:
    team_history = pd.DataFrame(
        columns=[
            "SEASON",
            "TEAM_ID",
            "GAME_DATE",
            "TEAM_WIN",
            "TEAM_PTS",
            "OPPONENT_PTS",
            "POINT_DIFFERENTIAL",
        ]
    )

    state = compute_team_rolling_state(
        team_history=team_history,
        team_id=TEAM_A,
        season="2026-27",
        as_of_date=pd.Timestamp("2026-10-21"),
    )

    assert state["PRIOR_GAMES_PLAYED"] == 0
    assert pd.isna(state["SEASON_WIN_PCT"])
    assert pd.isna(state["DAYS_REST"])
    assert state["IS_BACK_TO_BACK"] == 0


def test_compute_team_rolling_state_rejects_date_not_after_last_game() -> None:
    team_history = pd.DataFrame(
        make_team_history_rows(TEAM_A, "2025-26", ["2025-10-22", "2025-10-24"], [1, 0])
    )

    with pytest.raises(ValueError, match="already has a recorded"):
        compute_team_rolling_state(
            team_history=team_history,
            team_id=TEAM_A,
            season="2025-26",
            as_of_date=pd.Timestamp("2025-10-24"),
        )


def test_compute_team_current_elo_same_season_uses_post_game_rating() -> None:
    elo_ratings = pd.DataFrame(
        [
            {
                "SEASON": "2025-26",
                "TEAM_ID": TEAM_A,
                "GAME_DATE": pd.Timestamp("2025-10-24"),
                "ELO_RATING": 1490.0,
                "POST_GAME_ELO_RATING": 1512.5,
            }
        ]
    )

    rating = compute_team_current_elo(elo_ratings=elo_ratings, team_id=TEAM_A, season="2025-26")

    assert rating == pytest.approx(1512.5)


def test_compute_team_current_elo_new_season_applies_regression() -> None:
    elo_ratings = pd.DataFrame(
        [
            {
                "SEASON": "2025-26",
                "TEAM_ID": TEAM_A,
                "GAME_DATE": pd.Timestamp("2026-04-10"),
                "ELO_RATING": 1580.0,
                "POST_GAME_ELO_RATING": 1600.0,
            }
        ]
    )

    rating = compute_team_current_elo(elo_ratings=elo_ratings, team_id=TEAM_A, season="2026-27")

    assert rating == pytest.approx(carried_rating(1600.0))
    assert rating < 1600.0


def test_compute_team_current_elo_no_history_returns_initial() -> None:
    elo_ratings = pd.DataFrame(
        columns=["SEASON", "TEAM_ID", "GAME_DATE", "ELO_RATING", "POST_GAME_ELO_RATING"]
    )

    rating = compute_team_current_elo(elo_ratings=elo_ratings, team_id=TEAM_A, season="2015-16")

    assert rating == INITIAL_RATING


def test_build_computed_matchup_row_produces_expected_diffs() -> None:
    home_state = {
        "PRIOR_GAMES_PLAYED": 10,
        "DAYS_REST": 2.0,
        "IS_BACK_TO_BACK": 0,
        "SEASON_WIN_PCT": 0.7,
        "ROLLING_5_WIN_PCT": 0.6,
        "ROLLING_10_WIN_PCT": 0.7,
        "ROLLING_5_POINTS_SCORED": 115.0,
        "ROLLING_10_POINTS_SCORED": 112.0,
        "ROLLING_5_POINTS_ALLOWED": 105.0,
        "ROLLING_10_POINTS_ALLOWED": 108.0,
        "ROLLING_5_POINT_DIFFERENTIAL": 10.0,
        "ROLLING_10_POINT_DIFFERENTIAL": 4.0,
        "ELO_RATING": 1600.0,
    }
    away_state = {
        "PRIOR_GAMES_PLAYED": 10,
        "DAYS_REST": 0.0,
        "IS_BACK_TO_BACK": 1,
        "SEASON_WIN_PCT": 0.4,
        "ROLLING_5_WIN_PCT": 0.4,
        "ROLLING_10_WIN_PCT": 0.4,
        "ROLLING_5_POINTS_SCORED": 105.0,
        "ROLLING_10_POINTS_SCORED": 106.0,
        "ROLLING_5_POINTS_ALLOWED": 110.0,
        "ROLLING_10_POINTS_ALLOWED": 109.0,
        "ROLLING_5_POINT_DIFFERENTIAL": -5.0,
        "ROLLING_10_POINT_DIFFERENTIAL": -3.0,
        "ELO_RATING": 1500.0,
    }

    row = build_computed_matchup_row(
        home_state=home_state,
        away_state=away_state,
        home_team_id=TEAM_A,
        away_team_id=TEAM_B,
    )

    assert len(row) == 1
    result = row.iloc[0]
    assert result["ELO_RATING_DIFF"] == pytest.approx(100.0)
    assert result["SEASON_WIN_PCT_DIFF"] == pytest.approx(0.3)
    assert result["HOME_TEAM_ID"] == TEAM_A
    assert result["AWAY_TEAM_ID"] == TEAM_B


def write_minimal_dashboard_fixtures(project_root: Path) -> None:
    """Write minimal, real-shaped Phase 5 outputs for integration tests."""

    team_history = pd.DataFrame(
        make_team_history_rows(TEAM_A, "2025-26", ["2025-10-22", "2025-10-24"], [1, 0])
        + make_team_history_rows(TEAM_B, "2025-26", ["2025-10-22", "2025-10-25"], [0, 1])
    )
    history_path = team_history_output_path(project_root)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    team_history.to_parquet(history_path, index=False)

    elo_ratings = pd.DataFrame(
        [
            {
                "SEASON": "2025-26",
                "SEASON_ID": "22025",
                "GAME_ID": "0022500001",
                "GAME_DATE": pd.Timestamp("2025-10-24"),
                "TEAM_ID": TEAM_A,
                "ELO_RATING": 1495.0,
                "POST_GAME_ELO_RATING": 1505.0,
            },
            {
                "SEASON": "2025-26",
                "SEASON_ID": "22025",
                "GAME_ID": "0022500002",
                "GAME_DATE": pd.Timestamp("2025-10-25"),
                "TEAM_ID": TEAM_B,
                "ELO_RATING": 1498.0,
                "POST_GAME_ELO_RATING": 1508.0,
            },
        ]
    )
    elo_path = elo_ratings_output_path(project_root)
    elo_path.parent.mkdir(parents=True, exist_ok=True)
    elo_ratings.to_parquet(elo_path, index=False)

    modeling_row = {
        "SEASON": "2025-26",
        "SEASON_ID": "22025",
        "GAME_ID": "0022500001",
        "GAME_DATE": pd.Timestamp("2025-10-22"),
        "HOME_TEAM_ID": TEAM_A,
        "HOME_TEAM_ABBREVIATION": "LAL",
        "AWAY_TEAM_ID": TEAM_B,
        "AWAY_TEAM_ABBREVIATION": "BOS",
        "home_win": 1,
    }
    # A real modeling dataset row carries the full feature set. Fill every
    # numeric feature with a placeholder value so the historical lookup
    # path has something representative to return.
    for column in NUMERIC_FEATURE_COLUMNS:
        modeling_row.setdefault(column, 0.0)

    modeling_dataset = pd.DataFrame([modeling_row])
    modeling_path = modeling_dataset_output_path(project_root)
    modeling_path.parent.mkdir(parents=True, exist_ok=True)
    modeling_dataset.to_parquet(modeling_path, index=False)


def test_find_historical_matchup_returns_existing_row(tmp_path: Path) -> None:
    write_minimal_dashboard_fixtures(tmp_path)
    modeling_dataset = pd.read_parquet(modeling_dataset_output_path(tmp_path))

    match = find_historical_matchup(
        modeling_dataset=modeling_dataset,
        home_team_id=TEAM_A,
        away_team_id=TEAM_B,
        game_date="2025-10-22",
    )

    assert match is not None
    assert len(match) == 1


def test_find_historical_matchup_returns_none_when_absent(tmp_path: Path) -> None:
    write_minimal_dashboard_fixtures(tmp_path)
    modeling_dataset = pd.read_parquet(modeling_dataset_output_path(tmp_path))

    match = find_historical_matchup(
        modeling_dataset=modeling_dataset,
        home_team_id=TEAM_A,
        away_team_id=TEAM_B,
        game_date="2099-01-01",
    )

    assert match is None


def test_compute_matchup_features_prefers_historical_lookup(tmp_path: Path) -> None:
    write_minimal_dashboard_fixtures(tmp_path)

    feature_row, source, actual_home_win = compute_matchup_features(
        project_root=tmp_path,
        home_team_id=TEAM_A,
        away_team_id=TEAM_B,
        game_date="2025-10-22",
        season="2025-26",
    )

    assert source == "historical"
    assert actual_home_win == 1
    assert len(feature_row) == 1


def test_compute_matchup_features_computes_for_upcoming_matchup(tmp_path: Path) -> None:
    write_minimal_dashboard_fixtures(tmp_path)

    feature_row, source, actual_home_win = compute_matchup_features(
        project_root=tmp_path,
        home_team_id=TEAM_A,
        away_team_id=TEAM_B,
        game_date="2025-10-30",
        season="2025-26",
    )

    assert source == "computed"
    assert actual_home_win is None
    assert len(feature_row) == 1
    assert not feature_row.isna().any().any()
