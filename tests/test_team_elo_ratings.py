"""Tests for the cross-season Elo rating engine.

Unit tests check the math building blocks against independently reasoned
properties (not just re-deriving the same formula). Sequencing tests check
that compute_elo_ratings invokes those building blocks at the right time
across a small, hand-constructed schedule.
"""

from pathlib import Path

import pandas as pd
import pytest
from pipelines.features.build_team_elo_ratings import (
    HOME_COURT_ADVANTAGE,
    INITIAL_RATING,
    MEAN_RATING,
    carried_rating,
    compute_elo_ratings,
    elo_ratings_output_path,
    elo_ratings_summary_path,
    expected_home_win_probability,
    margin_of_victory_multiplier,
    update_ratings,
    write_elo_ratings_outputs,
)

# --- Math building blocks -----------------------------------------------


def test_home_court_advantage_favors_home_team_at_equal_ratings() -> None:
    assert expected_home_win_probability(1500.0, 1500.0) > 0.5


def test_expected_home_win_probability_matches_known_elo_fact() -> None:
    # A 100-point Elo edge is a well-known reference point: ~64% win
    # probability under the standard logistic formula.
    assert expected_home_win_probability(1500.0, 1500.0) == pytest.approx(0.64, abs=0.001)


def test_expected_home_win_probability_increases_with_home_strength() -> None:
    baseline = expected_home_win_probability(1500.0, 1500.0)
    stronger_home = expected_home_win_probability(1600.0, 1500.0)

    assert stronger_home > baseline


def test_expected_home_win_probability_can_favor_away_team() -> None:
    # Home court is worth HOME_COURT_ADVANTAGE points; a bigger true gap
    # than that should still favor the away team despite the home boost.
    assert expected_home_win_probability(1500.0, 1500.0 + HOME_COURT_ADVANTAGE + 100.0) < 0.5


def test_margin_of_victory_multiplier_grows_with_margin() -> None:
    assert margin_of_victory_multiplier(30.0, 0.0) > margin_of_victory_multiplier(5.0, 0.0)


def test_margin_of_victory_multiplier_grows_for_upsets() -> None:
    upset = margin_of_victory_multiplier(10.0, winner_rating_diff=-200.0)
    expected_win = margin_of_victory_multiplier(10.0, winner_rating_diff=200.0)

    assert upset > expected_win


def test_margin_of_victory_multiplier_stays_positive_for_extreme_inputs() -> None:
    assert margin_of_victory_multiplier(100.0, winner_rating_diff=-5000.0) > 0.0
    assert margin_of_victory_multiplier(1.0, winner_rating_diff=5000.0) > 0.0


def test_carried_rating_regresses_toward_mean() -> None:
    assert carried_rating(1600.0) == pytest.approx(1575.0)
    assert carried_rating(1400.0) == pytest.approx(1425.0)
    assert carried_rating(MEAN_RATING) == pytest.approx(MEAN_RATING)


def test_update_ratings_is_zero_sum() -> None:
    home_post, away_post = update_ratings(
        home_rating=1550.0, away_rating=1480.0, home_pts=110, away_pts=101, home_win=1
    )

    assert (home_post - 1550.0) == pytest.approx(-(away_post - 1480.0))


def test_update_ratings_winner_gains_loser_loses() -> None:
    home_post, away_post = update_ratings(
        home_rating=1500.0, away_rating=1500.0, home_pts=100, away_pts=90, home_win=1
    )

    assert home_post > 1500.0
    assert away_post < 1500.0


def test_update_ratings_away_win_direction() -> None:
    home_post, away_post = update_ratings(
        home_rating=1500.0, away_rating=1500.0, home_pts=90, away_pts=100, home_win=0
    )

    assert home_post < 1500.0
    assert away_post > 1500.0


def test_update_ratings_bigger_margin_moves_rating_more() -> None:
    close_home, _ = update_ratings(
        home_rating=1500.0, away_rating=1500.0, home_pts=101, away_pts=100, home_win=1
    )
    blowout_home, _ = update_ratings(
        home_rating=1500.0, away_rating=1500.0, home_pts=130, away_pts=100, home_win=1
    )

    assert (blowout_home - 1500.0) > (close_home - 1500.0)


# --- Sequential engine ----------------------------------------------------


def make_game(
    *,
    season: str,
    game_id: str,
    date: str,
    home_id: int,
    away_id: int,
    home_pts: int,
    away_pts: int,
) -> dict[str, object]:
    return {
        "SEASON": season,
        "SEASON_ID": f"2{season[:4]}",
        "GAME_ID": game_id,
        "GAME_DATE": date,
        "HOME_TEAM_ID": home_id,
        "AWAY_TEAM_ID": away_id,
        "HOME_PTS": home_pts,
        "AWAY_PTS": away_pts,
        "home_win": int(home_pts > away_pts),
    }


def test_first_game_ratings_start_at_initial_rating() -> None:
    games = pd.DataFrame(
        [
            make_game(
                season="2024-25",
                game_id="0001",
                date="2024-10-22",
                home_id=1,
                away_id=2,
                home_pts=110,
                away_pts=100,
            )
        ]
    )

    elo_ratings, _ = compute_elo_ratings(games)

    assert set(elo_ratings["ELO_RATING"]) == {INITIAL_RATING}


def test_same_season_second_game_carries_exact_prior_rating() -> None:
    games = pd.DataFrame(
        [
            make_game(
                season="2024-25",
                game_id="0001",
                date="2024-10-22",
                home_id=1,
                away_id=2,
                home_pts=110,
                away_pts=100,
            ),
            make_game(
                season="2024-25",
                game_id="0002",
                date="2024-10-24",
                home_id=3,
                away_id=1,
                home_pts=90,
                away_pts=95,
            ),
        ]
    )

    elo_ratings, _ = compute_elo_ratings(games)

    team_1_game_1_post = elo_ratings.loc[
        (elo_ratings["GAME_ID"] == "0001") & (elo_ratings["TEAM_ID"] == 1), "POST_GAME_ELO_RATING"
    ].item()
    team_1_game_2_pre = elo_ratings.loc[
        (elo_ratings["GAME_ID"] == "0002") & (elo_ratings["TEAM_ID"] == 1), "ELO_RATING"
    ].item()

    assert team_1_game_2_pre == pytest.approx(team_1_game_1_post)


def test_new_season_applies_regression_to_the_prior_rating() -> None:
    games = pd.DataFrame(
        [
            make_game(
                season="2024-25",
                game_id="0001",
                date="2024-10-22",
                home_id=1,
                away_id=2,
                home_pts=120,
                away_pts=90,
            ),
            make_game(
                season="2025-26",
                game_id="0002",
                date="2025-10-21",
                home_id=1,
                away_id=2,
                home_pts=100,
                away_pts=95,
            ),
        ]
    )

    elo_ratings, _ = compute_elo_ratings(games)

    team_1_season_1_post = elo_ratings.loc[
        (elo_ratings["GAME_ID"] == "0001") & (elo_ratings["TEAM_ID"] == 1), "POST_GAME_ELO_RATING"
    ].item()
    team_1_season_2_pre = elo_ratings.loc[
        (elo_ratings["GAME_ID"] == "0002") & (elo_ratings["TEAM_ID"] == 1), "ELO_RATING"
    ].item()

    assert team_1_season_2_pre == pytest.approx(carried_rating(team_1_season_1_post))
    # A blowout win should have pushed team 1 above the mean, so regression
    # must pull the new-season rating strictly back down toward it.
    assert team_1_season_1_post > MEAN_RATING
    assert team_1_season_2_pre < team_1_season_1_post


def test_compute_elo_ratings_output_structure() -> None:
    games = pd.DataFrame(
        [
            make_game(
                season="2024-25",
                game_id="0001",
                date="2024-10-22",
                home_id=1,
                away_id=2,
                home_pts=110,
                away_pts=100,
            ),
            make_game(
                season="2024-25",
                game_id="0002",
                date="2024-10-24",
                home_id=3,
                away_id=4,
                home_pts=95,
                away_pts=99,
            ),
        ]
    )

    elo_ratings, summary = compute_elo_ratings(games)

    assert len(elo_ratings) == 4
    assert summary.output_team_rows == 4
    assert summary.unique_teams == 4
    assert not elo_ratings[["ELO_RATING", "POST_GAME_ELO_RATING"]].isna().any().any()
    assert elo_ratings.duplicated(subset=["SEASON", "GAME_ID", "TEAM_ID"]).sum() == 0


def test_compute_elo_ratings_rejects_missing_columns() -> None:
    games = pd.DataFrame(
        [
            make_game(
                season="2024-25",
                game_id="0001",
                date="2024-10-22",
                home_id=1,
                away_id=2,
                home_pts=110,
                away_pts=100,
            )
        ]
    ).drop(columns=["HOME_PTS"])

    with pytest.raises(ValueError, match="missing required columns"):
        compute_elo_ratings(games)


def test_compute_elo_ratings_rejects_empty_input() -> None:
    games = pd.DataFrame(
        [
            make_game(
                season="2024-25",
                game_id="0001",
                date="2024-10-22",
                home_id=1,
                away_id=2,
                home_pts=110,
                away_pts=100,
            )
        ]
    ).iloc[0:0]

    with pytest.raises(ValueError, match="empty"):
        compute_elo_ratings(games)


def test_compute_elo_ratings_rejects_home_win_inconsistent_with_scores() -> None:
    game = make_game(
        season="2024-25",
        game_id="0001",
        date="2024-10-22",
        home_id=1,
        away_id=2,
        home_pts=90,
        away_pts=100,
    )
    game["home_win"] = 1  # Home actually lost by score.
    games = pd.DataFrame([game])

    with pytest.raises(ValueError, match="inconsistent"):
        compute_elo_ratings(games)


def test_compute_elo_ratings_rejects_team_playing_itself() -> None:
    games = pd.DataFrame(
        [
            make_game(
                season="2024-25",
                game_id="0001",
                date="2024-10-22",
                home_id=1,
                away_id=1,
                home_pts=110,
                away_pts=100,
            )
        ]
    )

    with pytest.raises(ValueError, match="identical"):
        compute_elo_ratings(games)


def test_write_elo_ratings_outputs_persists_reloadable_files(tmp_path: Path) -> None:
    games = pd.DataFrame(
        [
            make_game(
                season="2024-25",
                game_id="0001",
                date="2024-10-22",
                home_id=1,
                away_id=2,
                home_pts=110,
                away_pts=100,
            )
        ]
    )
    elo_ratings, summary = compute_elo_ratings(games)

    ratings_path, summary_path = write_elo_ratings_outputs(
        elo_ratings=elo_ratings, summary=summary, project_root=tmp_path
    )

    assert ratings_path == elo_ratings_output_path(tmp_path)
    assert summary_path == elo_ratings_summary_path(tmp_path)

    reloaded = pd.read_parquet(ratings_path)
    pd.testing.assert_frame_equal(reloaded, elo_ratings)
