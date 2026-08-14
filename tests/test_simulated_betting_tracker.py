"""Tests for the dashboard's simulated ("paper") betting tracker."""

from pathlib import Path

import pandas as pd
import pytest
from app.dashboard.picks_store import Pick, record_pick, update_actual_result
from app.dashboard.simulated_betting_tracker import (
    FIXED_AMERICAN_ODDS,
    FLAT_STAKE,
    MIN_BETS_FOR_SIGNAL,
    american_odds_profit,
    breakeven_hit_rate,
    compute_parlay,
    compute_round_robin,
    decimal_odds_from_american,
    run_parlay_tracker,
    run_round_robin_tracker,
    run_simulated_betting_tracker,
    simulate_strategy,
)


def write_resolved_picks(
    project_root: Path,
    count: int,
    user_picks: list[str],
    model_picks: list[str],
    actual_winners: list[str],
    id_prefix: str,
) -> None:
    for index in range(count):
        game_id = f"{id_prefix}{index:04d}"
        record_pick(
            project_root,
            Pick(
                game_id=game_id,
                game_date="2025-11-05",
                season="2025-26",
                home_team_id=1610612747,
                home_team_name="Los Angeles Lakers",
                away_team_id=1610612738,
                away_team_name="Boston Celtics",
                user_pick=user_picks[index],
                model_pick=model_picks[index],
                model_home_win_probability=0.6,
                feature_source="historical",
            ),
        )
        update_actual_result(project_root, game_id, actual_winners[index])


def write_slate_picks(
    project_root: Path,
    game_date: str,
    legs: list[tuple[str, str, str | None]],
    id_prefix: str,
) -> None:
    """Write one day's picks. Each leg is (user_pick, model_pick, actual_winner);
    actual_winner=None leaves that leg unresolved (still pending).
    """

    for index, (user_pick, model_pick, actual_winner) in enumerate(legs):
        game_id = f"{id_prefix}{index:04d}"
        record_pick(
            project_root,
            Pick(
                game_id=game_id,
                game_date=game_date,
                season="2025-26",
                home_team_id=1610612747,
                home_team_name="Los Angeles Lakers",
                away_team_id=1610612738,
                away_team_name="Boston Celtics",
                user_pick=user_pick,
                model_pick=model_pick,
                model_home_win_probability=0.6,
                feature_source="historical",
            ),
        )
        if actual_winner is not None:
            update_actual_result(project_root, game_id, actual_winner)


# --- american_odds_profit / breakeven_hit_rate -----------------------------------


def test_american_odds_profit_negative_odds() -> None:
    assert american_odds_profit(10.0, -110) == pytest.approx(10.0 * 100.0 / 110.0)


def test_american_odds_profit_positive_odds() -> None:
    assert american_odds_profit(10.0, 150) == pytest.approx(15.0)


def test_american_odds_profit_rejects_zero_odds() -> None:
    with pytest.raises(ValueError, match="cannot be zero"):
        american_odds_profit(10.0, 0)


def test_breakeven_hit_rate_matches_hand_computation() -> None:
    assert breakeven_hit_rate(-110) == pytest.approx(110.0 / 210.0)


def test_breakeven_hit_rate_rejects_positive_odds() -> None:
    with pytest.raises(ValueError, match="negative"):
        breakeven_hit_rate(110)


# --- simulate_strategy -----------------------------------


def test_simulate_strategy_reports_insufficient_below_minimum() -> None:
    resolved = pd.DataFrame(
        {
            "model_pick": ["HOME"] * (MIN_BETS_FOR_SIGNAL - 1),
            "actual_winner": ["HOME"] * (MIN_BETS_FOR_SIGNAL - 1),
        }
    )

    result = simulate_strategy(resolved, "model_pick", "Bet the model")

    assert result.sufficient_sample is False
    assert "fewer than" in result.reason
    assert result.total_profit is None


def test_simulate_strategy_matches_hand_computation() -> None:
    # 6 wins, 4 losses out of 10 bets.
    resolved = pd.DataFrame(
        {
            "model_pick": ["HOME"] * 6 + ["AWAY"] * 4,
            "actual_winner": ["HOME"] * 6 + ["HOME"] * 4,
        }
    )

    result = simulate_strategy(resolved, "model_pick", "Bet the model")

    win_profit = FLAT_STAKE * 100.0 / abs(FIXED_AMERICAN_ODDS)
    expected_profit = (6 * win_profit) - (4 * FLAT_STAKE)

    assert result.sufficient_sample is True
    assert result.bets_placed == 10
    assert result.wins == 6
    assert result.losses == 4
    assert result.hit_rate == pytest.approx(0.6)
    assert result.total_staked == pytest.approx(100.0)
    assert result.total_profit == pytest.approx(expected_profit)
    assert result.roi == pytest.approx(expected_profit / 100.0)
    assert result.beats_breakeven is True


def test_simulate_strategy_below_breakeven_is_a_loss() -> None:
    # 4 wins, 6 losses -- well below the -110 breakeven of ~52.4%.
    resolved = pd.DataFrame(
        {
            "model_pick": ["HOME"] * 4 + ["AWAY"] * 6,
            "actual_winner": ["HOME"] * 4 + ["HOME"] * 6,
        }
    )

    result = simulate_strategy(resolved, "model_pick", "Bet the model")

    assert result.total_profit < 0
    assert result.beats_breakeven is False


# --- run_simulated_betting_tracker (real picks_store fixture) -----------------------------------


def test_run_simulated_betting_tracker_excludes_unresolved(tmp_path: Path) -> None:
    write_resolved_picks(
        tmp_path,
        count=MIN_BETS_FOR_SIGNAL,
        user_picks=["HOME"] * MIN_BETS_FOR_SIGNAL,
        model_picks=["HOME"] * MIN_BETS_FOR_SIGNAL,
        actual_winners=["HOME"] * MIN_BETS_FOR_SIGNAL,
        id_prefix="r",
    )
    record_pick(
        tmp_path,
        Pick(
            game_id="unresolved0001",
            game_date="2025-11-06",
            season="2025-26",
            home_team_id=1610612747,
            home_team_name="Los Angeles Lakers",
            away_team_id=1610612738,
            away_team_name="Boston Celtics",
            user_pick="HOME",
            model_pick="HOME",
            model_home_win_probability=0.7,
            feature_source="historical",
        ),
    )

    results = run_simulated_betting_tracker(tmp_path)
    model_strategy = next(r for r in results if r.strategy_name == "Bet the model")

    assert model_strategy.bets_placed == MIN_BETS_FOR_SIGNAL


def test_run_simulated_betting_tracker_scores_model_and_user_separately(tmp_path: Path) -> None:
    count = MIN_BETS_FOR_SIGNAL
    # Model always picks HOME and is always right; user always picks AWAY
    # and is always wrong -- the two strategies must diverge.
    write_resolved_picks(
        tmp_path,
        count=count,
        user_picks=["AWAY"] * count,
        model_picks=["HOME"] * count,
        actual_winners=["HOME"] * count,
        id_prefix="d",
    )

    results = run_simulated_betting_tracker(tmp_path)
    by_name = {r.strategy_name: r for r in results}

    assert by_name["Bet the model"].wins == count
    assert by_name["Bet your own picks"].wins == 0
    assert by_name["Bet the model"].total_profit > 0
    assert by_name["Bet your own picks"].total_profit < 0


# --- decimal_odds_from_american -----------------------------------


def test_decimal_odds_from_american_negative_odds() -> None:
    assert decimal_odds_from_american(-110) == pytest.approx(1.0 + 100.0 / 110.0)


def test_decimal_odds_from_american_positive_odds() -> None:
    assert decimal_odds_from_american(150) == pytest.approx(2.5)


def test_decimal_odds_from_american_rejects_zero() -> None:
    with pytest.raises(ValueError, match="cannot be zero"):
        decimal_odds_from_american(0)


# --- compute_parlay -----------------------------------


def test_compute_parlay_with_no_picks() -> None:
    empty = pd.DataFrame(columns=["model_pick", "actual_winner"])

    result = compute_parlay(empty, "model_pick", stake=10.0, strategy_name="Test")

    assert result.status == "no_picks"
    assert result.leg_count == 0
    assert result.potential_payout is None


def test_compute_parlay_rejects_non_positive_stake() -> None:
    empty = pd.DataFrame(columns=["model_pick", "actual_winner"])

    with pytest.raises(ValueError, match="positive"):
        compute_parlay(empty, "model_pick", stake=0.0, strategy_name="Test")


def test_compute_parlay_all_legs_won_matches_hand_computation() -> None:
    # 3-leg parlay, every leg resolved and correct.
    legs = pd.DataFrame(
        {
            "model_pick": ["HOME", "HOME", "AWAY"],
            "actual_winner": ["HOME", "HOME", "AWAY"],
        }
    )

    result = compute_parlay(legs, "model_pick", stake=10.0, strategy_name="Test")

    expected_multiplier = decimal_odds_from_american(FIXED_AMERICAN_ODDS) ** 3
    expected_payout = 10.0 * expected_multiplier

    assert result.status == "won"
    assert result.leg_count == 3
    assert result.resolved_legs == 3
    assert result.pending_legs == 0
    assert result.potential_payout == pytest.approx(expected_payout)
    assert result.realized_profit == pytest.approx(expected_payout - 10.0)


def test_compute_parlay_one_loss_busts_it_even_with_pending_legs() -> None:
    # Leg 1 resolved and lost; leg 2 still pending. The parlay is dead
    # immediately -- it must not wait for leg 2 to "confirm" the loss.
    legs = pd.DataFrame(
        {
            "model_pick": ["HOME", "HOME"],
            "actual_winner": ["AWAY", None],
        }
    )

    result = compute_parlay(legs, "model_pick", stake=10.0, strategy_name="Test")

    assert result.status == "lost"
    assert result.realized_profit == pytest.approx(-10.0)


def test_compute_parlay_still_live_when_wins_so_far_but_legs_pending() -> None:
    legs = pd.DataFrame(
        {
            "model_pick": ["HOME", "HOME"],
            "actual_winner": ["HOME", None],
        }
    )

    result = compute_parlay(legs, "model_pick", stake=10.0, strategy_name="Test")

    assert result.status == "live"
    assert result.realized_profit is None
    assert result.potential_payout is not None
    assert result.resolved_legs == 1
    assert result.pending_legs == 1


# --- run_parlay_tracker (real picks_store fixture) -----------------------------------


def test_run_parlay_tracker_scopes_to_one_date(tmp_path: Path) -> None:
    write_slate_picks(
        tmp_path,
        game_date="2025-11-05",
        legs=[("HOME", "HOME", "HOME"), ("AWAY", "AWAY", "AWAY")],
        id_prefix="a",
    )
    write_slate_picks(
        tmp_path,
        game_date="2025-11-06",
        legs=[("HOME", "HOME", "AWAY")],
        id_prefix="b",
    )

    results = run_parlay_tracker(tmp_path, game_date="2025-11-05", stake=10.0)
    model_parlay = next(r for r in results if r.strategy_name == "Model parlay")

    assert model_parlay.leg_count == 2
    assert model_parlay.status == "won"


def test_run_parlay_tracker_diverges_by_strategy(tmp_path: Path) -> None:
    # Model always picks correctly; user always picks the other side.
    write_slate_picks(
        tmp_path,
        game_date="2025-11-05",
        legs=[("AWAY", "HOME", "HOME"), ("HOME", "AWAY", "AWAY")],
        id_prefix="c",
    )

    results = run_parlay_tracker(tmp_path, game_date="2025-11-05", stake=10.0)
    by_name = {r.strategy_name: r for r in results}

    assert by_name["Model parlay"].status == "won"
    assert by_name["Your parlay"].status == "lost"


def test_run_parlay_tracker_no_picks_for_date(tmp_path: Path) -> None:
    results = run_parlay_tracker(tmp_path, game_date="2099-01-01", stake=10.0)

    assert all(r.status == "no_picks" for r in results)


# --- compute_round_robin -----------------------------------


def test_compute_round_robin_rejects_non_positive_stake() -> None:
    legs = pd.DataFrame({"model_pick": ["HOME", "HOME"], "actual_winner": ["HOME", "HOME"]})

    with pytest.raises(ValueError, match="positive"):
        compute_round_robin(
            legs, "model_pick", stake_per_combo=0.0, combo_size=2, strategy_name="T"
        )


def test_compute_round_robin_rejects_combo_size_below_two() -> None:
    legs = pd.DataFrame({"model_pick": ["HOME", "HOME"], "actual_winner": ["HOME", "HOME"]})

    with pytest.raises(ValueError, match="at least 2"):
        compute_round_robin(
            legs, "model_pick", stake_per_combo=5.0, combo_size=1, strategy_name="T"
        )


def test_compute_round_robin_reports_insufficient_picks() -> None:
    legs = pd.DataFrame({"model_pick": ["HOME"], "actual_winner": ["HOME"]})

    result = compute_round_robin(
        legs, "model_pick", stake_per_combo=5.0, combo_size=2, strategy_name="T"
    )

    assert result.combo_count == 0
    assert result.reason is not None
    assert "need at least 2" in result.reason


def test_compute_round_robin_matches_hand_computation() -> None:
    # 3 picks, all resolved: legs 0 and 1 win, leg 2 loses.
    # C(3,2) = 3 combos: (0,1) won, (0,2) lost, (1,2) lost.
    legs = pd.DataFrame(
        {
            "model_pick": ["HOME", "HOME", "HOME"],
            "actual_winner": ["HOME", "HOME", "AWAY"],
        }
    )

    result = compute_round_robin(
        legs, "model_pick", stake_per_combo=5.0, combo_size=2, strategy_name="T"
    )

    payout_per_win = 5.0 * (decimal_odds_from_american(FIXED_AMERICAN_ODDS) ** 2)
    expected_profit = (1 * (payout_per_win - 5.0)) - (2 * 5.0)

    assert result.combo_count == 3
    assert result.wins == 1
    assert result.losses == 2
    assert result.pending_combos == 0
    assert result.total_staked == pytest.approx(15.0)
    assert result.realized_profit == pytest.approx(expected_profit)


def test_compute_round_robin_handles_pending_legs() -> None:
    # 3 picks: legs 0 and 1 resolved wins, leg 2 still pending.
    # (0,1) won; (0,2) and (1,2) each include the pending leg 2 -> live.
    legs = pd.DataFrame(
        {
            "model_pick": ["HOME", "HOME", "HOME"],
            "actual_winner": ["HOME", "HOME", None],
        }
    )

    result = compute_round_robin(
        legs, "model_pick", stake_per_combo=5.0, combo_size=2, strategy_name="T"
    )

    assert result.combo_count == 3
    assert result.wins == 1
    assert result.losses == 0
    assert result.pending_combos == 2
    assert result.resolved_combos == 1


def test_compute_round_robin_supports_three_leg_combos() -> None:
    legs = pd.DataFrame(
        {
            "model_pick": ["HOME", "HOME", "HOME", "HOME"],
            "actual_winner": ["HOME", "HOME", "HOME", "HOME"],
        }
    )

    result = compute_round_robin(
        legs, "model_pick", stake_per_combo=5.0, combo_size=3, strategy_name="T"
    )

    # C(4,3) = 4 combos, all resolved wins.
    assert result.combo_count == 4
    assert result.wins == 4
    assert result.losses == 0


# --- run_round_robin_tracker (real picks_store fixture) -----------------------------------


def test_run_round_robin_tracker_scopes_to_one_date(tmp_path: Path) -> None:
    write_slate_picks(
        tmp_path,
        game_date="2025-11-05",
        legs=[("HOME", "HOME", "HOME"), ("AWAY", "AWAY", "AWAY"), ("HOME", "HOME", "AWAY")],
        id_prefix="e",
    )
    write_slate_picks(
        tmp_path,
        game_date="2025-11-06",
        legs=[("HOME", "HOME", "AWAY")],
        id_prefix="f",
    )

    results = run_round_robin_tracker(tmp_path, game_date="2025-11-05", stake_per_combo=5.0)
    model_rr = next(r for r in results if r.strategy_name == "Model round robin")

    assert model_rr.total_picks == 3
    assert model_rr.combo_count == 3


def test_run_round_robin_tracker_diverges_by_strategy(tmp_path: Path) -> None:
    # Model always picks correctly; user always picks the opposite side.
    write_slate_picks(
        tmp_path,
        game_date="2025-11-05",
        legs=[("AWAY", "HOME", "HOME"), ("HOME", "AWAY", "AWAY"), ("AWAY", "HOME", "HOME")],
        id_prefix="g",
    )

    results = run_round_robin_tracker(tmp_path, game_date="2025-11-05", stake_per_combo=5.0)
    by_name = {r.strategy_name: r for r in results}

    assert by_name["Model round robin"].wins == 3
    assert by_name["Your round robin"].losses == 3
