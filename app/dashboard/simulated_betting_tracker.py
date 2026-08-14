"""Simulate a "bet the model" strategy against the dashboard's real picks.

This is entirely a paper exercise: no real money, no sportsbook account,
no external odds feed. It answers one question -- if every resolved pick
had been a flat $10 wager at a standard -110 American moneyline (the
typical sportsbook vig on either side of a close game), would following
the model have come out ahead? -110 breaks even at a 52.38% hit rate, so
this is a real bar, not an arbitrary one: beating it is the actual
threshold a strategy would need to clear against a real bookmaker's
typical juice, even though these are not real bookmaker lines.

Deliberately does NOT derive odds from the model's own predicted
probability. Betting at odds equal to your own stated confidence is
tautologically break-even in expectation (the "edge" and the "price" are
the same number) -- that would always average to roughly zero and say
nothing about whether the model is actually any good. A fixed, realistic
vig is what makes the result mean something.

This does not constitute betting advice, and nothing here interacts with
any real sportsbook, account, or payment method.

Also simulates parlays: combining one day's picks into a single bet where
every leg must win. Same fixed odds, same "no real money" scope -- one
resolved loss busts the whole parlay even if other legs haven't been
decided yet, exactly like a real one would.

And round robins: instead of one all-or-nothing parlay across every pick,
split the day's picks into every possible combination of a chosen size
(2-leg by default) and treat each combination as its own independent
mini-parlay with its own stake. One bad leg only kills the combinations
that included it, not the whole thing -- a softer risk shape than a
straight parlay, using the same fixed odds and no real money.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Final

import pandas as pd

from app.dashboard.picks_store import get_all_picks

# A standard sportsbook moneyline vig, applied uniformly regardless of how
# confident the model or the user was -- not a real market line.
FIXED_AMERICAN_ODDS: Final[int] = -110

# Purely illustrative; no real money is ever staked.
FLAT_STAKE: Final[float] = 10.0

# Below this many resolved bets, hit rate and ROI are dominated by sampling
# noise -- reporting a number would look precise without being meaningful.
MIN_BETS_FOR_SIGNAL: Final[int] = 10


def american_odds_profit(stake: float, odds: int) -> float:
    """Return the profit (not including stake return) from a winning bet.

    Standard American-odds convention: negative odds are "risk |odds| to
    win 100"; positive odds are "risk 100 to win odds". Handles both,
    even though this module only ever uses a fixed negative value.
    """

    if odds == 0:
        raise ValueError("odds cannot be zero")

    if odds < 0:
        return stake * (100.0 / abs(odds))

    return stake * (odds / 100.0)


@dataclass(frozen=True)
class BettingStrategyResult:
    """One strategy's simulated results across every resolved pick."""

    strategy_name: str
    bets_placed: int
    sufficient_sample: bool
    reason: str | None
    wins: int | None = None
    losses: int | None = None
    hit_rate: float | None = None
    breakeven_hit_rate: float | None = None
    beats_breakeven: bool | None = None
    total_staked: float | None = None
    total_profit: float | None = None
    roi: float | None = None


def breakeven_hit_rate(odds: int = FIXED_AMERICAN_ODDS) -> float:
    """Return the win rate at which flat betting at these odds nets zero."""

    if odds >= 0:
        raise ValueError("breakeven_hit_rate expects negative (favorite) odds")

    return abs(odds) / (abs(odds) + 100.0)


def simulate_strategy(
    resolved_picks: pd.DataFrame,
    pick_column: str,
    strategy_name: str,
) -> BettingStrategyResult:
    """Simulate flat-staking every resolved pick's chosen side."""

    bets_placed = len(resolved_picks)

    if bets_placed < MIN_BETS_FOR_SIGNAL:
        return BettingStrategyResult(
            strategy_name=strategy_name,
            bets_placed=bets_placed,
            sufficient_sample=False,
            reason=f"fewer than {MIN_BETS_FOR_SIGNAL} resolved bets",
        )

    wins = int((resolved_picks[pick_column] == resolved_picks["actual_winner"]).sum())
    losses = bets_placed - wins

    win_profit = american_odds_profit(FLAT_STAKE, FIXED_AMERICAN_ODDS)
    total_profit = (wins * win_profit) - (losses * FLAT_STAKE)
    total_staked = bets_placed * FLAT_STAKE
    hit_rate = wins / bets_placed
    breakeven = breakeven_hit_rate()

    return BettingStrategyResult(
        strategy_name=strategy_name,
        bets_placed=bets_placed,
        sufficient_sample=True,
        reason=None,
        wins=wins,
        losses=losses,
        hit_rate=hit_rate,
        breakeven_hit_rate=breakeven,
        beats_breakeven=hit_rate > breakeven,
        total_staked=total_staked,
        total_profit=total_profit,
        roi=(total_profit / total_staked) if total_staked else None,
    )


def decimal_odds_from_american(odds: int) -> float:
    """Convert American odds to a decimal payout multiplier (stake included).

    A $1 stake at these decimal odds returns $decimal_odds total if it
    wins -- e.g. 1.9091 for -110, meaning a winning $10 bet returns $19.09
    (a $9.09 profit, matching american_odds_profit's result).
    """

    if odds == 0:
        raise ValueError("odds cannot be zero")

    if odds < 0:
        return 1.0 + (100.0 / abs(odds))

    return 1.0 + (odds / 100.0)


@dataclass(frozen=True)
class ParlayResult:
    """One simulated same-day parlay: every leg must win, and any resolved
    loss busts the whole bet even if other legs are still pending.
    """

    strategy_name: str
    stake: float
    leg_count: int
    resolved_legs: int
    pending_legs: int
    status: str
    potential_payout: float | None
    potential_profit: float | None
    realized_profit: float | None


def compute_parlay(
    picks_for_date: pd.DataFrame,
    pick_column: str,
    stake: float,
    strategy_name: str,
) -> ParlayResult:
    """Simulate combining one date's picks into a single parlay.

    status is "no_picks" (nothing to combine), "lost" (at least one
    resolved leg lost -- the parlay is dead regardless of any pick still
    pending), "live" (every resolved leg so far has won, but at least one
    leg has not been decided yet), or "won" (every leg resolved and won).
    """

    if stake <= 0:
        raise ValueError("stake must be positive")

    leg_count = len(picks_for_date)

    if leg_count == 0:
        return ParlayResult(
            strategy_name=strategy_name,
            stake=stake,
            leg_count=0,
            resolved_legs=0,
            pending_legs=0,
            status="no_picks",
            potential_payout=None,
            potential_profit=None,
            realized_profit=None,
        )

    resolved = picks_for_date.loc[picks_for_date["actual_winner"].notna()]
    pending_legs = leg_count - len(resolved)
    resolved_losses = int((resolved[pick_column] != resolved["actual_winner"]).sum())

    combined_multiplier = decimal_odds_from_american(FIXED_AMERICAN_ODDS) ** leg_count
    potential_payout = stake * combined_multiplier
    potential_profit = potential_payout - stake

    if resolved_losses > 0:
        status = "lost"
        realized_profit = -stake
    elif pending_legs == 0:
        status = "won"
        realized_profit = potential_profit
    else:
        status = "live"
        realized_profit = None

    return ParlayResult(
        strategy_name=strategy_name,
        stake=stake,
        leg_count=leg_count,
        resolved_legs=len(resolved),
        pending_legs=pending_legs,
        status=status,
        potential_payout=potential_payout,
        potential_profit=potential_profit,
        realized_profit=realized_profit,
    )


def run_parlay_tracker(project_root: Path, game_date: str, stake: float) -> list[ParlayResult]:
    """Simulate a same-day parlay for both the model's picks and the user's picks."""

    picks = get_all_picks(project_root)
    picks_for_date = picks.loc[picks["game_date"].eq(game_date)]

    return [
        compute_parlay(picks_for_date, "model_pick", stake, "Model parlay"),
        compute_parlay(picks_for_date, "user_pick", stake, "Your parlay"),
    ]


@dataclass(frozen=True)
class RoundRobinResult:
    """Every possible combo_size-leg combination from one day's picks,
    each scored as its own independent mini-parlay.
    """

    strategy_name: str
    combo_size: int
    stake_per_combo: float
    total_picks: int
    combo_count: int
    resolved_combos: int
    pending_combos: int
    wins: int
    losses: int
    total_staked: float
    realized_profit: float | None
    reason: str | None = None


def compute_round_robin(
    picks_for_date: pd.DataFrame,
    pick_column: str,
    stake_per_combo: float,
    combo_size: int,
    strategy_name: str,
) -> RoundRobinResult:
    """Split one day's picks into every combo_size-leg combination and
    score each as an independent parlay -- one bad leg only kills the
    combinations that included it, not the whole round robin.
    """

    if stake_per_combo <= 0:
        raise ValueError("stake_per_combo must be positive")

    if combo_size < 2:
        raise ValueError("combo_size must be at least 2")

    total_picks = len(picks_for_date)

    if total_picks < combo_size:
        return RoundRobinResult(
            strategy_name=strategy_name,
            combo_size=combo_size,
            stake_per_combo=stake_per_combo,
            total_picks=total_picks,
            combo_count=0,
            resolved_combos=0,
            pending_combos=0,
            wins=0,
            losses=0,
            total_staked=0.0,
            realized_profit=None,
            reason=(
                f"need at least {combo_size} picks for a {combo_size}-team round robin, "
                f"have {total_picks}"
            ),
        )

    payout_per_win = stake_per_combo * (
        decimal_odds_from_american(FIXED_AMERICAN_ODDS) ** combo_size
    )

    wins = 0
    losses = 0
    pending = 0

    for combo_positions in combinations(range(total_picks), combo_size):
        combo = picks_for_date.iloc[list(combo_positions)]
        resolved = combo.loc[combo["actual_winner"].notna()]
        resolved_losses = int((resolved[pick_column] != resolved["actual_winner"]).sum())
        still_pending = len(combo) - len(resolved)

        if resolved_losses > 0:
            losses += 1
        elif still_pending == 0:
            wins += 1
        else:
            pending += 1

    combo_count = wins + losses + pending
    realized_profit = (wins * (payout_per_win - stake_per_combo)) - (losses * stake_per_combo)

    return RoundRobinResult(
        strategy_name=strategy_name,
        combo_size=combo_size,
        stake_per_combo=stake_per_combo,
        total_picks=total_picks,
        combo_count=combo_count,
        resolved_combos=wins + losses,
        pending_combos=pending,
        wins=wins,
        losses=losses,
        total_staked=combo_count * stake_per_combo,
        realized_profit=realized_profit,
        reason=None,
    )


def run_round_robin_tracker(
    project_root: Path,
    game_date: str,
    stake_per_combo: float,
    combo_size: int = 2,
) -> list[RoundRobinResult]:
    """Simulate a same-day round robin for both the model's and the user's picks."""

    picks = get_all_picks(project_root)
    picks_for_date = picks.loc[picks["game_date"].eq(game_date)]

    return [
        compute_round_robin(
            picks_for_date, "model_pick", stake_per_combo, combo_size, "Model round robin"
        ),
        compute_round_robin(
            picks_for_date, "user_pick", stake_per_combo, combo_size, "Your round robin"
        ),
    ]


def run_simulated_betting_tracker(project_root: Path) -> list[BettingStrategyResult]:
    """Simulate both the model's picks and the user's own picks as strategies."""

    picks = get_all_picks(project_root)
    resolved = picks.loc[picks["actual_winner"].notna()]

    return [
        simulate_strategy(resolved, "model_pick", "Bet the model"),
        simulate_strategy(resolved, "user_pick", "Bet your own picks"),
    ]


def betting_tracker_summary_path(project_root: Path) -> Path:
    """Return the betting tracker summary path."""

    return project_root / "artifacts" / "nba" / "betting_tracker" / "simulated_betting_summary.json"


def write_summary(results: list[BettingStrategyResult], project_root: Path) -> Path:
    """Persist the simulated betting summary as JSON."""

    output_path = betting_tracker_summary_path(project_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "note": (
            "Simulated only. No real money, sportsbook account, or payment "
            "method involved. Odds are a fixed -110 illustrative vig, not "
            "real market lines. Not betting advice."
        ),
        "fixed_american_odds": FIXED_AMERICAN_ODDS,
        "flat_stake": FLAT_STAKE,
        "strategies": [asdict(result) for result in results],
    }

    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    return output_path


def print_report(results: list[BettingStrategyResult]) -> None:
    """Print a readable simulated-betting report."""

    print(
        f"\nSimulated betting tracker (${FLAT_STAKE:.0f} flat stake, "
        f"{FIXED_AMERICAN_ODDS} illustrative odds, breakeven "
        f"{breakeven_hit_rate():.1%}):"
    )

    for result in results:
        if not result.sufficient_sample:
            print(f"  {result.strategy_name:<20} {result.bets_placed:>4} bets -- {result.reason}")
            continue

        verdict = "beats breakeven" if result.beats_breakeven else "below breakeven"
        print(
            f"  {result.strategy_name:<20} {result.bets_placed:>4} bets -- "
            f"hit rate {result.hit_rate:.1%} ({verdict}), "
            f"profit ${result.total_profit:+.2f}, ROI {result.roi:+.1%}"
        )


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the simulated betting tracker."""

    return argparse.ArgumentParser(description=__doc__).parse_args()


def main() -> None:
    """Run the simulated betting tracker from the command line."""

    parse_args()
    project_root = Path(__file__).resolve().parents[2]

    results = run_simulated_betting_tracker(project_root)
    output_path = write_summary(results, project_root)

    print_report(results)
    print(f"\nSummary: {output_path}")


if __name__ == "__main__":
    main()
