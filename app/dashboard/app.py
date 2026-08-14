"""Pick-the-winner dashboard: guess a slate of games against the frozen model.

Defaults to a historical date from the ingested seasons, since the
2026-27 regular season has not started yet (see the README's Results
section) -- there is nothing live to guess on today, but the exact same
code path works unchanged once real upcoming games exist: schedule
fetching, feature computation, and prediction do not know or care whether
a date is in the past or the future.

Run with: uv run streamlit run app/dashboard/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Unlike `python -m`, `streamlit run` does not add the project root to
# sys.path, so package-style imports below (app.dashboard.*, modeling.*,
# pipelines.*) would otherwise fail with "No module named 'app.dashboard'".
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import streamlit as st
from modeling.serving.matchup_features import compute_matchup_features
from modeling.serving.predict_matchup import (
    load_selected_model,
    predict_home_win_probability,
)
from pipelines.ingestion.fetch_schedule import (
    GAME_STATUS_FINAL,
    fetch_schedule,
    filter_regular_season_games,
    games_on_date,
    season_for_date,
)

from app.dashboard.live_calibration_monitor import (
    load_frozen_reference_metrics,
    run_live_calibration_monitor,
)
from app.dashboard.picks_store import (
    Pick,
    compute_scoreboard,
    get_all_picks,
    get_pick,
    record_pick,
    update_actual_result,
)
from app.dashboard.simulated_betting_tracker import (
    FIXED_AMERICAN_ODDS,
    FLAT_STAKE,
    breakeven_hit_rate,
    run_parlay_tracker,
    run_round_robin_tracker,
    run_simulated_betting_tracker,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DEMO_DATE = pd.Timestamp("2025-11-05")


@st.cache_data(ttl=3600, show_spinner="Fetching NBA schedule...")
def cached_season_schedule(season: str) -> pd.DataFrame:
    """Fetch and cache one season's full schedule."""

    return fetch_schedule(season=season)


@st.cache_resource(show_spinner="Loading the frozen model...")
def cached_model():
    """Load the frozen logistic regression pipeline once per app process."""

    return load_selected_model(PROJECT_ROOT)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_matchup_features(home_team_id: int, away_team_id: int, game_date: str, season: str):
    """Compute (or look up) one matchup's feature row once, then reuse it for
    both the pre-pick stat comparison and the post-pick prediction.

    ttl matches cached_season_schedule: for an upcoming ("computed")
    matchup, the underlying team_history/elo_ratings files can change if
    the ingestion pipeline is re-run while this process keeps running, and
    without a ttl this cache would keep serving the stale pre-refresh
    features indefinitely instead of picking up the new data.
    """

    return compute_matchup_features(
        project_root=PROJECT_ROOT,
        home_team_id=home_team_id,
        away_team_id=away_team_id,
        game_date=game_date,
        season=season,
    )


def format_record(win_pct: object, window: int) -> str:
    """Turn a win percentage plus a game count into a "W-L" string."""

    if window <= 0 or pd.isna(win_pct):
        return "—"

    wins = round(float(win_pct) * window)

    return f"{wins}-{window - wins}"


def format_stat(value: object, decimals: int = 1) -> str:
    """Format a numeric feature for display, or an em dash when missing."""

    if pd.isna(value):
        return "—"

    return f"{float(value):.{decimals}f}"


def render_stat_comparison(feature_row: pd.DataFrame, away_name: str, home_name: str) -> None:
    """Show the same pregame numbers the model sees, before any pick is made."""

    row = feature_row.iloc[0]

    home_games = int(row["HOME_PRIOR_GAMES_PLAYED"])
    away_games = int(row["AWAY_PRIOR_GAMES_PLAYED"])
    home_l10, away_l10 = min(home_games, 10), min(away_games, 10)

    comparison = pd.DataFrame(
        {
            away_name: [
                format_record(row["AWAY_SEASON_WIN_PCT"], away_games),
                format_record(row["AWAY_ROLLING_10_WIN_PCT"], away_l10),
                format_stat(row["AWAY_ROLLING_10_POINTS_SCORED"]),
                format_stat(row["AWAY_ROLLING_10_POINTS_ALLOWED"]),
                format_stat(row["AWAY_ROLLING_10_POINT_DIFFERENTIAL"]),
                "Yes" if row["AWAY_IS_BACK_TO_BACK"] else "No",
                format_stat(row["AWAY_ELO_RATING"], decimals=0),
            ],
            home_name: [
                format_record(row["HOME_SEASON_WIN_PCT"], home_games),
                format_record(row["HOME_ROLLING_10_WIN_PCT"], home_l10),
                format_stat(row["HOME_ROLLING_10_POINTS_SCORED"]),
                format_stat(row["HOME_ROLLING_10_POINTS_ALLOWED"]),
                format_stat(row["HOME_ROLLING_10_POINT_DIFFERENTIAL"]),
                "Yes" if row["HOME_IS_BACK_TO_BACK"] else "No",
                format_stat(row["HOME_ELO_RATING"], decimals=0),
            ],
        },
        index=[
            "Season record",
            "Last 10",
            "Pts/gm (L10)",
            "Pts allowed/gm (L10)",
            "Point diff (L10)",
            "On back-to-back",
            "Power rating",
        ],
    )

    st.table(comparison)
    st.caption(
        "Power rating is an Elo-style score: higher means stronger, adjusted for "
        "opponent quality -- the single most predictive number the model uses."
    )


def actual_winner_from_scores(home_score: object, away_score: object) -> str | None:
    """Derive HOME/AWAY from a finished schedule row's scores."""

    if home_score is None or away_score is None:
        return None

    try:
        home, away = int(home_score), int(away_score)
    except (TypeError, ValueError):
        return None

    return "HOME" if home > away else "AWAY"


def render_pick_prompt(
    game: pd.Series,
    season: str,
    feature_row: pd.DataFrame,
    source: str,
    historical_home_win: int | None,
) -> None:
    """Let the user pick a winner, then reveal the model and (if known) reality."""

    game_id = game["GAME_ID"]
    home_name, away_name = game["HOME_TEAM_NAME"], game["AWAY_TEAM_NAME"]

    choice_key = f"choice_{game_id}"
    choice = st.radio(
        "Your pick",
        options=["HOME", "AWAY"],
        format_func=lambda side: home_name if side == "HOME" else away_name,
        key=choice_key,
        index=None,
        horizontal=True,
    )

    if st.button("Lock in prediction", key=f"submit_{game_id}", disabled=choice is None):
        with st.spinner("Computing the model's prediction..."):
            model = cached_model()
            home_win_probability = predict_home_win_probability(model, feature_row)

        model_pick = "HOME" if home_win_probability >= 0.5 else "AWAY"

        actual_winner = (
            "HOME" if historical_home_win == 1 else "AWAY" if historical_home_win == 0 else None
        )
        if actual_winner is None and game["GAME_STATUS"] == GAME_STATUS_FINAL:
            actual_winner = actual_winner_from_scores(game["HOME_SCORE"], game["AWAY_SCORE"])

        record_pick(
            PROJECT_ROOT,
            Pick(
                game_id=game_id,
                game_date=str(game["GAME_DATE"].date()),
                season=season,
                home_team_id=int(game["HOME_TEAM_ID"]),
                home_team_name=home_name,
                away_team_id=int(game["AWAY_TEAM_ID"]),
                away_team_name=away_name,
                user_pick=choice,
                model_pick=model_pick,
                model_home_win_probability=home_win_probability,
                feature_source=source,
                actual_winner=actual_winner,
            ),
        )
        st.rerun()


def render_reveal(pick: dict[str, object]) -> None:
    """Show a previously recorded pick: yours, the model's, and reality."""

    home_name, away_name = pick["home_team_name"], pick["away_team_name"]
    probability = pick["model_home_win_probability"]

    columns = st.columns(3)

    with columns[0]:
        picked_name = home_name if pick["user_pick"] == "HOME" else away_name
        st.metric("Your pick", picked_name)

    with columns[1]:
        model_name = home_name if pick["model_pick"] == "HOME" else away_name
        st.metric("Model pick", model_name, f"{probability:.0%} home-win prob.")

    with columns[2]:
        if pick["actual_winner"] is None:
            st.metric("Actual result", "Pending")
        else:
            winner_name = home_name if pick["actual_winner"] == "HOME" else away_name
            st.metric("Actual winner", winner_name)

    if pick["actual_winner"] is not None:
        user_correct = pick["user_pick"] == pick["actual_winner"]
        model_correct = pick["model_pick"] == pick["actual_winner"]
        st.write(
            f"You were {'✅ right' if user_correct else '❌ wrong'} · "
            f"Model was {'✅ right' if model_correct else '❌ wrong'}"
        )

    if pick["feature_source"] == "computed":
        st.caption(
            "Features computed as-of the pick date -- this matchup wasn't in the historical set."
        )


def render_slate(day_slate: pd.DataFrame, season: str) -> None:
    """Render one pick block per game in the slate."""

    for _, game in day_slate.iterrows():
        header = f"{game['AWAY_TEAM_NAME']} @ {game['HOME_TEAM_NAME']}"

        with st.container(border=True):
            st.subheader(header)
            st.caption(game["GAME_STATUS_TEXT"].strip() or "Scheduled")

            feature_row, source, historical_home_win = cached_matchup_features(
                int(game["HOME_TEAM_ID"]),
                int(game["AWAY_TEAM_ID"]),
                str(game["GAME_DATE"].date()),
                season,
            )
            render_stat_comparison(feature_row, game["AWAY_TEAM_NAME"], game["HOME_TEAM_NAME"])

            existing_pick = get_pick(PROJECT_ROOT, game["GAME_ID"])

            if existing_pick is not None:
                render_reveal(existing_pick)
            else:
                render_pick_prompt(game, season, feature_row, source, historical_home_win)


def render_scoreboard() -> None:
    """Show cumulative accuracy across every resolved pick."""

    scoreboard = compute_scoreboard(PROJECT_ROOT)

    if scoreboard.resolved_picks == 0:
        st.info("No resolved picks yet -- lock in some predictions above.")
        return

    columns = st.columns(3)
    columns[0].metric("Resolved picks", scoreboard.resolved_picks)
    columns[1].metric("Your accuracy", f"{scoreboard.user_accuracy:.0%}")
    columns[2].metric("Model accuracy", f"{scoreboard.model_accuracy:.0%}")


def render_refresh_button(season: str) -> None:
    """Check any unresolved picks against the live schedule and update them."""

    if not st.button("Check for new results"):
        return

    picks = get_all_picks(PROJECT_ROOT)
    pending = picks.loc[picks["actual_winner"].isna()]

    if pending.empty:
        st.info("Nothing pending.")
        return

    schedule = cached_season_schedule(season)
    resolved_count = 0

    for _, pick in pending.iterrows():
        match = schedule.loc[schedule["GAME_ID"].eq(pick["game_id"])]

        if match.empty:
            continue

        row = match.iloc[0]

        if row["GAME_STATUS"] != GAME_STATUS_FINAL:
            continue

        winner = actual_winner_from_scores(row["HOME_SCORE"], row["AWAY_SCORE"])

        if winner is not None:
            update_actual_result(PROJECT_ROOT, pick["game_id"], winner)
            resolved_count += 1

    st.success(f"Resolved {resolved_count} pick(s).") if resolved_count else st.info(
        "No new final results yet."
    )
    st.rerun()


def render_live_calibration() -> None:
    """Compare the model's real, resolved predictions against its frozen
    test-set calibration -- not a demo number, whatever picks exist so far.
    """

    st.caption(
        "Every prediction this dashboard has made, scored against the real outcome once known, "
        "next to the frozen model's one-time test-set result. The 'computed' row is the only "
        "real-world evidence so far for the upcoming-matchup feature path."
    )

    frozen_reference = load_frozen_reference_metrics(PROJECT_ROOT)

    if frozen_reference is not None:
        columns = st.columns(3)
        columns[0].metric("Frozen test log loss", f"{frozen_reference['log_loss']:.4f}")
        columns[1].metric("Frozen test Brier", f"{frozen_reference['brier_score']:.4f}")
        columns[2].metric(
            "Frozen test ECE", f"{frozen_reference['expected_calibration_error']:.4f}"
        )
    else:
        st.caption("Frozen test reference unavailable -- run the Phase 8 evaluation first.")

    for result in run_live_calibration_monitor(PROJECT_ROOT):
        label = result.feature_source.capitalize()

        if not result.sufficient_sample:
            st.caption(
                f"**{label}**: {result.resolved_picks} resolved -- "
                f"not enough signal yet ({result.reason})."
            )
            continue

        metrics = result.metrics
        columns = st.columns(4)
        columns[0].metric(f"{label} resolved", result.resolved_picks)
        columns[1].metric("Live log loss", f"{metrics['log_loss']:.4f}")
        columns[2].metric("Live Brier", f"{metrics['brier_score']:.4f}")
        columns[3].metric("Live ECE", f"{metrics['expected_calibration_error']:.4f}")


def render_simulated_betting_tracker() -> None:
    """Show a paper-only "bet the model" simulation against real picks.

    No real money, sportsbook account, or payment method is involved
    anywhere in this app -- this is a backtest of a flat-staking strategy
    at a fixed, illustrative vig, not betting advice.
    """

    st.caption(
        f"**Simulated only** -- no real money, no sportsbook, nothing here places a real bet. "
        f"Assumes a flat ${FLAT_STAKE:.0f} stake per resolved pick at a fixed "
        f"{FIXED_AMERICAN_ODDS} American-odds vig (a typical sportsbook moneyline juice, not a "
        f"real market line), which breaks even at a {breakeven_hit_rate():.1%} hit rate."
    )

    for result in run_simulated_betting_tracker(PROJECT_ROOT):
        if not result.sufficient_sample:
            st.caption(
                f"**{result.strategy_name}**: {result.bets_placed} bets -- "
                f"not enough signal yet ({result.reason})."
            )
            continue

        st.markdown(f"**{result.strategy_name}**")
        columns = st.columns(4)
        columns[0].metric("Bets", result.bets_placed)
        columns[1].metric("Hit rate", f"{result.hit_rate:.1%}")
        columns[2].metric("Simulated profit", f"${result.total_profit:+.2f}")
        columns[3].metric("ROI", f"{result.roi:+.1%}")

        if result.beats_breakeven:
            st.caption(f"Beats the {breakeven_hit_rate():.1%} breakeven hit rate.")
        else:
            st.caption(f"Below the {breakeven_hit_rate():.1%} breakeven hit rate.")


def render_round_robin_tracker(picked_date: object) -> None:
    """Show a same-day round robin: every combo-size combination of this
    date's picks scored as its own independent mini-parlay.

    Paper only, same fixed odds as the other trackers. One bad leg only
    kills the combinations that included it, not every combination --
    a softer risk shape than a straight parlay across the same picks.
    """

    st.caption(
        "**Simulated round robin** -- paper only, no real bet placed. Splits this date's picks "
        f"into every possible combo of the chosen size, each its own mini-parlay at a fixed "
        f"{FIXED_AMERICAN_ODDS} illustrative vig per leg."
    )

    columns = st.columns(2)
    stake_per_combo = columns[0].number_input(
        "Stake per combo ($)", min_value=1.0, value=5.0, step=1.0, key="round_robin_stake"
    )
    combo_size = columns[1].selectbox("Combo size", options=[2, 3], key="round_robin_combo_size")

    for result in run_round_robin_tracker(
        PROJECT_ROOT,
        game_date=str(picked_date),
        stake_per_combo=stake_per_combo,
        combo_size=combo_size,
    ):
        if result.combo_count == 0:
            st.caption(f"**{result.strategy_name}**: {result.reason}.")
            continue

        st.markdown(f"**{result.strategy_name}**")
        result_columns = st.columns(5)
        result_columns[0].metric("Combos", result.combo_count)
        result_columns[1].metric("W-L", f"{result.wins}-{result.losses}")
        result_columns[2].metric("Pending", result.pending_combos)
        result_columns[3].metric("Staked", f"${result.total_staked:.2f}")
        result_columns[4].metric("Profit so far", f"${result.realized_profit:+.2f}")


def render_parlay_summary(picked_date: object, stake: float) -> None:
    """Show a same-day simulated parlay across every pick locked in so far.

    Paper only -- no real bet is ever placed. Every leg must win; one
    resolved loss busts the whole parlay even if other legs are still
    pending, matching how a real parlay actually works.
    """

    st.caption(
        f"**Simulated parlay** -- paper only, no real bet placed. Combines every pick locked in "
        f"below for this date into one parlay at a fixed {FIXED_AMERICAN_ODDS} illustrative vig "
        f"per leg (not a real market line)."
    )

    for result in run_parlay_tracker(PROJECT_ROOT, game_date=str(picked_date), stake=stake):
        if result.status == "no_picks":
            st.caption(f"**{result.strategy_name}**: lock in picks below to start a parlay.")
            continue

        columns = st.columns(4)
        columns[0].metric("Legs", f"{result.resolved_legs}/{result.leg_count} resolved")
        columns[1].metric("Stake", f"${result.stake:.2f}")

        if result.status == "won":
            columns[2].metric("Payout", f"${result.potential_payout:.2f}")
            columns[3].metric("Profit", f"${result.realized_profit:+.2f}")
            st.success(f"{result.strategy_name}: WON")
        elif result.status == "lost":
            columns[2].metric("Payout", "$0.00")
            columns[3].metric("Profit", f"${result.realized_profit:+.2f}")
            st.error(f"{result.strategy_name}: LOST")
        else:
            columns[2].metric("Potential payout", f"${result.potential_payout:.2f}")
            columns[3].metric("Potential profit", f"${result.potential_profit:+.2f}")
            st.info(f"{result.strategy_name}: still live ({result.pending_legs} leg(s) pending)")


def main() -> None:
    st.set_page_config(page_title="NBA Win Predictor", page_icon="🏀")
    st.title("🏀 Pick the winners")
    st.caption(
        "Guess a day's slate before seeing the model's prediction. Defaults to a historical "
        "date since the upcoming season hasn't started -- the same flow works unchanged once "
        "it does."
    )

    parlay_stake = st.number_input(
        "💰 Simulated parlay stake ($)",
        min_value=1.0,
        value=FLAT_STAKE,
        step=1.0,
        help="Paper money only -- no real bet is ever placed.",
    )

    picked_date = st.date_input("Slate date", value=DEFAULT_DEMO_DATE)
    season = season_for_date(pd.Timestamp(picked_date))
    st.caption(f"Season: {season}")

    render_parlay_summary(picked_date, parlay_stake)
    st.divider()

    try:
        schedule = cached_season_schedule(season)
    except Exception as error:
        st.error(f"Could not fetch the {season} schedule: {error}")
        return

    regular_season = filter_regular_season_games(schedule)
    day_slate = games_on_date(regular_season, picked_date)

    if day_slate.empty:
        st.warning(f"No regular-season games on {picked_date}. Try another date.")
    else:
        render_slate(day_slate, season)

    st.divider()
    st.subheader("Scoreboard")
    render_scoreboard()
    render_refresh_button(season)

    with st.expander("Live calibration monitor"):
        render_live_calibration()

    with st.expander("Simulated betting tracker (paper only, not real money)"):
        render_simulated_betting_tracker()

    with st.expander("Simulated round robin (paper only, not real money)"):
        render_round_robin_tracker(picked_date)

    with st.expander("Pick history"):
        picks = get_all_picks(PROJECT_ROOT)
        if picks.empty:
            st.write("No picks yet.")
        else:
            st.dataframe(picks, width="stretch")


if __name__ == "__main__":
    main()
