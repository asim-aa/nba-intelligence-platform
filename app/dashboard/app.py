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

from app.dashboard.picks_store import (
    Pick,
    compute_scoreboard,
    get_all_picks,
    get_pick,
    record_pick,
    update_actual_result,
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


def main() -> None:
    st.set_page_config(page_title="NBA Win Predictor", page_icon="🏀")
    st.title("🏀 Pick the winners")
    st.caption(
        "Guess a day's slate before seeing the model's prediction. Defaults to a historical "
        "date since the upcoming season hasn't started -- the same flow works unchanged once "
        "it does."
    )

    picked_date = st.date_input("Slate date", value=DEFAULT_DEMO_DATE)
    season = season_for_date(pd.Timestamp(picked_date))
    st.caption(f"Season: {season}")

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

    with st.expander("Pick history"):
        picks = get_all_picks(PROJECT_ROOT)
        if picks.empty:
            st.write("No picks yet.")
        else:
            st.dataframe(picks, use_container_width=True)


if __name__ == "__main__":
    main()
