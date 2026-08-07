"""Build a model-ready feature row for any matchup -- past or upcoming.

For a game that is already in the historical modeling dataset, this just
looks up its existing row directly: it was already computed and validated
by the Phase 5 pipeline, so recomputing it would only add risk.

For a matchup that is not in the historical dataset yet (a genuinely
upcoming game), there is no shortcut: pregame features have to be computed
from each team's most recent known state. The key insight is that this is
the exact same computation the Phase 5 pipeline already does, one step
further: build_pregame_team_features.py's shifted_expanding_mean and
shifted_rolling_mean functions compute "everything before the current
game" via shift(1). If a team's real history is extended with one
placeholder row dated at the target game, shift(1) at that placeholder
naturally evaluates to "everything through the team's most recent real
game" -- because shift(1) only ever looks backward, the placeholder's own
(unknown) values are never read, by the exact same construction that
already keeps today's score out of today's features. This reuses
build_pregame_team_features.py's real functions rather than
reimplementing rolling-window logic a second time.

Elo works the same way, one level simpler: a team's current rating is
just its most recent POST_GAME_ELO_RATING, regressed toward the mean via
build_team_elo_ratings.py's carried_rating() if the target game falls in
a new season.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pipelines.features.build_modeling_dataset import (
    CATEGORICAL_FEATURE_COLUMNS,
    NUMERIC_FEATURE_COLUMNS,
    SIDE_FEATURE_COLUMNS,
    add_matchup_comparison_features,
    modeling_dataset_output_path,
)
from pipelines.features.build_pregame_team_features import (
    add_outcome_history_features,
    add_schedule_features,
)
from pipelines.features.build_team_elo_ratings import (
    INITIAL_RATING,
    carried_rating,
    elo_ratings_output_path,
)
from pipelines.features.build_team_history import team_history_output_path

# The rolling/season-scoped features, excluding ELO_RATING which has its
# own separate, cross-season computation below.
ROLLING_STATE_COLUMNS = tuple(column for column in SIDE_FEATURE_COLUMNS if column != "ELO_RATING")

MODEL_ROW_COLUMNS = (*CATEGORICAL_FEATURE_COLUMNS, *NUMERIC_FEATURE_COLUMNS)


def load_team_history(project_root: Path) -> pd.DataFrame:
    """Load the completed-game team history built by Phase 5."""

    path = team_history_output_path(project_root)

    if not path.exists():
        raise FileNotFoundError(f"Team history does not exist: {path}")

    history = pd.read_parquet(path)
    history["GAME_DATE"] = pd.to_datetime(history["GAME_DATE"], errors="raise")

    return history


def load_elo_ratings(project_root: Path) -> pd.DataFrame:
    """Load the simulated Elo ratings built by Phase 5."""

    path = elo_ratings_output_path(project_root)

    if not path.exists():
        raise FileNotFoundError(f"Elo ratings do not exist: {path}")

    ratings = pd.read_parquet(path)
    ratings["GAME_DATE"] = pd.to_datetime(ratings["GAME_DATE"], errors="raise")

    return ratings


def load_modeling_dataset(project_root: Path) -> pd.DataFrame:
    """Load the final historical modeling dataset built by Phase 5."""

    path = modeling_dataset_output_path(project_root)

    if not path.exists():
        raise FileNotFoundError(f"Modeling dataset does not exist: {path}")

    modeling = pd.read_parquet(path)
    modeling["GAME_DATE"] = pd.to_datetime(modeling["GAME_DATE"], errors="raise")

    return modeling


def find_historical_matchup(
    modeling_dataset: pd.DataFrame,
    home_team_id: int,
    away_team_id: int,
    game_date: str | pd.Timestamp,
) -> pd.DataFrame | None:
    """Return the existing modeling row for this exact matchup, if any."""

    target_date = pd.Timestamp(game_date)

    match = modeling_dataset.loc[
        modeling_dataset["HOME_TEAM_ID"].eq(home_team_id)
        & modeling_dataset["AWAY_TEAM_ID"].eq(away_team_id)
        & modeling_dataset["GAME_DATE"].eq(target_date)
    ]

    return None if match.empty else match.reset_index(drop=True)


def compute_team_rolling_state(
    team_history: pd.DataFrame,
    team_id: int,
    season: str,
    as_of_date: pd.Timestamp,
) -> dict[str, float]:
    """Compute a team's pregame rolling/season state for a hypothetical
    next game on as_of_date, from its real same-season history so far.
    """

    state_columns = [
        "SEASON",
        "TEAM_ID",
        "GAME_DATE",
        "TEAM_WIN",
        "TEAM_PTS",
        "OPPONENT_PTS",
        "POINT_DIFFERENTIAL",
    ]

    team_season_history = team_history.loc[
        team_history["TEAM_ID"].eq(team_id) & team_history["SEASON"].eq(season),
        state_columns,
    ].sort_values(["GAME_DATE"], kind="stable")

    if not team_season_history.empty and (team_season_history["GAME_DATE"] >= as_of_date).any():
        raise ValueError(
            f"Team {team_id} already has a recorded {season} game on or after {as_of_date.date()}"
        )

    placeholder = pd.DataFrame(
        [
            {
                "SEASON": season,
                "TEAM_ID": team_id,
                "GAME_DATE": as_of_date,
                "TEAM_WIN": 0,
                "TEAM_PTS": 0,
                "OPPONENT_PTS": 0,
                "POINT_DIFFERENTIAL": 0,
            }
        ]
    )

    if team_season_history.empty:
        # Concatenating with a zero-row frame risks a dtype-inference
        # FutureWarning; there is nothing to combine with anyway.
        extended = placeholder
    else:
        extended = pd.concat([team_season_history, placeholder], ignore_index=True)

    extended = extended.sort_values(["GAME_DATE"], kind="stable").reset_index(drop=True)

    extended["PRIOR_GAMES_PLAYED"] = extended.groupby(["SEASON", "TEAM_ID"], sort=False).cumcount()

    extended = add_outcome_history_features(extended)
    extended = add_schedule_features(extended)

    placeholder_row = extended.iloc[-1]

    state = {column: placeholder_row[column] for column in ROLLING_STATE_COLUMNS}
    state["PRIOR_GAMES_PLAYED"] = int(placeholder_row["PRIOR_GAMES_PLAYED"])

    return state


def compute_team_current_elo(
    elo_ratings: pd.DataFrame,
    team_id: int,
    season: str,
) -> float:
    """Compute a team's current Elo rating, regressed for a new season."""

    team_ratings = elo_ratings.loc[elo_ratings["TEAM_ID"].eq(team_id)].sort_values(
        ["GAME_DATE"], kind="stable"
    )

    if team_ratings.empty:
        return INITIAL_RATING

    most_recent = team_ratings.iloc[-1]
    rating = float(most_recent["POST_GAME_ELO_RATING"])

    if most_recent["SEASON"] != season:
        rating = carried_rating(rating)

    return rating


def compute_team_state(
    team_history: pd.DataFrame,
    elo_ratings: pd.DataFrame,
    team_id: int,
    season: str,
    as_of_date: pd.Timestamp,
) -> dict[str, float]:
    """Compute a team's full pregame state (rolling stats + Elo)."""

    state = compute_team_rolling_state(
        team_history=team_history,
        team_id=team_id,
        season=season,
        as_of_date=as_of_date,
    )
    state["ELO_RATING"] = compute_team_current_elo(
        elo_ratings=elo_ratings,
        team_id=team_id,
        season=season,
    )

    return state


def build_computed_matchup_row(
    home_state: dict[str, float],
    away_state: dict[str, float],
    home_team_id: int,
    away_team_id: int,
) -> pd.DataFrame:
    """Assemble both teams' states into one HOME_/AWAY_/DIFF feature row.

    Reuses build_modeling_dataset.py's own add_matchup_comparison_features
    so the DIFF and history-flag logic is identical to the historical
    pipeline's, not a second, possibly-divergent implementation.
    """

    row: dict[str, object] = {
        "HOME_TEAM_ID": home_team_id,
        "AWAY_TEAM_ID": away_team_id,
    }

    for column in SIDE_FEATURE_COLUMNS:
        row[f"HOME_{column}"] = home_state[column]
        row[f"AWAY_{column}"] = away_state[column]

    frame = pd.DataFrame([row])
    frame = add_matchup_comparison_features(frame)

    return frame.loc[:, list(MODEL_ROW_COLUMNS)]


def compute_matchup_features(
    project_root: Path,
    home_team_id: int,
    away_team_id: int,
    game_date: str | pd.Timestamp,
    season: str,
) -> tuple[pd.DataFrame, str, int | None]:
    """Return (feature_row, source, actual_home_win) for one matchup.

    source is "historical" when the game already exists in the modeling
    dataset (feature_row is looked up, not recomputed) or "computed" when
    it was built from each team's current state. actual_home_win is the
    known outcome for a historical game, or None for a matchup that
    has not been played yet.
    """

    modeling_dataset = load_modeling_dataset(project_root)

    historical_match = find_historical_matchup(
        modeling_dataset=modeling_dataset,
        home_team_id=home_team_id,
        away_team_id=away_team_id,
        game_date=game_date,
    )

    if historical_match is not None:
        feature_row = historical_match.loc[:, list(MODEL_ROW_COLUMNS)]
        actual_home_win = int(historical_match.iloc[0]["home_win"])

        return feature_row, "historical", actual_home_win

    team_history = load_team_history(project_root)
    elo_ratings = load_elo_ratings(project_root)
    as_of_date = pd.Timestamp(game_date)

    home_state = compute_team_state(
        team_history=team_history,
        elo_ratings=elo_ratings,
        team_id=home_team_id,
        season=season,
        as_of_date=as_of_date,
    )
    away_state = compute_team_state(
        team_history=team_history,
        elo_ratings=elo_ratings,
        team_id=away_team_id,
        season=season,
        as_of_date=as_of_date,
    )

    feature_row = build_computed_matchup_row(
        home_state=home_state,
        away_state=away_state,
        home_team_id=home_team_id,
        away_team_id=away_team_id,
    )

    return feature_row, "computed", None
