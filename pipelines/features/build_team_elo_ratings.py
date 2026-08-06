"""Compute an opponent-adjusted Elo rating for every team going into every game.

The existing pregame features (season win percentage, rolling point
differential, ...) are season-scoped: they reset to missing at the start of
each season, and none of them account for the strength of the opponents a
team's record was built against. This module addresses both gaps with a
single running Elo rating per team:

1. Opponent-adjusted: beating a strong team moves a team's rating more than
   beating a weak one, unlike a raw win percentage.
2. Carried across season boundaries: instead of resetting to "unknown," a
   team's rating regresses partway toward the league mean at the start of a
   new season (rosters mostly carry over), so it is never missing -- every
   team has a defined rating before its very first game in the dataset,
   using the standard initial rating.

Ratings are updated sequentially, in true chronological order across all
seasons, which is why this lives in its own module rather than reusing the
groupby-and-shift machinery in build_pregame_team_features.py: each game's
update depends on the immediately preceding state for both teams, not a
vectorizable window function.

The rating recorded for each team-game row is the PRE-game rating -- the
value known before tipoff, safe to use as a model feature. POST_GAME_ELO_RATING
is kept only for inspection and must never be used as a feature.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

import pandas as pd

REQUIRED_GAME_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "SEASON",
        "SEASON_ID",
        "GAME_ID",
        "GAME_DATE",
        "HOME_TEAM_ID",
        "AWAY_TEAM_ID",
        "HOME_PTS",
        "AWAY_PTS",
        "home_win",
    }
)

INITIAL_RATING: Final[float] = 1500.0
MEAN_RATING: Final[float] = 1500.0

# Fraction of a team's deviation from the mean that carries into a new
# season; the remainder regresses to the mean. Rosters change but rarely
# enough to justify a full reset, so this is a partial reset, not a wipe.
SEASON_CARRYOVER: Final[float] = 0.75

# Elo points added to the home team's rating when computing the expected
# score, representing home-court advantage.
HOME_COURT_ADVANTAGE: Final[float] = 100.0

# Controls how much one game moves a rating; a standard chess/sports value.
K_FACTOR: Final[float] = 20.0

# Floor for the margin-of-victory dampening denominator so it can never
# reach zero or go negative for pathological synthetic inputs.
_MOV_DENOMINATOR_FLOOR: Final[float] = 0.1


@dataclass(frozen=True)
class EloRatingSummary:
    """Describe one completed Elo rating computation."""

    source_games: int
    output_team_rows: int
    unique_teams: int
    seasons: int
    initial_rating: float
    mean_rating: float
    season_carryover: float
    home_court_advantage: float
    k_factor: float
    first_game_date: str
    last_game_date: str


def elo_ratings_output_path(project_root: Path) -> Path:
    """Return the team-Elo-ratings Parquet path."""

    return project_root / "data" / "processed" / "nba" / "features" / "team_elo_ratings.parquet"


def elo_ratings_summary_path(project_root: Path) -> Path:
    """Return the Elo-ratings metadata path."""

    return (
        project_root / "data" / "processed" / "nba" / "features" / "team_elo_ratings_summary.json"
    )


def combined_games_input_path(project_root: Path) -> Path:
    """Return the combined Phase 4 game-dataset path."""

    return project_root / "data" / "processed" / "nba" / "games" / "all_seasons.parquet"


def validate_games_input(games: pd.DataFrame) -> None:
    """Validate the Phase 4 dataset before simulating ratings over it."""

    missing_columns = REQUIRED_GAME_COLUMNS - set(games.columns)

    if missing_columns:
        raise ValueError(f"Game dataset is missing required columns: {sorted(missing_columns)}")

    if games.empty:
        raise ValueError("Game dataset cannot be empty")

    duplicate_games = int(games.duplicated(subset=["SEASON", "GAME_ID"]).sum())

    if duplicate_games:
        raise ValueError(f"Game dataset contains {duplicate_games} duplicate SEASON/GAME_ID rows")

    if not games["home_win"].isin([0, 1]).all():
        raise ValueError("home_win must contain only 0 and 1")

    expected_home_win = (games["HOME_PTS"] > games["AWAY_PTS"]).astype("int8")

    if not expected_home_win.equals(games["home_win"].astype("int8")):
        raise ValueError("home_win is inconsistent with HOME_PTS/AWAY_PTS")

    same_team_mask = games["HOME_TEAM_ID"].astype("int64") == games["AWAY_TEAM_ID"].astype("int64")

    if same_team_mask.any():
        raise ValueError("Game dataset contains identical home and away teams")


def carried_rating(previous_rating: float) -> float:
    """Regress a rating partway toward the mean at a season boundary."""

    return MEAN_RATING + SEASON_CARRYOVER * (previous_rating - MEAN_RATING)


def expected_home_win_probability(home_rating: float, away_rating: float) -> float:
    """Return the pregame win probability implied by two Elo ratings."""

    rating_diff = (home_rating + HOME_COURT_ADVANTAGE) - away_rating

    return 1.0 / (1.0 + 10.0 ** (-rating_diff / 400.0))


def margin_of_victory_multiplier(margin: float, winner_rating_diff: float) -> float:
    """Scale a rating update by how surprising and lopsided the result was.

    A blowout moves ratings more than a narrow win, and an upset (the
    winner was rated lower) moves ratings more than a expected win of the
    same margin. This is the standard log-margin dampening formula used in
    published NBA Elo models, adapted here with a defensive floor on the
    denominator.
    """

    denominator = max(0.001 * winner_rating_diff + 2.2, _MOV_DENOMINATOR_FLOOR)

    return math.log(margin + 1.0) * (2.2 / denominator)


def update_ratings(
    home_rating: float,
    away_rating: float,
    home_pts: float,
    away_pts: float,
    home_win: int,
) -> tuple[float, float]:
    """Return the post-game (home, away) ratings after one game.

    The update is exactly zero-sum: whatever the home rating gains, the
    away rating loses, and vice versa.
    """

    expected_home = expected_home_win_probability(home_rating, away_rating)
    actual_home = float(home_win)

    margin = abs(float(home_pts) - float(away_pts))
    effective_home_rating = home_rating + HOME_COURT_ADVANTAGE

    winner_rating_diff = (
        effective_home_rating - away_rating
        if home_win == 1
        else away_rating - effective_home_rating
    )

    multiplier = margin_of_victory_multiplier(margin, winner_rating_diff)
    delta = K_FACTOR * multiplier * (actual_home - expected_home)

    return home_rating + delta, away_rating - delta


def compute_elo_ratings(games: pd.DataFrame) -> tuple[pd.DataFrame, EloRatingSummary]:
    """Simulate Elo ratings sequentially over every game in chronological order."""

    validate_games_input(games)

    working = games.copy()
    working["GAME_DATE"] = pd.to_datetime(working["GAME_DATE"], errors="raise")
    working = working.sort_values(["GAME_DATE", "GAME_ID"], kind="stable").reset_index(drop=True)

    current_rating: dict[int, float] = {}
    last_season_seen: dict[int, str] = {}
    records: list[dict[str, object]] = []

    for row in working.itertuples(index=False):
        season = row.SEASON
        home_id = int(row.HOME_TEAM_ID)
        away_id = int(row.AWAY_TEAM_ID)

        pregame_ratings: dict[int, float] = {}

        for team_id in (home_id, away_id):
            if team_id not in current_rating:
                pregame_ratings[team_id] = INITIAL_RATING
            elif last_season_seen[team_id] != season:
                pregame_ratings[team_id] = carried_rating(current_rating[team_id])
            else:
                pregame_ratings[team_id] = current_rating[team_id]

        home_pre = pregame_ratings[home_id]
        away_pre = pregame_ratings[away_id]

        home_post, away_post = update_ratings(
            home_rating=home_pre,
            away_rating=away_pre,
            home_pts=row.HOME_PTS,
            away_pts=row.AWAY_PTS,
            home_win=int(row.home_win),
        )

        for team_id, pre_rating, post_rating in (
            (home_id, home_pre, home_post),
            (away_id, away_pre, away_post),
        ):
            records.append(
                {
                    "SEASON": season,
                    "SEASON_ID": row.SEASON_ID,
                    "GAME_ID": row.GAME_ID,
                    "GAME_DATE": row.GAME_DATE,
                    "TEAM_ID": team_id,
                    "ELO_RATING": pre_rating,
                    "POST_GAME_ELO_RATING": post_rating,
                }
            )

        current_rating[home_id] = home_post
        current_rating[away_id] = away_post
        last_season_seen[home_id] = season
        last_season_seen[away_id] = season

    elo_ratings = pd.DataFrame.from_records(records)

    validate_elo_ratings(elo_ratings, expected_game_count=len(working))

    summary = EloRatingSummary(
        source_games=len(working),
        output_team_rows=len(elo_ratings),
        unique_teams=int(elo_ratings["TEAM_ID"].nunique()),
        seasons=int(elo_ratings["SEASON"].nunique()),
        initial_rating=INITIAL_RATING,
        mean_rating=MEAN_RATING,
        season_carryover=SEASON_CARRYOVER,
        home_court_advantage=HOME_COURT_ADVANTAGE,
        k_factor=K_FACTOR,
        first_game_date=(elo_ratings["GAME_DATE"].min().date().isoformat()),
        last_game_date=(elo_ratings["GAME_DATE"].max().date().isoformat()),
    )

    return elo_ratings, summary


def validate_elo_ratings(elo_ratings: pd.DataFrame, expected_game_count: int) -> None:
    """Enforce structural invariants on the simulated rating output."""

    expected_rows = expected_game_count * 2

    if len(elo_ratings) != expected_rows:
        raise ValueError(f"Expected {expected_rows} Elo rating rows, found {len(elo_ratings)}")

    duplicate_rows = int(elo_ratings.duplicated(subset=["SEASON", "GAME_ID", "TEAM_ID"]).sum())

    if duplicate_rows:
        raise ValueError(f"Elo ratings contain {duplicate_rows} duplicate team-game rows")

    if elo_ratings[["ELO_RATING", "POST_GAME_ELO_RATING"]].isna().any().any():
        raise ValueError("Elo ratings contain missing values")

    grouped = elo_ratings.groupby(["SEASON", "GAME_ID"], sort=False)

    if not grouped.size().eq(2).all():
        raise ValueError("Every game must produce exactly two Elo rating rows")

    # The post-game delta must be exactly zero-sum within each game: whatever
    # one team's rating gained, the other's lost by the same amount.
    deltas = grouped.apply(
        lambda group: (group["POST_GAME_ELO_RATING"] - group["ELO_RATING"]).sum(),
        include_groups=False,
    )

    if not (deltas.abs() < 1e-6).all():
        raise ValueError("Elo rating updates are not zero-sum within a game")


def write_elo_ratings_outputs(
    elo_ratings: pd.DataFrame,
    summary: EloRatingSummary,
    project_root: Path,
) -> tuple[Path, Path]:
    """Write the Elo ratings Parquet and its summary."""

    ratings_path = elo_ratings_output_path(project_root)
    summary_path = elo_ratings_summary_path(project_root)

    ratings_path.parent.mkdir(parents=True, exist_ok=True)

    elo_ratings.to_parquet(ratings_path, index=False)

    summary_path.write_text(
        json.dumps(asdict(summary), indent=2) + "\n",
        encoding="utf-8",
    )

    return ratings_path, summary_path


def build_team_elo_ratings_dataset(project_root: Path) -> EloRatingSummary:
    """Read the Phase 4 dataset and write simulated Elo ratings."""

    input_path = combined_games_input_path(project_root)

    if not input_path.exists():
        raise FileNotFoundError(f"Combined game dataset does not exist: {input_path}")

    games = pd.read_parquet(input_path)

    elo_ratings, summary = compute_elo_ratings(games)

    ratings_path, summary_path = write_elo_ratings_outputs(
        elo_ratings=elo_ratings,
        summary=summary,
        project_root=project_root,
    )

    print("\nElo rating build complete:")
    print(json.dumps(asdict(summary), indent=2))
    print(f"Elo ratings: {ratings_path}")
    print(f"Summary: {summary_path}")

    return summary


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the Elo-ratings build."""

    return argparse.ArgumentParser(description=__doc__).parse_args()


def main() -> None:
    """Run the Elo rating computation from the command line."""

    parse_args()
    project_root = Path(__file__).resolve().parents[2]

    build_team_elo_ratings_dataset(project_root=project_root)


if __name__ == "__main__":
    main()
