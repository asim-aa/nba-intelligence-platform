"""Build complete game-level datasets for multiple NBA seasons.

For each requested season, this module:

1. Reads the raw team-game Parquet created by historical ingestion.
2. Converts standard games into one row per game.
3. Reconciles ambiguous home/away assignments when necessary.
4. Validates the completed season dataset.
5. Combines all completed seasons into one chronological Parquet file.

This is the orchestration layer connecting ingestion, transformation,
reconciliation, validation, and future feature engineering.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from pipelines.transform.build_game_dataset import (
    build_season_game_dataset,
    games_output_path,
)
from pipelines.transform.reconcile_home_away import (
    create_layered_resolver,
    reconcile_season,
    validate_complete_game_dataset,
)


@dataclass(frozen=True)
class SeasonPipelineResult:
    """Describe the completed transformation for one NBA season."""

    season: str
    source_games: int
    initially_resolved_games: int
    ambiguous_games: int
    reconciled_games: int
    final_game_rows: int


@dataclass(frozen=True)
class HistoricalPipelineSummary:
    """Describe a completed multi-season transformation run."""

    seasons_processed: int
    total_game_rows: int
    season_results: tuple[SeasonPipelineResult, ...]
    combined_output_path: str


def combined_games_output_path(project_root: Path) -> Path:
    """Return the path for the combined multi-season game dataset."""

    return project_root / "data" / "processed" / "nba" / "games" / "all_seasons.parquet"


def historical_summary_output_path(project_root: Path) -> Path:
    """Return the metadata path for the multi-season transformation."""

    return (
        project_root
        / "data"
        / "processed"
        / "nba"
        / "games"
        / "historical_transformation_summary.json"
    )


def validate_requested_seasons(seasons: list[str]) -> None:
    """Validate the season list before starting expensive work."""

    if not seasons:
        raise ValueError("At least one season must be provided")

    duplicate_seasons = sorted({season for season in seasons if seasons.count(season) > 1})

    if duplicate_seasons:
        raise ValueError(f"Duplicate seasons were provided: {duplicate_seasons}")

    for season in seasons:
        parts = season.split("-")

        if (
            len(parts) != 2
            or len(parts[0]) != 4
            or len(parts[1]) != 2
            or not all(part.isdigit() for part in parts)
        ):
            raise ValueError(
                f"Invalid NBA season format: {season!r}. Expected a value such as '2024-25'."
            )


def load_completed_season(
    project_root: Path,
    season: str,
    expected_game_count: int,
) -> pd.DataFrame:
    """Load and validate one completed season-level game dataset."""

    path = games_output_path(
        project_root=project_root,
        season=season,
    )

    if not path.exists():
        raise FileNotFoundError(f"Completed season dataset does not exist: {path}")

    games = pd.read_parquet(path)

    validate_complete_game_dataset(
        games=games,
        expected_game_count=expected_game_count,
    )

    games = games.copy()

    # The readable season label is added to the combined dataset. SEASON_ID
    # remains available as the NBA source's internal season identifier.
    games.insert(
        0,
        "SEASON",
        season,
    )

    games["GAME_DATE"] = pd.to_datetime(
        games["GAME_DATE"],
        errors="raise",
    )

    return games


def process_season(
    season: str,
    project_root: Path,
    timeout: int,
    max_attempts: int,
) -> tuple[pd.DataFrame, SeasonPipelineResult]:
    """Run transformation and reconciliation for one NBA season."""

    print(f"\nProcessing season {season}...")

    transformation_summary = build_season_game_dataset(
        season=season,
        project_root=project_root,
    )

    reconciled_games = 0

    if transformation_summary.unresolved_games:
        print(f"{season}: reconciling {transformation_summary.unresolved_games} ambiguous games")

        resolver = create_layered_resolver(
            season=season,
            timeout=timeout,
            max_attempts=max_attempts,
        )

        reconciliation_summary = reconcile_season(
            season=season,
            project_root=project_root,
            resolver=resolver,
        )

        reconciled_games = reconciliation_summary.reconciled_games
        final_game_rows = reconciliation_summary.final_game_rows

    else:
        # Seasons without ambiguous source games already contain their full
        # game count after the standard transformation.
        final_game_rows = transformation_summary.output_game_rows

    completed_games = load_completed_season(
        project_root=project_root,
        season=season,
        expected_game_count=transformation_summary.source_games,
    )

    result = SeasonPipelineResult(
        season=season,
        source_games=transformation_summary.source_games,
        initially_resolved_games=(transformation_summary.resolved_games),
        ambiguous_games=transformation_summary.unresolved_games,
        reconciled_games=reconciled_games,
        final_game_rows=final_game_rows,
    )

    print(
        f"{season}: complete "
        f"({result.final_game_rows} game rows, "
        f"{result.reconciled_games} reconciled)"
    )

    return completed_games, result


def validate_combined_dataset(
    games: pd.DataFrame,
    expected_game_count: int,
) -> None:
    """Validate invariants specific to the combined historical dataset."""

    required_columns = {
        "SEASON",
        "GAME_ID",
        "GAME_DATE",
        "HOME_TEAM_ID",
        "AWAY_TEAM_ID",
        "home_win",
    }

    missing_columns = required_columns - set(games.columns)

    if missing_columns:
        raise ValueError(f"Combined dataset is missing required columns: {sorted(missing_columns)}")

    if len(games) != expected_game_count:
        raise ValueError(f"Expected {expected_game_count} combined game rows, found {len(games)}")

    duplicate_rows = int(games.duplicated(subset=["SEASON", "GAME_ID"]).sum())

    if duplicate_rows:
        raise ValueError(
            f"Combined dataset contains {duplicate_rows} duplicate SEASON/GAME_ID rows"
        )

    if games["SEASON"].isna().any():
        raise ValueError("Combined dataset contains missing SEASON values")

    if games["GAME_DATE"].isna().any():
        raise ValueError("Combined dataset contains missing GAME_DATE values")

    invalid_targets = ~games["home_win"].isin([0, 1])

    if invalid_targets.any():
        raise ValueError("Combined dataset contains home_win values outside {0, 1}")


def build_historical_game_datasets(
    seasons: list[str],
    project_root: Path,
    timeout: int = 30,
    max_attempts: int = 3,
) -> HistoricalPipelineSummary:
    """Build, reconcile, validate, and combine multiple NBA seasons."""

    validate_requested_seasons(seasons)

    season_frames: list[pd.DataFrame] = []
    season_results: list[SeasonPipelineResult] = []

    for season in seasons:
        completed_games, result = process_season(
            season=season,
            project_root=project_root,
            timeout=timeout,
            max_attempts=max_attempts,
        )

        season_frames.append(completed_games)
        season_results.append(result)

    combined_games = pd.concat(
        season_frames,
        ignore_index=True,
    )

    combined_games = combined_games.sort_values(
        ["GAME_DATE", "GAME_ID"],
        kind="stable",
    ).reset_index(drop=True)

    expected_game_count = sum(result.final_game_rows for result in season_results)

    validate_combined_dataset(
        games=combined_games,
        expected_game_count=expected_game_count,
    )

    combined_path = combined_games_output_path(project_root)
    summary_path = historical_summary_output_path(project_root)

    combined_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Outputs are written only after all requested seasons have completed and
    # the final combined dataset has passed its invariants.
    combined_games.to_parquet(
        combined_path,
        index=False,
    )

    summary = HistoricalPipelineSummary(
        seasons_processed=len(season_results),
        total_game_rows=len(combined_games),
        season_results=tuple(season_results),
        combined_output_path=str(combined_path),
    )

    summary_path.write_text(
        json.dumps(
            asdict(summary),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print("\nHistorical game-dataset build complete:")
    print(f"- Seasons processed: {summary.seasons_processed}")
    print(f"- Total game rows: {summary.total_game_rows}")
    print(f"- Combined dataset: {combined_path}")
    print(f"- Summary: {summary_path}")

    return summary


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the historical runner."""

    parser = argparse.ArgumentParser(
        description=__doc__,
    )

    parser.add_argument(
        "--seasons",
        nargs="+",
        required=True,
        help=("NBA seasons to process, such as 2023-24 2024-25"),
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Seconds allowed for each NBA request",
    )

    parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="Maximum attempts for each NBA data request",
    )

    return parser.parse_args()


def main() -> None:
    """Run the historical game-dataset pipeline from the CLI."""

    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]

    build_historical_game_datasets(
        seasons=args.seasons,
        project_root=project_root,
        timeout=args.timeout,
        max_attempts=args.max_attempts,
    )


if __name__ == "__main__":
    main()
