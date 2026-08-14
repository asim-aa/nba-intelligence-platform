"""Refresh the live season's feature data so serving doesn't go stale.

This is NOT retraining. The frozen model artifact is never touched here --
retraining is a deliberate, validated, rare decision (see docs/roadmap.md's
v1.1 note on how the roster-continuity feature was adopted), not something
a routine data refresh should ever trigger on its own.

What actually goes stale without this: the dashboard/API's "computed"
(upcoming-matchup) serving path reads team_history.parquet,
team_elo_ratings.parquet, and team_roster_continuity.parquet to compute
each team's *current* state. Those files are static snapshots from
whenever the pipeline last ran -- if games have been played since, an
upcoming prediction would use each team's state as of the last refresh,
not as of today, without any error or warning.

The past historical seasons (2015-16 through 2025-26) never change once
ingested, so re-ingesting them here would just waste time and network
calls -- ingest_season()'s existing cache-reuse behavior already handles
that correctly. What was missing is that nothing ever called it with
force=True for the one season that DOES change during the year: today's.

Steps, in order:
1. Force re-ingest only the current season's team and player game logs
   (past seasons stay cached, as intended).
2. Rebuild the combined game-level dataset across every season -- fast,
   local reprocessing of already-ingested data, not a network operation
   itself (only ambiguous-matchup reconciliation, if any, hits the network,
   the same as it already does in the historical pipeline).
3. Rebuild the full Phase 5 feature pipeline (team history, Elo ratings,
   roster continuity, pregame features, modeling dataset) -- also fast and
   local, identical to run_feature_pipeline.py.

Deliberately stops before modeling/data/split_dataset.py: the train/
validation/test season boundaries are fixed by project_spec.md's
provisional split and do not include the live season, and this refresh
must never retrain or re-select a model.

Run with: uv run python -m pipelines.refresh_current_season
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pandas as pd

from pipelines.features.run_feature_pipeline import (
    FeaturePipelineSummary,
    run_feature_pipeline,
)
from pipelines.ingestion.fetch_schedule import season_for_date
from pipelines.ingestion.ingest_historical_seasons import DEFAULT_SEASONS
from pipelines.ingestion.ingest_historical_seasons import ingest_season as ingest_team_season
from pipelines.ingestion.ingest_player_game_logs import ingest_season as ingest_player_season
from pipelines.transform.build_historical_game_datasets import (
    HistoricalPipelineSummary,
    build_historical_game_datasets,
)

# The exact message audit_nba_source.py's validate_schema() raises for a
# season with zero rows -- the normal, expected state before that season's
# first game has been played. Matched precisely (not "any ValueError")
# so a genuine data-integrity error, like a duplicate-row check failing,
# still propagates instead of being silently treated as a skip.
NO_GAMES_YET_MESSAGE: Final[str] = "NBA source returned an empty table"


@dataclass(frozen=True)
class SeasonRefreshSummary:
    """Describe one live-season refresh run."""

    current_season: str
    seasons_processed: int
    total_game_rows: int
    feature_pipeline_summary: FeaturePipelineSummary | None
    skipped_reason: str | None = None


def seasons_to_process(current_season: str) -> tuple[str, ...]:
    """Return the historical seasons plus the current one, deduplicated
    and chronologically sorted (season labels sort correctly as strings).
    """

    combined = {*DEFAULT_SEASONS, current_season}

    return tuple(sorted(combined))


def refresh_current_season(
    project_root: Path,
    as_of_date: pd.Timestamp,
    *,
    timeout: int = 30,
    max_attempts: int = 3,
) -> SeasonRefreshSummary:
    """Force-refresh the live season's raw data and rebuild all Phase 5
    features from it. Never touches the frozen model or the chronological
    splits.

    Before the season's first game, LeagueGameLog legitimately returns no
    rows for it yet -- audit_nba_source.py's schema validation rejects
    that as a ValueError, same as it would for a genuinely malformed
    response. In this one caller, that specific error means "nothing to
    refresh yet," not a data-integrity problem, so it is caught here and
    reported as a skip rather than left to crash the whole refresh.
    """

    current_season = season_for_date(as_of_date)

    print(f"Refreshing live season {current_season} (as of {as_of_date.date()})...")

    print(f"\nForce re-ingesting {current_season} team game log...")

    try:
        ingest_team_season(
            season=current_season,
            project_root=project_root,
            timeout=timeout,
            max_attempts=max_attempts,
            force=True,
        )
    except ValueError as error:
        if str(error) != NO_GAMES_YET_MESSAGE:
            raise

        reason = f"{current_season} has no completed games yet ({error})"
        print(f"\n{reason} -- nothing to refresh.")

        return SeasonRefreshSummary(
            current_season=current_season,
            seasons_processed=0,
            total_game_rows=0,
            feature_pipeline_summary=None,
            skipped_reason=reason,
        )

    print(f"\nForce re-ingesting {current_season} player game log...")

    ingest_player_season(
        season=current_season,
        project_root=project_root,
        timeout=timeout,
        max_attempts=max_attempts,
        force=True,
    )

    seasons = seasons_to_process(current_season)

    print(f"\nRebuilding the combined game-level dataset ({len(seasons)} seasons)...")

    historical_summary: HistoricalPipelineSummary = build_historical_game_datasets(
        seasons=list(seasons),
        project_root=project_root,
        timeout=timeout,
        max_attempts=max_attempts,
    )

    print("\nRebuilding the Phase 5 feature pipeline...")

    feature_summary = run_feature_pipeline(project_root=project_root)

    summary = SeasonRefreshSummary(
        current_season=current_season,
        seasons_processed=historical_summary.seasons_processed,
        total_game_rows=historical_summary.total_game_rows,
        feature_pipeline_summary=feature_summary,
    )

    print("\nLive-season refresh complete:")
    print(f"- Current season: {summary.current_season}")
    print(f"- Seasons processed: {summary.seasons_processed}")
    print(f"- Total game rows: {summary.total_game_rows}")
    print(f"- Modeling rows: {summary.feature_pipeline_summary.modeling_rows}")

    return summary


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the live-season refresh."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--as-of-date",
        default=None,
        help="Override today's date for season resolution, YYYY-MM-DD (mainly for testing).",
    )
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--max-attempts", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    """Run the live-season refresh from the command line."""

    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent

    as_of_date = pd.Timestamp(args.as_of_date) if args.as_of_date else pd.Timestamp.now()

    refresh_current_season(
        project_root=project_root,
        as_of_date=as_of_date,
        timeout=args.timeout,
        max_attempts=args.max_attempts,
    )


if __name__ == "__main__":
    main()
