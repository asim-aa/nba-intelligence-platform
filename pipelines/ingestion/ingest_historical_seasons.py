"""Download, validate, and store multiple NBA regular seasons.

This module turns the one-season source audit into a reusable ingestion
pipeline. It uses the validation functions from ``audit_nba_source.py``
before saving each season to the raw-data layer.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

import pandas as pd

from pipelines.ingestion.audit_nba_source import (
    AuditSummary,
    download_team_game_log,
    summarize,
)

# These seasons are used when the command is run without --seasons.
# We start with five seasons so the future model has enough historical data
# while keeping the initial download manageable.
DEFAULT_SEASONS: Final[tuple[str, ...]] = (
    "2020-21",
    "2021-22",
    "2022-23",
    "2023-24",
    "2024-25",
)


@dataclass(frozen=True)
class SeasonIngestionResult:
    """Record what happened when one season was processed."""

    season: str
    status: str
    raw_path: str
    metadata_path: str
    summary: AuditSummary


def season_directory(project_root: Path, season: str) -> Path:
    """Return the partition directory used to store one NBA season."""

    return project_root / "data" / "raw" / "nba" / "league_game_log" / f"season={season}"


def raw_data_path(project_root: Path, season: str) -> Path:
    """Return the Parquet path for one season's team-game records."""

    return season_directory(project_root, season) / "team_game_log.parquet"


def metadata_path(project_root: Path, season: str) -> Path:
    """Return the JSON metadata path for one ingested season."""

    return season_directory(project_root, season) / "ingestion_metadata.json"


def write_season_outputs(
    frame: pd.DataFrame,
    summary: AuditSummary,
    project_root: Path,
) -> tuple[Path, Path]:
    """Write one validated season and its audit metadata.

    The Parquet file preserves the raw team-game rows returned by the NBA
    source. The JSON file stores the validation summary so we can inspect
    ingestion quality without reopening the full dataset.
    """

    data_path = raw_data_path(project_root, summary.season)
    summary_path = metadata_path(project_root, summary.season)

    # Create the season partition before writing either output.
    data_path.parent.mkdir(parents=True, exist_ok=True)

    frame.to_parquet(data_path, index=False)
    summary_path.write_text(
        json.dumps(asdict(summary), indent=2) + "\n",
        encoding="utf-8",
    )

    return data_path, summary_path


def ingest_season(
    season: str,
    project_root: Path,
    *,
    timeout: int = 30,
    max_attempts: int = 3,
    backoff_seconds: float = 1.0,
    force: bool = False,
) -> SeasonIngestionResult:
    """Download, validate, and store one NBA regular season.

    Existing raw files are reused unless ``force=True``. Even reused files
    are validated again so corrupted or manually modified data cannot pass
    silently into later feature-engineering stages.
    """

    data_path = raw_data_path(project_root, season)

    if data_path.exists() and not force:
        # Reusing saved data avoids unnecessary NBA API requests.
        frame = pd.read_parquet(data_path)
        status = "reused"
    else:
        frame = download_team_game_log(
            season=season,
            timeout=timeout,
            max_attempts=max_attempts,
            backoff_seconds=backoff_seconds,
        )
        status = "downloaded"

    # The same schema, duplicate, and matchup checks used in Phase 2 are
    # applied before the season is accepted into the historical dataset.
    summary = summarize(frame, season=season)

    saved_data_path, saved_metadata_path = write_season_outputs(
        frame=frame,
        summary=summary,
        project_root=project_root,
    )

    return SeasonIngestionResult(
        season=season,
        status=status,
        raw_path=str(saved_data_path),
        metadata_path=str(saved_metadata_path),
        summary=summary,
    )


def ingest_seasons(
    seasons: list[str],
    project_root: Path,
    *,
    timeout: int = 30,
    max_attempts: int = 3,
    backoff_seconds: float = 1.0,
    pause_seconds: float = 1.0,
    force: bool = False,
) -> list[SeasonIngestionResult]:
    """Process multiple seasons sequentially.

    A pause is inserted between seasons to avoid sending rapid repeated
    requests to the NBA statistics service.
    """

    if not seasons:
        raise ValueError("At least one season must be provided")
    if pause_seconds < 0:
        raise ValueError("pause_seconds cannot be negative")

    results: list[SeasonIngestionResult] = []

    for index, season in enumerate(seasons):
        print(f"\nProcessing season {season}...")

        result = ingest_season(
            season=season,
            project_root=project_root,
            timeout=timeout,
            max_attempts=max_attempts,
            backoff_seconds=backoff_seconds,
            force=force,
        )
        results.append(result)

        print(
            f"{season}: {result.status}, "
            f"{result.summary.games} games, "
            f"{result.summary.rows} team-game rows"
        )

        # Do not sleep after the final season because no request follows it.
        if index < len(seasons) - 1:
            time.sleep(pause_seconds)

    return results


def parse_args() -> argparse.Namespace:
    """Parse command-line options for historical ingestion."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seasons",
        nargs="+",
        default=list(DEFAULT_SEASONS),
        help="NBA seasons to ingest, such as 2022-23 2023-24 2024-25",
    )
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--backoff-seconds", type=float, default=1.0)
    parser.add_argument("--pause-seconds", type=float, default=1.0)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Download seasons again even when local raw files already exist",
    )
    return parser.parse_args()


def main() -> None:
    """Run the historical ingestion pipeline from the command line."""

    args = parse_args()

    if args.timeout < 1:
        raise ValueError("--timeout must be at least 1")
    if args.max_attempts < 1:
        raise ValueError("--max-attempts must be at least 1")
    if args.backoff_seconds < 0:
        raise ValueError("--backoff-seconds cannot be negative")
    if args.pause_seconds < 0:
        raise ValueError("--pause-seconds cannot be negative")

    project_root = Path(__file__).resolve().parents[2]

    results = ingest_seasons(
        seasons=args.seasons,
        project_root=project_root,
        timeout=args.timeout,
        max_attempts=args.max_attempts,
        backoff_seconds=args.backoff_seconds,
        pause_seconds=args.pause_seconds,
        force=args.force,
    )

    print("\nHistorical ingestion complete:")
    for result in results:
        print(f"- {result.season}: {result.status} ({result.summary.games} games)")


if __name__ == "__main__":
    main()
