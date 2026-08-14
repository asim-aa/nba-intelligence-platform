"""Download, validate, and store multiple NBA seasons of player game logs.

Mirrors ingest_historical_seasons.py's structure exactly, one level down:
that module ingests one row per team per game (LeagueGameLog in team mode);
this one ingests one row per player per game (LeagueGameLog in player mode)
for the roster-continuity feature in build_roster_continuity_features.py.
A player with no row for a game did not appear in it.
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
    PlayerLogAuditSummary,
    download_player_game_log,
    summarize_player_log,
)

# Same target acquisition window as ingest_historical_seasons.py.
DEFAULT_SEASONS: Final[tuple[str, ...]] = (
    "2015-16",
    "2016-17",
    "2017-18",
    "2018-19",
    "2019-20",
    "2020-21",
    "2021-22",
    "2022-23",
    "2023-24",
    "2024-25",
    "2025-26",
)


@dataclass(frozen=True)
class PlayerLogIngestionResult:
    """Record what happened when one season's player log was processed."""

    season: str
    status: str
    raw_path: str
    metadata_path: str
    summary: PlayerLogAuditSummary


def season_directory(project_root: Path, season: str) -> Path:
    """Return the same partition directory the team log ingestion uses."""

    return project_root / "data" / "raw" / "nba" / "league_game_log" / f"season={season}"


def raw_data_path(project_root: Path, season: str) -> Path:
    """Return the Parquet path for one season's player-game records."""

    return season_directory(project_root, season) / "player_game_log.parquet"


def metadata_path(project_root: Path, season: str) -> Path:
    """Return the JSON metadata path for one ingested season's player log."""

    return season_directory(project_root, season) / "player_log_ingestion_metadata.json"


def write_season_outputs(
    frame: pd.DataFrame,
    summary: PlayerLogAuditSummary,
    project_root: Path,
) -> tuple[Path, Path]:
    """Write one validated season's player log and its audit metadata."""

    data_path = raw_data_path(project_root, summary.season)
    summary_path = metadata_path(project_root, summary.season)

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
) -> PlayerLogIngestionResult:
    """Download, validate, and store one season's player game log.

    Existing raw files are reused unless force=True, matching
    ingest_historical_seasons.py's caching behavior.
    """

    data_path = raw_data_path(project_root, season)

    if data_path.exists() and not force:
        frame = pd.read_parquet(data_path)
        status = "reused"
    else:
        frame = download_player_game_log(
            season=season,
            timeout=timeout,
            max_attempts=max_attempts,
            backoff_seconds=backoff_seconds,
        )
        status = "downloaded"

    summary = summarize_player_log(frame, season=season)

    saved_data_path, saved_metadata_path = write_season_outputs(
        frame=frame,
        summary=summary,
        project_root=project_root,
    )

    return PlayerLogIngestionResult(
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
) -> list[PlayerLogIngestionResult]:
    """Process multiple seasons sequentially, pausing between requests."""

    if not seasons:
        raise ValueError("At least one season must be provided")
    if pause_seconds < 0:
        raise ValueError("pause_seconds cannot be negative")

    results: list[PlayerLogIngestionResult] = []

    for index, season in enumerate(seasons):
        print(f"\nProcessing player log for season {season}...")

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
            f"{result.summary.players} players, "
            f"{result.summary.rows} player-game rows"
        )

        if index < len(seasons) - 1:
            time.sleep(pause_seconds)

    return results


def parse_args() -> argparse.Namespace:
    """Parse command-line options for player-log ingestion."""

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
    """Run the player-log ingestion pipeline from the command line."""

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

    print("\nPlayer-log ingestion complete:")
    for result in results:
        print(f"- {result.season}: {result.status} ({result.summary.games} games)")


if __name__ == "__main__":
    main()
