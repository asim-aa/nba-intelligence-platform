"""Download and validate a small NBA LeagueGameLog sample.

This module is intentionally limited to a source audit. The full historical ingestion pipeline is
implemented in a later phase.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

import pandas as pd

REQUIRED_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "GAME_ID",
        "GAME_DATE",
        "SEASON_ID",
        "TEAM_ID",
        "TEAM_ABBREVIATION",
        "MATCHUP",
        "WL",
        "PTS",
        "FG_PCT",
        "FG3_PCT",
        "FT_PCT",
        "REB",
        "AST",
        "TOV",
        "STL",
        "BLK",
    }
)


@dataclass(frozen=True)
class AuditSummary:
    """Compact validation summary written after a successful audit."""

    season: str
    rows: int
    games: int
    home_rows: int
    away_rows: int
    duplicate_team_game_rows: int
    invalid_game_pair_count: int


def classify_location(matchup: str) -> str:
    """Return ``home`` or ``away`` from an NBA matchup string."""
    if " vs. " in matchup:
        return "home"
    if " @ " in matchup:
        return "away"
    raise ValueError(f"Unrecognized MATCHUP format: {matchup!r}")


def validate_schema(frame: pd.DataFrame) -> None:
    """Raise when the source response is empty or misses required columns."""
    if frame.empty:
        raise ValueError("NBA source returned an empty table")

    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"NBA source is missing required columns: {missing_text}")


def summarize(frame: pd.DataFrame, season: str) -> AuditSummary:
    """Validate team-game uniqueness and home/away game pairing."""
    validate_schema(frame)

    audited = frame.copy()
    audited["LOCATION"] = audited["MATCHUP"].map(classify_location)

    duplicate_count = int(audited.duplicated(subset=["GAME_ID", "TEAM_ID"]).sum())
    if duplicate_count:
        raise ValueError(f"Found {duplicate_count} duplicate GAME_ID/TEAM_ID rows")

    grouped = audited.groupby("GAME_ID", sort=False)
    row_counts = grouped.size()
    home_counts = grouped["LOCATION"].apply(lambda values: int((values == "home").sum()))
    away_counts = grouped["LOCATION"].apply(lambda values: int((values == "away").sum()))

    invalid_mask = (row_counts != 2) | (home_counts != 1) | (away_counts != 1)
    invalid_pair_count = int(invalid_mask.sum())
    if invalid_pair_count:
        raise ValueError(
            f"Found {invalid_pair_count} games without exactly one home and one away row"
        )

    return AuditSummary(
        season=season,
        rows=len(audited),
        games=int(audited["GAME_ID"].nunique()),
        home_rows=int((audited["LOCATION"] == "home").sum()),
        away_rows=int((audited["LOCATION"] == "away").sum()),
        duplicate_team_game_rows=duplicate_count,
        invalid_game_pair_count=invalid_pair_count,
    )


def download_team_game_log(
    season: str,
    timeout: int = 30,
    max_attempts: int = 3,
    backoff_seconds: float = 1.0,
) -> pd.DataFrame:
    """Fetch regular-season team game logs with bounded exponential-backoff retries."""
    from nba_api.stats.endpoints import leaguegamelog

    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    if backoff_seconds < 0:
        raise ValueError("backoff_seconds cannot be negative")

    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            endpoint = leaguegamelog.LeagueGameLog(
                counter=0,
                direction="DESC",
                league_id="00",
                player_or_team_abbreviation="T",
                season=season,
                season_type_all_star="Regular Season",
                sorter="DATE",
                timeout=timeout,
            )
            frames = endpoint.get_data_frames()
            if not frames:
                raise ValueError("LeagueGameLog returned no result sets")
            return frames[0]
        except Exception as error:
            last_error = error
            if attempt == max_attempts:
                break
            time.sleep(backoff_seconds * (2 ** (attempt - 1)))

    raise RuntimeError(
        f"LeagueGameLog failed after {max_attempts} attempts for season {season}"
    ) from last_error


def write_outputs(
    frame: pd.DataFrame,
    summary: AuditSummary,
    sample_rows: int,
    project_root: Path,
) -> tuple[Path, Path, Path]:
    """Write the raw audit data, a small sample, and runtime metadata."""
    raw_path = (
        project_root
        / "data"
        / "raw"
        / "nba"
        / "league_game_log"
        / f"season={summary.season}"
        / "team_game_log.parquet"
    )
    sample_path = project_root / "data" / "samples" / f"team_game_log_{summary.season}.parquet"
    metadata_path = project_root / "docs" / "data_source_audit_runtime.json"

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    frame.to_parquet(raw_path, index=False)
    frame.head(sample_rows).to_parquet(sample_path, index=False)
    metadata_path.write_text(json.dumps(asdict(summary), indent=2) + "\n", encoding="utf-8")

    return raw_path, sample_path, metadata_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", default="2024-25")
    parser.add_argument("--sample-rows", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--backoff-seconds", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sample_rows < 1:
        raise ValueError("--sample-rows must be at least 1")
    if args.timeout < 1:
        raise ValueError("--timeout must be at least 1")

    project_root = Path(__file__).resolve().parents[2]
    frame = download_team_game_log(
        season=args.season,
        timeout=args.timeout,
        max_attempts=args.max_attempts,
        backoff_seconds=args.backoff_seconds,
    )
    summary = summarize(frame, season=args.season)
    raw_path, sample_path, metadata_path = write_outputs(
        frame=frame,
        summary=summary,
        sample_rows=args.sample_rows,
        project_root=project_root,
    )

    print(json.dumps(asdict(summary), indent=2))
    print(f"Raw data: {raw_path}")
    print(f"Sample: {sample_path}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
