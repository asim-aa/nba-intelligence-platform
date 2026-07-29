"""Tests for the historical NBA ingestion pipeline.

These tests use temporary files and small in-memory DataFrames so they do
not call the live NBA API or write into the project's real data directory.
"""

from pathlib import Path

import pandas as pd
import pytest
from pipelines.ingestion.audit_nba_source import summarize
from pipelines.ingestion.ingest_historical_seasons import (
    ingest_season,
    ingest_seasons,
    metadata_path,
    raw_data_path,
    write_season_outputs,
)


def make_valid_frame() -> pd.DataFrame:
    """Build a minimal valid two-team NBA game for ingestion tests."""

    shared = {
        "GAME_ID": "0022400001",
        "GAME_DATE": "2024-10-22",
        "SEASON_ID": "22024",
        "WL": "W",
        "PTS": 110,
        "FG_PCT": 0.50,
        "FG3_PCT": 0.40,
        "FT_PCT": 0.80,
        "REB": 45,
        "AST": 25,
        "TOV": 12,
        "STL": 8,
        "BLK": 5,
    }

    return pd.DataFrame(
        [
            {
                **shared,
                "TEAM_ID": 1,
                "TEAM_ABBREVIATION": "AAA",
                "MATCHUP": "AAA vs. BBB",
            },
            {
                **shared,
                "TEAM_ID": 2,
                "TEAM_ABBREVIATION": "BBB",
                "MATCHUP": "BBB @ AAA",
                "WL": "L",
                "PTS": 101,
            },
        ]
    )


def test_raw_data_path_uses_season_partition(tmp_path: Path) -> None:
    """Raw files should be stored inside a season-specific directory."""

    path = raw_data_path(tmp_path, "2024-25")

    assert path == (
        tmp_path
        / "data"
        / "raw"
        / "nba"
        / "league_game_log"
        / "season=2024-25"
        / "team_game_log.parquet"
    )


def test_metadata_path_uses_same_season_partition(tmp_path: Path) -> None:
    """Metadata should live beside the raw Parquet file it describes."""

    path = metadata_path(tmp_path, "2024-25")

    assert path == (
        tmp_path
        / "data"
        / "raw"
        / "nba"
        / "league_game_log"
        / "season=2024-25"
        / "ingestion_metadata.json"
    )


def test_write_season_outputs_creates_data_and_metadata(tmp_path: Path) -> None:
    """Validated season outputs should be written successfully."""

    frame = make_valid_frame()
    summary = summarize(frame, season="2024-25")

    data_path, summary_path = write_season_outputs(
        frame=frame,
        summary=summary,
        project_root=tmp_path,
    )

    assert data_path.exists()
    assert summary_path.exists()

    saved_frame = pd.read_parquet(data_path)

    assert len(saved_frame) == 2
    assert saved_frame["GAME_ID"].nunique() == 1
    assert '"games": 1' in summary_path.read_text(encoding="utf-8")


def test_ingest_season_reuses_existing_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An existing valid file should be reused instead of downloaded."""

    frame = make_valid_frame()
    existing_path = raw_data_path(tmp_path, "2024-25")
    existing_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(existing_path, index=False)

    def fail_if_downloaded(*args: object, **kwargs: object) -> pd.DataFrame:
        raise AssertionError("The downloader should not be called")

    # Replace the downloader inside the ingestion module for this test.
    monkeypatch.setattr(
        "pipelines.ingestion.ingest_historical_seasons.download_team_game_log",
        fail_if_downloaded,
    )

    result = ingest_season(
        season="2024-25",
        project_root=tmp_path,
    )

    assert result.status == "reused"
    assert result.summary.games == 1
    assert result.summary.rows == 2


def test_ingest_season_downloads_when_file_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing season should call the downloader and save the result."""

    frame = make_valid_frame()

    def fake_download(*args: object, **kwargs: object) -> pd.DataFrame:
        return frame

    monkeypatch.setattr(
        "pipelines.ingestion.ingest_historical_seasons.download_team_game_log",
        fake_download,
    )

    result = ingest_season(
        season="2024-25",
        project_root=tmp_path,
    )

    assert result.status == "downloaded"
    assert Path(result.raw_path).exists()
    assert Path(result.metadata_path).exists()


def test_ingest_seasons_requires_at_least_one_season(tmp_path: Path) -> None:
    """The multi-season runner should reject an empty season list."""

    with pytest.raises(ValueError, match="At least one season"):
        ingest_seasons(
            seasons=[],
            project_root=tmp_path,
        )
