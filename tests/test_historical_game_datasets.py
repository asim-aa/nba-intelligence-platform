"""Tests for the multi-season NBA game-dataset orchestration pipeline.

These tests replace season processing with small fake datasets. They verify
multi-season validation, chronological combination, output persistence, and
failure safety without calling the NBA API or reading real project data.
"""

from pathlib import Path

import pandas as pd
import pipelines.transform.build_historical_game_datasets as historical_module
import pytest
from pipelines.transform.build_historical_game_datasets import (
    SeasonPipelineResult,
    build_historical_game_datasets,
    combined_games_output_path,
    historical_summary_output_path,
    validate_combined_dataset,
    validate_requested_seasons,
)


def make_completed_game(
    season: str,
    game_id: str,
    game_date: str,
    home_win: int = 1,
) -> pd.DataFrame:
    """Create one completed season-labeled game row."""

    return pd.DataFrame(
        [
            {
                "SEASON": season,
                "GAME_ID": game_id,
                "GAME_DATE": pd.Timestamp(game_date),
                "SEASON_ID": f"2{season[:4]}",
                "HOME_TEAM_ID": 1610612747,
                "HOME_TEAM_ABBREVIATION": "LAL",
                "HOME_WL": "W" if home_win == 1 else "L",
                "HOME_PTS": 112 if home_win == 1 else 101,
                "AWAY_TEAM_ID": 1610612738,
                "AWAY_TEAM_ABBREVIATION": "BOS",
                "AWAY_WL": "L" if home_win == 1 else "W",
                "AWAY_PTS": 101 if home_win == 1 else 112,
                "home_win": home_win,
            }
        ]
    )


def make_season_result(
    season: str,
    game_count: int = 1,
) -> SeasonPipelineResult:
    """Create summary metadata for one fake completed season."""

    return SeasonPipelineResult(
        season=season,
        source_games=game_count,
        initially_resolved_games=game_count,
        ambiguous_games=0,
        reconciled_games=0,
        final_game_rows=game_count,
    )


def test_historical_output_paths_use_processed_games_directory(
    tmp_path: Path,
) -> None:
    """Combined data and metadata should share the processed games folder."""

    expected_directory = tmp_path / "data" / "processed" / "nba" / "games"

    assert combined_games_output_path(tmp_path) == (expected_directory / "all_seasons.parquet")

    assert historical_summary_output_path(tmp_path) == (
        expected_directory / "historical_transformation_summary.json"
    )


def test_validate_requested_seasons_accepts_valid_values() -> None:
    """Properly formatted unique NBA seasons should pass validation."""

    validate_requested_seasons(
        [
            "2023-24",
            "2024-25",
        ]
    )


def test_validate_requested_seasons_rejects_empty_list() -> None:
    """At least one season must be supplied."""

    with pytest.raises(
        ValueError,
        match="At least one season",
    ):
        validate_requested_seasons([])


def test_validate_requested_seasons_rejects_duplicates() -> None:
    """The same season should not be processed twice in one run."""

    with pytest.raises(
        ValueError,
        match="Duplicate seasons",
    ):
        validate_requested_seasons(
            [
                "2024-25",
                "2024-25",
            ]
        )


def test_validate_requested_seasons_rejects_invalid_format() -> None:
    """Season labels must follow the four-digit/two-digit convention."""

    invalid_seasons = [
        "2024",
        "24-25",
        "2024-2025",
        "season-2024",
    ]

    for invalid_season in invalid_seasons:
        with pytest.raises(
            ValueError,
            match="Invalid NBA season format",
        ):
            validate_requested_seasons([invalid_season])


def test_validate_combined_dataset_rejects_duplicate_games() -> None:
    """Duplicate SEASON and GAME_ID combinations should fail."""

    game = make_completed_game(
        season="2024-25",
        game_id="0022400001",
        game_date="2024-10-22",
    )

    games = pd.concat(
        [
            game,
            game,
        ],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match="duplicate SEASON/GAME_ID",
    ):
        validate_combined_dataset(
            games=games,
            expected_game_count=2,
        )


def test_validate_combined_dataset_rejects_invalid_target() -> None:
    """The combined home_win target must remain binary."""

    games = make_completed_game(
        season="2024-25",
        game_id="0022400001",
        game_date="2024-10-22",
    )

    games.loc[0, "home_win"] = 2

    with pytest.raises(
        ValueError,
        match="outside",
    ):
        validate_combined_dataset(
            games=games,
            expected_game_count=1,
        )


def test_build_historical_game_datasets_combines_and_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Completed seasons should be combined chronologically and persisted."""

    season_frames = {
        "2023-24": make_completed_game(
            season="2023-24",
            game_id="0022300001",
            game_date="2024-04-01",
        ),
        "2024-25": make_completed_game(
            season="2024-25",
            game_id="0022400001",
            game_date="2024-10-22",
        ),
    }

    def fake_process_season(
        *,
        season: str,
        project_root: Path,
        timeout: int,
        max_attempts: int,
    ) -> tuple[pd.DataFrame, SeasonPipelineResult]:
        """Return one fake completed season without network access."""

        assert project_root == tmp_path
        assert timeout == 15
        assert max_attempts == 1

        return (
            season_frames[season].copy(),
            make_season_result(season),
        )

    monkeypatch.setattr(
        historical_module,
        "process_season",
        fake_process_season,
    )

    summary = build_historical_game_datasets(
        seasons=[
            "2023-24",
            "2024-25",
        ],
        project_root=tmp_path,
        timeout=15,
        max_attempts=1,
    )

    combined_path = combined_games_output_path(tmp_path)
    summary_path = historical_summary_output_path(tmp_path)

    assert summary.seasons_processed == 2
    assert summary.total_game_rows == 2
    assert len(summary.season_results) == 2

    assert combined_path.exists()
    assert summary_path.exists()

    saved_games = pd.read_parquet(combined_path)

    assert len(saved_games) == 2
    assert saved_games["SEASON"].tolist() == [
        "2023-24",
        "2024-25",
    ]
    assert saved_games["GAME_ID"].tolist() == [
        "0022300001",
        "0022400001",
    ]

    metadata = summary_path.read_text(encoding="utf-8")

    assert '"seasons_processed": 2' in metadata
    assert '"total_game_rows": 2' in metadata


def test_build_historical_game_datasets_does_not_write_after_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed season should prevent misleading combined outputs."""

    def failing_process_season(
        *,
        season: str,
        project_root: Path,
        timeout: int,
        max_attempts: int,
    ) -> tuple[pd.DataFrame, SeasonPipelineResult]:
        """Complete the first season and fail during the second."""

        del project_root, timeout, max_attempts

        if season == "2023-24":
            return (
                make_completed_game(
                    season="2023-24",
                    game_id="0022300001",
                    game_date="2024-04-01",
                ),
                make_season_result(season),
            )

        raise RuntimeError("Simulated season-processing failure")

    monkeypatch.setattr(
        historical_module,
        "process_season",
        failing_process_season,
    )

    with pytest.raises(
        RuntimeError,
        match="Simulated season-processing failure",
    ):
        build_historical_game_datasets(
            seasons=[
                "2023-24",
                "2024-25",
            ],
            project_root=tmp_path,
        )

    assert not combined_games_output_path(tmp_path).exists()
    assert not historical_summary_output_path(tmp_path).exists()
