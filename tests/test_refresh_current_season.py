"""Tests for the live-season data refresh orchestrator."""

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pipelines.refresh_current_season as refresh_module
import pytest
from pipelines.refresh_current_season import (
    refresh_current_season,
    seasons_to_process,
)


def test_seasons_to_process_includes_current_season_once() -> None:
    seasons = seasons_to_process("2026-27")

    assert "2026-27" in seasons
    assert seasons.count("2026-27") == 1
    assert seasons == tuple(sorted(seasons))


def test_seasons_to_process_does_not_duplicate_an_already_known_season() -> None:
    """A current season already in the historical baseline (e.g. running
    this mid-way through an already-ingested season) must not duplicate.
    """

    seasons = seasons_to_process("2024-25")

    assert seasons.count("2024-25") == 1


def test_refresh_current_season_force_refreshes_only_the_current_season(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The current season's ingestion must be forced; the underlying
    historical-transform and feature-pipeline stages must run once each.
    """

    calls: dict[str, object] = {}

    def fake_ingest_team_season(season, project_root, timeout, max_attempts, force):
        calls["team_season"] = season
        calls["team_force"] = force
        return SimpleNamespace(status="downloaded")

    def fake_ingest_player_season(season, project_root, timeout, max_attempts, force):
        calls["player_season"] = season
        calls["player_force"] = force
        return SimpleNamespace(status="downloaded")

    def fake_build_historical_game_datasets(seasons, project_root, timeout, max_attempts):
        calls["historical_seasons"] = tuple(seasons)
        return SimpleNamespace(seasons_processed=len(seasons), total_game_rows=100)

    def fake_run_feature_pipeline(project_root):
        return SimpleNamespace(modeling_rows=50)

    monkeypatch.setattr(refresh_module, "ingest_team_season", fake_ingest_team_season)
    monkeypatch.setattr(refresh_module, "ingest_player_season", fake_ingest_player_season)
    monkeypatch.setattr(
        refresh_module, "build_historical_game_datasets", fake_build_historical_game_datasets
    )
    monkeypatch.setattr(refresh_module, "run_feature_pipeline", fake_run_feature_pipeline)

    summary = refresh_current_season(
        project_root=tmp_path,
        as_of_date=pd.Timestamp("2026-11-15"),
    )

    assert calls["team_season"] == "2026-27"
    assert calls["team_force"] is True
    assert calls["player_season"] == "2026-27"
    assert calls["player_force"] is True
    assert "2026-27" in calls["historical_seasons"]
    assert "2025-26" in calls["historical_seasons"]

    assert summary.current_season == "2026-27"
    assert summary.total_game_rows == 100
    assert summary.feature_pipeline_summary.modeling_rows == 50


def test_refresh_current_season_skips_gracefully_before_season_starts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Before the season's first game, LeagueGameLog legitimately returns
    no rows, which audit_nba_source.py's schema validation rejects as a
    ValueError. This caller must treat that as a clean skip, not a crash,
    and must not touch the transform or feature stages at all.
    """

    downstream_called = {"value": False}

    def fake_ingest_team_season(**kwargs):
        del kwargs
        raise ValueError("NBA source returned an empty table")

    def fail_if_called(**kwargs):
        del kwargs
        downstream_called["value"] = True
        raise AssertionError("downstream stage must not run when ingestion is skipped")

    monkeypatch.setattr(refresh_module, "ingest_team_season", fake_ingest_team_season)
    monkeypatch.setattr(refresh_module, "ingest_player_season", fail_if_called)
    monkeypatch.setattr(refresh_module, "build_historical_game_datasets", fail_if_called)
    monkeypatch.setattr(refresh_module, "run_feature_pipeline", fail_if_called)

    summary = refresh_current_season(
        project_root=tmp_path,
        as_of_date=pd.Timestamp("2026-08-13"),
    )

    assert summary.current_season == "2026-27"
    assert summary.seasons_processed == 0
    assert summary.feature_pipeline_summary is None
    assert summary.skipped_reason is not None
    assert "2026-27" in summary.skipped_reason
    assert not downstream_called["value"]


def test_refresh_current_season_reraises_other_value_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A different ValueError (a real data problem, not "season hasn't
    started") must not be silently swallowed as a skip.
    """

    def fake_ingest_team_season(**kwargs):
        del kwargs
        raise ValueError("Found 3 duplicate GAME_ID/TEAM_ID rows")

    monkeypatch.setattr(refresh_module, "ingest_team_season", fake_ingest_team_season)

    with pytest.raises(ValueError, match="duplicate"):
        refresh_current_season(project_root=tmp_path, as_of_date=pd.Timestamp("2026-11-15"))


def test_refresh_current_season_module_never_imports_splits_or_training() -> None:
    """A refresh must never import split_dataset or a training module --
    retraining stays a separate, deliberate step (see docs/roadmap.md's
    v1.1 note). Checked directly against the module's own import list
    rather than via monkeypatching, since an accidental import would
    succeed silently in a test that only replaces attributes afterward.
    """

    module_globals = vars(refresh_module)
    forbidden_substrings = ("split_dataset", "train_logistic_regression", "train_catboost")

    imported_names = [
        name
        for name, value in module_globals.items()
        if not name.startswith("__") and hasattr(value, "__module__")
    ]

    for name in imported_names:
        value = module_globals[name]
        origin = getattr(value, "__module__", "") or ""

        for forbidden in forbidden_substrings:
            assert forbidden not in origin, f"{name} originates from forbidden module {origin}"
