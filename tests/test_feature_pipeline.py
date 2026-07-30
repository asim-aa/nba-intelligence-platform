"""Tests for the complete Phase 5 feature-pipeline orchestration."""

from pathlib import Path
from types import SimpleNamespace

import pipelines.features.run_feature_pipeline as pipeline_module
import pytest
from pipelines.features.build_modeling_dataset import (
    modeling_dataset_output_path,
)
from pipelines.features.run_feature_pipeline import (
    feature_pipeline_summary_path,
    run_feature_pipeline,
)


def test_feature_pipeline_summary_path_uses_features_directory(
    tmp_path: Path,
) -> None:
    """The orchestration summary should live beside Phase 5 metadata."""

    assert feature_pipeline_summary_path(tmp_path) == (
        tmp_path / "data" / "processed" / "nba" / "features" / "feature_pipeline_summary.json"
    )


def test_run_feature_pipeline_writes_summary_after_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Successful stages should produce one final pipeline summary."""

    team_history_summary = SimpleNamespace(
        output_team_rows=4,
        seasons=1,
    )

    pregame_summary = SimpleNamespace(
        source_team_rows=4,
        output_feature_rows=4,
        seasons=1,
    )

    modeling_summary = SimpleNamespace(
        source_game_rows=2,
        source_team_feature_rows=4,
        output_model_rows=2,
        seasons=1,
        numeric_feature_count=41,
        categorical_feature_count=2,
    )

    monkeypatch.setattr(
        pipeline_module,
        "build_team_history_dataset",
        lambda project_root: team_history_summary,
    )

    monkeypatch.setattr(
        pipeline_module,
        "build_pregame_team_feature_dataset",
        lambda project_root: pregame_summary,
    )

    monkeypatch.setattr(
        pipeline_module,
        "build_modeling_dataset_from_files",
        lambda project_root: modeling_summary,
    )

    final_path = modeling_dataset_output_path(tmp_path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    final_path.write_bytes(b"test-output")

    summary = run_feature_pipeline(project_root=tmp_path)

    assert summary.source_game_rows == 2
    assert summary.team_history_rows == 4
    assert summary.pregame_team_feature_rows == 4
    assert summary.modeling_rows == 2
    assert summary.numeric_feature_count == 41
    assert summary.categorical_feature_count == 2

    summary_path = feature_pipeline_summary_path(tmp_path)

    assert summary_path.exists()
    assert '"modeling_rows": 2' in summary_path.read_text(encoding="utf-8")


def test_run_feature_pipeline_does_not_write_summary_after_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed feature stage must not create success metadata."""

    team_history_summary = SimpleNamespace(
        output_team_rows=4,
        seasons=1,
    )

    pregame_summary = SimpleNamespace(
        source_team_rows=4,
        output_feature_rows=4,
        seasons=1,
    )

    monkeypatch.setattr(
        pipeline_module,
        "build_team_history_dataset",
        lambda project_root: team_history_summary,
    )

    monkeypatch.setattr(
        pipeline_module,
        "build_pregame_team_feature_dataset",
        lambda project_root: pregame_summary,
    )

    def fail_modeling_build(project_root: Path) -> None:
        """Simulate a failure in the final feature stage."""

        del project_root
        raise RuntimeError("Simulated modeling-dataset failure")

    monkeypatch.setattr(
        pipeline_module,
        "build_modeling_dataset_from_files",
        fail_modeling_build,
    )

    with pytest.raises(
        RuntimeError,
        match="Simulated modeling-dataset failure",
    ):
        run_feature_pipeline(project_root=tmp_path)

    assert not feature_pipeline_summary_path(tmp_path).exists()
