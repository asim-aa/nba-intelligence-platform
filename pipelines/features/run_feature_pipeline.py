"""Run the complete leakage-safe NBA feature engineering pipeline.

This orchestration module rebuilds every Phase 5 artifact in dependency order:

1. Convert game rows into chronological team history.
2. Simulate cross-season Elo ratings (independent of team history; both
   read the Phase 4 game dataset directly).
3. Build shifted, leakage-safe pregame team features.
4. Join home, away, and Elo features into one modeling row per game.

The final output is ready for chronological splitting and baseline modeling.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from pipelines.features.build_modeling_dataset import (
    ModelingDatasetSummary,
    build_modeling_dataset_from_files,
    modeling_dataset_output_path,
)
from pipelines.features.build_pregame_team_features import (
    PregameFeatureSummary,
    build_pregame_team_feature_dataset,
)
from pipelines.features.build_team_elo_ratings import (
    EloRatingSummary,
    build_team_elo_ratings_dataset,
)
from pipelines.features.build_team_history import (
    TeamHistorySummary,
    build_team_history_dataset,
)


@dataclass(frozen=True)
class FeaturePipelineSummary:
    """Describe one complete Phase 5 pipeline execution."""

    source_game_rows: int
    team_history_rows: int
    elo_rating_rows: int
    pregame_team_feature_rows: int
    modeling_rows: int
    seasons: int
    numeric_feature_count: int
    categorical_feature_count: int
    final_output_path: str


def feature_pipeline_summary_path(project_root: Path) -> Path:
    """Return the Phase 5 orchestration summary path."""

    return (
        project_root / "data" / "processed" / "nba" / "features" / "feature_pipeline_summary.json"
    )


def validate_stage_counts(
    team_history_summary: TeamHistorySummary,
    elo_summary: EloRatingSummary,
    pregame_summary: PregameFeatureSummary,
    modeling_summary: ModelingDatasetSummary,
) -> None:
    """Verify that every feature stage preserved the expected row grain."""

    expected_team_rows = modeling_summary.source_game_rows * 2

    if team_history_summary.output_team_rows != expected_team_rows:
        raise ValueError("Team-history row count is inconsistent with source games")

    if elo_summary.output_team_rows != expected_team_rows:
        raise ValueError("Elo rating row count is inconsistent with source games")

    if pregame_summary.source_team_rows != expected_team_rows:
        raise ValueError("Pregame feature input count is inconsistent")

    if pregame_summary.output_feature_rows != expected_team_rows:
        raise ValueError("Pregame feature output count is inconsistent")

    if modeling_summary.source_team_feature_rows != expected_team_rows:
        raise ValueError("Modeling source feature count is inconsistent")

    if modeling_summary.output_model_rows != (modeling_summary.source_game_rows):
        raise ValueError("Modeling output must contain one row per source game")

    if team_history_summary.seasons != pregame_summary.seasons:
        raise ValueError("Team-history and pregame feature season counts differ")

    if elo_summary.seasons != pregame_summary.seasons:
        raise ValueError("Elo rating and pregame feature season counts differ")

    if pregame_summary.seasons != modeling_summary.seasons:
        raise ValueError("Pregame and modeling season counts differ")


def run_feature_pipeline(
    project_root: Path,
) -> FeaturePipelineSummary:
    """Execute all Phase 5 transformations and write final metadata."""

    print("Building chronological team history...")

    team_history_summary = build_team_history_dataset(
        project_root=project_root,
    )

    print("\nSimulating cross-season Elo ratings...")

    elo_summary = build_team_elo_ratings_dataset(
        project_root=project_root,
    )

    print("\nBuilding leakage-safe pregame team features...")

    pregame_summary = build_pregame_team_feature_dataset(
        project_root=project_root,
    )

    print("\nBuilding final one-row-per-game modeling dataset...")

    modeling_summary = build_modeling_dataset_from_files(
        project_root=project_root,
    )

    validate_stage_counts(
        team_history_summary=team_history_summary,
        elo_summary=elo_summary,
        pregame_summary=pregame_summary,
        modeling_summary=modeling_summary,
    )

    final_output_path = modeling_dataset_output_path(project_root)

    if not final_output_path.exists():
        raise FileNotFoundError(f"Final modeling dataset was not created: {final_output_path}")

    summary = FeaturePipelineSummary(
        source_game_rows=modeling_summary.source_game_rows,
        team_history_rows=team_history_summary.output_team_rows,
        elo_rating_rows=elo_summary.output_team_rows,
        pregame_team_feature_rows=(pregame_summary.output_feature_rows),
        modeling_rows=modeling_summary.output_model_rows,
        seasons=modeling_summary.seasons,
        numeric_feature_count=(modeling_summary.numeric_feature_count),
        categorical_feature_count=(modeling_summary.categorical_feature_count),
        final_output_path=str(final_output_path),
    )

    summary_path = feature_pipeline_summary_path(project_root)

    summary_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # The orchestration summary is written only after all three stages have
    # completed and their row-count invariants agree.
    summary_path.write_text(
        json.dumps(
            asdict(summary),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print("\nPhase 5 feature pipeline complete:")
    print(json.dumps(asdict(summary), indent=2))
    print(f"Pipeline summary: {summary_path}")

    return summary


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the Phase 5 pipeline."""

    return argparse.ArgumentParser(
        description=__doc__,
    ).parse_args()


def main() -> None:
    """Run all Phase 5 feature transformations."""

    parse_args()
    project_root = Path(__file__).resolve().parents[2]

    run_feature_pipeline(
        project_root=project_root,
    )


if __name__ == "__main__":
    main()
