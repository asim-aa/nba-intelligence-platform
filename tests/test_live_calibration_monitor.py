"""Tests for the dashboard's live calibration monitor."""

import json
from pathlib import Path

import numpy as np
from app.dashboard.live_calibration_monitor import (
    MIN_RESOLVED_PICKS_FOR_SIGNAL,
    live_calibration_summary_path,
    load_frozen_reference_metrics,
    resolved_picks_with_outcomes,
    run_live_calibration_monitor,
)
from app.dashboard.picks_store import Pick, record_pick, update_actual_result
from modeling.evaluation.run_final_evaluation import final_evaluation_summary_path


def write_resolved_picks(
    project_root: Path,
    count: int,
    feature_source: str,
    probabilities: np.ndarray,
    actual_winners: list[str],
    id_prefix: str,
) -> None:
    for index in range(count):
        game_id = f"{id_prefix}{index:04d}"
        record_pick(
            project_root,
            Pick(
                game_id=game_id,
                game_date="2025-11-05",
                season="2025-26",
                home_team_id=1610612747,
                home_team_name="Los Angeles Lakers",
                away_team_id=1610612738,
                away_team_name="Boston Celtics",
                user_pick="HOME",
                model_pick="HOME" if probabilities[index] >= 0.5 else "AWAY",
                model_home_win_probability=float(probabilities[index]),
                feature_source=feature_source,
            ),
        )
        update_actual_result(project_root, game_id, actual_winners[index])


def write_unresolved_pick(project_root: Path, game_id: str = "unresolved0001") -> None:
    record_pick(
        project_root,
        Pick(
            game_id=game_id,
            game_date="2025-11-06",
            season="2025-26",
            home_team_id=1610612747,
            home_team_name="Los Angeles Lakers",
            away_team_id=1610612738,
            away_team_name="Boston Celtics",
            user_pick="HOME",
            model_pick="HOME",
            model_home_win_probability=0.7,
            feature_source="historical",
        ),
    )


def make_well_calibrated_outcomes(
    rng: np.random.Generator, count: int
) -> tuple[np.ndarray, list[str]]:
    """Probabilities and outcomes where the model's confidence roughly matches reality."""

    probabilities = rng.uniform(0.2, 0.8, size=count)
    home_wins = rng.binomial(1, probabilities)
    winners = ["HOME" if win else "AWAY" for win in home_wins]

    return probabilities, winners


# --- resolved_picks_with_outcomes -----------------------------------


def test_resolved_picks_with_outcomes_excludes_unresolved(tmp_path: Path) -> None:
    rng = np.random.default_rng(seed=0)
    probabilities, winners = make_well_calibrated_outcomes(rng, 3)
    write_resolved_picks(tmp_path, 3, "historical", probabilities, winners, id_prefix="r")
    write_unresolved_pick(tmp_path)

    resolved = resolved_picks_with_outcomes(tmp_path)

    assert len(resolved) == 3
    assert resolved["home_win"].isin([0, 1]).all()


def test_resolved_picks_with_outcomes_derives_home_win_from_actual_winner(tmp_path: Path) -> None:
    write_resolved_picks(
        tmp_path, 2, "historical", np.array([0.6, 0.4]), ["HOME", "AWAY"], id_prefix="w"
    )

    resolved = resolved_picks_with_outcomes(tmp_path).sort_values("game_id").reset_index(drop=True)

    assert resolved.loc[0, "home_win"] == 1
    assert resolved.loc[1, "home_win"] == 0


# --- run_live_calibration_monitor: sample-size and class-diversity gates ---


def test_run_live_calibration_monitor_reports_insufficient_below_minimum(tmp_path: Path) -> None:
    rng = np.random.default_rng(seed=1)
    probabilities, winners = make_well_calibrated_outcomes(rng, 5)
    write_resolved_picks(tmp_path, 5, "historical", probabilities, winners, id_prefix="small")

    results = run_live_calibration_monitor(tmp_path)

    all_group = next(r for r in results if r.feature_source == "all")
    assert all_group.resolved_picks == 5
    assert all_group.sufficient_sample is False
    assert "fewer than" in all_group.reason
    assert all_group.metrics is None


def test_run_live_calibration_monitor_reports_insufficient_for_single_class(tmp_path: Path) -> None:
    probabilities = np.full(MIN_RESOLVED_PICKS_FOR_SIGNAL, 0.9)
    winners = ["HOME"] * MIN_RESOLVED_PICKS_FOR_SIGNAL
    write_resolved_picks(
        tmp_path,
        MIN_RESOLVED_PICKS_FOR_SIGNAL,
        "historical",
        probabilities,
        winners,
        id_prefix="one",
    )

    results = run_live_calibration_monitor(tmp_path)

    all_group = next(r for r in results if r.feature_source == "all")
    assert all_group.sufficient_sample is False
    assert "same outcome" in all_group.reason


def test_run_live_calibration_monitor_computes_metrics_when_sufficient(tmp_path: Path) -> None:
    rng = np.random.default_rng(seed=2)
    count = MIN_RESOLVED_PICKS_FOR_SIGNAL + 10
    probabilities, winners = make_well_calibrated_outcomes(rng, count)
    write_resolved_picks(tmp_path, count, "historical", probabilities, winners, id_prefix="ok")

    results = run_live_calibration_monitor(tmp_path)

    all_group = next(r for r in results if r.feature_source == "all")
    assert all_group.sufficient_sample is True
    assert all_group.resolved_picks == count
    assert all_group.metrics["rows"] == count
    assert all_group.metrics["log_loss"] >= 0.0


def test_run_live_calibration_monitor_groups_by_feature_source(tmp_path: Path) -> None:
    rng = np.random.default_rng(seed=3)
    count = MIN_RESOLVED_PICKS_FOR_SIGNAL + 5

    hist_probs, hist_winners = make_well_calibrated_outcomes(rng, count)
    write_resolved_picks(tmp_path, count, "historical", hist_probs, hist_winners, id_prefix="hist")

    computed_probs, computed_winners = make_well_calibrated_outcomes(rng, 4)
    write_resolved_picks(
        tmp_path, 4, "computed", computed_probs, computed_winners, id_prefix="comp"
    )

    results = run_live_calibration_monitor(tmp_path)
    by_source = {r.feature_source: r for r in results}

    assert by_source["all"].resolved_picks == count + 4
    assert by_source["historical"].resolved_picks == count
    assert by_source["historical"].sufficient_sample is True
    assert by_source["computed"].resolved_picks == 4
    assert by_source["computed"].sufficient_sample is False


# --- frozen reference lookup -----------------------------------


def test_load_frozen_reference_metrics_returns_none_when_missing(tmp_path: Path) -> None:
    assert load_frozen_reference_metrics(tmp_path) is None


def test_load_frozen_reference_metrics_reads_existing_summary(tmp_path: Path) -> None:
    summary_path = final_evaluation_summary_path(tmp_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            {
                "selected_model": "logistic_regression",
                "selected_model_test_metrics": {
                    "model_name": "logistic_regression",
                    "split_name": "test",
                    "rows": 2460,
                    "log_loss": 0.5996,
                    "brier_score": 0.2067,
                    "roc_auc": 0.7347,
                    "accuracy": 0.6789,
                    "expected_calibration_error": 0.0213,
                    "mean_predicted_probability": 0.566,
                    "actual_home_win_rate": 0.549,
                },
            }
        ),
        encoding="utf-8",
    )

    reference = load_frozen_reference_metrics(tmp_path)

    assert reference["log_loss"] == 0.5996
    assert reference["rows"] == 2460


def test_live_calibration_summary_path_is_gitignored_artifacts_location(tmp_path: Path) -> None:
    path = live_calibration_summary_path(tmp_path)

    assert "artifacts" in path.parts
    assert path.name == "live_calibration_summary.json"
