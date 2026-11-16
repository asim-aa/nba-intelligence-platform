"""Tests for the dashboard agent's allowlisted tool layer.

The tools are thin wrappers, so (like the API route tests) the wrapped
business logic is monkeypatched for the valid-input cases. Missing-data
cases run the real wrapped functions against an empty tmp_path instead,
so they prove the real FileNotFoundError is turned into an error result.
"""

from pathlib import Path

import app.agent.tools as tools_module
import pytest
from app.agent.tools import TOOLS, execute_tool, tool_specs
from app.api.main import MatchupPrediction
from modeling.evaluation.run_calibration_by_segment import SegmentResult
from modeling.evaluation.run_robustness_backtest import (
    BacktestFoldResult,
    MetricSpread,
    RobustnessBacktestSummary,
)

LAKERS = 1610612747
CELTICS = 1610612738


def make_prediction(**overrides: object) -> MatchupPrediction:
    fields = {
        "home_team_id": LAKERS,
        "away_team_id": CELTICS,
        "game_date": "2025-11-05",
        "season": "2025-26",
        "selected_model": "logistic_regression",
        "home_win_probability": 0.62,
        "model_pick": "HOME",
        "feature_source": "historical",
    }

    return MatchupPrediction(**{**fields, **overrides})


def make_backtest_summary(seasons: list[str]) -> RobustnessBacktestSummary:
    folds = [
        BacktestFoldResult(
            test_season=season,
            train_seasons=("2015-16",),
            train_rows=100,
            test_rows=50,
            metrics={"accuracy": 0.6 + index / 100},
        )
        for index, season in enumerate(seasons)
    ]

    return RobustnessBacktestSummary(
        folds=folds,
        metric_spread={"accuracy": MetricSpread(mean=0.61, std=0.01, min=0.6, max=0.63)},
        note="test",
    )


# --- registry -------------------------------------------------------------


def test_tool_specs_expose_exactly_the_allowlist() -> None:
    specs = tool_specs()

    assert [spec["function"]["name"] for spec in specs] == list(TOOLS)
    assert all(spec["type"] == "function" for spec in specs)
    assert all(spec["function"]["parameters"]["additionalProperties"] is False for spec in specs)


def test_execute_tool_rejects_unknown_tool(tmp_path: Path) -> None:
    result = execute_tool(tmp_path, "run_sql", {"query": "SELECT 1"})

    assert "Unknown tool 'run_sql'" in result["error"]
    assert "find_team" in result["error"]


def test_execute_tool_rejects_non_object_arguments(tmp_path: Path) -> None:
    result = execute_tool(tmp_path, "find_team", ["Lakers"])

    assert result["error"].startswith("Invalid arguments for find_team")


def test_execute_tool_rejects_unknown_fields(tmp_path: Path) -> None:
    result = execute_tool(tmp_path, "find_team", {"query": "Lakers", "path": "/etc/passwd"})

    assert "path: Extra inputs are not permitted" in result["error"]


# --- find_team ------------------------------------------------------------


@pytest.mark.parametrize("query", ["Lakers", "lal", "Los Angeles Lakers", str(LAKERS)])
def test_find_team_resolves_name_forms(tmp_path: Path, query: str) -> None:
    result = execute_tool(tmp_path, "find_team", {"query": query})

    assert result == {
        "matches": [{"team_id": LAKERS, "full_name": "Los Angeles Lakers", "abbreviation": "LAL"}]
    }


def test_find_team_returns_every_match_for_a_shared_city(tmp_path: Path) -> None:
    result = execute_tool(tmp_path, "find_team", {"query": "Los Angeles"})

    assert {team["abbreviation"] for team in result["matches"]} == {"LAL", "LAC"}


def test_find_team_reports_no_match(tmp_path: Path) -> None:
    result = execute_tool(tmp_path, "find_team", {"query": "Seattle"})

    assert result == {
        "error": "No NBA team matches 'Seattle'. Use a full name, nickname, city or abbreviation."
    }


def test_find_team_rejects_empty_query(tmp_path: Path) -> None:
    result = execute_tool(tmp_path, "find_team", {"query": ""})

    assert result["error"].startswith("Invalid arguments for find_team: query:")


# --- predict_matchup ------------------------------------------------------


def test_predict_matchup_resolves_teams_and_infers_season(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []

    def fake_prediction(*args: object, **kwargs: object) -> MatchupPrediction:
        calls.append((args, kwargs))
        return make_prediction()

    monkeypatch.setattr(tools_module, "get_matchup_prediction", fake_prediction)

    result = execute_tool(
        tmp_path,
        "predict_matchup",
        {"home_team": "Lakers", "away_team": "BOS", "game_date": "2025-11-05"},
    )

    assert calls == [
        (
            (tmp_path, LAKERS, CELTICS, "2025-11-05", "2025-26"),
            {"home_team_name": "Los Angeles Lakers", "away_team_name": "Boston Celtics"},
        )
    ]
    assert result["home_win_probability"] == 0.62
    assert result["model_pick"] == "HOME"


def test_predict_matchup_rejects_malformed_date(tmp_path: Path) -> None:
    result = execute_tool(
        tmp_path,
        "predict_matchup",
        {"home_team": "Lakers", "away_team": "Celtics", "game_date": "tonight"},
    )

    assert result["error"].startswith("Invalid arguments for predict_matchup: game_date:")


def test_predict_matchup_rejects_ambiguous_team(tmp_path: Path) -> None:
    result = execute_tool(
        tmp_path,
        "predict_matchup",
        {"home_team": "Los Angeles", "away_team": "Celtics", "game_date": "2025-11-05"},
    )

    assert "'Los Angeles' is ambiguous" in result["error"]
    assert "Los Angeles Clippers" in result["error"]


def test_predict_matchup_rejects_a_team_playing_itself(tmp_path: Path) -> None:
    result = execute_tool(
        tmp_path,
        "predict_matchup",
        {"home_team": "Lakers", "away_team": "LAL", "game_date": "2025-11-05"},
    )

    assert result == {"error": "home_team_id and away_team_id must differ"}


def test_predict_matchup_reports_missing_data(tmp_path: Path) -> None:
    result = execute_tool(
        tmp_path,
        "predict_matchup",
        {"home_team": "Lakers", "away_team": "Celtics", "game_date": "2025-11-05"},
    )

    assert "does not exist" in result["error"]


# --- get_slate_predictions ------------------------------------------------


def test_slate_predictions_returns_every_game(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        tools_module,
        "get_slate_predictions",
        lambda project_root, date: [make_prediction(game_id="0022500001", game_date=date)],
    )

    result = execute_tool(tmp_path, "get_slate_predictions", {"date": "2025-11-05"})

    assert result["date"] == "2025-11-05"
    assert [game["game_id"] for game in result["games"]] == ["0022500001"]


def test_slate_predictions_returns_empty_list_for_a_day_without_games(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tools_module, "get_slate_predictions", lambda project_root, date: [])

    result = execute_tool(tmp_path, "get_slate_predictions", {"date": "2025-07-04"})

    assert result == {"date": "2025-07-04", "games": []}


def test_slate_predictions_rejects_malformed_date(tmp_path: Path) -> None:
    result = execute_tool(tmp_path, "get_slate_predictions", {"date": "11/05/2025"})

    assert result["error"].startswith("Invalid arguments for get_slate_predictions: date:")


def test_slate_predictions_reports_schedule_fetch_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def raise_fetch_failure(project_root: Path, date: str) -> list[MatchupPrediction]:
        raise RuntimeError("Failed to fetch schedule for 2025-26 after 3 attempts")

    monkeypatch.setattr(tools_module, "get_slate_predictions", raise_fetch_failure)

    result = execute_tool(tmp_path, "get_slate_predictions", {"date": "2025-11-05"})

    assert result == {"error": "Failed to fetch schedule for 2025-26 after 3 attempts"}


# --- get_live_calibration -------------------------------------------------


def test_live_calibration_reports_insufficient_sample_when_no_picks_exist(
    tmp_path: Path,
) -> None:
    result = execute_tool(tmp_path, "get_live_calibration", {})

    assert result["frozen_test_reference"] is None
    assert [group["feature_source"] for group in result["live_groups"]] == [
        "all",
        "historical",
        "computed",
    ]
    assert all(group["sufficient_sample"] is False for group in result["live_groups"])
    assert all(group["metrics"] is None for group in result["live_groups"])


def test_live_calibration_rejects_arguments(tmp_path: Path) -> None:
    result = execute_tool(tmp_path, "get_live_calibration", {"feature_source": "computed"})

    assert result["error"].startswith("Invalid arguments for get_live_calibration")


# --- get_calibration_by_segment -------------------------------------------


def test_calibration_by_segment_returns_only_the_requested_dimension(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    canned = [
        SegmentResult("season_phase", "early (<=20 games played)", {"rows": 10}),
        SegmentResult(
            "back_to_back", "back-to-back (either team)", {"expected_calibration_error": 0.04}
        ),
        SegmentResult("back_to_back", "rest (neither team)", {"expected_calibration_error": 0.02}),
    ]
    monkeypatch.setattr(tools_module, "run_calibration_by_segment", lambda project_root: canned)

    result = execute_tool(tmp_path, "get_calibration_by_segment", {"dimension": "back_to_back"})

    assert result == {
        "dimension": "back_to_back",
        "segments": [
            {"segment_label": "back-to-back (either team)", "expected_calibration_error": 0.04},
            {"segment_label": "rest (neither team)", "expected_calibration_error": 0.02},
        ],
    }


def test_calibration_by_segment_rejects_unknown_dimension(tmp_path: Path) -> None:
    result = execute_tool(tmp_path, "get_calibration_by_segment", {"dimension": "weekday"})

    assert "dimension: Input should be 'back_to_back' or 'season_phase'" in result["error"]


def test_calibration_by_segment_reports_missing_data(tmp_path: Path) -> None:
    result = execute_tool(tmp_path, "get_calibration_by_segment", {"dimension": "season_phase"})

    assert result["error"].startswith("Training split does not exist")


# --- get_backtest_results -------------------------------------------------


def test_backtest_results_returns_every_fold_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    summary = make_backtest_summary(["2022-23", "2023-24", "2024-25", "2025-26"])
    monkeypatch.setattr(tools_module, "run_robustness_backtest", lambda project_root: summary)

    result = execute_tool(tmp_path, "get_backtest_results", {})

    assert result["total_folds"] == 4
    assert [fold["test_season"] for fold in result["folds"]] == [
        "2022-23",
        "2023-24",
        "2024-25",
        "2025-26",
    ]
    assert result["folds"][0] == {
        "test_season": "2022-23",
        "train_seasons": ["2015-16"],
        "train_rows": 100,
        "test_rows": 50,
        "accuracy": 0.6,
    }
    assert result["metric_spread_all_folds"] == {
        "accuracy": {"mean": 0.61, "std": 0.01, "min": 0.6, "max": 0.63}
    }


def test_backtest_results_keeps_only_the_last_n_folds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    summary = make_backtest_summary(["2022-23", "2023-24", "2024-25", "2025-26"])
    monkeypatch.setattr(tools_module, "run_robustness_backtest", lambda project_root: summary)

    result = execute_tool(tmp_path, "get_backtest_results", {"last_n": 3})

    assert result["total_folds"] == 4
    assert [fold["test_season"] for fold in result["folds"]] == ["2023-24", "2024-25", "2025-26"]


def test_backtest_results_rejects_non_positive_last_n(tmp_path: Path) -> None:
    result = execute_tool(tmp_path, "get_backtest_results", {"last_n": 0})

    assert "last_n: Input should be greater than or equal to 1" in result["error"]


def test_backtest_results_reports_missing_data(tmp_path: Path) -> None:
    result = execute_tool(tmp_path, "get_backtest_results", {})

    assert result["error"].startswith("Modeling dataset does not exist")
