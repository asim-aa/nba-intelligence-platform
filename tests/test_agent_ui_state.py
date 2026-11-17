"""Tests for the layer that turns agent tool results into dashboard state."""

import pytest
from app.agent.tools import TOOLS
from app.agent.ui_state import DashboardState, Panel, apply_tool_result

LAKERS = 1610612747
CELTICS = 1610612738

PREDICTION = {
    "game_id": None,
    "home_team_id": LAKERS,
    "home_team_name": "Los Angeles Lakers",
    "away_team_id": CELTICS,
    "away_team_name": "Boston Celtics",
    "game_date": "2025-11-05",
    "season": "2025-26",
    "selected_model": "logistic_regression",
    "home_win_probability": 0.38,
    "model_pick": "AWAY",
    "feature_source": "computed",
    "actual_home_win": None,
}

METRICS = {
    "model_name": "logistic_regression",
    "rows": 750,
    "accuracy": 0.66,
    "log_loss": 0.60,
    "brier_score": 0.21,
    "roc_auc": 0.73,
    "expected_calibration_error": 0.039,
}

HEADLINE_METRICS = {
    "accuracy": 0.66,
    "log_loss": 0.60,
    "brier_score": 0.21,
    "roc_auc": 0.73,
    "expected_calibration_error": 0.039,
}

EXISTING = DashboardState(
    slate_date="2025-01-01", selected_game=(1, 2), panel=Panel("Earlier", [{"a": 1}])
)


def test_predict_matchup_selects_the_game_and_moves_the_slate_date() -> None:
    state = apply_tool_result(DashboardState(), "predict_matchup", PREDICTION)

    assert state.slate_date == "2025-11-05"
    assert state.selected_game == (LAKERS, CELTICS)
    assert state.panel == Panel(
        "Matchup prediction",
        [
            {
                "Matchup": "Boston Celtics @ Los Angeles Lakers",
                "Date": "2025-11-05",
                "Model pick": "Boston Celtics",
                "Home win probability": 0.38,
                "Feature source": "computed",
            }
        ],
    )


def test_slate_predictions_move_the_date_and_clear_the_selected_game() -> None:
    home_pick = {**PREDICTION, "model_pick": "HOME", "home_win_probability": 0.71}
    result = {"date": "2025-11-06", "games": [PREDICTION, home_pick]}

    state = apply_tool_result(EXISTING, "get_slate_predictions", result)

    assert state.slate_date == "2025-11-06"
    assert state.selected_game is None
    assert state.panel.title == "Model picks for 2025-11-06"
    assert [row["Model pick"] for row in state.panel.rows] == [
        "Boston Celtics",
        "Los Angeles Lakers",
    ]
    assert state.panel.chart is None


def test_slate_with_no_games_shows_an_empty_panel() -> None:
    state = apply_tool_result(
        DashboardState(), "get_slate_predictions", {"date": "2025-07-04", "games": []}
    )

    assert state.slate_date == "2025-07-04"
    assert state.panel == Panel("Model picks for 2025-07-04", [])


def test_calibration_by_segment_shows_a_table_and_an_ece_chart() -> None:
    result = {
        "dimension": "back_to_back",
        "segments": [{"segment_label": "back-to-back (either team)", **METRICS}],
    }

    state = apply_tool_result(EXISTING, "get_calibration_by_segment", result)

    assert state.panel == Panel(
        "Calibration by back to back",
        [{"Segment": "back-to-back (either team)", "Games": 750, **HEADLINE_METRICS}],
        chart=("Segment", "expected_calibration_error"),
    )
    # A filter-style tool leaves the date and selected game alone.
    assert (state.slate_date, state.selected_game) == ("2025-01-01", (1, 2))


def test_backtest_results_show_one_row_per_cutoff_and_an_accuracy_chart() -> None:
    fold = {
        "test_season": "2025-26",
        "train_seasons": ["2015-16"],
        "train_rows": 100,
        "test_rows": 50,
        **METRICS,
    }
    result = {"total_folds": 8, "folds": [fold], "metric_spread_all_folds": {}}

    state = apply_tool_result(EXISTING, "get_backtest_results", result)

    assert state.panel == Panel(
        "Walk-forward backtest (1 of 8 season cutoffs)",
        [{"Test season": "2025-26", "Train games": 100, "Test games": 50, **HEADLINE_METRICS}],
        chart=("Test season", "accuracy"),
    )
    assert (state.slate_date, state.selected_game) == ("2025-01-01", (1, 2))


def test_live_calibration_lists_the_frozen_reference_then_each_group() -> None:
    result = {
        "frozen_test_reference": {**METRICS, "rows": 2460},
        "live_groups": [
            {
                "feature_source": "all",
                "resolved_picks": 34,
                "sufficient_sample": True,
                "reason": None,
                "metrics": METRICS,
            },
            {
                "feature_source": "computed",
                "resolved_picks": 0,
                "sufficient_sample": False,
                "reason": "fewer than 30 resolved picks",
                "metrics": None,
            },
        ],
    }

    state = apply_tool_result(DashboardState(), "get_live_calibration", result)

    assert state.panel.title == "Live calibration vs. frozen test set"
    assert state.panel.rows == [
        {
            "Group": "frozen test reference",
            "Resolved picks": 2460,
            "Status": "scored",
            **HEADLINE_METRICS,
        },
        {"Group": "all", "Resolved picks": 34, "Status": "scored", **HEADLINE_METRICS},
        {"Group": "computed", "Resolved picks": 0, "Status": "fewer than 30 resolved picks"},
    ]


def test_live_calibration_without_a_frozen_reference_lists_only_the_groups() -> None:
    result = {
        "frozen_test_reference": None,
        "live_groups": [
            {
                "feature_source": "all",
                "resolved_picks": 0,
                "sufficient_sample": False,
                "reason": "fewer than 30 resolved picks",
                "metrics": None,
            }
        ],
    }

    state = apply_tool_result(DashboardState(), "get_live_calibration", result)

    assert [row["Group"] for row in state.panel.rows] == ["all"]


def test_find_team_shows_the_matches_without_touching_date_or_game() -> None:
    matches = [{"team_id": LAKERS, "full_name": "Los Angeles Lakers", "abbreviation": "LAL"}]

    state = apply_tool_result(EXISTING, "find_team", {"matches": matches})

    assert state.panel == Panel("Team lookup", matches)
    assert (state.slate_date, state.selected_game) == ("2025-01-01", (1, 2))


@pytest.mark.parametrize("tool_name", list(TOOLS))
def test_a_failed_tool_call_changes_nothing(tool_name: str) -> None:
    assert apply_tool_result(EXISTING, tool_name, {"error": "boom"}) is EXISTING


def test_an_unknown_tool_changes_nothing() -> None:
    assert apply_tool_result(EXISTING, "not_a_tool", {"anything": 1}) is EXISTING


def test_later_results_build_on_earlier_ones() -> None:
    state = apply_tool_result(DashboardState(), "predict_matchup", PREDICTION)
    state = apply_tool_result(
        state,
        "get_backtest_results",
        {"total_folds": 8, "folds": [], "metric_spread_all_folds": {}},
    )

    assert state.selected_game == (LAKERS, CELTICS)
    assert state.panel.title == "Walk-forward backtest (0 of 8 season cutoffs)"
