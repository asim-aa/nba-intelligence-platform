"""The dashboard state the agent is allowed to change, and how each tool changes it.

This is the only place that knows what a tool result means for the UI.
It is pure data in, pure data out -- no Streamlit, no LLM -- so the
dashboard renders a DashboardState without knowing an agent produced it,
and the agent loop never imports anything UI-related.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Final

METRIC_COLUMNS: Final[tuple[str, ...]] = (
    "accuracy",
    "log_loss",
    "brier_score",
    "roc_auc",
    "expected_calibration_error",
)


@dataclass(frozen=True)
class Panel:
    """One table (and optionally one bar chart) for the dashboard to show.

    chart is (x column, y column) into rows.
    """

    title: str
    rows: list[dict[str, object]]
    chart: tuple[str, str] | None = None


@dataclass(frozen=True)
class DashboardState:
    """What the agent last pointed the dashboard at.

    slate_date drives the date picker; selected_game is
    (home_team_id, away_team_id) for the game to highlight in the slate.
    """

    slate_date: str | None = None
    selected_game: tuple[int, int] | None = None
    panel: Panel | None = None


def prediction_row(prediction: dict[str, object]) -> dict[str, object]:
    """Flatten one prediction into the columns the dashboard table shows."""

    home, away = prediction["home_team_name"], prediction["away_team_name"]

    return {
        "Matchup": f"{away} @ {home}",
        "Date": prediction["game_date"],
        "Model pick": home if prediction["model_pick"] == "HOME" else away,
        "Home win probability": prediction["home_win_probability"],
        "Feature source": prediction["feature_source"],
    }


def metric_columns(metrics: dict[str, object] | None) -> dict[str, object]:
    """Keep just the headline metrics from a metrics record, if there is one."""

    return {name: metrics[name] for name in METRIC_COLUMNS if name in metrics} if metrics else {}


def apply_tool_result(
    state: DashboardState, tool_name: str, result: dict[str, object]
) -> DashboardState:
    """Return the state after one tool call. A failed call changes nothing."""

    if "error" in result:
        return state

    if tool_name == "find_team":
        return replace(state, panel=Panel("Team lookup", result["matches"]))

    if tool_name == "predict_matchup":
        return DashboardState(
            slate_date=result["game_date"],
            selected_game=(result["home_team_id"], result["away_team_id"]),
            panel=Panel("Matchup prediction", [prediction_row(result)]),
        )

    if tool_name == "get_slate_predictions":
        return DashboardState(
            slate_date=result["date"],
            selected_game=None,
            panel=Panel(
                f"Model picks for {result['date']}",
                [prediction_row(game) for game in result["games"]],
            ),
        )

    if tool_name == "get_live_calibration":
        rows = [
            {
                "Group": group["feature_source"],
                "Resolved picks": group["resolved_picks"],
                "Status": group["reason"] or "scored",
                **metric_columns(group["metrics"]),
            }
            for group in result["live_groups"]
        ]

        if result["frozen_test_reference"] is not None:
            reference = result["frozen_test_reference"]
            rows.insert(
                0,
                {
                    "Group": "frozen test reference",
                    "Resolved picks": reference["rows"],
                    "Status": "scored",
                    **metric_columns(reference),
                },
            )

        return replace(state, panel=Panel("Live calibration vs. frozen test set", rows))

    if tool_name == "get_calibration_by_segment":
        return replace(
            state,
            panel=Panel(
                f"Calibration by {result['dimension'].replace('_', ' ')}",
                [
                    {
                        "Segment": segment["segment_label"],
                        "Games": segment.get("rows"),
                        **metric_columns(segment),
                    }
                    for segment in result["segments"]
                ],
                chart=("Segment", "expected_calibration_error"),
            ),
        )

    if tool_name == "get_backtest_results":
        return replace(
            state,
            panel=Panel(
                f"Walk-forward backtest ({len(result['folds'])} of "
                f"{result['total_folds']} season cutoffs)",
                [
                    {
                        "Test season": fold["test_season"],
                        "Train games": fold["train_rows"],
                        "Test games": fold["test_rows"],
                        **metric_columns(fold),
                    }
                    for fold in result["folds"]
                ],
                chart=("Test season", "accuracy"),
            ),
        )

    return state
