"""UI tests for the dashboard's agent chat panel, using Streamlit's AppTest.

The panel is exercised the way app.py uses it: render_agent_chat() runs
before the keyed slate date picker, then render_agent_view(). The agent
itself is replaced with a scripted run_agent, so no test reaches a model.
"""

import datetime as dt
from pathlib import Path

import app.dashboard.agent_panel as panel_module
import pytest
from app.agent.loop import MODEL_ENV_VAR, AgentMetrics, AgentResult, ToolCallRecord
from app.agent.ui_state import DashboardState, Panel
from openai import OpenAIError
from streamlit.testing.v1 import AppTest

LAKERS, CELTICS = 1610612747, 1610612738

PREDICTION = {
    "game_id": None,
    "home_team_id": LAKERS,
    "home_team_name": "Los Angeles Lakers",
    "away_team_id": CELTICS,
    "away_team_name": "Boston Celtics",
    "game_date": "2025-11-07",
    "season": "2025-26",
    "selected_model": "logistic_regression",
    "home_win_probability": 0.62,
    "model_pick": "HOME",
    "feature_source": "historical",
    "actual_home_win": None,
}

SEGMENT_RESULT = {
    "dimension": "back_to_back",
    "segments": [
        {
            "segment_label": "back-to-back (either team)",
            "rows": 750,
            "accuracy": 0.66,
            "expected_calibration_error": 0.039,
        },
        {
            "segment_label": "rest (neither team)",
            "rows": 1710,
            "accuracy": 0.69,
            "expected_calibration_error": 0.023,
        },
    ],
}

APP_SCRIPT = """
import datetime as dt
from pathlib import Path

import streamlit as st

from app.dashboard.agent_panel import SLATE_DATE_KEY, render_agent_chat, render_agent_view

render_agent_chat(Path("."))
render_agent_view()
st.session_state.setdefault(SLATE_DATE_KEY, dt.date(2025, 11, 5))
st.date_input("Slate date", key=SLATE_DATE_KEY)
"""


@pytest.fixture
def script_path(tmp_path: Path) -> str:
    path = tmp_path / "panel_app.py"
    path.write_text(APP_SCRIPT, encoding="utf-8")

    return str(path)


@pytest.fixture
def configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv(MODEL_ENV_VAR, "test-model")


def scripted_agent(monkeypatch: pytest.MonkeyPatch, *results: AgentResult) -> list[dict]:
    """Replace run_agent with one that returns each result in turn; log its calls."""

    calls: list[dict] = []
    queue = list(results)

    def fake_run_agent(client, model, message, project_root, today, **kwargs):
        calls.append({"model": model, "message": message, "history": kwargs["history"]})
        return queue.pop(0)

    monkeypatch.setattr(panel_module, "run_agent", fake_run_agent)

    return calls


def matchup_result() -> AgentResult:
    call = ToolCallRecord(
        "predict_matchup",
        {"home_team": "Lakers", "away_team": "Celtics", "game_date": "2025-11-07"},
        PREDICTION,
    )

    return AgentResult("The model favours the Lakers.", [call], completed=True)


def segment_result() -> AgentResult:
    call = ToolCallRecord(
        "get_calibration_by_segment", {"dimension": "back_to_back"}, SEGMENT_RESULT
    )

    return AgentResult("See the table.", [call], completed=True)


def ask(at: AppTest, message: str) -> AppTest:
    at.chat_input[0].set_value(message).run()

    return at


def test_chat_is_disabled_upfront_when_no_model_is_configured(
    script_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv(MODEL_ENV_VAR, raising=False)

    at = AppTest.from_file(script_path).run()

    assert not at.exception
    assert "Chat is disabled" in at.sidebar.info[0].value
    assert at.chat_input[0].disabled is True


def test_chat_is_enabled_without_a_notice_when_configured(
    script_path: str, configured: None
) -> None:
    at = AppTest.from_file(script_path).run()

    assert not at.exception
    assert len(at.sidebar.info) == 0
    assert at.chat_input[0].disabled is False


def test_a_turn_shows_the_answer_the_actions_and_the_agent_view(
    script_path: str, configured: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = scripted_agent(monkeypatch, matchup_result())

    at = ask(AppTest.from_file(script_path).run(), "Who wins Lakers vs Celtics?")

    assert not at.exception
    assert calls[0]["message"] == "Who wins Lakers vs Celtics?"
    assert calls[0]["model"] == "test-model"
    assert [message.markdown[0].value for message in at.sidebar.chat_message] == [
        "Who wins Lakers vs Celtics?",
        "The model favours the Lakers.",
    ]
    assert [expander.label for expander in at.sidebar.expander] == ["Actions taken (1)"]
    assert any("Agent view: Matchup prediction" in m.value for m in at.markdown)
    assert len(at.dataframe) == 1


def test_a_slate_date_result_moves_the_date_picker(
    script_path: str, configured: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    scripted_agent(monkeypatch, matchup_result())

    at = ask(AppTest.from_file(script_path).run(), "Lakers vs Celtics")

    assert at.date_input[0].value == dt.date(2025, 11, 7)


def test_a_turn_without_a_date_leaves_the_users_date_alone(
    script_path: str, configured: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: an st.rerun() after each turn once reset the picker to its
    default whenever the tool returned no date."""

    scripted_agent(monkeypatch, matchup_result(), segment_result())
    at = ask(AppTest.from_file(script_path).run(), "Lakers vs Celtics")

    at.date_input[0].set_value(dt.date(2025, 12, 1)).run()
    ask(at, "Show calibration for back-to-back games")

    assert at.date_input[0].value == dt.date(2025, 12, 1)
    assert any("Calibration by back to back" in m.value for m in at.markdown)
    assert len(at.get("vega_lite_chart")) == 1


def test_follow_up_turns_receive_the_earlier_messages_as_history(
    script_path: str, configured: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = scripted_agent(monkeypatch, matchup_result(), segment_result())

    at = ask(AppTest.from_file(script_path).run(), "first question")
    ask(at, "second question")

    assert calls[0]["history"] == []
    assert calls[1]["history"] == [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "The model favours the Lakers."},
    ]


def test_a_failed_model_request_shows_an_error_and_records_no_turn(
    script_path: str, configured: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing_run_agent(*args: object, **kwargs: object) -> AgentResult:
        raise OpenAIError("connection refused")

    monkeypatch.setattr(panel_module, "run_agent", failing_run_agent)

    at = ask(AppTest.from_file(script_path).run(), "anything")

    assert not at.exception
    assert "The model request failed: connection refused" in at.sidebar.error[0].value
    assert len(at.sidebar.chat_message) == 0


def test_an_empty_panel_says_so_instead_of_rendering_a_table(script_path: str) -> None:
    at = AppTest.from_file(script_path)
    at.session_state[panel_module.STATE_KEY] = DashboardState(panel=Panel("Model picks", []))

    at.run()

    assert not at.exception
    assert len(at.dataframe) == 0
    assert any("Nothing to show" in caption.value for caption in at.caption)


def test_a_turn_shows_what_the_request_cost(
    script_path: str, configured: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    metrics = AgentMetrics(
        model_calls=2, tool_calls=1, tool_errors=0, seconds=1.25,
        prompt_tokens=300, completion_tokens=40,
    )  # fmt: skip
    result = AgentResult("ok", matchup_result().tool_calls, True, metrics)
    scripted_agent(monkeypatch, result)

    at = ask(AppTest.from_file(script_path).run(), "hello")

    assert "1.2s · 2 model call(s) · 1 tool call(s) · 340 tokens" in [
        caption.value for caption in at.sidebar.caption
    ]


def test_unsupported_numbers_are_called_out_under_the_answer(
    script_path: str, configured: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = AgentResult("It is 71%.", [], True, ungrounded_numbers=["71%"])
    scripted_agent(monkeypatch, result)

    at = ask(AppTest.from_file(script_path).run(), "hello")

    assert "Not backed by any tool result" in at.sidebar.warning[0].value
    assert "71%" in at.sidebar.warning[0].value


def test_a_clean_answer_has_no_warning(
    script_path: str, configured: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    scripted_agent(monkeypatch, matchup_result())

    at = ask(AppTest.from_file(script_path).run(), "hello")

    assert len(at.sidebar.warning) == 0
