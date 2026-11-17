"""Tests for the agent eval's labels and scoring -- not a run of the eval itself.

The eval's accuracy numbers only mean something against a real model;
these tests only make sure the labels are valid tool calls and the
scoring arithmetic is right, using a scripted fake client.
"""

from types import SimpleNamespace

import pytest
from app.agent.eval import (
    CASES,
    EvalCase,
    arguments_match,
    main,
    run_eval,
    score_case,
)
from app.agent.loop import MODEL_ENV_VAR, ToolCallRecord
from app.agent.tools import TOOLS


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.request[:40])
def test_every_label_is_a_valid_call_to_an_allowlisted_tool(case: EvalCase) -> None:
    arguments = {name: value for name, value in case.arguments.items() if value is not None}

    TOOLS[case.tool].input_model.model_validate(arguments)


def test_arguments_match_compares_teams_by_identity_not_spelling() -> None:
    expected = {"home_team": "Lakers", "away_team": "Celtics", "game_date": "2025-12-25"}

    assert arguments_match(
        expected,
        {"home_team": "LAL", "away_team": "Boston Celtics", "game_date": "2025-12-25"},
    )
    # Home and away swapped is a wrong answer, not a spelling difference.
    assert not arguments_match(
        expected, {"home_team": "Celtics", "away_team": "Lakers", "game_date": "2025-12-25"}
    )
    assert not arguments_match(
        expected, {"home_team": "Lakers", "away_team": "Celtics", "game_date": "2025-12-26"}
    )
    assert not arguments_match(
        expected, {"home_team": "Nobody", "away_team": "Celtics", "game_date": "2025-12-25"}
    )


def test_arguments_match_treats_expected_none_as_omitted_or_null() -> None:
    assert arguments_match({"last_n": None}, {})
    assert arguments_match({"last_n": None}, {"last_n": None})
    assert not arguments_match({"last_n": None}, {"last_n": 3})
    assert not arguments_match({"last_n": 3}, {})


def test_arguments_match_rejects_missing_or_non_object_arguments() -> None:
    assert not arguments_match({"query": "Spurs"}, {})
    assert not arguments_match({"query": "Spurs"}, '{"query": "Spurs"}')


def test_score_case_separates_wrong_tool_from_wrong_arguments() -> None:
    case = EvalCase("Compare the last 3 cutoffs.", "get_backtest_results", {"last_n": 3})

    right = score_case(case, [ToolCallRecord("get_backtest_results", {"last_n": 3}, {})])
    wrong_arguments = score_case(case, [ToolCallRecord("get_backtest_results", {}, {})])
    wrong_tool = score_case(case, [ToolCallRecord("get_live_calibration", {}, {})])
    no_call = score_case(case, [])

    assert (right.tool_correct, right.arguments_correct) == (True, True)
    assert (wrong_arguments.tool_correct, wrong_arguments.arguments_correct) == (True, False)
    assert (wrong_tool.tool_correct, wrong_tool.arguments_correct) == (False, False)
    assert (no_call.tool_correct, no_call.arguments_correct) == (False, False)


def test_run_eval_scores_the_first_step_of_every_case() -> None:
    """A fake model that always calls get_live_calibration() gets exactly
    the get_live_calibration cases right and nothing else."""

    call = SimpleNamespace(
        id="call_1", function=SimpleNamespace(name="get_live_calibration", arguments="{}")
    )
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=None, tool_calls=[call]))]
    )
    requests = []

    def create(**kwargs: object) -> SimpleNamespace:
        requests.append(kwargs)
        return response

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    scores = run_eval(client, "test-model")

    expected_hits = sum(case.tool == "get_live_calibration" for case in CASES)
    assert len(scores) == len(requests) == len(CASES)
    assert sum(score.tool_correct for score in scores) == expected_hits
    assert sum(score.arguments_correct for score in scores) == expected_hits


def test_main_exits_with_a_clear_message_when_unconfigured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv(MODEL_ENV_VAR, raising=False)

    with pytest.raises(SystemExit, match="Cannot run the eval"):
        main()
