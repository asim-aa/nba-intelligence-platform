"""Tests for the agent loop, driven by a scripted fake LLM client.

No test here touches a real model: FakeClient replays canned
chat-completion responses in order and records every request it was sent,
so the tests can assert both what the loop did and what it told the model.
"""

import datetime as dt
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from app.agent.loop import (
    MAX_ITERATIONS_ANSWER,
    MODEL_ENV_VAR,
    client_from_env,
    run_agent,
)
from app.agent.tools import TOOLS

TODAY = dt.date(2025, 11, 5)


def text_reply(content: str) -> SimpleNamespace:
    message = SimpleNamespace(content=content, tool_calls=None)

    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def tool_reply(*calls: tuple[str, str, str]) -> SimpleNamespace:
    """Build a response asking for tool calls, each (call_id, tool_name, raw_arguments)."""

    tool_calls = [
        SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=arguments))
        for call_id, name, arguments in calls
    ]
    message = SimpleNamespace(content=None, tool_calls=tool_calls)

    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeClient:
    """Replays scripted responses; repeats the last one if asked again."""

    def __init__(self, *script: SimpleNamespace) -> None:
        self.script = list(script)
        self.requests: list[dict[str, object]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs: object) -> SimpleNamespace:
        # messages is mutated by the loop after this call, so snapshot it.
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})

        return self.script.pop(0) if len(self.script) > 1 else self.script[0]


def test_answers_directly_when_no_tool_is_needed(tmp_path: Path) -> None:
    client = FakeClient(text_reply("I can only help with NBA predictions."))

    result = run_agent(client, "test-model", "What's the weather?", tmp_path, TODAY)

    assert result.answer == "I can only help with NBA predictions."
    assert result.tool_calls == []
    assert result.completed is True
    assert len(client.requests) == 1


def test_sends_system_prompt_with_today_then_history_then_the_request(tmp_path: Path) -> None:
    client = FakeClient(text_reply("ok"))
    history = [
        {"role": "user", "content": "earlier question"},
        {"role": "assistant", "content": "earlier answer"},
    ]

    run_agent(client, "test-model", "follow-up", tmp_path, TODAY, history=history)

    request = client.requests[0]
    assert request["model"] == "test-model"
    assert [tool["function"]["name"] for tool in request["tools"]] == list(TOOLS)
    assert request["messages"][0]["role"] == "system"
    assert "Today's date is 2025-11-05" in request["messages"][0]["content"]
    assert request["messages"][1:] == [*history, {"role": "user", "content": "follow-up"}]


def test_single_tool_call_feeds_the_result_back_to_the_model(tmp_path: Path) -> None:
    client = FakeClient(
        tool_reply(("call_1", "find_team", '{"query": "Lakers"}')),
        text_reply("The Lakers are team 1610612747."),
    )

    result = run_agent(client, "test-model", "What's the Lakers' team ID?", tmp_path, TODAY)

    expected = {
        "matches": [
            {"team_id": 1610612747, "full_name": "Los Angeles Lakers", "abbreviation": "LAL"}
        ]
    }
    assert result.completed is True
    assert result.answer == "The Lakers are team 1610612747."
    assert [(call.name, call.arguments, call.result) for call in result.tool_calls] == [
        ("find_team", {"query": "Lakers"}, expected)
    ]

    followup = client.requests[1]["messages"]
    assert followup[-2]["role"] == "assistant"
    assert followup[-2]["tool_calls"][0]["id"] == "call_1"
    assert followup[-1] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": json.dumps(expected),
    }


def test_multi_step_request_runs_tools_in_order(tmp_path: Path) -> None:
    executed = []

    def fake_execute(project_root: Path, name: str, arguments: object) -> dict[str, object]:
        executed.append((project_root, name, arguments))
        return {"step": len(executed)}

    client = FakeClient(
        tool_reply(("call_1", "get_slate_predictions", '{"date": "2025-11-05"}')),
        tool_reply(("call_2", "get_calibration_by_segment", '{"dimension": "back_to_back"}')),
        text_reply("Done."),
    )

    result = run_agent(
        client, "test-model", "Tonight's games and b2b calibration", tmp_path, TODAY,
        execute=fake_execute,
    )  # fmt: skip

    assert executed == [
        (tmp_path, "get_slate_predictions", {"date": "2025-11-05"}),
        (tmp_path, "get_calibration_by_segment", {"dimension": "back_to_back"}),
    ]
    assert [call.result for call in result.tool_calls] == [{"step": 1}, {"step": 2}]
    assert result.answer == "Done."
    assert result.completed is True
    assert len(client.requests) == 3


def test_parallel_tool_calls_in_one_response_are_all_answered(tmp_path: Path) -> None:
    client = FakeClient(
        tool_reply(
            ("call_a", "find_team", '{"query": "Lakers"}'),
            ("call_b", "find_team", '{"query": "Celtics"}'),
        ),
        text_reply("Found both."),
    )

    result = run_agent(client, "test-model", "IDs for Lakers and Celtics?", tmp_path, TODAY)

    assert [call.arguments for call in result.tool_calls] == [
        {"query": "Lakers"},
        {"query": "Celtics"},
    ]
    tool_messages = [m for m in client.requests[1]["messages"] if m["role"] == "tool"]
    assert [message["tool_call_id"] for message in tool_messages] == ["call_a", "call_b"]


def test_recovers_after_a_tool_error(tmp_path: Path) -> None:
    client = FakeClient(
        tool_reply(("call_1", "find_team", '{"query": "Seattle"}')),
        tool_reply(("call_2", "find_team", '{"query": "Thunder"}')),
        text_reply("Seattle has no team; the franchise is now the Oklahoma City Thunder."),
    )

    result = run_agent(client, "test-model", "Seattle's team?", tmp_path, TODAY)

    first, second = result.tool_calls
    assert "No NBA team matches 'Seattle'" in first.result["error"]
    assert second.result["matches"][0]["abbreviation"] == "OKC"
    assert result.completed is True

    # The error itself went back to the model, which is what let it retry.
    error_message = client.requests[1]["messages"][-1]
    assert error_message["role"] == "tool"
    assert "No NBA team matches 'Seattle'" in error_message["content"]


def test_malformed_json_arguments_become_a_tool_error_without_running_anything(
    tmp_path: Path,
) -> None:
    executed = []
    client = FakeClient(
        tool_reply(("call_1", "find_team", '{"query": ')),
        text_reply("Sorry, that lookup failed."),
    )

    result = run_agent(
        client, "test-model", "Lakers?", tmp_path, TODAY,
        execute=lambda *args: executed.append(args) or {},
    )  # fmt: skip

    assert executed == []
    assert result.tool_calls[0].result["error"].startswith("Tool arguments were not valid JSON")
    assert result.completed is True


def test_stops_at_max_iterations_with_a_fixed_notice(tmp_path: Path) -> None:
    client = FakeClient(tool_reply(("call_1", "find_team", '{"query": "Lakers"}')))

    result = run_agent(client, "test-model", "Loop forever", tmp_path, TODAY, max_iterations=3)

    assert result.completed is False
    assert result.answer == MAX_ITERATIONS_ANSWER
    assert len(result.tool_calls) == 3
    assert len(client.requests) == 3


def test_client_from_env_requires_key_and_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv(MODEL_ENV_VAR, "some-model")

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        client_from_env()

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv(MODEL_ENV_VAR)

    with pytest.raises(ValueError, match=MODEL_ENV_VAR):
        client_from_env()


def test_client_from_env_reads_model_and_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:9999/v1")
    monkeypatch.setenv(MODEL_ENV_VAR, "some-model")

    client, model = client_from_env()

    assert model == "some-model"
    assert str(client.base_url) == "http://localhost:9999/v1/"
