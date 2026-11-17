"""The agent loop: one user request in, tool calls and a short answer out.

A plain loop over the OpenAI chat-completions tool-calling API, capped at
MAX_ITERATIONS model calls -- no agent framework. The client is passed in,
so tests drive it with a scripted fake and never touch a real model.

The model is told to state only numbers that came from a tool result and
to say so when a tool errors or returns nothing. That is an instruction,
not an enforced guarantee: nothing here checks the final answer's numbers
against the tool results.
"""

from __future__ import annotations

import datetime as dt
import json
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from openai import OpenAI

from app.agent.tools import execute_tool, tool_specs

MODEL_ENV_VAR: Final[str] = "NBA_AGENT_MODEL"
MAX_ITERATIONS: Final[int] = 6

MAX_ITERATIONS_ANSWER: Final[str] = (
    "I stopped before finishing: this request needed more steps than I am allowed to take. "
    "The actions I did take are listed below."
)

SYSTEM_PROMPT: Final[str] = """\
You are the assistant inside an NBA win-probability dashboard. Today's date is {today}.

You answer by calling the provided tools; the dashboard shows the tables and charts for
whatever you call, so keep your written answer to a few sentences.

Rules:
- Every number you state must come from a tool result in this conversation. Never estimate,
  recall or invent a number. The one calculation allowed is the away team's win probability
  as 1 minus home_win_probability.
- If a tool returns an error or an empty result, say so plainly. You may retry once with
  corrected arguments, but do not guess an answer.
- For a real game on a date ("tonight", "tomorrow"), call get_slate_predictions for that
  date to find who is actually playing and who is at home. Use predict_matchup only for a
  matchup the user fully specifies or a hypothetical one.
- home_win_probability is the model's confidence in the home team; model_pick says which
  side it favours.
- These are model estimates, not betting advice.
"""


@dataclass(frozen=True)
class ToolCallRecord:
    """One tool call the agent made, with what came back."""

    name: str
    arguments: object
    result: dict[str, object]


@dataclass(frozen=True)
class AgentResult:
    """The agent's answer plus every action it took to get there.

    completed is False when the loop hit max_iterations before the model
    produced a final answer; answer is then a fixed notice, not model text.
    """

    answer: str
    tool_calls: list[ToolCallRecord]
    completed: bool


def configuration_error() -> str | None:
    """Return what is missing from the environment, or None when the agent can run."""

    if os.environ.get("OPENAI_API_KEY") and os.environ.get(MODEL_ENV_VAR):
        return None

    return (
        f"The agent needs OPENAI_API_KEY and {MODEL_ENV_VAR} set "
        "(plus OPENAI_BASE_URL for a non-OpenAI endpoint)."
    )


def client_from_env() -> tuple[OpenAI, str]:
    """Build the LLM client and model name from environment variables.

    The OpenAI SDK reads OPENAI_API_KEY and (optionally) OPENAI_BASE_URL
    itself; only the model name is this project's own variable.
    """

    error = configuration_error()

    if error is not None:
        raise ValueError(error)

    return OpenAI(), os.environ[MODEL_ENV_VAR]


def run_agent(
    client: OpenAI,
    model: str,
    user_message: str,
    project_root: Path,
    today: dt.date,
    *,
    history: Sequence[dict[str, str]] = (),
    max_iterations: int = MAX_ITERATIONS,
    execute: Callable[[Path, str, object], dict[str, object]] = execute_tool,
) -> AgentResult:
    """Run one user request to a final answer or the iteration cap.

    history is earlier {"role", "content"} chat turns, so follow-up
    questions keep their context.
    """

    messages: list[dict[str, object]] = [
        {"role": "system", "content": SYSTEM_PROMPT.format(today=today.isoformat())},
        *history,
        {"role": "user", "content": user_message},
    ]
    records: list[ToolCallRecord] = []

    for _ in range(max_iterations):
        response = client.chat.completions.create(
            model=model, messages=messages, tools=tool_specs()
        )
        message = response.choices[0].message

        if not message.tool_calls:
            return AgentResult(answer=message.content or "", tool_calls=records, completed=True)

        messages.append(
            {
                "role": "assistant",
                "content": message.content,
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.function.name,
                            "arguments": call.function.arguments,
                        },
                    }
                    for call in message.tool_calls
                ],
            }
        )

        for call in message.tool_calls:
            try:
                arguments = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError as error:
                arguments = call.function.arguments
                result = {"error": f"Tool arguments were not valid JSON: {error}"}
            else:
                result = execute(project_root, call.function.name, arguments)

            records.append(ToolCallRecord(call.function.name, arguments, result))
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(result, default=str),
                }
            )

    return AgentResult(answer=MAX_ITERATIONS_ANSWER, tool_calls=records, completed=False)
