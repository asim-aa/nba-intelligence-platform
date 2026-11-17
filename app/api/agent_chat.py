"""Business logic behind the agent endpoints.

Plain functions that take project_root explicitly, like the rest of
app/api, so they are testable without HTTP. Imports the agent at module
top, so app/api/main.py imports this lazily inside its route handlers
(see the note there) -- app.agent.tools imports main.py.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import asdict
from pathlib import Path

from app.agent.loop import client_from_env, configuration_error, run_agent
from app.agent.tools import TOOLS
from app.api.agent_schemas import ChatMetrics, ChatRequest, ChatResponse, ToolCall, ToolInfo

__all__ = ["configuration_error", "list_tools", "run_chat"]


def list_tools() -> list[ToolInfo]:
    """Describe every tool the agent is allowed to call."""

    return [
        ToolInfo(
            name=tool.name,
            description=tool.description,
            input_schema=tool.input_model.model_json_schema(),
        )
        for tool in TOOLS.values()
    ]


def run_chat(project_root: Path, request: ChatRequest) -> ChatResponse:
    """Run the agent on one request. Raises ValueError if no model is configured."""

    client, model = client_from_env()
    result = run_agent(
        client,
        model,
        request.message,
        project_root,
        dt.date.today(),
        history=[turn.model_dump() for turn in request.history],
    )

    return ChatResponse(
        answer=result.answer,
        completed=result.completed,
        # Round-tripped through JSON the same way the model saw each result,
        # so anything not natively JSON (a numpy scalar) cannot break the response.
        tool_calls=[
            ToolCall(
                name=call.name,
                arguments=call.arguments,
                result=json.loads(json.dumps(call.result, default=str)),
            )
            for call in result.tool_calls
        ],
        metrics=ChatMetrics(**asdict(result.metrics)),
        ungrounded_numbers=result.ungrounded_numbers,
    )
