"""Request and response models for the agent endpoints.

Pydantic only, with no agent imports, so app/api/main.py can import them at
module load without pulling in the agent (which itself imports main.py).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

MAX_MESSAGE_CHARS = 2000
MAX_HISTORY_TURNS = 20


class ToolInfo(BaseModel):
    """One allowlisted tool, as the model sees it."""

    name: str
    description: str
    input_schema: dict[str, Any]


class ChatTurn(BaseModel):
    """One earlier message in the conversation."""

    role: Literal["user", "assistant"]
    content: str = Field(max_length=10 * MAX_MESSAGE_CHARS)


class ChatRequest(BaseModel):
    """One user request, with the conversation so far for follow-ups."""

    message: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)
    history: list[ChatTurn] = Field(default_factory=list, max_length=MAX_HISTORY_TURNS)


class ToolCall(BaseModel):
    """One tool call the agent made, and what came back."""

    name: str
    arguments: Any
    result: dict[str, Any]


class ChatMetrics(BaseModel):
    """What the request cost. Token counts are null when the endpoint reports none."""

    model_calls: int
    tool_calls: int
    tool_errors: int
    seconds: float
    prompt_tokens: int | None
    completion_tokens: int | None


class ChatResponse(BaseModel):
    """The agent's answer and everything it did to get there.

    completed is false when the agent hit its step cap; answer is then a
    fixed notice. ungrounded_numbers are figures in the answer that no tool
    result supports -- treat the tool_calls results as the source of truth.
    """

    answer: str
    completed: bool
    tool_calls: list[ToolCall]
    metrics: ChatMetrics
    ungrounded_numbers: list[str]
