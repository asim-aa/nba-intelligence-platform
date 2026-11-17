"""Flag numbers in the agent's answer that no tool result supports.

The system prompt tells the model to state only numbers that came from a
tool result. This is the check that actually looks: it pulls every number
out of the written answer and reports the ones it cannot find in the
grounding -- the tool results, the tool arguments, and what the user wrote.

A number is grounded when, rounded to the precision the answer shows, it
equals a grounded value v, 100 * v (a percentage), 1 - v or 100 * (1 - v)
(the away team's probability, which the prompt allows), or the length of a
result list (a count). Dates and season labels are grounded when they
appear verbatim.

This is a tripwire, not a proof: it catches an invented or misquoted
figure, but not a correct figure attached to the wrong claim. It can also
flag a legitimate derived number (a difference, a mean), which is the safe
direction to be wrong in.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from typing import Final

DATE_OR_SEASON: Final[re.Pattern[str]] = re.compile(r"\d{4}-\d{2}(?:-\d{2})?")
NUMBER: Final[re.Pattern[str]] = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?%?")
LIST_MARKER: Final[re.Pattern[str]] = re.compile(r"^\s*\d+[.)]\s+", re.MULTILINE)


def parse_number(token: str) -> tuple[float, int, bool]:
    """Return (value, decimals shown, is_percent) for one NUMBER match."""

    percent = token.endswith("%")
    text = token.rstrip("%").replace(",", "")
    decimals = len(text.split(".")[1]) if "." in text else 0

    return float(text), decimals, percent


def numbers_in(text: str) -> list[str]:
    """Return the number tokens in text, ignoring dates, seasons and list markers."""

    text = LIST_MARKER.sub("", DATE_OR_SEASON.sub(" ", text))

    return NUMBER.findall(text)


def numeric_values(value: object) -> Iterable[float]:
    """Yield every number a tool result holds: numeric leaves, numbers inside
    strings, and the length of every list."""

    if isinstance(value, bool) or value is None:
        return

    if isinstance(value, int | float):
        yield float(value)
    elif isinstance(value, str):
        for token in numbers_in(value):
            yield parse_number(token)[0]
    elif isinstance(value, dict):
        for item in value.values():
            yield from numeric_values(item)
    elif isinstance(value, list | tuple):
        yield float(len(value))
        for item in value:
            yield from numeric_values(item)


def is_grounded(token: str, grounded: set[float]) -> bool:
    """True when token matches a grounded value at the precision it shows."""

    shown, decimals, percent = parse_number(token)
    tolerance = 0.5 * 10**-decimals + 1e-9

    for value in grounded:
        forms = (100 * value, 100 * (1 - value)) if percent else (
            value, 1 - value, 100 * value, 100 * (1 - value)
        )  # fmt: skip

        if any(abs(form - shown) <= tolerance for form in forms):
            return True

    return False


def check_faithfulness(
    answer: str,
    user_messages: Iterable[str],
    tool_calls: Iterable[tuple[object, dict[str, object]]],
) -> list[str]:
    """Return the numbers, dates and seasons in answer that nothing grounds.

    tool_calls is (arguments, result) pairs. Each ungrounded token appears
    once, in the order the answer first shows it.
    """

    user_text = " ".join(user_messages)
    evidence = [user_text, *(pair for call in tool_calls for pair in call)]
    grounding_text = " ".join(json.dumps(item, default=str) for item in evidence)

    grounded = {parse_number(token)[0] for token in numbers_in(user_text)}
    for item in evidence[1:]:
        grounded.update(numeric_values(item))

    ungrounded: list[str] = []

    for label in DATE_OR_SEASON.findall(answer):
        if label not in grounding_text and label not in ungrounded:
            ungrounded.append(label)

    for token in numbers_in(answer):
        if not is_grounded(token, grounded) and token not in ungrounded:
            ungrounded.append(token)

    return ungrounded
