"""Tool-selection eval for the dashboard agent, run against a real model.

Each labeled request is sent once and the model's first step of tool
calls is scored against the expected tool and arguments. Tools are never
executed (a stub answers), so the only network call is the LLM itself and
the score does not depend on NBA data or the schedule endpoint.

Two numbers are reported, both out of all cases:
- tool-selection accuracy: the expected tool was among the first-step calls
- argument accuracy: one of those calls also had the expected arguments

"Today" is pinned to EVAL_TODAY so relative dates ("tonight", "tomorrow")
have one right answer. This measures tool routing only -- not whether the
final written answer is faithful to the tool results.

Needs OPENAI_API_KEY and NBA_AGENT_MODEL (see app/agent/loop.py).

Run with: uv run python -m app.agent.eval
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from openai import OpenAI

from app.agent.loop import ToolCallRecord, client_from_env, run_agent
from app.agent.tools import resolve_teams

EVAL_TODAY: Final[dt.date] = dt.date(2025, 11, 5)

# Compared by which team they resolve to, so "Lakers", "LAL" and
# "Los Angeles Lakers" all count as the same argument.
TEAM_ARGUMENTS: Final[frozenset[str]] = frozenset({"home_team", "away_team", "query"})


@dataclass(frozen=True)
class EvalCase:
    """One request and the tool call it should lead to first.

    An expected argument of None means "omitted or null".
    """

    request: str
    tool: str
    arguments: dict[str, object]


CASES: Final[tuple[EvalCase, ...]] = (
    EvalCase(
        "Who wins Lakers vs Celtics tonight and how confident is the model?",
        "get_slate_predictions",
        {"date": "2025-11-05"},
    ),
    EvalCase(
        "Pick the winner for tomorrow's games and show the table.",
        "get_slate_predictions",
        {"date": "2025-11-06"},
    ),
    EvalCase("What games are on tonight?", "get_slate_predictions", {"date": "2025-11-05"}),
    EvalCase("Show yesterday's slate.", "get_slate_predictions", {"date": "2025-11-04"}),
    EvalCase(
        "Show me the model's picks for November 12, 2025.",
        "get_slate_predictions",
        {"date": "2025-11-12"},
    ),
    EvalCase(
        "Predict Celtics at Lakers on 2025-12-25.",
        "predict_matchup",
        {"home_team": "Lakers", "away_team": "Celtics", "game_date": "2025-12-25"},
    ),
    EvalCase(
        "If the Knicks hosted the Thunder on January 10, 2026, who would the model favor?",
        "predict_matchup",
        {"home_team": "Knicks", "away_team": "Thunder", "game_date": "2026-01-10"},
    ),
    EvalCase(
        "What's the home win probability for the Warriors at home against the Suns on 2026-02-01?",
        "predict_matchup",
        {"home_team": "Warriors", "away_team": "Suns", "game_date": "2026-02-01"},
    ),
    EvalCase(
        "Hypothetically, Denver visiting Miami on 2025-11-20 -- who wins?",
        "predict_matchup",
        {"home_team": "Heat", "away_team": "Nuggets", "game_date": "2025-11-20"},
    ),
    EvalCase(
        "Show calibration for back-to-back games vs rested games.",
        "get_calibration_by_segment",
        {"dimension": "back_to_back"},
    ),
    EvalCase(
        "Is the model less well calibrated when a team is on a back-to-back?",
        "get_calibration_by_segment",
        {"dimension": "back_to_back"},
    ),
    EvalCase(
        "How does calibration change between early, mid and late season?",
        "get_calibration_by_segment",
        {"dimension": "season_phase"},
    ),
    EvalCase(
        "Break down calibration error by season phase.",
        "get_calibration_by_segment",
        {"dimension": "season_phase"},
    ),
    EvalCase("Compare the last 3 walk-forward cutoffs.", "get_backtest_results", {"last_n": 3}),
    EvalCase(
        "How did the model do in the two most recent backtest seasons?",
        "get_backtest_results",
        {"last_n": 2},
    ),
    EvalCase(
        "Show the full walk-forward backtest, every cutoff.",
        "get_backtest_results",
        {"last_n": None},
    ),
    EvalCase(
        "How stable is accuracy across all the season cutoffs?",
        "get_backtest_results",
        {"last_n": None},
    ),
    EvalCase("How well calibrated are the live picks so far?", "get_live_calibration", {}),
    EvalCase("Compare live pick calibration to the frozen test set.", "get_live_calibration", {}),
    EvalCase("What's the team ID for the Spurs?", "find_team", {"query": "Spurs"}),
    EvalCase("Which team has the abbreviation OKC?", "find_team", {"query": "OKC"}),
)


@dataclass(frozen=True)
class CaseScore:
    """How one case went: what was called, and whether it matched."""

    case: EvalCase
    calls: list[ToolCallRecord]
    tool_correct: bool
    arguments_correct: bool


def arguments_match(expected: dict[str, object], actual: object) -> bool:
    """True when every expected argument has the expected value in actual.

    Extra arguments in actual are not penalised.
    """

    if not isinstance(actual, dict):
        return False

    for name, value in expected.items():
        given = actual.get(name)

        if name in TEAM_ARGUMENTS and given is not None:
            try:
                if resolve_teams(str(given)) != resolve_teams(str(value)):
                    return False
            except ValueError:
                return False
        elif given != value:
            return False

    return True


def score_case(case: EvalCase, calls: list[ToolCallRecord]) -> CaseScore:
    """Score one case's first-step tool calls against its label."""

    matching_tool = [call for call in calls if call.name == case.tool]

    return CaseScore(
        case=case,
        calls=calls,
        tool_correct=bool(matching_tool),
        arguments_correct=any(
            arguments_match(case.arguments, call.arguments) for call in matching_tool
        ),
    )


def run_eval(client: OpenAI, model: str) -> list[CaseScore]:
    """Send every case to the model once and score its first step."""

    scores = []

    for case in CASES:
        result = run_agent(
            client,
            model,
            case.request,
            Path("."),
            EVAL_TODAY,
            max_iterations=1,
            execute=lambda project_root, name, arguments: {},
        )
        scores.append(score_case(case, result.tool_calls))

    return scores


def print_report(model: str, scores: list[CaseScore]) -> None:
    """Print one line per case, then the two accuracy numbers."""

    print(f"\nAgent tool-selection eval -- model={model}, today pinned to {EVAL_TODAY}\n")

    for score in scores:
        mark = "PASS" if score.arguments_correct else "TOOL" if score.tool_correct else "FAIL"
        called = ", ".join(f"{call.name}({call.arguments})" for call in score.calls) or "no tool"
        print(f"  [{mark}] {score.case.request}\n         -> {called}")

    total = len(scores)
    tool_hits = sum(score.tool_correct for score in scores)
    argument_hits = sum(score.arguments_correct for score in scores)

    print(f"\nTool-selection accuracy: {tool_hits}/{total} = {tool_hits / total:.1%}")
    print(f"Argument accuracy:       {argument_hits}/{total} = {argument_hits / total:.1%}")


def main() -> None:
    """Run the eval from the command line."""

    try:
        client, model = client_from_env()
    except ValueError as error:
        raise SystemExit(f"Cannot run the eval: {error}") from error

    print_report(model, run_eval(client, model))


if __name__ == "__main__":
    main()
