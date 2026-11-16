"""The allowlisted tools the dashboard agent may call.

Every tool is a thin wrapper over a function that already exists --
app/api's prediction logic, the live calibration monitor, and the two
supplementary evaluation diagnostics -- so the agent can only do what the
dashboard and API already do. Nothing here reimplements feature or model
logic, runs arbitrary code, takes SQL, or touches the file system beyond
what those wrapped functions already read.

Each tool has a Pydantic input model (unknown fields rejected) whose JSON
schema is what the LLM sees. execute_tool() is the single entry point: it
always returns a dict, and reports any bad argument or missing data as
{"error": "..."} so the agent can read the message and recover instead of
the loop crashing.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final, Literal

import pandas as pd
from modeling.evaluation.run_calibration_by_segment import run_calibration_by_segment
from modeling.evaluation.run_robustness_backtest import run_robustness_backtest
from nba_api.stats.static import teams as nba_teams
from pipelines.ingestion.fetch_schedule import season_for_date
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.api.main import get_matchup_prediction, get_slate_predictions
from app.dashboard.live_calibration_monitor import (
    load_frozen_reference_metrics,
    run_live_calibration_monitor,
)


class ToolInput(BaseModel):
    """Base for every tool's arguments: unknown fields are an error."""

    model_config = ConfigDict(extra="forbid")


class FindTeamInput(ToolInput):
    query: str = Field(
        min_length=1,
        description="Team full name, nickname, city, or abbreviation, e.g. 'Lakers' or 'BOS'.",
    )


class PredictMatchupInput(ToolInput):
    home_team: str = Field(min_length=1, description="Home team name, nickname, or abbreviation.")
    away_team: str = Field(min_length=1, description="Away team name, nickname, or abbreviation.")
    game_date: dt.date = Field(description="Game date, YYYY-MM-DD.")


class SlatePredictionsInput(ToolInput):
    date: dt.date = Field(description="Calendar date of the slate, YYYY-MM-DD.")


class LiveCalibrationInput(ToolInput):
    pass


class CalibrationBySegmentInput(ToolInput):
    dimension: Literal["back_to_back", "season_phase"] = Field(
        description=(
            "'back_to_back' compares games where either team is on a back-to-back against "
            "rested games; 'season_phase' compares early, mid and late season."
        )
    )


class BacktestResultsInput(ToolInput):
    last_n: int | None = Field(
        default=None,
        ge=1,
        description="Only return the most recent N season cutoffs. Omit for all of them.",
    )


def resolve_teams(query: str) -> list[dict[str, object]]:
    """Return every NBA team a name, nickname, city, abbreviation or ID matches.

    Raises a ValueError when nothing matches.
    """

    needle = query.strip().casefold()
    teams = nba_teams.get_teams()

    matches = [
        team
        for team in teams
        if needle
        in {
            str(team["id"]),
            team["abbreviation"].casefold(),
            team["nickname"].casefold(),
            team["city"].casefold(),
            team["full_name"].casefold(),
        }
    ]

    if not matches:
        matches = [team for team in teams if needle in team["full_name"].casefold()]

    if not matches:
        raise ValueError(
            f"No NBA team matches {query!r}. Use a full name, nickname, city or abbreviation."
        )

    return [
        {
            "team_id": team["id"],
            "full_name": team["full_name"],
            "abbreviation": team["abbreviation"],
        }
        for team in matches
    ]


def resolve_one_team(query: str) -> dict[str, object]:
    """Resolve a query to exactly one team, or raise a ValueError if it is ambiguous."""

    matches = resolve_teams(query)

    if len(matches) > 1:
        names = ", ".join(str(team["full_name"]) for team in matches)
        raise ValueError(f"{query!r} is ambiguous: it matches {names}. Be more specific.")

    return matches[0]


def find_team(project_root: Path, args: FindTeamInput) -> dict[str, object]:
    return {"matches": resolve_teams(args.query)}


def predict_matchup(project_root: Path, args: PredictMatchupInput) -> dict[str, object]:
    home = resolve_one_team(args.home_team)
    away = resolve_one_team(args.away_team)
    game_date = args.game_date.isoformat()

    prediction = get_matchup_prediction(
        project_root,
        home["team_id"],
        away["team_id"],
        game_date,
        season_for_date(pd.Timestamp(game_date)),
        home_team_name=home["full_name"],
        away_team_name=away["full_name"],
    )

    return prediction.model_dump()


def slate_predictions(project_root: Path, args: SlatePredictionsInput) -> dict[str, object]:
    date = args.date.isoformat()
    games = [game.model_dump() for game in get_slate_predictions(project_root, date)]

    return {"date": date, "games": games}


def live_calibration(project_root: Path, args: LiveCalibrationInput) -> dict[str, object]:
    return {
        "frozen_test_reference": load_frozen_reference_metrics(project_root),
        "live_groups": [asdict(result) for result in run_live_calibration_monitor(project_root)],
    }


def calibration_by_segment(
    project_root: Path, args: CalibrationBySegmentInput
) -> dict[str, object]:
    segments = [
        {"segment_label": result.segment_label, **result.metrics}
        for result in run_calibration_by_segment(project_root)
        if result.segment_dimension == args.dimension
    ]

    return {"dimension": args.dimension, "segments": segments}


def backtest_results(project_root: Path, args: BacktestResultsInput) -> dict[str, object]:
    summary = run_robustness_backtest(project_root)
    folds = summary.folds[-args.last_n :] if args.last_n else summary.folds

    return {
        "total_folds": len(summary.folds),
        "folds": [
            {
                "test_season": fold.test_season,
                "train_seasons": list(fold.train_seasons),
                "train_rows": fold.train_rows,
                "test_rows": fold.test_rows,
                **fold.metrics,
            }
            for fold in folds
        ],
        "metric_spread_all_folds": {
            name: asdict(spread) for name, spread in summary.metric_spread.items()
        },
    }


@dataclass(frozen=True)
class Tool:
    """One allowlisted tool: its name, what the LLM is told, and what it runs."""

    name: str
    description: str
    input_model: type[ToolInput]
    run: Callable[[Path, ToolInput], dict[str, object]]


TOOLS: Final[dict[str, Tool]] = {
    tool.name: tool
    for tool in (
        Tool(
            "find_team",
            "Look up an NBA team's ID, full name and abbreviation.",
            FindTeamInput,
            find_team,
        ),
        Tool(
            "predict_matchup",
            "Home-win probability from the frozen model for one matchup on one date. Works for "
            "hypothetical matchups too -- it does not check the game is really scheduled, so "
            "use get_slate_predictions to find real games and who is at home.",
            PredictMatchupInput,
            predict_matchup,
        ),
        Tool(
            "get_slate_predictions",
            "Every real regular-season game on one date with the frozen model's prediction "
            "for each. An empty games list means no games that day.",
            SlatePredictionsInput,
            slate_predictions,
        ),
        Tool(
            "get_live_calibration",
            "Calibration of the model's real, resolved dashboard picks next to its frozen "
            "test-set reference. Groups with too few picks report why instead of metrics.",
            LiveCalibrationInput,
            live_calibration,
        ),
        Tool(
            "get_calibration_by_segment",
            "The frozen model's test-set metrics (including expected calibration error) "
            "broken down by one segment dimension.",
            CalibrationBySegmentInput,
            calibration_by_segment,
        ),
        Tool(
            "get_backtest_results",
            "Walk-forward backtest: the same model retrained at each season cutoff and scored "
            "on the following season, oldest first, plus the metric spread across all cutoffs.",
            BacktestResultsInput,
            backtest_results,
        ),
    )
}


def tool_specs() -> list[dict[str, object]]:
    """Return the tools in the OpenAI chat-completions tool-calling format."""

    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_model.model_json_schema(),
            },
        }
        for tool in TOOLS.values()
    ]


def execute_tool(project_root: Path, name: str, arguments: object) -> dict[str, object]:
    """Validate arguments and run one allowlisted tool.

    Never raises for a bad request: an unknown tool, invalid arguments,
    bad input or missing data all come back as {"error": message}.
    """

    tool = TOOLS.get(name)

    if tool is None:
        return {"error": f"Unknown tool {name!r}. Available tools: {', '.join(TOOLS)}."}

    try:
        return tool.run(project_root, tool.input_model.model_validate(arguments))
    except ValidationError as error:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in problem['loc']) or 'arguments'}: {problem['msg']}"
            for problem in error.errors()
        )
        return {"error": f"Invalid arguments for {name}: {problems}"}
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        return {"error": str(error)}
