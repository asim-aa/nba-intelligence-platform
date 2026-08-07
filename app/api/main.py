"""Prediction-serving API: the service boundary app/dashboard never had.

The dashboard talks directly to modeling/serving/ and pipelines/ingestion/
because it runs in the same process. This module wraps the exact same
functions in HTTP -- no reimplementation, no duplicated feature logic --
so any other client (a script, a future frontend, curl) can get a
prediction without importing this project's Python at all.

Business logic lives in plain, synchronous functions that take
project_root explicitly, so they can be unit-tested against a tmp_path
fixture the same way every other module in this project is; the FastAPI
route handlers below are thin wrappers that supply the real PROJECT_ROOT
and translate exceptions into HTTP status codes.

Run with: uv run uvicorn app.api.main:app --reload
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from modeling.evaluation.run_final_evaluation import SELECTED_MODEL
from modeling.serving.matchup_features import compute_matchup_features
from modeling.serving.predict_matchup import load_selected_model, predict_home_win_probability
from pipelines.ingestion.fetch_schedule import (
    fetch_schedule,
    filter_regular_season_games,
    games_on_date,
    season_for_date,
)
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class MatchupPrediction(BaseModel):
    """One matchup's prediction from the frozen, selected model."""

    game_id: str | None = None
    home_team_id: int
    home_team_name: str | None = None
    away_team_id: int
    away_team_name: str | None = None
    game_date: str
    season: str
    selected_model: str
    home_win_probability: float = Field(ge=0.0, le=1.0)
    model_pick: str
    feature_source: str
    actual_home_win: int | None = None


@lru_cache(maxsize=1)
def get_cached_model(project_root: Path):
    """Load the frozen model once per process, not once per request."""

    return load_selected_model(project_root)


def get_matchup_prediction(
    project_root: Path,
    home_team_id: int,
    away_team_id: int,
    game_date: str,
    season: str,
    *,
    game_id: str | None = None,
    home_team_name: str | None = None,
    away_team_name: str | None = None,
) -> MatchupPrediction:
    """Predict one matchup. Raises ValueError for bad input."""

    if home_team_id == away_team_id:
        raise ValueError("home_team_id and away_team_id must differ")

    feature_row, source, actual_home_win = compute_matchup_features(
        project_root=project_root,
        home_team_id=home_team_id,
        away_team_id=away_team_id,
        game_date=game_date,
        season=season,
    )

    model = get_cached_model(project_root)
    probability = predict_home_win_probability(model, feature_row)

    return MatchupPrediction(
        game_id=game_id,
        home_team_id=home_team_id,
        home_team_name=home_team_name,
        away_team_id=away_team_id,
        away_team_name=away_team_name,
        game_date=str(pd.Timestamp(game_date).date()),
        season=season,
        selected_model=SELECTED_MODEL,
        home_win_probability=probability,
        model_pick="HOME" if probability >= 0.5 else "AWAY",
        feature_source=source,
        actual_home_win=actual_home_win,
    )


def get_slate_predictions(project_root: Path, date: str) -> list[MatchupPrediction]:
    """Predict every real regular-season game on one calendar date."""

    season = season_for_date(pd.Timestamp(date))
    schedule = fetch_schedule(season=season)
    regular_season = filter_regular_season_games(schedule)
    day_slate = games_on_date(regular_season, date)

    predictions = []

    for _, game in day_slate.iterrows():
        predictions.append(
            get_matchup_prediction(
                project_root=project_root,
                home_team_id=int(game["HOME_TEAM_ID"]),
                away_team_id=int(game["AWAY_TEAM_ID"]),
                game_date=str(game["GAME_DATE"].date()),
                season=season,
                game_id=game["GAME_ID"],
                home_team_name=game["HOME_TEAM_NAME"],
                away_team_name=game["AWAY_TEAM_NAME"],
            )
        )

    return predictions


app = FastAPI(
    title="NBA Win Predictor API",
    description="Pregame home-win probability from the Phase 8 frozen model.",
)


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness check."""

    return {"status": "ok"}


@app.get("/predict", response_model=MatchupPrediction)
def predict(
    home_team_id: int = Query(..., description="NBA.com home team ID"),
    away_team_id: int = Query(..., description="NBA.com away team ID"),
    game_date: str = Query(..., description="YYYY-MM-DD"),
    season: str | None = Query(None, description="NBA season, e.g. 2025-26. Inferred if omitted."),
) -> MatchupPrediction:
    """Predict one matchup, by team ID and date."""

    resolved_season = season or season_for_date(pd.Timestamp(game_date))

    try:
        return get_matchup_prediction(
            PROJECT_ROOT,
            home_team_id,
            away_team_id,
            game_date,
            resolved_season,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.get("/slate", response_model=list[MatchupPrediction])
def slate(date: str = Query(..., description="YYYY-MM-DD")) -> list[MatchupPrediction]:
    """Predict every real regular-season game on one date."""

    try:
        return get_slate_predictions(PROJECT_ROOT, date)
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
