"""Tests for the prediction-serving API.

Business logic (get_matchup_prediction, get_slate_predictions) is tested
directly against tmp_path fixtures, same as every other module. The
FastAPI routes are tested through TestClient with that same business
logic monkeypatched out, so route-wiring tests (status codes, response
shape) don't depend on real trained-model data being present on disk.
"""

from pathlib import Path

import app.api.main as api_module
import numpy as np
import pandas as pd
import pytest
from app.api.main import MatchupPrediction, app, get_matchup_prediction, get_slate_predictions
from fastapi.testclient import TestClient
from modeling.training.train_logistic_regression import (
    train_logistic_regression,
    write_logistic_regression_artifacts,
)
from pipelines.features.build_modeling_dataset import (
    NUMERIC_FEATURE_COLUMNS,
    modeling_dataset_output_path,
)
from pipelines.features.build_team_elo_ratings import elo_ratings_output_path
from pipelines.features.build_team_history import team_history_output_path
from pipelines.ingestion.fetch_schedule import GAME_STATUS_FINAL

TEAM_A = 1610612747
TEAM_B = 1610612738


def make_team_history_rows(
    team_id: int, season: str, dates: list[str], wins: list[int]
) -> list[dict]:
    rows = []
    for date, win in zip(dates, wins, strict=True):
        rows.append(
            {
                "SEASON": season,
                "TEAM_ID": team_id,
                "GAME_DATE": pd.Timestamp(date),
                "TEAM_WIN": win,
                "TEAM_PTS": 110 if win else 100,
                "OPPONENT_PTS": 100 if win else 110,
                "POINT_DIFFERENTIAL": 10 if win else -10,
            }
        )
    return rows


def make_training_frame(rows: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(seed=0)
    win_pct_diff = rng.uniform(-0.4, 0.4, size=rows)
    home_win = rng.binomial(1, 1.0 / (1.0 + np.exp(-(win_pct_diff * 4.0))))

    return pd.DataFrame(
        {
            "SEASON_WIN_PCT_DIFF": win_pct_diff,
            "ROLLING_10_WIN_PCT_DIFF": rng.uniform(-0.4, 0.4, size=rows),
            "ROLLING_10_POINT_DIFFERENTIAL_DIFF": rng.uniform(-10.0, 10.0, size=rows),
            "DAYS_REST_DIFF": rng.integers(-2, 3, size=rows).astype("float64"),
            "IS_BACK_TO_BACK_DIFF": rng.integers(-1, 2, size=rows).astype("float64"),
            "ELO_RATING_DIFF": rng.uniform(-200.0, 200.0, size=rows),
            "home_win": home_win,
        }
    )


def write_serving_fixtures(project_root: Path) -> None:
    """Write a trained model plus minimal Phase 5 outputs the API relies on."""

    train = make_training_frame()
    pipeline, summary = train_logistic_regression(train)
    write_logistic_regression_artifacts(
        pipeline=pipeline, summary=summary, project_root=project_root
    )

    team_history = pd.DataFrame(
        make_team_history_rows(TEAM_A, "2025-26", ["2025-10-22", "2025-10-24"], [1, 0])
        + make_team_history_rows(TEAM_B, "2025-26", ["2025-10-22", "2025-10-25"], [0, 1])
    )
    history_path = team_history_output_path(project_root)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    team_history.to_parquet(history_path, index=False)

    elo_ratings = pd.DataFrame(
        [
            {
                "SEASON": "2025-26",
                "SEASON_ID": "22025",
                "GAME_ID": "0022500001",
                "GAME_DATE": pd.Timestamp("2025-10-24"),
                "TEAM_ID": TEAM_A,
                "ELO_RATING": 1495.0,
                "POST_GAME_ELO_RATING": 1505.0,
            },
            {
                "SEASON": "2025-26",
                "SEASON_ID": "22025",
                "GAME_ID": "0022500002",
                "GAME_DATE": pd.Timestamp("2025-10-25"),
                "TEAM_ID": TEAM_B,
                "ELO_RATING": 1498.0,
                "POST_GAME_ELO_RATING": 1508.0,
            },
        ]
    )
    elo_path = elo_ratings_output_path(project_root)
    elo_path.parent.mkdir(parents=True, exist_ok=True)
    elo_ratings.to_parquet(elo_path, index=False)

    modeling_row = {
        "SEASON": "2025-26",
        "SEASON_ID": "22025",
        "GAME_ID": "0022500001",
        "GAME_DATE": pd.Timestamp("2025-10-22"),
        "HOME_TEAM_ID": TEAM_A,
        "HOME_TEAM_ABBREVIATION": "LAL",
        "AWAY_TEAM_ID": TEAM_B,
        "AWAY_TEAM_ABBREVIATION": "BOS",
        "home_win": 1,
    }
    for column in NUMERIC_FEATURE_COLUMNS:
        modeling_row.setdefault(column, 0.0)

    modeling_path = modeling_dataset_output_path(project_root)
    modeling_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([modeling_row]).to_parquet(modeling_path, index=False)


def make_fake_schedule() -> pd.DataFrame:
    """A schedule containing exactly the historical game the fixtures know."""

    return pd.DataFrame(
        [
            {
                "GAME_ID": "0022500001",
                "GAME_DATE": pd.Timestamp("2025-10-22"),
                "GAME_LABEL": "",
                "GAME_SUB_LABEL": "",
                "GAME_STATUS": GAME_STATUS_FINAL,
                "GAME_STATUS_TEXT": "Final",
                "HOME_TEAM_ID": TEAM_A,
                "HOME_TEAM_TRICODE": "LAL",
                "HOME_TEAM_NAME": "Los Angeles Lakers",
                "HOME_SCORE": "110",
                "AWAY_TEAM_ID": TEAM_B,
                "AWAY_TEAM_TRICODE": "BOS",
                "AWAY_TEAM_NAME": "Boston Celtics",
                "AWAY_SCORE": "100",
            }
        ]
    )


# --- Business logic (tmp_path, no HTTP) -----------------------------------


def test_get_matchup_prediction_uses_historical_lookup(tmp_path: Path) -> None:
    write_serving_fixtures(tmp_path)

    prediction = get_matchup_prediction(tmp_path, TEAM_A, TEAM_B, "2025-10-22", "2025-26")

    assert prediction.feature_source == "historical"
    assert prediction.actual_home_win == 1
    assert prediction.model_pick in {"HOME", "AWAY"}
    assert 0.0 <= prediction.home_win_probability <= 1.0
    assert prediction.selected_model == "logistic_regression"


def test_get_matchup_prediction_computes_for_upcoming_matchup(tmp_path: Path) -> None:
    write_serving_fixtures(tmp_path)

    prediction = get_matchup_prediction(tmp_path, TEAM_A, TEAM_B, "2025-10-30", "2025-26")

    assert prediction.feature_source == "computed"
    assert prediction.actual_home_win is None


def test_get_matchup_prediction_rejects_identical_teams(tmp_path: Path) -> None:
    write_serving_fixtures(tmp_path)

    with pytest.raises(ValueError, match="must differ"):
        get_matchup_prediction(tmp_path, TEAM_A, TEAM_A, "2025-10-22", "2025-26")


def test_get_slate_predictions_returns_one_per_game(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_serving_fixtures(tmp_path)
    monkeypatch.setattr(api_module, "fetch_schedule", lambda season: make_fake_schedule())

    predictions = get_slate_predictions(tmp_path, "2025-10-22")

    assert len(predictions) == 1
    assert predictions[0].home_team_name == "Los Angeles Lakers"
    assert predictions[0].feature_source == "historical"


# --- HTTP routes (TestClient, business logic monkeypatched) ---------------


def test_health_endpoint() -> None:
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_endpoint_returns_prediction(monkeypatch: pytest.MonkeyPatch) -> None:
    canned = MatchupPrediction(
        home_team_id=TEAM_A,
        away_team_id=TEAM_B,
        game_date="2025-10-22",
        season="2025-26",
        selected_model="logistic_regression",
        home_win_probability=0.62,
        model_pick="HOME",
        feature_source="historical",
        actual_home_win=1,
    )
    monkeypatch.setattr(api_module, "get_matchup_prediction", lambda *a, **k: canned)

    client = TestClient(app)
    response = client.get(
        "/predict",
        params={
            "home_team_id": TEAM_A,
            "away_team_id": TEAM_B,
            "game_date": "2025-10-22",
            "season": "2025-26",
        },
    )

    assert response.status_code == 200
    assert response.json()["home_win_probability"] == pytest.approx(0.62)


def test_predict_endpoint_returns_400_for_bad_input(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_value_error(*_args, **_kwargs):
        raise ValueError("home_team_id and away_team_id must differ")

    monkeypatch.setattr(api_module, "get_matchup_prediction", raise_value_error)

    client = TestClient(app)
    response = client.get(
        "/predict",
        params={
            "home_team_id": TEAM_A,
            "away_team_id": TEAM_A,
            "game_date": "2025-10-22",
        },
    )

    assert response.status_code == 400


def test_predict_endpoint_returns_503_when_model_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_missing_model(*_args, **_kwargs):
        raise FileNotFoundError("Selected model artifact does not exist")

    monkeypatch.setattr(api_module, "get_matchup_prediction", raise_missing_model)

    client = TestClient(app)
    response = client.get(
        "/predict",
        params={
            "home_team_id": TEAM_A,
            "away_team_id": TEAM_B,
            "game_date": "2025-10-22",
        },
    )

    assert response.status_code == 503


def test_slate_endpoint_returns_predictions(monkeypatch: pytest.MonkeyPatch) -> None:
    canned = MatchupPrediction(
        home_team_id=TEAM_A,
        away_team_id=TEAM_B,
        game_date="2025-10-22",
        season="2025-26",
        selected_model="logistic_regression",
        home_win_probability=0.55,
        model_pick="HOME",
        feature_source="historical",
        actual_home_win=1,
    )
    monkeypatch.setattr(api_module, "get_slate_predictions", lambda *a, **k: [canned])

    client = TestClient(app)
    response = client.get("/slate", params={"date": "2025-10-22"})

    assert response.status_code == 200
    assert len(response.json()) == 1


def test_slate_endpoint_returns_503_when_data_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_missing_data(*_args, **_kwargs):
        raise FileNotFoundError("Team history does not exist")

    monkeypatch.setattr(api_module, "get_slate_predictions", raise_missing_data)

    client = TestClient(app)
    response = client.get("/slate", params={"date": "2025-10-22"})

    assert response.status_code == 503
