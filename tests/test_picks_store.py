"""Tests for the dashboard's SQLite picks store."""

from pathlib import Path

import pytest
from app.dashboard.picks_store import (
    Pick,
    compute_scoreboard,
    get_all_picks,
    get_pick,
    init_db,
    picks_db_path,
    record_pick,
    update_actual_result,
)


def make_pick(
    game_id: str = "0022500001", user_pick: str = "HOME", model_pick: str = "HOME"
) -> Pick:
    return Pick(
        game_id=game_id,
        game_date="2025-11-05",
        season="2025-26",
        home_team_id=1610612747,
        home_team_name="Los Angeles Lakers",
        away_team_id=1610612738,
        away_team_name="Boston Celtics",
        user_pick=user_pick,
        model_pick=model_pick,
        model_home_win_probability=0.62,
        feature_source="historical",
    )


def test_init_db_creates_database_file(tmp_path: Path) -> None:
    db_path = init_db(tmp_path)

    assert db_path == picks_db_path(tmp_path)
    assert db_path.exists()


def test_record_and_get_pick_round_trip(tmp_path: Path) -> None:
    record_pick(tmp_path, make_pick())

    stored = get_pick(tmp_path, "0022500001")

    assert stored is not None
    assert stored["user_pick"] == "HOME"
    assert stored["model_pick"] == "HOME"
    assert stored["model_home_win_probability"] == pytest.approx(0.62)
    assert stored["actual_winner"] is None


def test_get_pick_returns_none_for_unknown_game(tmp_path: Path) -> None:
    assert get_pick(tmp_path, "does-not-exist") is None


def test_record_pick_rejects_invalid_side(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="user_pick"):
        record_pick(tmp_path, make_pick(user_pick="HOMETEAM"))


def test_record_pick_ignores_duplicate_game_id(tmp_path: Path) -> None:
    record_pick(tmp_path, make_pick(user_pick="HOME"))
    record_pick(tmp_path, make_pick(user_pick="AWAY"))  # Second pick for the same game.

    stored = get_pick(tmp_path, "0022500001")

    assert stored["user_pick"] == "HOME"  # First pick wins; second is ignored.


def test_update_actual_result_sets_winner(tmp_path: Path) -> None:
    record_pick(tmp_path, make_pick())
    update_actual_result(tmp_path, "0022500001", "AWAY")

    stored = get_pick(tmp_path, "0022500001")

    assert stored["actual_winner"] == "AWAY"


def test_update_actual_result_rejects_invalid_side(tmp_path: Path) -> None:
    record_pick(tmp_path, make_pick())

    with pytest.raises(ValueError, match="actual_winner"):
        update_actual_result(tmp_path, "0022500001", "TIE")


def test_get_all_picks_returns_every_pick(tmp_path: Path) -> None:
    record_pick(tmp_path, make_pick(game_id="0022500001"))
    record_pick(tmp_path, make_pick(game_id="0022500002"))

    picks = get_all_picks(tmp_path)

    assert len(picks) == 2


def test_compute_scoreboard_counts_only_resolved_picks(tmp_path: Path) -> None:
    # User right, model right.
    record_pick(tmp_path, make_pick(game_id="g1", user_pick="HOME", model_pick="HOME"))
    update_actual_result(tmp_path, "g1", "HOME")

    # User wrong, model right.
    record_pick(tmp_path, make_pick(game_id="g2", user_pick="AWAY", model_pick="HOME"))
    update_actual_result(tmp_path, "g2", "HOME")

    # Not yet resolved -- should not count.
    record_pick(tmp_path, make_pick(game_id="g3", user_pick="HOME", model_pick="AWAY"))

    scoreboard = compute_scoreboard(tmp_path)

    assert scoreboard.resolved_picks == 2
    assert scoreboard.user_correct == 1
    assert scoreboard.model_correct == 2
    assert scoreboard.user_accuracy == pytest.approx(0.5)
    assert scoreboard.model_accuracy == pytest.approx(1.0)


def test_compute_scoreboard_handles_no_resolved_picks(tmp_path: Path) -> None:
    record_pick(tmp_path, make_pick())

    scoreboard = compute_scoreboard(tmp_path)

    assert scoreboard.resolved_picks == 0
    assert scoreboard.user_accuracy is None
    assert scoreboard.model_accuracy is None
