"""SQLite persistence for the dashboard's pick-the-winner mini game.

Tracks, per game: what the user picked, what the frozen model picked, and
(once known) who actually won -- so accuracy can be compared over time.
The schema lives in sql/picks_schema.sql as the readable source of truth;
this module only applies it.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import pandas as pd

VALID_SIDES = ("HOME", "AWAY")

# Unlike data files, the schema is a fixed, committed repo asset -- it does
# not live under a configurable project_root the way generated data does,
# so it is resolved relative to this module rather than taking project_root
# as a parameter. That keeps tests free to point project_root at a tmp_path
# for the database file while still loading the one real schema.
SCHEMA_PATH: Final[Path] = Path(__file__).resolve().parents[2] / "sql" / "picks_schema.sql"


def picks_db_path(project_root: Path) -> Path:
    """Return the SQLite database path (gitignored, generated locally)."""

    return project_root / "artifacts" / "nba" / "dashboard" / "picks.db"


def init_db(project_root: Path) -> Path:
    """Create the picks database and table if they do not exist yet."""

    db_path = picks_db_path(project_root)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    schema_sql = SCHEMA_PATH.read_text(encoding="utf-8")

    with closing(sqlite3.connect(db_path)) as connection:
        connection.executescript(schema_sql)
        connection.commit()

    return db_path


@dataclass(frozen=True)
class Pick:
    """One recorded pick: the user's, the model's, and (maybe) reality's."""

    game_id: str
    game_date: str
    season: str
    home_team_id: int
    home_team_name: str
    away_team_id: int
    away_team_name: str
    user_pick: str
    model_pick: str
    model_home_win_probability: float
    feature_source: str
    actual_winner: str | None = None


@dataclass(frozen=True)
class Scoreboard:
    """Summarize accuracy across every pick with a known outcome."""

    resolved_picks: int
    user_correct: int
    model_correct: int
    user_accuracy: float | None
    model_accuracy: float | None


def record_pick(project_root: Path, pick: Pick) -> None:
    """Insert one pick. A game already picked is left untouched."""

    if pick.user_pick not in VALID_SIDES:
        raise ValueError(f"user_pick must be one of {VALID_SIDES}, got {pick.user_pick!r}")

    if pick.model_pick not in VALID_SIDES:
        raise ValueError(f"model_pick must be one of {VALID_SIDES}, got {pick.model_pick!r}")

    db_path = init_db(project_root)

    with closing(sqlite3.connect(db_path)) as connection:
        connection.execute(
            """
            INSERT OR IGNORE INTO picks (
                game_id, game_date, season, home_team_id, home_team_name,
                away_team_id, away_team_name, user_pick, model_pick,
                model_home_win_probability, feature_source, actual_winner, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                pick.game_id,
                pick.game_date,
                pick.season,
                pick.home_team_id,
                pick.home_team_name,
                pick.away_team_id,
                pick.away_team_name,
                pick.user_pick,
                pick.model_pick,
                pick.model_home_win_probability,
                pick.feature_source,
                pick.actual_winner,
                datetime.now(UTC).isoformat(),
            ),
        )
        connection.commit()


def get_pick(project_root: Path, game_id: str) -> dict[str, object] | None:
    """Return the existing pick for one game, if the user already made one."""

    db_path = init_db(project_root)

    with closing(sqlite3.connect(db_path)) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            "SELECT * FROM picks WHERE game_id = ?",
            (game_id,),
        ).fetchone()

    return dict(row) if row is not None else None


def update_actual_result(project_root: Path, game_id: str, actual_winner: str) -> None:
    """Record the real outcome for a previously picked game."""

    if actual_winner not in VALID_SIDES:
        raise ValueError(f"actual_winner must be one of {VALID_SIDES}, got {actual_winner!r}")

    db_path = init_db(project_root)

    with closing(sqlite3.connect(db_path)) as connection:
        connection.execute(
            "UPDATE picks SET actual_winner = ? WHERE game_id = ?",
            (actual_winner, game_id),
        )
        connection.commit()


def get_all_picks(project_root: Path) -> pd.DataFrame:
    """Return every recorded pick, most recent first."""

    db_path = init_db(project_root)

    with closing(sqlite3.connect(db_path)) as connection:
        return pd.read_sql_query("SELECT * FROM picks ORDER BY created_at DESC", connection)


def compute_scoreboard(project_root: Path) -> Scoreboard:
    """Summarize user vs. model accuracy across every resolved pick."""

    picks = get_all_picks(project_root)
    resolved = picks.loc[picks["actual_winner"].notna()]

    resolved_count = len(resolved)
    user_correct = int((resolved["user_pick"] == resolved["actual_winner"]).sum())
    model_correct = int((resolved["model_pick"] == resolved["actual_winner"]).sum())

    return Scoreboard(
        resolved_picks=resolved_count,
        user_correct=user_correct,
        model_correct=model_correct,
        user_accuracy=(user_correct / resolved_count) if resolved_count else None,
        model_accuracy=(model_correct / resolved_count) if resolved_count else None,
    )
