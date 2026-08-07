-- Schema for the dashboard's pick-the-winner mini game.
-- Applied by app/dashboard/picks_store.py:init_db(); kept here as the
-- readable source of truth rather than only inline in Python.

CREATE TABLE IF NOT EXISTS picks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id TEXT NOT NULL UNIQUE,
    game_date TEXT NOT NULL,
    season TEXT NOT NULL,
    home_team_id INTEGER NOT NULL,
    home_team_name TEXT NOT NULL,
    away_team_id INTEGER NOT NULL,
    away_team_name TEXT NOT NULL,
    user_pick TEXT NOT NULL CHECK (user_pick IN ('HOME', 'AWAY')),
    model_pick TEXT NOT NULL CHECK (model_pick IN ('HOME', 'AWAY')),
    model_home_win_probability REAL NOT NULL CHECK (
        model_home_win_probability >= 0.0 AND model_home_win_probability <= 1.0
    ),
    feature_source TEXT NOT NULL CHECK (feature_source IN ('historical', 'computed')),
    actual_winner TEXT CHECK (actual_winner IN ('HOME', 'AWAY')),
    created_at TEXT NOT NULL
);
