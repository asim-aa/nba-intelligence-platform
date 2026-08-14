"""Tests for the leakage-safe roster-continuity feature."""

import numpy as np
import pandas as pd
import pytest
from pipelines.features.build_roster_continuity_features import (
    CONTINUITY_ROLLING_WINDOW,
    CORE_SIZE,
    MINUTES_ROLLING_WINDOW,
    build_roster_continuity_features,
    compute_core_realized_ratio,
    compute_player_rolling_minutes,
    compute_rolling_continuity,
)

SEASON = "2020-21"
TEAM = 100


def date(day: int) -> pd.Timestamp:
    return pd.Timestamp(f"2021-01-{day:02d}")


# --- Stage 1: player rolling minutes -----------------------------------


def test_compute_player_rolling_minutes_is_shift1_safe() -> None:
    rows = pd.DataFrame(
        {
            "SEASON": [SEASON] * 3,
            "TEAM_ID": [TEAM] * 3,
            "PLAYER_ID": [1, 1, 1],
            "GAME_ID": ["G1", "G2", "G3"],
            "GAME_DATE": [date(1), date(2), date(3)],
            "MIN": [30.0, 32.0, 34.0],
        }
    )

    result = compute_player_rolling_minutes(rows).sort_values("GAME_DATE").reset_index(drop=True)

    assert pd.isna(result.loc[0, "PLAYER_ROLLING_MIN"])
    assert result.loc[1, "PLAYER_ROLLING_MIN"] == pytest.approx(30.0)
    assert result.loc[2, "PLAYER_ROLLING_MIN"] == pytest.approx((30.0 + 32.0) / 2)


def test_compute_player_rolling_minutes_resets_per_team() -> None:
    """A player traded mid-season starts a fresh rolling window for the new team."""

    rows = pd.DataFrame(
        {
            "SEASON": [SEASON, SEASON, SEASON],
            "TEAM_ID": [TEAM, TEAM, 200],
            "PLAYER_ID": [1, 1, 1],
            "GAME_ID": ["G1", "G2", "G3"],
            "GAME_DATE": [date(1), date(2), date(3)],
            "MIN": [30.0, 32.0, 20.0],
        }
    )

    result = compute_player_rolling_minutes(rows)
    new_team_row = result.loc[result["TEAM_ID"].eq(200)].iloc[0]

    assert pd.isna(new_team_row["PLAYER_ROLLING_MIN"])


# --- Stage 2: core realized ratio -----------------------------------


def test_compute_core_realized_ratio_full_strength_game() -> None:
    rows = pd.DataFrame(
        {
            "SEASON": [SEASON] * 4,
            "TEAM_ID": [TEAM] * 4,
            "PLAYER_ID": [1, 2, 3, 4],
            "GAME_ID": ["G1"] * 4,
            "GAME_DATE": [date(5)] * 4,
            "MIN": [32.0, 30.0, 27.0, 12.0],
            "PLAYER_ROLLING_MIN": [30.0, 28.0, 26.0, 10.0],
        }
    )

    result = compute_core_realized_ratio(rows)
    row = result.iloc[0]

    expected = (32.0 + 30.0 + 27.0) / (30.0 + 28.0 + 26.0)
    assert row["CORE_MINUTES_REALIZED_RATIO"] == pytest.approx(expected)


def test_compute_core_realized_ratio_detects_missing_star() -> None:
    """Player 1 has the highest established role but sits this game out --
    no row at all for (PLAYER_ID=1, GAME_ID=G2). The core must still be
    identified as players 1/2/3 (their established top-3), not silently
    re-ranked among whoever showed up.
    """

    rows = pd.DataFrame(
        {
            "SEASON": [SEASON] * 6,
            "TEAM_ID": [TEAM] * 6,
            "PLAYER_ID": [1, 2, 3, 4, 2, 3],
            "GAME_ID": ["G1", "G1", "G1", "G1", "G2", "G2"],
            "GAME_DATE": [date(1)] * 4 + [date(2)] * 2,
            "MIN": [30.0, 28.0, 26.0, 10.0, 29.0, 25.0],
            # Established roles as of G2 (player 1 has the highest, 31).
            "PLAYER_ROLLING_MIN": [31.0, 27.0, 25.0, 9.0, 27.0, 25.0],
        }
    )

    result = compute_core_realized_ratio(rows)
    g2_row = result.loc[result["GAME_ID"].eq("G2")].iloc[0]

    # Core (by established role) is players 1, 2, 3: expected 31+27+25=83.
    # Player 1 is absent (0 realized minutes); 2 and 3 play their game-2 minutes.
    expected = (0.0 + 29.0 + 25.0) / (31.0 + 27.0 + 25.0)
    assert g2_row["CORE_MINUTES_REALIZED_RATIO"] == pytest.approx(expected)
    assert expected < 1.0


def test_compute_core_realized_ratio_requires_full_core() -> None:
    """Fewer than CORE_SIZE players with an established role yet -> NaN,
    not a ratio computed from a smaller core.
    """

    rows = pd.DataFrame(
        {
            "SEASON": [SEASON, SEASON],
            "TEAM_ID": [TEAM, TEAM],
            "PLAYER_ID": [1, 2],
            "GAME_ID": ["G1", "G1"],
            "GAME_DATE": [date(1), date(1)],
            "MIN": [30.0, 28.0],
            "PLAYER_ROLLING_MIN": [29.0, 27.0],
        }
    )

    result = compute_core_realized_ratio(rows)

    assert pd.isna(result.iloc[0]["CORE_MINUTES_REALIZED_RATIO"])


def test_compute_core_realized_ratio_keeps_every_team_game_row() -> None:
    """A team's first game (nobody has an established role yet) must still
    appear as a row -- NaN, not missing -- so the rolling window downstream
    has no silent gap in the chronological sequence.
    """

    rows = pd.DataFrame(
        {
            "SEASON": [SEASON, SEASON],
            "TEAM_ID": [TEAM, TEAM],
            "PLAYER_ID": [1, 2],
            "GAME_ID": ["G1", "G1"],
            "GAME_DATE": [date(1), date(1)],
            "MIN": [30.0, 28.0],
            "PLAYER_ROLLING_MIN": [np.nan, np.nan],
        }
    )

    result = compute_core_realized_ratio(rows)

    assert len(result) == 1
    assert result.iloc[0]["GAME_ID"] == "G1"
    assert pd.isna(result.iloc[0]["CORE_MINUTES_REALIZED_RATIO"])


# --- Stage 3: rolling continuity -----------------------------------


def test_compute_rolling_continuity_is_shift1_safe() -> None:
    ratio_table = pd.DataFrame(
        {
            "SEASON": [SEASON] * 3,
            "TEAM_ID": [TEAM] * 3,
            "GAME_ID": ["G1", "G2", "G3"],
            "GAME_DATE": [date(1), date(2), date(3)],
            "CORE_MINUTES_REALIZED_RATIO": [1.0, 0.5, 0.9],
        }
    )

    result = compute_rolling_continuity(ratio_table).sort_values("GAME_DATE").reset_index(drop=True)

    assert pd.isna(result.loc[0, "ROLLING_CORE_CONTINUITY"])
    assert result.loc[1, "ROLLING_CORE_CONTINUITY"] == pytest.approx(1.0)
    assert result.loc[2, "ROLLING_CORE_CONTINUITY"] == pytest.approx((1.0 + 0.5) / 2)


def test_compute_rolling_continuity_ignores_the_current_games_own_ratio() -> None:
    """Changing G3's own ratio must not change G3's own continuity value --
    only later games' values."""

    base = pd.DataFrame(
        {
            "SEASON": [SEASON] * 3,
            "TEAM_ID": [TEAM] * 3,
            "GAME_ID": ["G1", "G2", "G3"],
            "GAME_DATE": [date(1), date(2), date(3)],
            "CORE_MINUTES_REALIZED_RATIO": [1.0, 0.5, 0.9],
        }
    )
    altered = base.copy()
    altered.loc[altered["GAME_ID"].eq("G3"), "CORE_MINUTES_REALIZED_RATIO"] = 0.1

    base_result = compute_rolling_continuity(base).set_index("GAME_ID")
    altered_result = compute_rolling_continuity(altered).set_index("GAME_ID")

    assert base_result.loc["G3", "ROLLING_CORE_CONTINUITY"] == pytest.approx(
        altered_result.loc["G3", "ROLLING_CORE_CONTINUITY"]
    )


# --- End-to-end -----------------------------------


def make_full_player_log() -> pd.DataFrame:
    """A small, realistic player log: 4 players, 3 games, one absence."""

    rows = []
    for game_index, game_id in enumerate(["G1", "G2", "G3"], start=1):
        game_date = date(game_index)
        for player_id, base_min in [(1, 30.0), (2, 28.0), (3, 26.0), (4, 10.0)]:
            if game_id == "G3" and player_id == 1:
                continue  # player 1 sits out game 3
            rows.append(
                {
                    "SEASON_ID": "22020",
                    "PLAYER_ID": player_id,
                    "PLAYER_NAME": f"Player {player_id}",
                    "TEAM_ID": TEAM,
                    "TEAM_ABBREVIATION": "AAA",
                    "GAME_ID": game_id,
                    "GAME_DATE": game_date,
                    "MIN": base_min,
                }
            )
    return pd.DataFrame(rows)


def test_build_roster_continuity_features_end_to_end() -> None:
    player_rows = make_full_player_log()
    player_rows["SEASON"] = SEASON

    output, summary = build_roster_continuity_features(player_rows)

    assert set(output.columns) == {
        "SEASON",
        "SEASON_ID",
        "GAME_ID",
        "GAME_DATE",
        "TEAM_ID",
        "ROLLING_CORE_CONTINUITY",
        "CORE_MINUTES_REALIZED_RATIO",
    }
    assert summary.output_team_game_rows == 3
    assert summary.unique_players == 4
    # G1 is the team's first game: no established core yet -> NaN.
    g1 = output.loc[output["GAME_ID"].eq("G1")].iloc[0]
    assert pd.isna(g1["ROLLING_CORE_CONTINUITY"])


def test_build_roster_continuity_features_rejects_missing_columns() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        build_roster_continuity_features(pd.DataFrame({"SEASON": [SEASON]}))


def test_build_roster_continuity_features_rejects_empty() -> None:
    player_rows = make_full_player_log()
    player_rows["SEASON"] = SEASON

    with pytest.raises(ValueError, match="cannot be empty"):
        build_roster_continuity_features(player_rows.iloc[0:0])


def test_build_roster_continuity_features_no_duplicate_team_game_rows() -> None:
    player_rows = make_full_player_log()
    player_rows["SEASON"] = SEASON

    output, _ = build_roster_continuity_features(player_rows)

    assert output.duplicated(subset=["SEASON", "TEAM_ID", "GAME_ID"]).sum() == 0


def test_module_constants_are_reasonable() -> None:
    assert MINUTES_ROLLING_WINDOW > 0
    assert CORE_SIZE > 0
    assert CONTINUITY_ROLLING_WINDOW > 0
