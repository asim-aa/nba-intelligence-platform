import pandas as pd
import pytest
from pipelines.ingestion.audit_nba_source import (
    classify_location,
    summarize,
    validate_schema,
)


def make_valid_frame() -> pd.DataFrame:
    shared = {
        "GAME_DATE": "2024-10-22",
        "SEASON_ID": "22024",
        "WL": "W",
        "PTS": 110,
        "FG_PCT": 0.5,
        "FG3_PCT": 0.4,
        "FT_PCT": 0.8,
        "REB": 45,
        "AST": 25,
        "TOV": 12,
        "STL": 8,
        "BLK": 5,
    }
    return pd.DataFrame(
        [
            {
                **shared,
                "GAME_ID": "0022400061",
                "TEAM_ID": 1,
                "TEAM_ABBREVIATION": "AAA",
                "MATCHUP": "AAA vs. BBB",
            },
            {
                **shared,
                "GAME_ID": "0022400061",
                "TEAM_ID": 2,
                "TEAM_ABBREVIATION": "BBB",
                "MATCHUP": "BBB @ AAA",
                "WL": "L",
                "PTS": 101,
            },
        ]
    )


def make_neutral_frame() -> pd.DataFrame:
    frame = make_valid_frame()
    frame.loc[1, "MATCHUP"] = "BBB vs. AAA"
    return frame


def make_all_away_frame() -> pd.DataFrame:
    frame = make_valid_frame()
    frame.loc[0, "MATCHUP"] = "AAA @ BBB"
    return frame


def test_classify_location() -> None:
    assert classify_location("AAA vs. BBB") == "vs"
    assert classify_location("BBB @ AAA") == "away"


def test_classify_location_rejects_unknown_format() -> None:
    with pytest.raises(ValueError, match="Unrecognized MATCHUP"):
        classify_location("AAA - BBB")


def test_validate_schema_rejects_missing_column() -> None:
    frame = make_valid_frame().drop(columns="BLK")

    with pytest.raises(ValueError, match="BLK"):
        validate_schema(frame)


def test_summarize_valid_game_pair() -> None:
    summary = summarize(make_valid_frame(), season="2024-25")

    assert summary.rows == 2
    assert summary.games == 1
    assert summary.standard_home_rows == 1
    assert summary.away_rows == 1
    assert summary.neutral_site_games == 0
    assert summary.ambiguous_all_away_games == 0
    assert summary.duplicate_team_game_rows == 0
    assert summary.invalid_game_pair_count == 0


def test_summarize_accepts_neutral_site_pair() -> None:
    summary = summarize(make_neutral_frame(), season="2024-25")

    assert summary.games == 1
    assert summary.standard_home_rows == 0
    assert summary.away_rows == 0
    assert summary.neutral_site_games == 1
    assert summary.ambiguous_all_away_games == 0
    assert summary.invalid_game_pair_count == 0


def test_summarize_accepts_all_away_pair() -> None:
    summary = summarize(make_all_away_frame(), season="2024-25")

    assert summary.games == 1
    assert summary.standard_home_rows == 0
    assert summary.away_rows == 2
    assert summary.neutral_site_games == 0
    assert summary.ambiguous_all_away_games == 1
    assert summary.invalid_game_pair_count == 0


def test_summarize_rejects_duplicate_team_game() -> None:
    frame = pd.concat([make_valid_frame(), make_valid_frame().iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate"):
        summarize(frame, season="2024-25")


def test_summarize_rejects_incomplete_pair() -> None:
    with pytest.raises(ValueError, match="invalid team-row pairing"):
        summarize(make_valid_frame().iloc[[0]], season="2024-25")
