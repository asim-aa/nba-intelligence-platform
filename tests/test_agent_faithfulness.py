"""Tests for the numbers-faithfulness check."""

import pytest
from app.agent.faithfulness import check_faithfulness, is_grounded, numbers_in, numeric_values

PREDICTION = {
    "home_team_name": "Los Angeles Lakers",
    "away_team_name": "Boston Celtics",
    "game_date": "2025-11-07",
    "season": "2025-26",
    "home_win_probability": 0.6234,
    "model_pick": "HOME",
}
CALLS = [({"home_team": "Lakers", "game_date": "2025-11-07"}, PREDICTION)]


def check(answer: str, user: str = "Who wins?", calls: list = CALLS) -> list[str]:
    return check_faithfulness(answer, [user], calls)


def test_numbers_in_skips_dates_seasons_and_list_markers() -> None:
    text = "1. On 2025-11-07 (2025-26) the Lakers are 62.3% likely, 1,230 games, -4 pts"

    assert numbers_in(text) == ["62.3%", "1,230", "-4"]


def test_numeric_values_collects_leaves_string_numbers_and_list_lengths() -> None:
    result = {"a": 0.5, "flag": True, "none": None, "label": "fewer than 30 picks", "rows": [1, 2]}

    assert sorted(numeric_values(result)) == [0.5, 1.0, 2.0, 2.0, 30.0]


@pytest.mark.parametrize("token", ["62%", "62.3%", "0.62", "0.623", "38%", "37.7%", "0.38"])
def test_a_probability_is_grounded_as_a_fraction_percent_or_complement(token: str) -> None:
    assert is_grounded(token, {0.6234})


@pytest.mark.parametrize("token", ["63%", "62.4%", "0.65", "70%", "0.5"])
def test_a_figure_that_does_not_round_to_a_grounded_value_is_not(token: str) -> None:
    assert not is_grounded(token, {0.6234})


def test_a_correct_answer_has_nothing_ungrounded() -> None:
    answer = (
        "On 2025-11-07 the model favours the Lakers at 62% (so about 38% for the Celtics) "
        "in the 2025-26 season."
    )

    assert check(answer) == []


def test_an_invented_number_is_reported() -> None:
    assert check("The Lakers win 71% of the time and average 118.2 points.") == ["71%", "118.2"]


def test_each_ungrounded_figure_is_reported_once_in_order() -> None:
    assert check("It is 71%, or 71% again, then 9.9.") == ["71%", "9.9"]


def test_an_invented_date_or_season_is_reported() -> None:
    answer = "That was 2024-25, and the game is on 2025-12-01."

    assert check(answer) == ["2024-25", "2025-12-01"]


def test_numbers_the_user_wrote_are_grounded() -> None:
    assert check("Looking at the last 3 cutoffs.", user="Compare the last 3 cutoffs") == []


def test_list_lengths_count_as_grounded_counts() -> None:
    calls = [({}, {"games": [{}, {}, {}, {}, {}]})]

    assert check("There are 5 games tonight.", calls=calls) == []
    assert check("There are 6 games tonight.", calls=calls) == ["6"]


def test_numbers_inside_tool_arguments_are_grounded() -> None:
    calls = [({"last_n": 3}, {"folds": []})]

    assert check("Showing the 3 most recent seasons.", calls=calls) == []


def test_a_number_with_no_tool_results_at_all_is_reported() -> None:
    assert check("The Lakers have an 80% chance.", calls=[]) == ["80%"]


def test_an_answer_with_no_numbers_is_clean() -> None:
    assert check("The model favours the Lakers.") == []
