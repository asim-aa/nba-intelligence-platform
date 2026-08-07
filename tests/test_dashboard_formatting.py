"""Tests for the dashboard's stat-comparison formatting helpers."""

import numpy as np
import pytest
from app.dashboard.app import format_record, format_stat


def test_format_record_converts_win_pct_to_win_loss() -> None:
    assert format_record(0.7, 10) == "7-3"
    assert format_record(0.0, 5) == "0-5"
    assert format_record(1.0, 5) == "5-0"


def test_format_record_handles_missing_value() -> None:
    assert format_record(np.nan, 5) == "—"


def test_format_record_handles_zero_games() -> None:
    assert format_record(0.5, 0) == "—"


def test_format_stat_formats_with_default_precision() -> None:
    assert format_stat(112.345) == "112.3"


def test_format_stat_respects_decimals_argument() -> None:
    assert format_stat(1580.4, decimals=0) == "1580"


def test_format_stat_handles_missing_value() -> None:
    assert format_stat(np.nan) == "—"


@pytest.mark.parametrize("win_pct,window", [(0.55, 10), (0.333, 3), (0.9, 1)])
def test_format_record_wins_plus_losses_equals_window(win_pct: float, window: int) -> None:
    record = format_record(win_pct, window)
    wins, losses = (int(part) for part in record.split("-"))
    assert wins + losses == window
