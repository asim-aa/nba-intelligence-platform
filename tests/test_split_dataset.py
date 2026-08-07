"""Tests for the chronological train/validation/test split.

This is the module that enforces the project's core anti-leakage
guarantee -- games stay in time order and no season appears in more than
one partition -- so it gets exercised more thoroughly than a typical
module: every validation gate, plus a deliberately pathological case
that proves the chronological-overlap check actually fires rather than
being trivially true given the current season configuration.
"""

import json
from pathlib import Path

import pandas as pd
import pytest
from modeling.data.split_dataset import (
    TEST_SEASONS,
    TRAIN_SEASONS,
    VALIDATION_SEASONS,
    build_chronological_splits,
    build_partition_summary,
    modeling_dataset_input_path,
    split_dataset,
    split_output_directory,
    split_summary_output_path,
    train_output_path,
    validate_source_dataset,
    validate_split_configuration,
    validation_output_path,
    write_split_outputs,
)
from modeling.data.split_dataset import test_output_path as split_test_output_path

ALL_CONFIGURED_SEASONS = (*TRAIN_SEASONS, *VALIDATION_SEASONS, *TEST_SEASONS)


def make_game_row(season: str, index: int, game_date: str, home_win: int) -> dict:
    return {
        "SEASON": season,
        "SEASON_ID": f"2{season[:4]}",
        "GAME_ID": f"{season.replace('-', '')}{index:04d}",
        "GAME_DATE": pd.Timestamp(game_date),
        "HOME_TEAM_ID": 1610612747,
        "AWAY_TEAM_ID": 1610612738,
        "home_win": home_win,
    }


def make_full_dataset(games_per_season: int = 2) -> pd.DataFrame:
    """Build a synthetic dataset covering every configured season.

    Each season's games are dated in late October of its own start year,
    so seasons are naturally in chronological order without any special
    casing -- exactly like the real historical data.
    """

    rows = []

    for season in ALL_CONFIGURED_SEASONS:
        start_year = int(season[:4])

        for index in range(games_per_season):
            rows.append(
                make_game_row(
                    season=season,
                    index=index,
                    game_date=f"{start_year}-10-{25 + index}",
                    home_win=index % 2,
                )
            )

    return pd.DataFrame(rows)


def test_split_dataset_creates_correct_partitions() -> None:
    dataset = make_full_dataset(games_per_season=2)

    train, validation, test, summary = split_dataset(dataset)

    assert len(train) == len(TRAIN_SEASONS) * 2
    assert len(validation) == len(VALIDATION_SEASONS) * 2
    assert len(test) == len(TEST_SEASONS) * 2
    assert summary.source_rows == len(dataset)

    assert set(train["SEASON"].unique()) == set(TRAIN_SEASONS)
    assert set(validation["SEASON"].unique()) == set(VALIDATION_SEASONS)
    assert set(test["SEASON"].unique()) == set(TEST_SEASONS)


def test_split_dataset_partitions_are_chronologically_sorted() -> None:
    dataset = make_full_dataset(games_per_season=3)

    train, validation, test, _ = split_dataset(dataset)

    for partition in (train, validation, test):
        assert partition["GAME_DATE"].is_monotonic_increasing


def test_split_dataset_rejects_missing_required_columns() -> None:
    dataset = make_full_dataset().drop(columns=["HOME_TEAM_ID"])

    with pytest.raises(ValueError, match="missing required columns"):
        split_dataset(dataset)


def test_split_dataset_rejects_empty_dataset() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        split_dataset(make_full_dataset().iloc[0:0])


def test_split_dataset_rejects_duplicate_games() -> None:
    dataset = make_full_dataset()
    dataset = pd.concat([dataset, dataset.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate games"):
        split_dataset(dataset)


def test_split_dataset_rejects_missing_game_date() -> None:
    dataset = make_full_dataset()
    dataset.loc[0, "GAME_DATE"] = pd.NaT

    with pytest.raises(ValueError, match="missing GAME_DATE"):
        split_dataset(dataset)


def test_split_dataset_rejects_invalid_home_win() -> None:
    dataset = make_full_dataset()
    dataset.loc[0, "home_win"] = 2

    with pytest.raises(ValueError, match="outside \\{0, 1\\}"):
        split_dataset(dataset)


def test_split_dataset_rejects_dataset_missing_a_configured_season() -> None:
    dataset = make_full_dataset()
    dataset = dataset.loc[dataset["SEASON"] != TRAIN_SEASONS[0]]

    with pytest.raises(ValueError, match="do not match the split configuration"):
        split_dataset(dataset)


def test_split_dataset_rejects_unexpected_season() -> None:
    dataset = make_full_dataset()
    extra_season = "1996-97"
    dataset = pd.concat(
        [dataset, pd.DataFrame([make_game_row(extra_season, 0, "1996-11-01", 1)])],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="do not match the split configuration"):
        split_dataset(dataset)


def test_split_dataset_detects_chronological_overlap() -> None:
    """A train-season game dated after validation must be rejected, even
    though its SEASON label is a valid configured training season --
    proving the date check is real, not just season-label bookkeeping.
    """

    dataset = make_full_dataset(games_per_season=2)

    last_train_season = TRAIN_SEASONS[-1]
    row_to_break = dataset.index[
        (dataset["SEASON"] == last_train_season) & (dataset["HOME_TEAM_ID"] == 1610612747)
    ][0]
    dataset.loc[row_to_break, "GAME_DATE"] = pd.Timestamp("2099-01-01")

    with pytest.raises(ValueError, match="Training dates overlap or follow validation dates"):
        split_dataset(dataset)


def test_validate_source_dataset_rejects_missing_columns() -> None:
    dataset = make_full_dataset().drop(columns=["GAME_ID"])

    with pytest.raises(ValueError, match="missing required columns"):
        validate_source_dataset(dataset)


def test_validate_source_dataset_rejects_empty() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        validate_source_dataset(make_full_dataset().iloc[0:0])


def test_validate_split_configuration_passes_for_current_seasons() -> None:
    validate_split_configuration()  # Should not raise.


def test_build_partition_summary_computes_correct_stats() -> None:
    partition = pd.DataFrame(
        [
            make_game_row("2023-24", 0, "2023-10-25", home_win=1),
            make_game_row("2023-24", 1, "2023-10-27", home_win=0),
            make_game_row("2023-24", 2, "2023-10-29", home_win=1),
        ]
    )

    summary = build_partition_summary(
        name="validation",
        seasons=("2023-24",),
        partition=partition,
    )

    assert summary.rows == 3
    assert summary.home_wins == 2
    assert summary.away_wins == 1
    assert summary.home_win_rate == pytest.approx(2 / 3)
    assert summary.first_game_date == "2023-10-25"
    assert summary.last_game_date == "2023-10-29"


def test_write_split_outputs_persists_all_files(tmp_path: Path) -> None:
    dataset = make_full_dataset()
    train, validation, test, summary = split_dataset(dataset)

    train_path, validation_path, test_path, summary_path = write_split_outputs(
        train=train,
        validation=validation,
        test=test,
        summary=summary,
        project_root=tmp_path,
    )

    assert train_path == train_output_path(tmp_path)
    assert validation_path == validation_output_path(tmp_path)
    assert test_path == split_test_output_path(tmp_path)
    assert summary_path == split_summary_output_path(tmp_path)

    assert pd.read_parquet(train_path).equals(train)
    assert pd.read_parquet(validation_path).equals(validation)
    assert pd.read_parquet(test_path).equals(test)

    saved_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert saved_summary["source_rows"] == len(dataset)


def test_build_chronological_splits_end_to_end(tmp_path: Path) -> None:
    dataset = make_full_dataset()
    input_path = modeling_dataset_input_path(tmp_path)
    input_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(input_path, index=False)

    summary = build_chronological_splits(project_root=tmp_path)

    assert summary.source_rows == len(dataset)
    assert train_output_path(tmp_path).exists()
    assert validation_output_path(tmp_path).exists()
    assert split_test_output_path(tmp_path).exists()
    assert split_summary_output_path(tmp_path).exists()


def test_build_chronological_splits_requires_input_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Modeling dataset does not exist"):
        build_chronological_splits(project_root=tmp_path)


def test_path_helpers_use_expected_locations(tmp_path: Path) -> None:
    splits_dir = split_output_directory(tmp_path)

    assert splits_dir == tmp_path / "data" / "processed" / "nba" / "modeling" / "splits"
    assert train_output_path(tmp_path) == splits_dir / "train.parquet"
    assert validation_output_path(tmp_path) == splits_dir / "validation.parquet"
    assert split_test_output_path(tmp_path) == splits_dir / "test.parquet"
    assert split_summary_output_path(tmp_path) == splits_dir / "split_summary.json"
    assert modeling_dataset_input_path(tmp_path) == (
        tmp_path / "data" / "processed" / "nba" / "modeling" / "pregame_modeling_dataset.parquet"
    )
