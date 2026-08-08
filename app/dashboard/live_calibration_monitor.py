"""Monitor whether the live model stays calibrated, using real picks.

Phase 8 measured calibration once, on a frozen historical test set.
Nothing else in this project looks at whether that calibration holds up
for real, live predictions -- not because it can't be checked, but because
nobody had summarized the data for it. That data already exists: every
pick recorded through the dashboard (picks_store.py) stores the model's
predicted probability alongside the real outcome, once known.

This reads that same picks database, splits resolved picks by
feature_source, and scores each group with the exact same probability
metrics Phase 8 used -- next to the frozen test-set reference numbers, for
comparison. This is the part of "no monitoring" that's actually
addressable without a deployed service: there's no live server to
instrument, but there is a growing table of real predictions with real
resolved outcomes, and until now nothing computed anything from it beyond
plain accuracy (see picks_store.compute_scoreboard).

The feature_source="computed" group deserves particular attention: it is
the upcoming-matchup code path that, per docs/roadmap.md's known gaps, has
only ever been exercised against synthetic test fixtures. The first real
games run through it and resolved here are also the first real-world
evidence of whether that path's calibration matches the frozen model's.

Below MIN_RESOLVED_PICKS_FOR_SIGNAL, or when a group's outcomes are all
one class, this reports the sample size honestly rather than compute
metrics that would just be noise dressed up as a number.

Run with: uv run python -m app.dashboard.live_calibration_monitor
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

import pandas as pd
from modeling.evaluation.probability_metrics import evaluate_probabilities, metrics_to_record
from modeling.evaluation.run_final_evaluation import final_evaluation_summary_path

from app.dashboard.picks_store import get_all_picks

MODEL_NAME: Final[str] = "logistic_regression"

# Below this many resolved picks, per-sample noise dominates any
# probability metric -- reporting a number here would look precise
# without being meaningful.
MIN_RESOLVED_PICKS_FOR_SIGNAL: Final[int] = 30


@dataclass(frozen=True)
class LiveCalibrationResult:
    """One feature_source group's live calibration, or why it couldn't be scored."""

    feature_source: str
    resolved_picks: int
    sufficient_sample: bool
    reason: str | None
    metrics: dict[str, object] | None


def resolved_picks_with_outcomes(project_root: Path) -> pd.DataFrame:
    """Return every resolved pick with a numeric home_win outcome column."""

    picks = get_all_picks(project_root)
    resolved = picks.loc[picks["actual_winner"].notna()].copy()
    resolved["home_win"] = resolved["actual_winner"].eq("HOME").astype("int64")

    return resolved


def evaluate_live_group(feature_source: str, subset: pd.DataFrame) -> LiveCalibrationResult:
    """Score one feature_source group, or explain why it can't be scored yet."""

    rows = len(subset)

    if rows < MIN_RESOLVED_PICKS_FOR_SIGNAL:
        return LiveCalibrationResult(
            feature_source=feature_source,
            resolved_picks=rows,
            sufficient_sample=False,
            reason=f"fewer than {MIN_RESOLVED_PICKS_FOR_SIGNAL} resolved picks",
            metrics=None,
        )

    if subset["home_win"].nunique() < 2:
        return LiveCalibrationResult(
            feature_source=feature_source,
            resolved_picks=rows,
            sufficient_sample=False,
            reason="every resolved pick in this group has the same outcome",
            metrics=None,
        )

    metrics = evaluate_probabilities(
        model_name=MODEL_NAME,
        split_name=f"live_{feature_source}",
        targets=subset["home_win"],
        probabilities=subset["model_home_win_probability"].to_numpy(dtype="float64"),
    )

    return LiveCalibrationResult(
        feature_source=feature_source,
        resolved_picks=rows,
        sufficient_sample=True,
        reason=None,
        metrics=metrics_to_record(metrics),
    )


def run_live_calibration_monitor(project_root: Path) -> list[LiveCalibrationResult]:
    """Score every feature_source group of resolved, real-world picks."""

    resolved = resolved_picks_with_outcomes(project_root)

    results = [evaluate_live_group("all", resolved)]

    for feature_source in ("historical", "computed"):
        subset = resolved.loc[resolved["feature_source"].eq(feature_source)]
        results.append(evaluate_live_group(feature_source, subset))

    return results


def load_frozen_reference_metrics(project_root: Path) -> dict[str, object] | None:
    """Load Phase 8's frozen test-set metrics for the selected model, if run."""

    summary_path = final_evaluation_summary_path(project_root)

    if not summary_path.exists():
        return None

    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    return summary["selected_model_test_metrics"]


def live_calibration_summary_path(project_root: Path) -> Path:
    """Return the live monitoring summary path."""

    return project_root / "artifacts" / "nba" / "live_monitoring" / "live_calibration_summary.json"


def write_summary(
    results: list[LiveCalibrationResult],
    frozen_reference: dict[str, object] | None,
    project_root: Path,
) -> Path:
    """Persist the live monitoring summary as JSON."""

    output_path = live_calibration_summary_path(project_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "note": (
            "Real predictions and real resolved outcomes recorded by the "
            "dashboard, scored with the same metrics Phase 8 used on the "
            "frozen historical test set -- not a retraining or a new look "
            "at the held-out test split."
        ),
        "frozen_test_reference": frozen_reference,
        "live_groups": [asdict(result) for result in results],
    }

    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    return output_path


def print_report(
    results: list[LiveCalibrationResult],
    frozen_reference: dict[str, object] | None,
) -> None:
    """Print a readable live-vs-frozen calibration report."""

    print("\nLive calibration monitor (real dashboard picks vs. frozen Phase 8 test set):")

    if frozen_reference is not None:
        print(
            f"  Frozen test reference: log_loss={frozen_reference['log_loss']:.4f}  "
            f"brier={frozen_reference['brier_score']:.4f}  "
            f"ece={frozen_reference['expected_calibration_error']:.4f}  "
            f"(n={frozen_reference['rows']})"
        )
    else:
        print("  Frozen test reference: unavailable (run run_final_evaluation.py first)")

    for result in results:
        if not result.sufficient_sample:
            print(
                f"  {result.feature_source:<12} {result.resolved_picks:>4} resolved -- "
                f"not enough signal yet ({result.reason})"
            )
            continue

        metrics = result.metrics
        print(
            f"  {result.feature_source:<12} {result.resolved_picks:>4} resolved -- "
            f"log_loss={metrics['log_loss']:.4f}  brier={metrics['brier_score']:.4f}  "
            f"ece={metrics['expected_calibration_error']:.4f}"
        )


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the live calibration monitor."""

    return argparse.ArgumentParser(description=__doc__).parse_args()


def main() -> None:
    """Run the live calibration monitor from the command line."""

    parse_args()
    project_root = Path(__file__).resolve().parents[2]

    results = run_live_calibration_monitor(project_root)
    frozen_reference = load_frozen_reference_metrics(project_root)
    output_path = write_summary(results, frozen_reference, project_root)

    print_report(results, frozen_reference)
    print(f"\nSummary: {output_path}")


if __name__ == "__main__":
    main()
