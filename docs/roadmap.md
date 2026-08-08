# Roadmap and status

The single source of truth for what's built, what isn't, and what's known
to be incomplete. `README.md` is the pitch and the results; this file is
for whoever (including future you) needs to know exactly where things
stand before picking the work back up. **If this file and the README ever
disagree, this file wins** — update it whenever a phase or gap changes,
in the same change that changes the code.

Last updated: 2026-08-08.

## Core prediction pipeline (Phases 0-8) — complete

| Phase | What it delivered | Where |
| --- | --- | --- |
| 0 | Prediction problem, scope, leakage policy | `docs/project_spec.md` |
| 1 | Repository and dev environment | — |
| 2 | Source-design audit, networked sample validator | `pipelines/ingestion/audit_nba_source.py`, `docs/data_source_audit.md`, `docs/phase_2_runbook.md` |
| 3 | Full historical ingestion, 2015-16 through 2025-26 | `pipelines/ingestion/ingest_historical_seasons.py` |
| 4 | Game-level dataset transformation, ambiguous-matchup reconciliation | `pipelines/transform/` |
| 5 | Leakage-safe pregame features, incl. opponent-adjusted Elo ratings | `pipelines/features/`, `docs/feature_engineering.md` |
| 6 | Chronological train/validation/test split, probability metrics | `modeling/data/split_dataset.py`, `modeling/evaluation/probability_metrics.py` |
| 7 | Baselines, logistic regression, CatBoost, validation comparison | `modeling/baselines/`, `modeling/training/`, `modeling/evaluation/run_model_comparison.py` |
| 8 | One-time held-out test evaluation | `modeling/evaluation/run_final_evaluation.py` |

**Selected model**: logistic regression on the compact feature set
(including the Elo rating differential). Chosen on 2023-24 validation
performance, before the test seasons were read. Frozen test result: log
loss 0.5996, Brier 0.2067, ROC-AUC 0.7347, accuracy 0.6789. This decision
is final — per `docs/project_spec.md` section 7 rule 8, the test seasons
are not to be re-evaluated to inform any further modeling choice.

## Application layer — complete

Both pieces below are thin layers over the same two shared modules
(`modeling/serving/` for feature computation and prediction,
`pipelines/ingestion/fetch_schedule.py` for the live NBA schedule) so
neither duplicates the other's logic.

- **Dashboard** (`app/dashboard/`) — interactive pick-the-winner mini
  game. Shows a stat comparison before you pick, tracks your picks vs.
  the model vs. reality in SQLite (`sql/picks_schema.sql`), and has a
  "Live calibration monitor" expander (`live_calibration_monitor.py`)
  scoring real, resolved picks against the frozen test-set reference.
- **API** (`app/api/`) — `GET /predict` (one matchup) and `GET /slate`
  (a day's games), both backed by the frozen model. Run with
  `uv run uvicorn app.api.main:app --reload`.

## Live calibration monitoring — the addressable half of "no monitoring"

There is no deployed service to instrument, so traditional production
monitoring doesn't apply -- but the dashboard's own picks database
(`sql/picks_schema.sql`) already records every real prediction alongside
its real outcome once known, and nothing was summarizing it. Now
`app/dashboard/live_calibration_monitor.py` does: it scores resolved
picks with the same metrics Phase 8 used, grouped by `feature_source`
(`historical` vs. `computed`), next to the frozen test-set reference.
Below 30 resolved picks in a group, or when a group's outcomes are all
one class, it reports the sample size honestly instead of a misleading
number -- as of this writing, real usage has only produced 24 resolved
picks (all `historical`; `computed` has 0), so it is correctly reporting
"not enough signal yet" rather than a number. The `computed` group is the
first place real-world evidence for the untested "upcoming matchup" path
(see the gap below) will ever show up, once the 2026-27 season starts and
picks accumulate against it.

## Robustness and calibration diagnostics — supplementary, not Phase 8

Two additional analyses go deeper than the single Phase 8 number without
reopening or superseding it — neither retunes anything, and neither
changes the selected model or its official test result:

- **`modeling/evaluation/run_robustness_backtest.py`** — walks the exact
  same logistic regression (same features, same fixed hyperparameters, no
  search) forward across 8 expanding-window season cutoffs (2018-19
  through 2025-26), instead of the single 2024-25/2025-26 cutoff Phase 8
  uses. Finding: performance is **not** stable across cutoffs — accuracy
  ranges from 0.621 (2020-21 cutoff) to 0.682 (2025-26 cutoff), log loss
  from 0.596 to 0.647. The officially reported Phase 8 result sits near
  the *favorable* end of that spread, not the middle, which is useful
  context missing from the single-number headline. (As a sanity check,
  this backtest's own 2024-25 and 2025-26 folds bracket the official
  Phase 8 test log loss almost exactly, which is reassuring but not a
  substitute for the deliberately narrower official result.)
- **`modeling/evaluation/run_calibration_by_segment.py`** — re-slices the
  same frozen test-set predictions Phase 8 already scored once, broken
  down by season phase and rest status. Finding: back-to-back games have
  more than double the calibration error of rested games (ECE 0.0395 vs.
  0.0180), and late-season games (51+ games played) have the best log
  loss/Brier score but the *worst* calibration error (0.0722) — the model
  is more accurate late in the season but its stated probabilities are
  less trustworthy at face value then, which the single test-set-wide ECE
  in the README doesn't surface.

## Known gaps (honest, as of this writing)

- `sql/` only has `picks_schema.sql` for the dashboard; no other use yet.
- No CI job exercises the dashboard, the API, or the live NBA.com
  endpoints — by design, since those need network access CI shouldn't
  depend on. Only the synthetic-fixture pytest suite (249 tests as of
  this writing) runs in CI.
- No deployment story. Both the API and dashboard are meant to run
  locally; nothing here provisions hosting, TLS, or auth.
- The 2026-27 season hasn't started, so the dashboard/API's "upcoming
  matchup" code path (`feature_source: "computed"` in
  `modeling/serving/matchup_features.py`) has only been exercised against
  synthetic test fixtures, never a real future game. It should work
  unchanged once real games exist, but that's an expectation, not yet an
  observation.
- No cross-validated variance estimate exists for the frozen Phase 8
  result itself (only for the supplementary walk-forward backtest above,
  which necessarily uses different, larger training windows per fold than
  Phase 8's fixed one) — a true apples-to-apples confidence interval on
  the exact frozen model would need bootstrap resampling of the test set,
  which hasn't been built.
