# Roadmap and status

The single source of truth for what's built, what isn't, and what's known
to be incomplete. `README.md` is the pitch and the results; this file is
for whoever (including future you) needs to know exactly where things
stand before picking the work back up. **If this file and the README ever
disagree, this file wins** — update it whenever a phase or gap changes,
in the same change that changes the code.

Last updated: 2026-08-06.

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
  the model vs. reality in SQLite (`sql/picks_schema.sql`).
- **API** (`app/api/`) — `GET /predict` (one matchup) and `GET /slate`
  (a day's games), both backed by the frozen model. Run with
  `uv run uvicorn app.api.main:app --reload`.

## Known gaps (honest, as of this writing)

- `sql/` only has `picks_schema.sql` for the dashboard; no other use yet.
- `shap` is a declared dependency but unused. It would matter more had a
  tree-based model been selected; logistic regression's coefficients are
  already directly interpretable, so this was never revisited.
- `fetch_schedule.py`'s regular-season filter reproduces 1,224 of the
  2025-26 season's 1,230 known games (99.5%). The 6-game gap doesn't fall
  into any recognized excluded-label bucket; documented as a known,
  unchased gap in the module's own docstring, since this module serves
  the dashboard/API, not model training, where the historical ingestion
  pipeline remains authoritative.
- No CI job exercises the dashboard, the API, or the live NBA.com
  endpoints — by design, since those need network access CI shouldn't
  depend on. Only the synthetic-fixture pytest suite (232 tests as of
  this writing) runs in CI.
- No deployment story. Both the API and dashboard are meant to run
  locally; nothing here provisions hosting, TLS, or auth.
- The 2026-27 season hasn't started, so the dashboard/API's "upcoming
  matchup" code path (`feature_source: "computed"` in
  `modeling/serving/matchup_features.py`) has only been exercised against
  synthetic test fixtures, never a real future game. It should work
  unchanged once real games exist, but that's an expectation, not yet an
  observation.
