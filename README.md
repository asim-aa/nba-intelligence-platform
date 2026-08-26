# NBA Intelligence and Win-Prediction Platform

An end-to-end machine-learning project that estimates the probability that the home team wins an
NBA regular-season game using only information available before tipoff.

**[🏀 Try the live Streamlit dashboard](https://nba-intelligence-platform-uhspx8ztbxf4algcfyujvb.streamlit.app/)**

## Interactive demo

The deployed dashboard turns the model into a pick-the-winner game. Choose a date, compare each
matchup's pregame statistics, lock in your picks before seeing the model's probabilities, and then
compare your calls with the model and the actual outcomes.

For a more game-like experience, the dashboard includes a **💰 simulated parlay** feature. Enter a
paper stake and every pick locked in for that day's slate is combined into a parlay using a fixed
-110 illustrative vig per leg. The app tracks both **your parlay** and the **model parlay**, making it
an easy way to compare human judgment with the frozen ML model across an entire slate. It also
includes simulated round-robin and flat-stake tracking for exploring how prediction quality behaves
under different combinations.

> **Simulation only:** no real money, sportsbook, payment method, or betting transaction is involved.
> These features are for interactive model evaluation and entertainment, not betting advice.

## Current phase status

- [x] Phase 0: prediction problem and leakage policy
- [x] Phase 1: repository and development environment
- [x] Phase 2: source-design audit and networked sample validator
- [x] Phase 3: full historical ingestion pipeline (2015-16 through 2025-26)
- [x] Phase 4: game-level dataset transformation and ambiguous-matchup reconciliation
- [x] Phase 5: leakage-safe pregame feature engineering, including opponent-adjusted Elo ratings
- [x] Phase 6: chronological train/validation/test split and probability evaluation metrics
- [x] Phase 7: baseline, logistic regression, and CatBoost models with a validation-based comparison
- [x] Phase 8: one-time final evaluation on the held-out test set

Version 1 is feature-complete: every phase above is implemented and tested. The application
layer beyond it -- a prediction API (`app/api/`) and an interactive dashboard (`app/dashboard/`)
-- is also built. See [`docs/roadmap.md`](docs/roadmap.md) for the maintained, authoritative
status of every piece, including known gaps this README doesn't get into.

## Results

The selected model is logistic regression on a compact, interpretable feature set: season win
percentage, recent form, recent point differential, rest, back-to-backs, an opponent-adjusted Elo
rating differential (see `pipelines/features/build_team_elo_ratings.py`), and a roster-continuity
differential (see `pipelines/features/build_roster_continuity_features.py` — a leakage-safe proxy
for pregame injury/availability, built from player game logs since the NBA's real injury report has
no clean historical API). It was selected on 2023-24 validation performance, before the test seasons
were read, after beating both required baselines and a regularized CatBoost model on every
validation metric.

**v1.1 note:** the roster-continuity feature was added after v1's initial freeze, validated the same
way (train/validation only, before touching test), and improved log loss, Brier score, and ROC-AUC
on the held-out test set — so Phase 8 was deliberately re-run once as a new, documented iteration.
This does not violate `docs/project_spec.md` section 7 (rule 8)'s one-time-test rule: that rule
prevents using test performance to *choose* between models or features, and no such choice was made
using test data here — the decision to add the feature was made entirely on validation, and Phase 8
answered "how good is this new frozen model" exactly once, the same as v1 did.

Final one-time evaluation on the held-out 2024-25 through 2025-26 test set (2,460 games):

| Metric | v1 | v1.1 (current) |
| --- | --- | --- |
| Log loss | 0.5996 | 0.5965 |
| Brier score | 0.2067 | 0.2054 |
| ROC-AUC | 0.7347 | 0.7381 |
| Accuracy (0.50 threshold) | 0.6789 | 0.6780 |
| Expected calibration error | 0.0213 | 0.0214 |

Test performance matched or exceeded validation on every metric both times, with no sign that model
selection had overfit to the validation season. The full comparison table, calibration table, and a
reliability diagram are written to `artifacts/nba/final_evaluation/` when you run the pipeline
(gitignored, generated locally — not present in a fresh clone).

This one Phase 8 number is deliberately narrow: it's from one specific season cutoff, and it's a
single test-set-wide calibration figure. Two supplementary, non-reopening diagnostics go further
(see `docs/roadmap.md` for the full methodology and findings): a walk-forward backtest across 8
season cutoffs shows accuracy ranging from 0.62 to 0.68 depending on where the cutoff falls — the
official result sits near the favorable end, not the middle — and a calibration breakdown by
segment shows back-to-back games are markedly less well-calibrated than rested games (ECE 0.0390
vs. 0.0233).

## Scope

- NBA regular season only
- one model row per game
- team-level pregame features
- probability prediction, not betting advice
- time-based evaluation
- no live or player-prop predictions in version 1

See [`docs/project_spec.md`](docs/project_spec.md) for the full definition.

## Data-source decision

The first implementation uses NBA.com statistics through the community-maintained `nba_api`
client. Raw NBA records are ignored by Git by default. See
[`docs/data_source_audit.md`](docs/data_source_audit.md) for endpoint mapping, stability risks,
validation rules, and publication controls.

## Setup

```bash
uv sync
uv run pytest
uv run ruff check .
```

## Run the Phase 2 network audit

```bash
uv run python -m pipelines.ingestion.audit_nba_source \
  --season 2024-25 \
  --sample-rows 20
```

When NBA.com is reachable, the command validates the response and writes local Parquet files plus a
runtime JSON summary. Downloaded data is intentionally excluded from Git.

## Run the full pipeline

Each stage reuses local output when it already exists, so re-running an earlier stage after adding
seasons or features is cheap.

```bash
uv run python -m pipelines.ingestion.ingest_historical_seasons
uv run python -m pipelines.ingestion.ingest_player_game_logs
uv run python -m pipelines.transform.build_historical_game_datasets \
  --seasons 2015-16 2016-17 2017-18 2018-19 2019-20 2020-21 2021-22 2022-23 2023-24 2024-25 2025-26
uv run python -m pipelines.features.run_feature_pipeline
uv run python -m modeling.data.split_dataset
uv run python -m modeling.evaluation.run_model_comparison
```

The last command fits the baselines, logistic regression, and CatBoost on the training split and
scores them on train and validation only — it never reads the test split, and is safe to re-run as
often as you like while iterating.

```bash
uv run python -m modeling.evaluation.run_final_evaluation
```

This one reads the held-out test split. Per `docs/project_spec.md` section 7 (rule 8), that is only
meant to happen once, after the modeling approach is frozen based on validation results alone —
re-running it is harmless (it is deterministic), but its result should not go on to inform any
further modeling decision.

```bash
uv run python -m modeling.evaluation.run_robustness_backtest
uv run python -m modeling.evaluation.run_calibration_by_segment
```

Two supplementary diagnostics, safe to re-run any time: the first re-walks the same frozen model
across multiple season cutoffs to show how much the headline metrics move with a different test
split; the second re-slices the same frozen test-set predictions Phase 8 already scored once,
broken down by season phase and rest status. Neither retunes anything or reopens Phase 8's result.

## Keep live predictions from going stale

The dashboard/API's "computed" (upcoming-matchup) path reads each team's *current* rolling
stats, Elo rating, and roster continuity from files that are static snapshots as of whenever the
pipeline last ran. If games have been played since, predictions for upcoming games would quietly
use each team's state as of the last refresh, not as of today.

```bash
uv run python -m pipelines.refresh_current_season
```

Run this periodically (daily is plenty) during a live season. It force-refreshes only the current
season's raw data — past seasons never change once ingested, so they stay cached — then rebuilds
the full feature pipeline from it. Before the season's first game it skips cleanly (there's
nothing to refresh yet) rather than erroring. This never touches the frozen model or the
train/validation/test splits: retraining stays a separate, deliberate, validated decision, the
same way the roster-continuity feature above was adopted.

## Run the dashboard

**Live app:** https://nba-intelligence-platform-uhspx8ztbxf4algcfyujvb.streamlit.app/

An interactive pick-the-winner game: see a day's slate, compare the two teams' pregame stats,
lock in your pick, then see the model's prediction and the real outcome. Tracks your accuracy
against the model's over time.

### Dashboard features

- **Pick before the model:** lock in a winner before the model prediction is revealed.
- **Pregame comparison:** inspect season/L10 form, recent scoring, rest, Elo, and other model context.
- **💰 Simulated parlay:** combine every locked pick for a slate into one paper parlay and compare
  your parlay with the model's parlay.
- **Simulated round robin:** split a slate into smaller combinations so one missed pick does not
  invalidate every combination.
- **Live calibration monitor:** compare resolved live predictions with the frozen test-set reference.
- **Simulated betting tracker:** evaluate a fixed-stake, fixed-odds model-following strategy on paper.

All betting-style features are **simulation only**: no real money, sportsbook integration, payment
method, or wagering transaction exists anywhere in the application.

```bash
uv run streamlit run app/dashboard/app.py
```

The standalone analysis trackers can also be run directly:

```bash
uv run python -m app.dashboard.live_calibration_monitor
uv run python -m app.dashboard.simulated_betting_tracker
```

## Run the API

Serves the same frozen model over HTTP, for any client that isn't this project's own Python.

```bash
uv run uvicorn app.api.main:app --reload
```

Then `GET /predict` for one matchup, or `GET /slate` for a whole day:

```bash
curl "http://127.0.0.1:8000/slate?date=2025-11-05"
```

Interactive API docs are at `http://127.0.0.1:8000/docs` once it's running.

## Repository structure

```text
nba-intelligence-platform/
├── app/
│   ├── api/              # FastAPI prediction service
│   └── dashboard/        # Streamlit pick-the-winner game
├── data/
│   ├── raw/
│   ├── interim/
│   └── samples/
├── pipelines/
│   ├── ingestion/        # historical + player-log ingestion, live schedule fetch
│   ├── transform/        # game-level dataset + matchup reconciliation
│   └── features/         # rolling stats, Elo ratings, roster continuity, modeling dataset
├── modeling/
│   ├── data/              # chronological train/validation/test split
│   ├── baselines/
│   ├── training/          # logistic regression, CatBoost
│   ├── evaluation/        # shared metrics, validation comparison, final evaluation
│   └── serving/           # as-of-date features + prediction for app/
├── sql/                    # picks-tracking schema
├── tests/
├── artifacts/              # trained models and evaluation outputs (gitignored)
├── docs/                   # spec, roadmap, audits, runbooks
├── pyproject.toml
└── README.md
```
