# Roadmap and status

The single source of truth for what's built, what isn't, and what's known
to be incomplete. `README.md` is the pitch and the results; this file is
for whoever (including future you) needs to know exactly where things
stand before picking the work back up. **If this file and the README ever
disagree, this file wins** — update it whenever a phase or gap changes,
in the same change that changes the code.

Last updated: 2026-08-13.

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
(including the Elo rating differential and, since v1.1, the
roster-continuity differential — see below). Chosen on 2023-24 validation
performance, before the test seasons were read. Frozen test result (v1.1):
log loss 0.5965, Brier 0.2054, ROC-AUC 0.7381, accuracy 0.6780, ECE 0.0214.
This decision is final — per `docs/project_spec.md` section 7 rule 8, the
test seasons are not to be re-evaluated to inform any further modeling
choice. (v1's frozen result, superseded by the v1.1 re-evaluation
documented below: log loss 0.5996, Brier 0.2067, ROC-AUC 0.7347, accuracy
0.6789, ECE 0.0213.)

## v1.1: roster-continuity feature (an injury-data proxy) — complete

The "capped ceiling" gap below used to note that the model has no
injury/lineup information, and that real historical injury-report data
doesn't exist as a clean API (the NBA only publishes it as PDFs). Rather
than accept an unaudited third-party scrape of those PDFs, this builds a
leakage-safe **proxy** entirely from data already in the pipeline:

- **New raw source**: `pipelines/ingestion/ingest_player_game_logs.py` —
  player-level `LeagueGameLog` (mirrors the existing team-level ingestion
  exactly), stored at `data/raw/nba/league_game_log/season=X/player_game_log.parquet`.
- **New feature**: `pipelines/features/build_roster_continuity_features.py`
  computes, per team-game, the ratio of a team's established top-3 core
  players' *actual* combined minutes to their *expected* combined minutes
  (a shift(1)-safe rolling average of each player's own recent minutes,
  carried forward through games they don't play so a genuine absence is
  detectable — ranking only among players present in a game would make an
  absence undetectable by construction). That per-game ratio is itself
  shift(1)-safe rolled into the final pregame feature,
  `ROLLING_CORE_CONTINUITY`, using the exact same `shifted_rolling_mean`
  helper every other rolling feature in this project uses. Sanity check on
  real data: the lowest-continuity games cluster at each season's final
  week — exactly when teams rest healthy stars for meaningless games, a
  well-known real pattern, not an artifact.
- **Serving layer**: `modeling/serving/matchup_features.py` computes this
  feature for upcoming ("computed") matchups the same placeholder-and-reroll
  way it already does for the rolling win/points features, reusing
  `build_roster_continuity_features.py`'s own `compute_rolling_continuity()`
  rather than a second implementation.
- **Validation-only decision**: adding `ROLLING_CORE_CONTINUITY_DIFF` to
  `COMPACT_FEATURE_COLUMNS` was tested on train/validation only, before
  touching test. It improved log loss, Brier score, ROC-AUC, and accuracy
  on validation, with a small calibration-error regression there (0.0259 →
  0.0306). Adopted anyway since log loss (the project's designated primary
  ranking metric) improved along with three of the four other metrics, and
  logistic regression still beat CatBoost on every validation metric with
  the new feature included (it hadn't, on ROC-AUC, before).
- **Phase 8 re-run**: per `docs/project_spec.md` section 7 rule 8, the test
  seasons may not be re-evaluated to *choose* between models or features —
  and no such choice was made using test data. The decision to adopt the
  feature was made entirely on validation; Phase 8 was then re-run exactly
  once, as a new frozen iteration, to answer "how good is this new frozen
  model" — the same question it answered for v1, not a re-opened
  comparison. On the actual test set, the calibration regression seen on
  validation did not carry over (ECE 0.0213 → 0.0214, essentially flat);
  log loss, Brier, and ROC-AUC all improved.

## Live-season data refresh — complete

The application layer's "computed" (upcoming-matchup) serving path reads
each team's *current* rolling stats, Elo rating, and roster continuity
from `data/processed/nba/features/*.parquet` -- static snapshots as of
whenever the pipeline last ran. Nothing kept those current once a season
actually starts. `pipelines/refresh_current_season.py` closes this:

- Force re-ingests only the current season's team and player game logs
  (`season_for_date()` on the given/current date decides which season);
  past seasons stay cached, since they never change once ingested --
  `ingest_season()`'s existing reuse-if-present behavior was already
  correct for them, it just needed a caller that force-refreshes the one
  season that *does* change.
- Rebuilds the combined game-level dataset and the full Phase 5 feature
  pipeline from the refreshed raw data -- fast, local, no meaningful cost
  to reprocessing every season each time (~15 seconds end to end for all
  11-12 seasons, verified against real data).
- Deliberately stops before `modeling/data/split_dataset.py` and never
  touches the frozen model artifact. Retraining stays a separate,
  validated, rare decision (see the v1.1 note above), not something a
  routine data refresh should ever trigger.
- Before a season's first game, `LeagueGameLog` legitimately returns zero
  rows for it; that's matched precisely and treated as a clean skip
  ("nothing to refresh yet") rather than a crash -- verified for real
  against the pre-season 2026-27 state. A different `ValueError` (an
  actual data-integrity problem) still propagates rather than being
  silently swallowed as the same case.
- Verified live against both real states: the pre-season skip path, and
  the full force-refresh-and-rebuild path against an already-completed
  season (2024-25) -- deterministic, byte-for-byte consistent with the
  data already on disk, no spurious diffs.

`.github/workflows/refresh.yml` was written to run this daily at 10:00 UTC
(and on manual `workflow_dispatch`), caching `data/raw`/`data/interim` and
committing refreshed `data/processed` files back to `main`. **It does not
work from GitHub's runners:** the 2026-10-24 and 2026-10-25 runs failed with
`LeagueGameLog failed after 3 attempts for season 2026-27` (stats.nba.com
refusing the runner's datacenter IP, see known gaps), and GitHub then
disabled the workflow for inactivity. It passed only as a local rehearsal
(including the pre-season skip path). Until it is moved somewhere
stats.nba.com will answer, run
`uv run python -m pipelines.refresh_current_season` by hand and push the
result; the deployed app's team stats are otherwise as of the last
successful refresh.

The frozen model artifact (`artifacts/nba/logistic_regression/model.joblib`,
~4KB) is now committed too -- a narrowly-scoped `.gitignore` exception, since
a deployed instance needs it to serve predictions and retraining it is a
deliberate decision (see the v1.1 note above), not something to regenerate
at deploy time. Everything else under `artifacts/` (picks.db, the unselected
CatBoost model, evaluation JSONs) stays gitignored and locally generated.

## Application layer — complete

Both pieces below are thin layers over the same two shared modules
(`modeling/serving/` for feature computation and prediction,
`pipelines/ingestion/fetch_schedule.py` for the live NBA schedule) so
neither duplicates the other's logic.

- **Dashboard** (`app/dashboard/`) — interactive pick-the-winner mini
  game. Shows a stat comparison before you pick, tracks your picks vs.
  the model vs. reality in SQLite (`sql/picks_schema.sql`), and has a
  "Live calibration monitor" expander (`live_calibration_monitor.py`)
  scoring real, resolved picks against the frozen test-set reference,
  plus a "Simulated betting tracker" expander (`simulated_betting_tracker.py`)
  -- see below.
- **API** (`app/api/`) — `GET /predict` (one matchup) and `GET /slate`
  (a day's games), both backed by the frozen model. Run with
  `uv run uvicorn app.api.main:app --reload`.

## Live calibration monitoring — the addressable half of "no monitoring"

The dashboard is deployed (see the README for the Streamlit Community
Cloud link), but there is no service-level instrumentation: no
metrics, alerting or uptime checks. What can be monitored is the model --
the dashboard's own picks database
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

## Simulated betting tracker — paper only, no real money

`app/dashboard/simulated_betting_tracker.py` answers one question: if
every resolved pick had been a flat $10 wager at a standard -110 American
moneyline (a typical sportsbook vig, not a real market line), would
following the model have come out ahead? Deliberately does not derive
odds from the model's own predicted probability -- betting at your own
stated confidence is tautologically break-even in expectation, which
would say nothing about whether the model is actually any good. A fixed,
realistic vig gives the result an actual bar to clear: -110 breaks even
at a 52.38% hit rate.

Nothing here touches a real sportsbook, account, or payment method --
this is a backtest, not betting advice, and the dashboard says so
explicitly. Below 10 resolved bets it reports the sample size honestly
instead of a number. As of this writing, real usage shows both "bet the
model" and "bet your own picks" at a 66.7% hit rate (24 resolved picks,
user and model picks happened to coincide throughout), clearing the
52.4% breakeven with room to spare -- interesting, but still a small
sample.

**Parlay mode**: the same module also simulates combining one day's picks
into a single parlay -- every leg must win, and one resolved loss busts
the whole bet immediately, even if other legs are still pending (matching
how a real parlay actually works, not waiting for every leg to settle
before declaring it dead). Uses the same fixed -110 vig per leg, compounded
across legs. A "💰 Simulated parlay stake" input sits at the top of the
dashboard, with the parlay result shown immediately below it, before the
day's slate -- paper only, same as the flat-betting tracker.

**Round robin mode**: splits one day's picks into every possible
combination of a chosen size (2-leg by default, 3-leg optional) and scores
each combination as its own independent mini-parlay with its own stake --
one bad leg only kills the combinations that included it, not the whole
round robin, a softer risk shape than a straight parlay across the same
picks. Real data makes the contrast concrete: the Nov 5 slate's straight
11-leg parlay was a total loss (one bad leg sinks 11-for-11), but the
same picks as a 2-leg round robin (`C(11,2)` = 55 combos, $5 per combo)
showed a net profit of $107.69 on the model's picks despite losing more
combos (34) than it won (21) -- because the winning combos pay out more
than 2x while losses only cost their own $5 stake.

## Robustness and calibration diagnostics — supplementary, not Phase 8

Two additional analyses go deeper than the single Phase 8 number without
reopening or superseding it — neither retunes anything, and neither
changes the selected model or its official test result:

- **`modeling/evaluation/run_robustness_backtest.py`** — walks the exact
  same logistic regression (same features, same fixed hyperparameters, no
  search) forward across 8 expanding-window season cutoffs (2018-19
  through 2025-26), instead of the single 2024-25/2025-26 cutoff Phase 8
  uses. Finding: performance is **not** stable across cutoffs — accuracy
  ranges from 0.620 (2020-21 cutoff) to 0.680 (2025-26 cutoff), log loss
  from 0.594 to 0.646. The officially reported Phase 8 result sits near
  the *favorable* end of that spread, not the middle, which is useful
  context missing from the single-number headline. (As a sanity check,
  this backtest's own 2024-25 and 2025-26 folds bracket the official
  Phase 8 test log loss almost exactly, which is reassuring but not a
  substitute for the deliberately narrower official result.)
- **`modeling/evaluation/run_calibration_by_segment.py`** — re-slices the
  same frozen test-set predictions Phase 8 already scored once, broken
  down by season phase and rest status. Finding: back-to-back games have
  markedly higher calibration error than rested games (ECE 0.0390 vs.
  0.0233 — roughly 1.7x), and late-season games (51+ games played) have
  the best log loss/Brier score but the *worst* calibration error (0.0485)
  — the model is more accurate late in the season but its stated
  probabilities are less trustworthy at face value then, which the single
  test-set-wide ECE in the README doesn't surface.

## Dashboard agent — built, not yet run against a real model

A chat panel in the dashboard sidebar and two REST endpoints drive the
dashboard through an LLM agent (`app/agent/`). See the README's "Ask the
dashboard" section for the tool list, configuration and limitations; the
structure is:

- **`tools.py`** — six allowlisted tools, each a thin wrapper over an
  existing function (API prediction logic, live calibration monitor,
  calibration by segment, walk-forward backtest, plus a team lookup on
  `nba_api`'s static team list). Pydantic input models reject unknown
  fields, and every failure comes back as `{"error": ...}` instead of
  raising, so the agent can correct itself.
- **`loop.py`** — a plain OpenAI chat-completions tool-calling loop capped
  at 6 model calls, with the client injected so tests never reach a real
  model. Each request returns `AgentMetrics` (model calls, tool calls,
  failed tool calls, seconds, tokens when the endpoint reports them) and
  logs one line on the `nba.agent` logger.
- **`faithfulness.py`** — checks the written answer: every number, date
  and season must be found in a tool result, tool argument or the user's
  messages (percent/fraction forms, the away-team complement and list
  counts allowed). It is a tripwire for invented or misquoted figures, not
  a proof; it cannot tell a correct figure attached to the wrong claim,
  and it may flag a legitimate derived number.
- **`ui_state.py`** — the only place that knows what a tool result means
  for the UI (which date, which game, which table/chart). Pure data, no
  Streamlit or LLM import; `app/dashboard/agent_panel.py` renders it, with
  the cost line and an unsupported-numbers warning under each answer.
- **`app/api/`** — `GET /agent/tools` and `POST /agent/chat` (503 when no
  model is configured, 502 when the model call fails, 422 for a malformed
  request). Their schemas and logic are in `agent_schemas.py` and
  `agent_chat.py`, imported lazily by `main.py` to avoid a circular import.
- **`eval.py`** — 21 labeled requests scored on the model's first tool
  call (tool-selection and argument accuracy).

What has and hasn't been verified: the tools were run against the real
data on disk; the loop, faithfulness check, UI-state layer, endpoints and
eval scoring are unit-tested with a scripted fake client; the chat panel
has Streamlit AppTest coverage and was also driven in a browser against a
scripted OpenAI-compatible endpoint, which caught a real bug (an
`st.rerun()` after each turn reset the date picker whenever a non-date
tool ran; a test now fails if it comes back). No real model has been
called, so the eval has no results and nothing is known yet about how well
a real model routes requests or how often the faithfulness check fires on
real answers.

## Known gaps (honest, as of this writing)

- The dashboard agent has never been run against a real model, and its
  eval has never produced a number — see the section above.
- `sql/` only has `picks_schema.sql` for the dashboard; no other use yet.
- No CI job exercises the dashboard, the API, the agent's real LLM calls,
  or the live NBA.com endpoints — by design, since those need network
  access CI shouldn't depend on. Only the synthetic-fixture pytest suite
  (448 tests as of this writing) runs in CI.
- The dashboard is live on Streamlit Community Cloud, but only the
  dashboard: the FastAPI service is not hosted, and there is no auth or
  rate limiting. The free tier sleeps the app when idle. The
  redeploy that added the agent panel and the `openai` dependency has not
  been verified live. Streamlit Cloud's filesystem is not durable, so
  `picks.db` (and with it the live calibration monitor's data) likely
  resets when the app restarts -- not verified. If a model key is ever added
  to the public app's secrets, any visitor can spend it through the chat
  panel; there is no per-visitor limit.
- The live-season refresh is not running. `stats.nba.com` blocks GitHub's
  datacenter IPs (the retries in `max_attempts` don't help), the workflow
  failed on 2026-10-24/25 and was then auto-disabled, so the deployed
  team stats, Elo ratings and roster continuity are stale for the 2026-27
  season. Fix: run the refresh from a machine stats.nba.com answers (a
  scheduled local job, or a self-hosted runner) and push the result.
- The 2026-27 season's schedule was published shortly after this was
  written, which retired one gap and surfaced another. The "upcoming
  matchup" code path (`feature_source: "computed"`) was exercised against
  a real future game for the first time (2026-10-20, Boston Celtics @
  Detroit Pistons) and worked end to end: correct blank season stats,
  correctly carried-over Elo ratings, a real prediction, and correct
  `actual_winner: pending`. Getting there required a real fix, not zero
  changes: `fetch_schedule.py` crashed on the initial schedule pull,
  because a season published before the Emirates NBA Cup knockout bracket
  is set includes 7 placeholder games with `teamId` 0 on both sides
  ("TBD") -- treated as incomplete data to skip now, not corrupt data to
  raise on, matching how a missing team was already handled. This is a
  good reminder that "should work unchanged" claims about an untested path
  are worth verifying the moment real data exists, not just trusting.
- No cross-validated variance estimate exists for the frozen Phase 8
  result itself (only for the supplementary walk-forward backtest above,
  which necessarily uses different, larger training windows per fold than
  Phase 8's fixed one) — a true apples-to-apples confidence interval on
  the exact frozen model would need bootstrap resampling of the test set,
  which hasn't been built.
