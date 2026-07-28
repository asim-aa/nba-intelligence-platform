# Project Specification: NBA Intelligence and Win-Prediction Platform

## 1. Prediction objective

Estimate the probability that the **home team wins an NBA regular-season game**, using only information available before the scheduled tipoff.

The model target is:

- `home_win = 1` when `home_score > away_score`
- `home_win = 0` when `home_score < away_score`

The model output is a probability in `[0, 1]`, not a guaranteed winner. Example:

- Boston Celtics home-win probability: `0.67`
- Opponent win probability: `0.33`

## 2. Unit of analysis and table grain

The model-ready table has **one row per NBA game**.

Each row contains:

- stable game and team identifiers;
- the scheduled game date and home/away assignment;
- pregame team-level features computed from earlier games only;
- `home_win` for completed games.

The raw team game log may contain two rows per game—one for each team—but the analytical model table must collapse those records to one row per game.

## 3. Initial scope

### Included

- NBA games only (`LeagueID = 00`).
- Regular-season games only.
- Completed historical games for training and evaluation.
- Scheduled regular-season games for future inference.
- Team-level pregame features.
- Probability estimates generated before tipoff.
- Target acquisition window: **2015-16 through 2025-26**, subject to endpoint availability and quality checks during the source audit.

### Excluded from version 1

- Preseason, All-Star, Play-In, and playoff games.
- Games without a reliable final score or home/away assignment.
- Cancelled, postponed, or abandoned games until their status is resolved.
- Live or in-game predictions.
- Player-prop or exact-score predictions.
- Betting recommendations, odds optimization, or gambling use.
- Injury, lineup, transaction, and player-tracking features.
- Predictions based on information timestamped after tipoff.

## 4. Evaluation design

### Primary metrics

1. **Log loss** — evaluates the quality of predicted probabilities and heavily penalizes confident incorrect predictions.
2. **Brier score** — measures mean squared error between predicted probabilities and observed outcomes.
3. **Calibration** — checks whether games predicted near a probability level occur near that empirical frequency.

Lower log loss and Brier score are better. Calibration will be evaluated with reliability curves, probability bins, and calibration-error summaries.

### Secondary metrics

- Accuracy using a documented classification threshold, normally `0.50`.
- ROC-AUC for ranking performance.

Accuracy and ROC-AUC do not replace probability-quality metrics.

## 5. Baselines

Every trained model must be compared with at least:

1. always predicting the home team;
2. a rule based on the better pregame season record;
3. logistic regression using a compact, interpretable feature set.

A more complex model is selected only when it improves out-of-time probability performance and remains acceptably calibrated.

## 6. Time-based splitting

Games must remain in chronological order. Random train/test splitting is prohibited.

The provisional split is:

- Training: 2015-16 through 2022-23
- Validation: 2023-24
- Test: 2024-25 through 2025-26

This split may be adjusted after the data audit, but the final test period must remain untouched until modeling decisions are complete.

## 7. Data-leakage prevention rules

1. Every feature for game `G` must be calculated from records timestamped before `G` begins.
2. Rolling team statistics must exclude the current game, normally with a one-row lag such as `shift(1)` before `rolling(...)`.
3. Season-to-date statistics must use only earlier games from the same season.
4. Final scores, final box-score statistics, and `home_win` from the current game may never enter its feature vector.
5. Whole-season aggregates may not be joined to games occurring within that season unless the aggregate is recomputed as of each game date.
6. Data-cleaning and imputation parameters must be fit on the training period only when they depend on learned distributions.
7. Model selection and threshold decisions use training and validation data only.
8. The final test period may be evaluated only after the modeling pipeline is frozen.
9. Upcoming-game predictions must store the model version, prediction timestamp, and feature snapshot so later results cannot overwrite the original evidence.
10. Postponed games must use the actual rescheduled tipoff and only information available before that tipoff.

## 8. Reproducibility requirements

Each data or model run must eventually record:

- source and endpoint;
- retrieval timestamp;
- season/date parameters;
- row count and schema;
- code/model version;
- feature list;
- training and evaluation windows;
- metric values;
- artifact checksum or immutable revision.

## 9. Intended use and limitations

This project is an educational, noncommercial portfolio system for learning data engineering, data science, ML engineering, and Hugging Face workflows. Predictions are uncertain statistical estimates. Version 1 omits injuries, roster availability, travel details, and other information that may materially affect game outcomes.
