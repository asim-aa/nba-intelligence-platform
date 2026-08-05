# NBA Intelligence and Win-Prediction Platform

An end-to-end machine-learning project that estimates the probability that the home team wins an
NBA regular-season game using only information available before tipoff.

## Current phase status

- [x] Phase 0: prediction problem and leakage policy
- [x] Phase 1: repository and development environment
- [x] Phase 2: source-design audit and networked sample validator
- [x] Phase 3: full historical ingestion pipeline

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

## Repository structure

```text
nba-intelligence-platform/
├── app/
│   ├── api/
│   └── dashboard/
├── data/
│   ├── raw/
│   ├── interim/
│   └── samples/
├── pipelines/
│   ├── ingestion/
│   ├── transformations/
│   └── features/
├── modeling/
│   ├── baselines/
│   ├── training/
│   └── evaluation/
├── sql/
├── tests/
├── artifacts/
├── docs/
├── pyproject.toml
└── README.md
```
