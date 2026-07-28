# Phase 2 Runbook

## 1. Pull the latest code

```bash
git pull origin main
```

## 2. Apply the Phase 1 Ruff fix

```bash
uv run ruff check . --fix
```

## 3. Run local quality checks

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

## 4. Run the live NBA.com source audit

```bash
uv run python -m pipelines.ingestion.audit_nba_source \
  --season 2024-25 \
  --sample-rows 20 \
  --timeout 30 \
  --max-attempts 3
```

Expected local outputs:

```text
data/raw/nba/league_game_log/season=2024-25/team_game_log.parquet
data/samples/team_game_log_2024-25.parquet
docs/data_source_audit_runtime.json
```

Raw and sample data are ignored by Git. The runtime JSON may also remain local because it describes a
specific execution rather than a stable project specification.

## 5. Completion test

Phase 2 is complete when:

- all unit tests pass;
- Ruff passes;
- the live audit returns a nonempty table;
- every game has exactly two team rows;
- every game has one home and one away row;
- no duplicate `(GAME_ID, TEAM_ID)` rows exist;
- the three expected output files are created.
