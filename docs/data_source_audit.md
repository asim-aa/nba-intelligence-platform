# Phase 2 — NBA Data-Source Audit

## Decision

Use NBA.com statistics through the community-maintained `nba_api` package as the primary source.
Keep BALLDONTLIE as a fallback source for basic game and team metadata if NBA.com access is
unavailable.

## Why this source

NBA.com exposes official league data and stable identifiers for games and teams. The Python client
reduces request-construction overhead, but it does not eliminate upstream risk: NBA endpoints,
headers, schemas, and availability may change without notice.

## Required coverage

### Historical games

Primary endpoint: `LeagueGameLog`

Required fields:

- `GAME_ID`
- `GAME_DATE`
- `SEASON_ID`
- `TEAM_ID`
- `TEAM_ABBREVIATION`
- `MATCHUP`
- `WL`
- `PTS`
- `FG_PCT`
- `FG3_PCT`
- `FT_PCT`
- `REB`
- `AST`
- `TOV`
- `STL`
- `BLK`

The endpoint returns one team-game row for each team in a game. A valid completed game should
therefore have exactly two rows sharing the same `GAME_ID`.

### Upcoming schedule

Preferred endpoints:

- `ScheduleLeagueV2` for league schedules
- `ScoreboardV3` for date-level game status and near-term schedules

### Detailed box scores

Later phases may use:

- `BoxScoreTraditionalV3`
- `BoxScoreAdvancedV3`
- `BoxScoreFourFactorsV3`

Phase 2 does not depend on these endpoints.

## Stable identifiers

- `GAME_ID` is the canonical game key.
- `TEAM_ID` is the canonical team key.
- Team names and abbreviations are descriptive attributes, not primary keys.

## Home and away inference

`MATCHUP` encodes location:

- `TEAM vs. OPP` means the listed team is home.
- `TEAM @ OPP` means the listed team is away.

Every completed game must resolve to one home row and one away row.

## Reliability controls

The ingestion layer must:

1. use an explicit request timeout;
2. retry transient failures with exponential backoff;
3. pause between season requests;
4. validate required columns before writing files;
5. reject duplicate team-game rows;
6. verify two team rows per completed game;
7. write raw responses without silently changing source values;
8. record runtime metadata and validation results.

## Storage policy

Raw downloads are written locally under `data/raw/nba/` as Parquet. Small samples may be written
under `data/samples/` for local inspection, but source data is ignored by Git by default.

Recommended partition pattern:

```text
data/raw/nba/league_game_log/season=2024-25/team_game_log.parquet
```

## Publication and licensing controls

This repository publishes code, schemas, documentation, tests, and model artifacts—not a mirrored,
continuously updated NBA statistics database. Users are responsible for reviewing NBA.com terms and
any source-specific restrictions before redistributing downloaded data.

## Phase 2 acceptance criteria

- [x] Primary and backup sources identified
- [x] Historical, schedule, and box-score endpoints mapped
- [x] Stable identifiers documented
- [x] Required fields documented
- [x] Validation policy documented
- [x] Executable source-audit script added
- [x] Unit tests added for schema and game-pair validation
- [ ] Live sample downloaded and validated locally
