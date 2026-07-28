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

- `ScheduleLeagueV2` for league schedules and official home/away assignments
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

## Home, away, and special-event inference

`MATCHUP` normally encodes location:

- `TEAM vs. OPP` marks the listed team with a `vs.` location indicator.
- `TEAM @ OPP` marks the listed team as away.

A standard completed game has one `vs.` row and one `@` row. The live 2024-25 audit revealed two
additional valid source patterns:

- NBA.com can use `vs.` for both team rows in some neutral-site games.
- NBA.com can use `@` for both team rows in some special-event games.

These are still valid two-team records, but `LeagueGameLog.MATCHUP` alone cannot determine the
official home team. The audit preserves them and records the ambiguous pattern instead of silently
guessing or deleting the games.

The audit therefore:

1. accepts one-`vs.`/one-`@` standard pairs;
2. accepts two-`vs.` pairs and counts them as neutral-site games;
3. accepts two-`@` pairs and counts them as ambiguous all-away games;
4. rejects every other pairing;
5. defers official home/away resolution for ambiguous games to `ScheduleLeagueV2`;
6. requires a special-site or ambiguous-location indicator in the future feature pipeline.

## Reliability controls

The ingestion layer must:

1. use an explicit request timeout;
2. retry transient failures with exponential backoff;
3. pause between season requests;
4. validate required columns before writing files;
5. reject duplicate team-game rows;
6. verify two team rows per completed game;
7. distinguish standard, neutral-site, and ambiguous all-away matchup pairs;
8. write raw responses without silently changing source values;
9. record runtime metadata and validation results.

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
