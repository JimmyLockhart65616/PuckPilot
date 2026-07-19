# PuckPilot

**An in-depth analytics companion for the data-driven fantasy hockey manager.**

PuckPilot reads your league's real settings, scoring categories, rosters, and
matchups, then combines them with public NHL data and advanced stats to turn the
numbers into decisions you can act on — every recommendation backed by
validation, not gut feel.

## What it does

- **Category-aware player valuation** — z-score / VORP valuations tuned to *your*
  league's exact scoring categories, built from projections tested walk-forward
  against past seasons, so you know how far to trust each number.
- **Draft assistant** — a live board with value-over-replacement rankings and
  pick suggestions, validated by thousands of simulated drafts against ADP and
  punt strategies and replayed on real historical seasons.
- **Daily lineup optimization** — starts the players who skate tonight, benches
  scratches, and slots the right goalies, solved as an optimization problem to
  wring the most value from every roster spot.
- **Waiver & free-agent engine** — nightly pickup proposals you review and
  approve, then executed through the Yahoo Fantasy API.

## Why PuckPilot

- **Research-grade, not guesswork.** Projections, rankings, and draft strategy
  are validated empirically — simulated drafts, walk-forward backtests, and
  full-season replays — before they reach your lineup.
- **Your league, your categories.** Everything is computed against your league's
  actual scoring settings, not a generic top-200 list.
- **Built on open data.** Public NHL APIs and MoneyPuck advanced stats —
  transparent, reproducible inputs.

## Status

The valuation, projection, draft, and lineup engines are built and validated
against historical seasons. Yahoo OAuth is implemented; full Fantasy API access
(reads and writes) is pending approval of our Yahoo Developer API application.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
cp .env.example .env        # then add your Yahoo app credentials
ppilot league show          # first run walks you through Yahoo OAuth
```

## Testing

```bash
pytest              # unit + recorded-fixture tests (no network)
pytest -m live      # live read-only contract tests against real APIs
```

## License

MIT / Apache-2.0 dependencies only — no GPL. Unlicensed third-party repositories
are design references only; no code is copied.
