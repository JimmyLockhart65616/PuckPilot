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

## Yahoo access and conduct

PuckPilot integrates with Yahoo Fantasy through the **official Fantasy Sports
API over OAuth 2.0** (`src/puckpilot/yahoo/client.py`, the only module permitted
to import `yahoo_fantasy_api`). Full read/write scope is pending approval of our
Yahoo Developer API application.

While that application is pending, two read-only fallbacks exist so development
can continue. Both are marked `DIAGNOSTIC` in their module docstrings, neither is
required by any engine, and both are written to retire themselves:

- `yahoo/session.py` reads the same `/fantasy/v2` paths the Yahoo web app calls,
  authenticated by the user's own logged-in browser session. It is read-only by
  construction (no write method exists, and a test asserts none appears), covers
  only the signed-in user's own leagues, throttles its requests, and **refuses to
  run once OAuth returns 200** — approval switches it off automatically.
- `draft/capture.py` and `draft/wsfeed.py` observe a draft room the user has
  joined by hand. They never click, type, submit, or navigate, and issue no
  requests of their own — they read what that browser already receives. Captured
  output stays local and git-ignored, with cookies and auth headers redacted.

`draft/farm.py` sits through public mock drafts to measure when players actually
go. It never picks and never clicks; the seat is the user's own and is played
exactly as it would be without the tool running. Harvest sessions are capped
because each one occupies a seat in a room of real people.

On draft night the engine does not draft. It ranks the board, shows the case for
and against each option, and a human makes every pick.

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
