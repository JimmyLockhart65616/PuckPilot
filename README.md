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
- **Waiver & free-agent engine** — ranked pickup and drop proposals, scored
  against a weekly backtest. Proposals are recorded for you to review; executing
  them through the Yahoo Fantasy API is designed but not built, and is gated on
  API approval.

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

**Nothing in PuckPilot writes to Yahoo.** There is no add, drop, trade, lineup
or draft-pick submission anywhere in the codebase, and no code path that sends
anything to Yahoo other than the OAuth token exchange itself.

Three components read Yahoo. They differ in status, so they are described
separately rather than lumped together:

**`yahoo/session.py` — an interim fallback, and it retires itself.** It reads
the same `/fantasy/v2` paths Yahoo's own web app calls, authenticated by the
user's existing logged-in browser session. It is read-only by construction (no
write method exists, and a test asserts none appears), covers only the
signed-in user's own leagues, throttles its requests, and **refuses to run once
OAuth returns 200** — approval switches it off automatically, without anyone
having to remember. It is marked `DIAGNOSTIC` in its docstring.

**`draft/wsfeed.py` — permanent, and passive.** This is how the draft-night
console learns which players are gone. It is not a diagnostic and has no kill
switch, because it is the product. It opens no connection of its own: it
attaches to the socket the user's own draft-room session already has, reads
what arrives, and never sends.

**`draft/capture.py` — a diagnostic, kept as an instrument.** It records a
draft room to disk so the pick feed could be built from evidence rather than
guesswork. Output stays local and git-ignored, with cookies and auth headers
redacted.

What those three do to a browser, stated precisely: **none of them submits a
form, enters text, or takes any action that changes your league or roster
state.** Two things they do do, and both are worth naming rather than glossing:

- They open a Yahoo URL you name — `capture.py --url`, and the lobby or draft
  room for `live --yahoo`.
- `capture.py` reads the draft room's visible text on an interval, by
  evaluating a one-line expression in the page (`document.body.innerText`).
  That is a read of what is already rendered; it sends nothing and changes
  nothing. No other component does this.

`draft/farm.py` sits through public mock drafts to record the order players
come off the board, which is what calibrates the survival model. **It never
picks.** The seat is the user's own, and Yahoo plays it exactly as it
would if the tool were not running — including autopicking it if the user steps
away, which is ordinary Yahoo behaviour for any idle seat. Sessions are capped,
because each run occupies a seat in a room of real people. It records picks and
nothing else: the advice values the room broadcasts mid-draft are Yahoo's own
analytics, and are deliberately not captured.

On draft night the engine does not draft. It ranks the board, shows the case
for and against each option, and a human makes every pick.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
cp .env.example .env        # then add your Yahoo app credentials
ppilot league show          # first run walks you through Yahoo OAuth
```

## Sharing the draft view

Two managers in the same league draft from the same room, so they can share one
board: the picks are universal, and only the roster and the shortlist differ by
seat. One person runs the console; the other opens a link and needs no install,
no database and no Yahoo session of their own.

```bash
# loopback only (the default, unchanged)
ppilot draft live --seat 3 --web

# reachable on the local network, behind generated owner/guest keys
ppilot draft live --seat 3 --web --share --seats 3,7

# ...and pushed to a relay, so the link works from anywhere
ppilot draft live --seat 3 --web --seats 3,7 --publish https://<relay-host>
```

Guests may read any seat but cannot undo, which is the only destructive control
on the view. `?seat=` selects a view, it is not an isolation boundary.

The **feed never moves**. It reads the draft room's own websocket through the
browser on the drafter's machine, so the board is always computed there; the
relay (`deploy/relay/`) holds nothing but the last snapshot pushed to it and
imports no part of the engine. That also means the drafter's machine is a single
point of failure — hosting buys a stable URL, not resilience.

## Testing

```bash
pytest              # unit + recorded-fixture tests (no network)
pytest -m live      # live read-only contract tests against real APIs
```

## License

MIT / Apache-2.0 dependencies only — no GPL. Unlicensed third-party repositories
are design references only; no code is copied.
