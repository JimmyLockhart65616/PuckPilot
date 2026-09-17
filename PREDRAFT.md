# Pre-draft hardening — PuckPilot, draft 2026-09-19

## Context

The draft is in two days. The engine, valuation, and board are built and
validated; the last week of commits was a crunch that fixed real bugs
(keepers missing from the live board, stale `team_abbrev` corrupting goalie
projections by up to 13 wins, `/undo` reachable by GET). What is *not* done is
the operational layer around draft night: the console has no way to recover
when its single pick source is wrong, several inputs degrade silently rather
than loudly, and keeper ownership — which drives every roster-aware
recommendation — is not yet declared.

This plan is ordered by what breaks the draft, not by what is interesting.
Tier 1 must land before Friday. Tier 2 lands if Tier 1 is green. Tier 3 is
after the draft.

Drafting happens from the **web view** (`ppilot draft live --seat N --yahoo
<league_key> --web`), so the web surface takes priority over the terminal
console everywhere the two differ.

---

## What I verified in the current code

Established facts this plan rests on, with references:

| Finding | Where | Consequence |
|---|---|---|
| An unmapped Yahoo id is recorded and `continue`d — no `PickEvent` is emitted | `draft/wsfeed.py:172-178` | `board.made` never advances. If that player *is* in our universe he stays on the shortlist after being drafted. Worst-case board error. |
| The only mutation in the web view is `POST /undo` | `web/server.py:329-351` | There is no way to add a pick. Undo cannot fix "the room is ahead of us". |
| Terminal console reads only `feed.last_error` | `draft/live.py:158-160` | Gaps/unmapped are computed but not shown mid-draft; the drift summary prints only *after* the draft (`cli.py:445-450`). The **web view already shows gaps and unmapped** (`web/server.py:169-195`, `web/page.py:127-128`). |
| `--yahoo` with no league key silently disables real ADP | `cli.py:294-295` — `key = args.yahoo if "." in str(args.yahoo) else None` | `--yahoo` bare → `const="auto"` → no dot → `adp=None`. Board falls back to pseudo-ADP (prior-season actual value order, `sim.py:206-219`), `market_ids` is not applied, and `build_live_board` prints its "using Yahoo ADP for N players" line only on success — **absence is signalled by silence**. Every `p_survive` and the `survival_discount` term are then computed off a proxy. |
| `load_adp` filters on an exact `league_key` string | `yahoo/playermap.py:269-278` | A key that doesn't match what `playermap` stored returns `{}`, which is falsy → same silent fallback as above. |
| `resolve_keeper_ids` resolves on normalized name with `setdefault` | `keepers.py:36` | First row wins on a normalized-name collision, with no ambiguity report — unlike `playermap._resolve`, which detects and reports it. Two NHL players can share a name (e.g. Sebastian Aho). |
| `sim.keepers_for` looks up owner names with the raw string | `draft/sim.py:127-131` — `resolved.get(n)` | `resolved` is keyed by the `by_season` spelling. A different spelling in `[keepers.owners]` sends that keeper to a random seat (warned, but the board is still wrong). |
| `yahoo_player_map.positions` holds Yahoo's real multi-position eligibility | schema `data/store.py:102`, written `playermap.py:235-245` | Never read anywhere. The board uses MoneyPuck's single position (`Universe.pos`, `draft/engine.py:47`), so a Yahoo-dual-eligible C/LW cannot fill an LW hole in our model. |
| `DraftRules.caps`/`mins` are hardcoded, not derived from the league | `draft/engine.py:22-23`; `league.draft_rules()` passes only `shape` and `rounds` (`league.py:80-81`) | Defaults assume an 18-round draft. For a 16-man roster with keepers, `mins` sum to 12 against ~13 live picks, so `eligible_positions` starts force-constraining the **shortlist** early. The full board is unconstrained (`web/server.py:106-111`), which mitigates it. |
| A player with no projection is never in the universe at all | `sim.build_universe:213-228` — `market_ids` can only re-include rows already in `ranked` | Rookies below `MIN_TRAIN_GP_SKATER=10` (`engine/projections.py:15`) are invisible to the board regardless of Yahoo ADP. `board.py:445-447` already measures this: **"11 of the top-163 Yahoo-priced players currently off our board"** — so this is a live condition, not a hypothetical, and it is what `record_unknown` exists for. |
| No `TODO`/`FIXME`/`NotImplementedError` anywhere in `src/` | repo-wide grep | Known gaps live in docstrings and commit messages, not in a checkable list. |

Environment note: this checkout has no `data/`, no `secrets/`, no `.env`, and
no real `leagues/*.toml`. All verification below runs on the user's machine.

---

## Tier 1 — must land before the draft

### 0. Keeper ownership — do this first, it is config not code

The keeper *pool* is set and resolves cleanly (29 names, zero unmatched).
Ownership is not. With `[keepers.owners.20262027]` empty, `keeper_seats`
deals all 29 round-robin to the emptiest seats (`keepers.py:79-86`) —
including onto our own. `board.counts[seat]` then drives `needs()`,
`eligible_positions()`, `_fills_starter()` and `bench_factor`, so **every
recommendation for the entire draft is computed against a roster that is
partly fiction.** Of everything here this is the largest single distortion to
recommendation quality, and it is the cheapest to fix.

`build_live_board` already detects and reports this exact case
(`live.py:268-273`), so the tool is telling the truth about it today — it
just cannot fix it for you.

- **Minimum viable, do it by hand tonight:** declare *our own seat* in
  `[keepers.owners.20262027]`. That alone corrects every panel that feeds our
  recommendations. The other eleven seats only need their keepers to be
  *unavailable*, which `[keepers.by_season]` already achieves. Rival roster
  composition affects nothing we score.
- **Better, if Yahoo exposes pre-draft rosters:** a discovery command. All the
  pieces exist — `YahooSession.get(path)` (`yahoo/session.py:161`) is a
  generic `/fantasy/v2` getter, so `league/<key>/teams/roster` needs no new
  transport; match names through the existing `yahoo_player_map` rather than a
  new matcher. Output a ready-to-paste TOML block; **print, do not write** —
  the league file is private and hand-maintained. Cross-check both directions
  and report roster names absent from `by_season`, and `by_season` names on no
  roster. Both are real pre-draft errors.
- **Two correctness fixes while in here:**
  - `sim.keepers_for:127-131` looks up owner names with `resolved.get(n)`
    using the raw string, while `resolved` is keyed by the `by_season`
    spelling. A spelling difference between the two blocks silently sends
    that keeper to a random seat. Normalize both sides through
    `keepers._norm`.
  - `resolve_keeper_ids` (`keepers.py:32-46`) uses `setdefault`, so a
    normalized-name collision silently takes the first row with no ambiguity
    report — unlike `playermap._resolve`, which detects and reports it.
    Report collisions instead.

### 1. Manual "mark taken" — a second pick source

The explicit ask, and the right first item: it is the recovery hatch for every
other failure. Today `record()` is reachable only from the feed.

This is smaller than it sounds, because **the board primitives already
exist and are already correct**:

- `record(player_id, seat=None, source="manual")` (`board.py:408`) — note
  the default `source` is *already* `"manual"`; it takes an explicit seat,
  validates it against the team count, and raises `DraftBoardError` rather
  than crashing on an out-of-range seat or an already-drafted player.
- `record_unknown(seat=None, label="")` (`board.py:434`) — already advances
  `made` without touching `avail`/`counts`, which is exactly right for "the
  room picked someone we can't identify".
- `undo()` (`board.py:468`) — already the recovery hatch.

So the work is **a name resolver plus UI wiring**, not new board mechanics.

- **Resolver** (`draft/board.py`): `find(query) -> list[row]` — prefix and
  substring match over `u.names`, normalized through `keepers._norm`
  (reuse it; do not write a third name matcher). Returns candidates so an
  ambiguous query can be disambiguated by click rather than guessed at.
- **Web** (`web/server.py`, `web/page.py`): add `POST /taken` and
  `POST /unknown`, owner-only, mirroring `/undo` exactly — the same
  `_route()` exact-match (`server.py:215-222`), the same Host pinning
  (`server.py:279-306`), the same `role == "owner"` gate (`server.py:349`).
  A search-as-you-type box over the already-rendered board rows, plus a
  one-click "taken" control on each board row, which is faster on a clock
  than typing. An "unknown pick (+1)" button advances the clock only.
- **Terminal** (`draft/live.py:186-203`): add `t <name>` / `taken <name>`
  and `x` / `unknown` to the input loop and to `HELP` (`live.py:34-40`,
  whose text currently says "there is nothing to type in" — that line
  changes).
- **Idempotency**: recording someone already off the board must return a
  message, not raise. Confirm it composes with `feed.apply()`, which already
  catches `DraftBoardError` "already off the board" and reports it as
  `rejected` without consuming a slot (`draft/feed.py:208-228`) — so a feed
  that later re-delivers a manually-entered pick is already handled.

### 2. Make feed health impossible to miss

The web view already renders `gaps` and an `unmapped` count. Two additions:

- **Drift**: compare `board.made` against `feed.status()["highest_pick"]` and
  surface the delta in the snapshot (`web/server.py:169-195`). A board two or
  more picks behind the room is the signal that matters, and nothing computes
  it today.
- **Name the unmapped players.** `state.unmapped` holds bare Yahoo ids
  (`wsfeed.py:177`). Resolve each against `yahoo_player_map.full_name` — which
  has the name even when `nhl_player_id IS NULL` — so the panel reads
  "unmapped: Ivan Demidov" instead of "unmapped: 1". That turns an unusable
  counter into a one-click "mark taken".
- Mirror both into the terminal `render()` status area (`live.py:104-109`).

### 3. Stop the silent ADP fallback

- Add `--adp-league-key` (falling back to `--yahoo`'s value when it contains a
  dot) so the ADP source is nameable independently of the feed.
- In `_cmd_draft_live` (`cli.py:294-295`): when a key is given but
  `load_adp` returns `{}`, say so and list the `league_key` values actually
  present in `yahoo_player_map` — that is the exact typo this catches.
- In `build_live_board` (`draft/live.py:238-251`): when `adp` is falsy, print
  a **warning** naming the consequence ("no Yahoo ADP — survival
  probabilities are computed against a prior-season-value proxy"), not
  silence. Surface the same fact in the web snapshot so it is visible on the
  screen being used, not just in scrollback.

### 4. `ppilot draft preflight` — one command, fails loudly

A single pre-draft check that exits non-zero on anything that would silently
corrupt the board. Everything it needs already exists; this wires it together
and states the numbers.

| Check | Source |
|---|---|
| League file actually loaded (not the generic 12-team fallback) | `league.load_default_league` (`league.py:169-191`) already warns to stderr; make it an error here |
| Roster slots / categories / `scoring.type` echoed back for eyeball confirmation | `LeagueConfig` |
| Keepers: count, resolved, **unmatched by name**, **outside the ranked universe**, **normalized-name collisions** | `keepers.resolve_keeper_ids`, `sim.keepers_for` |
| `[keepers.owners.<season>]` present, and **our seat declared** | `league.keeper_owners_for_season` |
| Live pick count and rounds (expect 192 − keepers) | `board.slots` |
| Playermap: row count, match rate, `updated_at` age, and coverage of the **top 250 by ADP** | `yahoo_player_map` |
| Yahoo ADP resolves for the league key being used, with coverage count | `playermap.load_adp` |
| Data freshness: newest `sync_meta.updated_at`, and whether `data rosters` has been run for `20262027` | `store.get_meta` (`store.py:131`), `sync.sync_current_rosters` |
| Projection coverage of Yahoo's priced pool — which priced players have **no projection** and so cannot appear on the board at all | `build_universe` vs `pool_adp` |
| `mins` sum vs live picks left, i.e. the round from which the shortlist becomes position-forced | `engine.eligible_positions` |
| `yahoo probe` verdict, so it is known which Yahoo path is live | `yahoo/probe.py` |

Files: new `src/puckpilot/preflight.py`, wired at `cli.py:873-884` alongside
the other `draft` subcommands. Tests in `tests/test_preflight.py` built on the
in-memory-DB fixtures in `tests/conftest.py`.

---

## Tier 2 — accuracy, if Tier 1 is green

### 5. Yahoo multi-position eligibility

`yahoo_player_map.positions` is populated and never read. Read it into the
universe and use it for **roster accounting only** — which slots a player can
fill (`_bump`, `needs`, `_fills_starter`, `eligible_positions`) — while
leaving VORP's replacement level on the primary position, so valuation is
untouched. Follow this repo's own established discipline: put it behind a
flag, run `draft sim --n 200 --seed 8675309`, and **ship it ON only if the
gate holds**, exactly as `replacement_depth` was measured and shipped OFF
(`draft/engine.py:216`, commit `424b2cb`).

### 6. Roster rules — check, probably do not change

`DraftRules.caps`/`mins` are hardcoded and not derived from the league
(`draft/engine.py:22-23`; `league.draft_rules()` passes only `shape` and
`rounds`). Worth stating plainly after checking the numbers: for *this*
league the defaults happen to be right. `mins` = `{C:2,L:2,R:2,D:4,G:2}`
is exactly the league's starting slots, and `caps` summing to 21 against a
16-man roster means caps effectively never bind. So this is a **preflight
assertion, not a code change** — have preflight fail if `mins` ever stops
matching the league's starting slots. Deriving them properly is Tier 3 work
for the next league, not something to touch two days out.

---

## Tier 3 — after the draft

- Harvest staleness: `farm.load_all` globs `mock-*.json` with no season or
  date filter, so a prior-year harvest would be reused silently for
  `display_spread` calibration.
- `data/goalies.py:10-17` — `GoalieStartSource` is a `Protocol` with only
  hindsight implementations. There is **no live projected-starter feed**, which
  the daily lineup optimizer needs all season.
- Zero test coverage: `draft/mock.py`, `shadow.py`, `yahoo/client.py`.
- `web/relay.py` holds state in memory only — a redeploy mid-draft loses the
  board.
- Per-season team history, so 2024-25 is a usable out-of-sample control for
  goalie-shaped changes again (`f2d0dcf`).

---

## Suggested sequencing against a two-day clock

- **Tonight:** declare our own seat in `[keepers.owners.20262027]` (Tier 1
  item 0, by hand — five minutes, largest accuracy win). Then build
  preflight (item 4), because it tells you what else is actually wrong on
  the real data rather than on my reading of the code.
- **Tomorrow:** manual "mark taken" (item 1) and feed health (item 2) — the
  two that decide whether a bad draft night is recoverable. Then the ADP
  warning (item 3), which is small. Then the replay rehearsal and the
  deliberate failure drill.
- **Day of:** the re-sync sequence in step 6 below, preflight again, and
  ship nothing new.

Tier 2 only if all of that is green and the sim gate still passes. I would
rather go into Friday with Tier 1 rehearsed than Tier 2 untested.

## Verification

Nothing here is verifiable in this checkout — there is no `data/`, no
`secrets/`, and no real league file. All of it runs on the drafting machine.

1. **Unit** — `pytest -q` plus `ruff check . && ruff format --check .`
   (what CI runs). New tests: manual-pick recording/idempotency/seat
   attribution and its interaction with a later feed re-delivery; the two new
   web routes including the owner-only and Host-pinning cases (match the style
   of `tests/test_web_server.py:155`); preflight pass/fail per check; keeper
   name-collision and owner-spelling cases.
2. **Regression gate** — `ppilot draft sim --n 200 --seed 8675309` must still
   pass. Required before Tier 2 item 6 ships ON.
3. **Preflight against the real league** — `ppilot draft preflight`, exits 0.
4. **Full dress rehearsal** — `ppilot draft live --seat N --web --replay
   data/mocks`, which is the only end-to-end path that has ever been exercised
   (commit `b218768`, and it found a crash the first time it was run). During
   the replay, exercise every new control: mark a player taken, mark an
   unknown pick, undo each, and confirm the drift indicator moves and clears.
5. **Deliberate failure drill** — kill the feed mid-replay and drive the board
   forward by hand for ten picks. This is the scenario the whole of Tier 1
   exists for, and it should be rehearsed once before it matters.
6. **Day-of** — re-run `ppilot data sync`, `ppilot data rosters --season
   20262027` (a separate command, easily forgotten, and the one whose absence
   corrupted goalie projections by up to 13 wins), `ppilot yahoo playermap`
   against the real league key, then `ppilot draft preflight` again.
