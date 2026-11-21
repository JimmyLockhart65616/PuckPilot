# Handoff: cloud session → PC session (2026-11-21)

For the Claude session on the PC: the machine with the private
`managers/<name>.toml`, the database and the scheduled runner. The cloud
session that wrote this has none of those. Work through it with the user, then
**delete this file** (it lives only on `claude/pre-draft-scope-59z9oo`).

---

## 1. Land `open_spot` on `in-season`

**Why.** A card read "Add Jake Neighbours … Needs your tap: it drops nobody -
only a Rotation player's drop is covered". Sennecke went to IR, which opened a
spot, so the add had no drop. Standing approval only ever covered moves whose
*drop* is a Rotation player, so a first-come-first-served free agent sat
waiting for a tap. The user expected the tier system to have made this
automatic.

**What changed** (commit `3f33920` on this branch; the full suite was green,
1313 passed, and ruff was clean):

- `authority.StandingApproval.open_spot: bool = False`, and
  `COVERS["open_spot"] = "adds into an open spot"`.
- `standing.move_kind`: an ADD_DROP with no drop is `open_spot`; it used to be
  `None`, which meant "never covered".
- `standing.refusal`: an `open_spot` card skips the tier checks because nobody
  leaves the roster. It keeps every other bound: the executor, waiver claims
  (still always a tap) and `max_per_week`. The league caps and the playoff
  reserve still apply upstream in `week.py`.
- `standing.offer` no longer assumes a drop exists; `tier_source` is None.
- `snapshot._standing`: the card names its approval ("…for adds into an open
  spot" vs "…for a Rotation player's drop").
- `managers/example.toml`: the new key, documented.
- Tests in `tests/test_season_standing.py`.

**Do this.** Cherry-pick rather than merge: the branch also carries
September's `PREDRAFT.md` and a merge commit that `in-season` does not need.

```bash
git fetch origin
git checkout in-season && git pull
git cherry-pick 3f33920
pytest -q && ruff check . && ruff format --check .
git push origin in-season
```

Then make sure the checkout the scheduled run uses is on the new `in-season`.
No relay redeploy is needed: `web/season_page.py` did not change.

## 2. Turn it on (needs the user's say-so)

It is off by default because it widens what the user consented to in advance.
They have said they want it. Add this under `[authority.standing]` in their
`managers/<name>.toml`:

```toml
open_spot = true
```

Ask them about `veto_minutes` before touching it. It is 120 in the template
and applies to *every* covered kind, not just open-spot adds. A shorter window
helps in a free-agent race but also shortens the time to stop a Rotation drop.

**Check it worked:**
- `ppilot season run` (or the next scheduled run) reports a `standing` line
  like "1 card(s) made 120 min after the page shows them, unless stopped"
  whenever a pure add is proposed.
- The card says "PuckPilot makes this at … unless you stop it - your standing
  approval for adds into an open spot", with Stop and Make it now buttons
  instead of Approve and Reject.
- A card that is still not covered shows the reason after "Needs your tap:".
  That text is the answer to "why is it still asking".

## 3. The deferred analysis

The user's original request, verbatim:

> Do an analysis of where this model is working and not. Lets address the
> gaps that keep causing low scores. Analyze what other players are doing
> better or worse. Come up with fundamental changes which may assist in
> winning more categories per week, or strengthening my roster long term.

The user deferred this to the desktop because the evidence is in the local
DB. Start with the data, not the code:

- `ppilot season g3`: every team-week's start-of-week projection against
  Yahoo's final totals. Use it for Brier per category, expected minus actual
  categories, and whether *our* team is projected differently from the rest.
- `ppilot season league`: standings and every team's adds. Look for who wins,
  how much they stream, and what share of their adds are goalies.
- `ppilot season moves`: the ledger. Did each executed move beat the player it
  dropped, on real games?
- `ppilot season week`: the current week's plan and odds.
- Tables, for direct SQL: `league_week_stats` (per-team weekly category
  totals), `league_transactions`, `league_standings`, `matchup_snapshots`,
  `week_odds_log`.

Useful cuts:
- Our W/L/T by category across weeks. Which categories do we lose most, and
  are they close losses or blowouts?
- Category profiles of the top teams against ours.
- Calibration by category: where are the odds over- or under-confident?

Structural gaps the cloud session saw in the code. Treat these as hypotheses
to check against the data, not conclusions:

- **No trade evaluation anywhere.** The only roster lever is the waiver wire.
- **The goalie-start model is about 62% accurate** with no announcement feed
  (`season/goalies.py`; `ChainedGoalieSource` is the seam for one). This
  feeds the W, SV and SV% odds and goalie adds.
- **Goalie rates are preseason only.** `season/form.py` measured and shrinks
  only skater rates.
- **Short add horizon.** The search prices this week, or this week plus next
  on the last days (`preload`/`-p`, `-h`). Nothing values a player over the
  next several weeks or over the playoff weeks' schedule.
- **G2's rivals are frozen by default** (`add_gate`, `rivals="frozen"`). The
  arm gains are upper bounds unless re-run with `rivals="stream"` and the
  measured rates from `league_feed`.

The user left one question open: when this season and next season pull in
different directions, which wins (win now, balanced, or build for next year)?
Ask before proposing anything that trades one for the other.

This repo's own discipline applies. A new knob ships off, with a gate
measurement in its docstring, and the default changes only when the gate says
so.
