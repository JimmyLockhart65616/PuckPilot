"""What a correct draft-night view is, stated as checks.

The console computes a snapshot per seat, serves it locally, and pushes it to
the relay, which serves it to a second manager. Each hop has its own way to be
quietly wrong: a shortlist that offers a drafted player, a count that drifts
from the board, a number the browser cannot parse, a relay that hands back
something other than what it was given. None of those crash; all of them are a
drafter making a pick off a wrong screen.

So the checks live here, once, and both the test suite and `ppilot draft e2e`
call them - a harness with its own copy of the rules would drift from the tests
the way two name matchers already did.

Every function returns a list of violations (empty means sound) rather than
raising, so a run can report every problem at a pick instead of the first.
"""

from __future__ import annotations

import json
import math
from collections import Counter

from puckpilot.draft.advice import recommend
from puckpilot.draft.board import DraftBoard
from puckpilot.draft.engine import RosterValuePolicy
from puckpilot.web import wire

# Keys whose value legitimately differs between the console's view and the
# relay's copy of it: who may undo, how old the relay copy is, and the pick
# clock the relay advances between pushes.
RELAY_VOLATILE = frozenset({"can_undo", "relay_age", "stale", "seconds_since_pick"})


def non_finite_paths(obj, path: str = "$") -> list[str]:
    """Where a NaN/Infinity (or a type JSON cannot carry) sits in a payload."""
    if isinstance(obj, bool) or obj is None or isinstance(obj, (str, int)):
        return []
    if isinstance(obj, float):
        return [] if math.isfinite(obj) else [path]
    if isinstance(obj, dict):
        out = []
        for k, v in obj.items():
            out += non_finite_paths(v, f"{path}.{k}")
        return out
    if isinstance(obj, (list, tuple)):
        out = []
        for i, v in enumerate(obj):
            out += non_finite_paths(v, f"{path}[{i}]")
        return out
    return [f"{path} ({type(obj).__name__})"]


def wire_violations(snap: dict) -> list[str]:
    """Would a browser parse this exactly as the console meant it?"""
    bad = non_finite_paths(snap)
    if bad:
        return [f"not strict JSON at {', '.join(bad[:5])}"]
    try:
        text = json.dumps(snap, allow_nan=False)
        if wire.loads(text) != json.loads(text):
            return ["payload does not survive a strict JSON round trip"]
    except (TypeError, ValueError) as e:
        return [f"not serializable: {e}"]
    return []


def snapshot_violations(
    board: DraftBoard,
    snap: dict,
    seat: int,
    policy: RosterValuePolicy | None = None,
    top: int | None = None,
    board_rows: int | None = None,
) -> list[str]:
    """Everything a snapshot claims, checked against the board it came from.

    Must be called with the board unchanged since the snapshot was taken.
    """
    policy = policy or RosterValuePolicy()
    out: list[str] = []

    def bad(msg: str) -> None:
        out.append(f"seat {seat} at pick {board.made}: {msg}")

    for w in wire_violations(snap):
        bad(w)

    # ---- the clock ----------------------------------------------------------
    on_clock = board.on_the_clock()
    nxt = board.next_pick_no(seat)
    expect = {
        "seat": seat,
        "made": board.made,
        "total": len(board.slots),
        "on_clock": on_clock,
        "my_turn": on_clock == seat,
        "round": board.current_round(),
        "picks_away": None if nxt is None else nxt - board.made,
        "n_left": int(board.avail.sum()),
    }
    for key, want in expect.items():
        if snap.get(key) != want:
            bad(f"{key} is {snap.get(key)!r}, board says {want!r}")

    u = board.u
    source = getattr(u, "source", None)
    is_market = (lambda r: False) if source is None else (lambda r: str(source[r]) == "market")
    avail_rows_by_name: dict[str, list[int]] = {}
    for r in range(len(u)):
        if board.avail[r]:
            avail_rows_by_name.setdefault(str(u.names[r]), []).append(r)

    def check_row(kind: str, row: dict, market_expected: bool) -> None:
        name = row.get("name")
        rows = avail_rows_by_name.get(name, [])
        if not rows:
            bad(f"{kind} offers {name!r}, who is not available")
            return
        if not any(str(u.pos[r]) == row.get("position") for r in rows):
            bad(f"{kind} lists {name!r} at {row.get('position')!r}, the board does not")
        if any(is_market(r) for r in rows) != market_expected:
            bad(
                f"{kind} {'must' if market_expected else 'must not'} hold market-only "
                f"rows, but {name!r} is {'not ' if market_expected else ''}one"
            )
        p = row.get("p_survive")
        if p is not None and not (isinstance(p, (int, float)) and 0.0 <= p <= 1.0):
            bad(f"{kind} {name!r} has p_survive {p!r} outside [0, 1]")

    # ---- the shortlist ------------------------------------------------------
    shortlist = snap.get("shortlist") or []
    names = [c.get("name") for c in shortlist]
    if len(names) != len(set(names)):
        bad(f"shortlist repeats a player: {names}")
    if top is not None and len(shortlist) > top:
        bad(f"shortlist has {len(shortlist)} cards, top is {top}")
    for c in shortlist:
        check_row("shortlist", c, market_expected=False)
        if not isinstance(c.get("vorp"), (int, float)):
            bad(f"shortlist {c.get('name')!r} has no numeric VORP")
        reasons = c.get("reasons") or []
        if not reasons:
            bad(f"shortlist {c.get('name')!r} carries no reasons")
        for r in reasons:
            if r.get("kind") not in ("pro", "con") or not r.get("text"):
                bad(f"shortlist {c.get('name')!r} has a malformed reason {r!r}")
    if board.picks_left(seat) > 0:
        want = [c.name for c in recommend(board, policy, n=len(shortlist) or 1, seat=seat)]
        if shortlist and names != want[: len(names)]:
            bad(f"shortlist {names} is not what the engine ranks first {want}")
        if not shortlist and want:
            bad(f"shortlist is empty while the engine would take {want[0]!r}")
    elif shortlist:
        bad("seat has no picks left but the shortlist still offers players")

    # ---- the full board -----------------------------------------------------
    rows = snap.get("board") or []
    keys = [(r.get("name"), r.get("position"), r.get("team")) for r in rows]
    if len(keys) != len(set(keys)):
        dupes = [k for k, n in Counter(keys).items() if n > 1]
        bad(f"board repeats rows: {dupes[:3]}")
    blocked = board.blocked(seat)
    for r in rows:
        check_row("board", r, market_expected=False)
        want_block = blocked.get(r.get("position"), "")
        if r.get("blocked", "") != want_block:
            bad(f"board row {r.get('name')!r} blocked={r.get('blocked')!r}, want {want_block!r}")
    if board_rows is not None:
        rankable = sum(1 for rr in range(len(u)) if board.avail[rr] and not is_market(rr))
        if len(rows) != min(board_rows, rankable):
            bad(f"board shows {len(rows)} rows, expected {min(board_rows, rankable)}")

    # ---- roster, needs, supply ---------------------------------------------
    want_roster = [p.name for p in board.roster(seat)]
    if [p.get("name") for p in snap.get("roster") or []] != want_roster:
        bad("roster does not match the board")
    need = board.needs(seat)
    if snap.get("needs") != [f"{p}x{n}" for p, n in sorted(need.items())]:
        bad(f"needs {snap.get('needs')} do not match the board {need}")
    want_supply = [[pos, n, pos in need] for pos, n in sorted(board.supply().items())]
    if snap.get("supply") != want_supply:
        bad("supply does not match the board")

    # ---- the room panels ----------------------------------------------------
    for p in snap.get("market_watchlist") or []:
        check_row("market watchlist", p, market_expected=True)
    gaps = snap.get("market_gaps") or {}
    for bucket in ("sleeping", "rated"):
        for g in gaps.get(bucket) or []:
            check_row(f"market gap ({bucket})", g, market_expected=False)
            for rank in ("our_rank", "market_rank"):
                if not (isinstance(g.get(rank), int) and g[rank] >= 1):
                    bad(f"market gap {g.get('name')!r} has {rank}={g.get(rank)!r}")
    return out


def relay_violations(local: dict, remote: dict, where: str = "") -> list[str]:
    """Did the relay hand back what the console pushed?

    Compared after both sides go through the wire, because that is what the two
    browsers see. The pick clock may only have moved forward on the relay.
    """
    prefix = f"{where}: " if where else ""
    a = wire.loads(wire.dumps(local))
    b = wire.loads(wire.dumps(remote))
    out: list[str] = []
    if b.get("error"):
        return [f"{prefix}relay answered an error: {b['error']}"]
    for key in sorted((set(a) | set(b)) - RELAY_VOLATILE):
        if a.get(key) != b.get(key):
            out.append(f"{prefix}relay differs on {key!r}")
    la, rb = a.get("seconds_since_pick"), b.get("seconds_since_pick")
    if (la is None) != (rb is None):
        out.append(f"{prefix}relay pick clock {rb!r} vs console {la!r}")
    elif la is not None and rb + CLOCK_SLACK_S < la:
        out.append(f"{prefix}relay pick clock went backwards ({rb:.1f} < {la:.1f})")
    return out


# The local snapshot is taken a moment after the push it is compared with, so
# the relay's clock may trail it by that moment without anything being wrong.
CLOCK_SLACK_S = 2.0
