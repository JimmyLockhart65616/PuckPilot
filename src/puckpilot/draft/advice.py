"""Turn live draft state into a ranked shortlist.

`RosterValuePolicy` answers "which one player", which is all a simulated bot
needs. A human on a 30-second clock needs the top handful and enough of the
reasoning to disagree — so this returns the same score the engine ranks on,
alongside the pieces that produced it.

Everything here is numpy over a pre-built board. Measured on the real universe
this is well under a millisecond, which is the whole reason the draft-time path
contains no model call: the board is computed once before the draft, and every
pick after that is arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from puckpilot.draft.board import Candidate, DraftBoard
from puckpilot.draft.engine import RosterValuePolicy, eligible_positions
from puckpilot.engine.categories import CATALOG

# Shown next to each name so a pick can be sanity-checked against the projection
# it rests on, in the league's own categories.
# Every category the catalog knows, so a Candidate carries whatever its league
# happens to score. A hardcoded list silently dropped SA and SV% - both scored
# in the Ajaxians league - from the goalie cards. Callers filter this down to
# their own league's categories; the pool is deliberately wider than any one
# league's set.
DISPLAY_CATS = tuple(sorted({c.key for c in CATALOG.values()}))


def _opt(v) -> float | None:
    """NaN and missing both mean "we do not know", and must not read as zero."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f else f


def _fills_starter(board: DraftBoard, seat: int, position: str) -> bool:
    """Would this player step straight into a starting slot (or util)?"""
    slots = dict(board.rules.shape.slots)
    counts = board.counts[seat]
    if counts.get(position, 0) < slots.get(position, 0):
        return True
    if position == "G":
        return False
    used = sum(max(0, counts.get(p, 0) - s) for p, s in slots.items() if p != "G")
    return used < board.rules.shape.util_slots


def recommend(
    board: DraftBoard,
    policy: RosterValuePolicy | None = None,
    n: int = 15,
    seat: int | None = None,
    enforce_eligibility: bool = True,
) -> list[Candidate]:
    """Top `n` available players for `seat`, best first.

    Ranked on exactly what `RosterValuePolicy.pick` would maximize, so the
    console's first row is always the pick the engine would make — the shortlist
    and the sim can never disagree.
    """
    seat = board.my_seat if seat is None else seat
    policy = policy or RosterValuePolicy()
    u = board.u

    picks_left = board.picks_left(seat)
    # Built by the board, not here: `avail` lets the policy re-base replacement
    # level against who is actually left, and a caller that assembles its own
    # ctx scores a different board than the console shows.
    ctx = board.pick_context(seat)
    score = policy.score(u, board.counts[seat], board.rules, ctx)
    # The displayed probability is fitted to real rooms, not to the scoring
    # knob - see RosterValuePolicy.display_spread. Everything downstream of
    # here is human-facing, so it gets the honest number.
    p_survive = policy.survival(u, ctx, spread=policy.display_spread)

    mask = board.avail.copy()
    if enforce_eligibility and picks_left > 0:
        allowed = eligible_positions(board.counts[seat], board.rules, picks_left)
        pos_ok = np.isin(u.pos, list(allowed))
        # Mirrors _pick_best: if the roster rules leave nothing, rules yield
        # rather than the board going empty.
        if (mask & pos_ok).any():
            mask &= pos_ok
    if not mask.any():
        return []

    rows = np.flatnonzero(mask)
    top = rows[np.argsort(-score[rows], kind="stable")[:n]]

    frame = u.frame
    out: list[Candidate] = []
    for row in top:
        r = int(row)
        record = frame.iloc[r]
        projected = {
            c: float(record[c])
            for c in DISPLAY_CATS
            if c in frame.columns and record[c] == record[c]  # skip NaN
        }
        out.append(
            Candidate(
                row=r,
                player_id=int(u.ids[r]),
                name=str(u.names[r]),
                position=str(u.pos[r]),
                team=str(record.get("team") or "?"),
                vorp=float(u.vorp[r]),
                z_total=float(u.z_total[r]),
                score=float(score[r]),
                adp_rank=float(u.adp_rank[r]),
                p_survive=float(p_survive[r]),
                fills_starter=_fills_starter(board, seat, str(u.pos[r])),
                projected=projected,
                age=_opt(record.get("age")),
                train_gp=_opt(record.get("train_gp")),
            )
        )
    return out


def survivors(board: DraftBoard, candidates: list[Candidate], threshold: float = 0.6) -> list[str]:
    """Names likely still there next turn — the ones NOT to spend this pick on."""
    return [c.name for c in candidates if c.p_survive >= threshold]


def format_board(board: DraftBoard, candidates: list[Candidate], width: int = 96) -> str:
    """Plain-text shortlist, for the terminal and for draft-night logs."""
    seat = board.my_seat
    on_clock = board.on_the_clock()
    rnd = board.current_round()
    header = [
        f"Round {rnd}  ·  pick {board.made + 1}/{len(board.slots)}  ·  "
        f"seat {on_clock}{' (YOU)' if on_clock == seat else ''}",
    ]
    nxt = board.next_pick_no(seat)
    if nxt is not None:
        header.append(f"Your next pick: #{nxt + 1}  ({nxt - board.made} picks away)")
    needs = board.needs(seat)
    if needs:
        header.append("Must still fill: " + ", ".join(f"{p}x{n}" for p, n in sorted(needs.items())))

    lines = [
        *header,
        "",
        f"{'#':>2} {'Name':<24}{'Pos':<4}{'Tm':<4}{'VORP':>6}"
        f"{'Score':>7}{'ADP':>6}{'Survive':>8}  Fills",
    ]
    for i, c in enumerate(candidates, start=1):
        lines.append(
            f"{i:>2} {c.name:<24}{c.position:<4}{c.team:<4}{c.vorp:>6.2f}"
            f"{c.score:>7.2f}{c.adp_rank:>6.0f}{c.p_survive:>7.0%}  "
            f"{'starter' if c.fills_starter else 'bench'}"
        )
    return "\n".join(lines)[: width * 400]


# A projection standing on less than about three-quarters of a season is thin
# enough that the market's read is probably better than ours; past 32 the age
# curve is fading a player the market may still be paying a name premium for.
THIN_EVIDENCE_GP = 60.0
FADING_AGE = 32.0
# How far apart the two boards must be before it counts as a disagreement.
# Without this the "room rates them higher" side fills with the room's best
# remaining player at each position - ours #2 against room #1 is not a
# disagreement, it is two boards agreeing.
MIN_RANK_GAP = 5


@dataclass(frozen=True)
class Gap:
    """One disagreement between our board and the room's."""

    name: str
    position: str
    team: str
    our_rank: int  # among available players AT HIS POSITION
    market_rank: int
    vorp: float
    flag: str  # "" | "thin history" | "fading?"


def _position_ranks(values: np.ndarray, mask: np.ndarray) -> dict[int, int]:
    """1-based rank within `mask`, best first. Ties broken stably."""
    rows = np.flatnonzero(mask)
    order = rows[np.argsort(values[rows], kind="stable")]
    return {int(r): i + 1 for i, r in enumerate(order)}


def market_disagreement(board: DraftBoard, n: int = 5) -> tuple[list[Gap], list[Gap]]:
    """Where our board and the room disagree, ranked WITHIN position.

    Returns (room_is_sleeping_on, room_rates_above_us).

    Ranking within position is not a refinement, it is the whole thing working.
    Replacement level differs enormously by position - the 28th-best centre sits
    around z -0.9 while the 48th-best defenceman is near -6 - so an overall
    rank-vs-ADP comparison measures that structural offset and almost nothing
    else. Done naively this panel leads with Brayden Point, our #214 against ADP
    60, which is not a disagreement about Brayden Point. Asked within position
    it becomes "we have him 4th-best D left, the room has him 11th", which is
    both true and directly actionable.

    Deliberately does not go through `recommend()`: that returns the top rows by
    score, and a player the market rates far above us is by construction below
    that cut. The panel has to see the whole board.
    """
    u = board.u
    avail = board.avail
    has_market = getattr(u, "has_market", None)
    if has_market is None:
        has_market = u.adp_rank < len(u)
    live = avail & has_market
    if not live.any():
        return [], []

    ours: dict[int, int] = {}
    theirs: dict[int, int] = {}
    for pos in {str(p) for p in u.pos}:
        at_pos = live & (u.pos == pos)
        if not at_pos.any():
            continue
        ours |= _position_ranks(-u.vorp, at_pos)
        theirs |= _position_ranks(u.adp_rank, at_pos)

    def _flag(row: int) -> str:
        rec = u.frame.iloc[row]
        gp, age = rec.get("train_gp"), rec.get("age")
        # Evidence first: a 26-year-old with 20 NHL games is the same problem as
        # a 20-year-old with 20, and age alone would call only one of them out.
        if gp is not None and gp == gp and float(gp) < THIN_EVIDENCE_GP:
            return "thin history"
        if age is not None and age == age and float(age) >= FADING_AGE:
            return "fading?"
        return ""

    def _gap(row: int) -> Gap:
        rec = u.frame.iloc[row]
        return Gap(
            name=str(u.names[row]),
            position=str(u.pos[row]),
            team=str(rec.get("team") or "?"),
            our_rank=ours[row],
            market_rank=theirs[row],
            vorp=float(u.vorp[row]),
            flag=_flag(row),
        )

    rows = list(ours)
    # Our side additionally requires the player to be startable: a bargain at a
    # position where the 6th-best left is sub-replacement is not a bargain.
    sleeping = sorted(
        (r for r in rows if u.vorp[r] > 0 and theirs[r] - ours[r] >= MIN_RANK_GAP),
        key=lambda r: (ours[r] - theirs[r], ours[r]),
    )
    # Sorted by how SOON the room takes him, not by the size of the gap. A
    # player the room reaches for at #9 is a decision you face in a few minutes;
    # one it takes at #47 is an argument you will never have to have. The gap
    # only breaks ties.
    rated = sorted(
        (r for r in rows if ours[r] - theirs[r] >= MIN_RANK_GAP),
        key=lambda r: (theirs[r], theirs[r] - ours[r]),
    )
    return [_gap(r) for r in sleeping[:n]], [_gap(r) for r in rated[:n]]
