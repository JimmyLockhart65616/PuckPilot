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
    p_survive = policy.survival(u, ctx)

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
