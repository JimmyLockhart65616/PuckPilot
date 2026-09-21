from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from puckpilot.engine.valuation import LeagueShape

# which slots each position may fill; UTIL is any skater, Yahoo-style.
POSITION_SLOTS = {
    "C": ("C", "UTIL"),
    "L": ("L", "UTIL"),
    "R": ("R", "UTIL"),
    "D": ("D", "UTIL"),
    "G": ("G",),
}

_INELIGIBLE = -1e9


def _slots_for(pos: str | frozenset[str] | set[str] | tuple[str, ...]) -> frozenset[str]:
    """Startable slots for one position, or for a set of them.

    A player's eligibility may arrive as a single NHL position (the sims and
    replays, which have only ever known one) or as the set Yahoo allows (the
    live roster, where 172 of 562 mapped players are multi-eligible). Passing a
    set widens the slots it can fill; it can never narrow them, so a
    single-position caller gets exactly the old behaviour.
    """
    if isinstance(pos, str):
        return frozenset(POSITION_SLOTS.get(pos, ()))
    out: set[str] = set()
    for p in pos:
        out.update(POSITION_SLOTS.get(p, ()))
    return frozenset(out)


def slot_instances(shape: LeagueShape) -> list[str]:
    """Expand the shape into one entry per startable slot, e.g. C,C,L,L,...,UTIL,UTIL,G,G."""
    out: list[str] = []
    for pos, n in shape.slots:
        out.extend([pos] * n)
    out.extend(["UTIL"] * shape.util_slots)
    return out


def optimize_lineup(
    players: list[tuple[int, str | frozenset[str], float]],
    shape: LeagueShape,
) -> dict[int, str]:
    """Assign players to starting slots maximizing total expected value.

    players: (player_id, position, expected value tonight), where position is
    either one position or the set of them the league allows. Solved as an
    assignment problem because with multi-position eligibility greedy fill is
    no longer optimal: a C/RW taking the last C slot can strand a C-only
    player the RW slot cannot hold. The LP cost is negligible (~18x14).

    Returns {player_id: slot_name} for assigned starters; everyone else sits.
    Zero/negative-value players may occupy otherwise-empty slots harmlessly.
    """
    if not players:
        return {}
    slots = slot_instances(shape)
    value = np.full((len(players), len(slots)), _INELIGIBLE)
    for i, (_pid, pos, v) in enumerate(players):
        allowed = _slots_for(pos)
        for j, slot in enumerate(slots):
            if slot in allowed:
                value[i, j] = v

    # pad with one dummy player per slot so any slot can stay empty instead of
    # force-taking an ineligible (or negative-value) player
    value = np.vstack([value, np.zeros((len(slots), len(slots)))])

    rows, cols = linear_sum_assignment(value, maximize=True)
    out: dict[int, str] = {}
    for i, j in zip(rows, cols, strict=True):
        if i < len(players) and value[i, j] > _INELIGIBLE / 2:
            out[players[i][0]] = slots[j]
    return out
