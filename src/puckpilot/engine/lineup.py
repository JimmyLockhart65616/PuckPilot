from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from puckpilot.engine.valuation import LeagueShape

# which slots each (NHL) position may fill; UTIL is any skater, Yahoo-style.
# Multi-position Yahoo eligibility slots in here once the Yahoo API lands.
POSITION_SLOTS = {
    "C": ("C", "UTIL"),
    "L": ("L", "UTIL"),
    "R": ("R", "UTIL"),
    "D": ("D", "UTIL"),
    "G": ("G",),
}

_INELIGIBLE = -1e9


def slot_instances(shape: LeagueShape) -> list[str]:
    """Expand the shape into one entry per startable slot, e.g. C,C,L,L,...,UTIL,UTIL,G,G."""
    out: list[str] = []
    for pos, n in shape.slots:
        out.extend([pos] * n)
    out.extend(["UTIL"] * shape.util_slots)
    return out


def optimize_lineup(
    players: list[tuple[int, str, float]],
    shape: LeagueShape,
) -> dict[int, str]:
    """Assign players to starting slots maximizing total expected value.

    players: (player_id, position, expected value tonight). Solved as an
    assignment problem so future multi-slot eligibility keeps working; with
    today's single NHL positions it reduces to greedy fill, but the LP cost
    is negligible (~18x14 matrix).

    Returns {player_id: slot_name} for assigned starters; everyone else sits.
    Zero/negative-value players may occupy otherwise-empty slots harmlessly.
    """
    if not players:
        return {}
    slots = slot_instances(shape)
    value = np.full((len(players), len(slots)), _INELIGIBLE)
    for i, (_pid, pos, v) in enumerate(players):
        for j, slot in enumerate(slots):
            if slot in POSITION_SLOTS.get(pos, ()):
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
