"""Multi-position eligibility: which slots a roster can actually fill.

Yahoo lets a player fill every position he is eligible at - Draisaitl C/LW,
Robertson LW/RW - and a third of the top 200 by ADP carry more than one. The
valuation deliberately stays on one primary position (replacement level is a
per-position quantity, and VORP is what the draft sim was tuned on). This
module is for roster ACCOUNTING only: whether a candidate would start, which
positions a roster still needs, and who plays on a given night.

With single-position players a greedy fill is optimal. With overlapping
eligibility it is not - a C/LW placed at C can block a C-only player who
arrives next - so slots are filled as a bipartite matching. Rosters are tiny
(16 players, 13 slots), so plain augmenting paths are fast enough to run inside
the season replay.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from functools import lru_cache

SKATER_POSITIONS = frozenset({"C", "L", "R", "D"})
UTIL = "U"
# Yahoo's position codes -> this codebase's NHL-style ones. Util, IR and friends
# are slot types, not positions, and are dropped.
YAHOO_TO_POS = {"C": "C", "LW": "L", "RW": "R", "D": "D", "G": "G"}


def parse_yahoo_positions(raw: str | None) -> frozenset[str]:
    """ "C,LW,Util,IR+" -> frozenset({"C", "L"})."""
    return frozenset(YAHOO_TO_POS[p] for p in str(raw or "").split(",") if p in YAHOO_TO_POS)


def slot_units(slots: Iterable[tuple[str, int]], util_slots: int) -> tuple[str, ...]:
    """(("C", 2), ("D", 4)), 1 -> ("C", "C", "D", "D", "D", "D", "U")."""
    units: list[str] = []
    for pos, n in slots:
        units += [pos] * int(n)
    return tuple(units + [UTIL] * int(util_slots))


def _fits(unit: str, elig: frozenset[str]) -> bool:
    return unit in elig or (unit == UTIL and bool(elig & SKATER_POSITIONS))


def _augment(p: int, eligs: Sequence[frozenset[str]], units, owner: list, seen: set) -> bool:
    for j, unit in enumerate(units):
        if j in seen or not _fits(unit, eligs[p]):
            continue
        seen.add(j)
        if owner[j] is None or _augment(owner[j], eligs, units, owner, seen):
            owner[j] = p
            return True
    return False


def _matching(eligs: Sequence[frozenset[str]], units: Sequence[str]) -> list:
    owner: list = [None] * len(units)
    for p in range(len(eligs)):
        _augment(p, eligs, units, owner, set())
    return owner


def max_starters(eligs: Sequence[frozenset[str]], units: Sequence[str]) -> int:
    return sum(o is not None for o in _matching(eligs, units))


@lru_cache(maxsize=200_000)
def starts_in_order(eligs: tuple[frozenset[str], ...], units: tuple[str, ...]) -> tuple[bool, ...]:
    """Who starts, taking players in priority order.

    A player starts if an augmenting path can seat him without unseating anyone
    already placed - they may move to another slot they are eligible for. On a
    transversal matroid that greedy is optimal for any priority order, which is
    what makes it a fair stand-in for a manager setting the best lineup.

    Cached on the eligibility pattern: across a season replay the same shapes
    recur constantly, and the answer depends on nothing else.
    """
    owner: list = [None] * len(units)
    out = []
    for p in range(len(eligs)):
        out.append(_augment(p, eligs, units, owner, set()))
    return tuple(out)


def open_positions(
    roster: Sequence[frozenset[str]], units: Sequence[str], positions: Iterable[str]
) -> set[str]:
    """Positions at which one more player would START on this roster.

    Asked of a hypothetical single-position player for each position, so a
    candidate with several eligibilities starts if ANY of them is open - after
    the existing roster reshuffles itself, which is exactly the flexibility a
    per-position count cannot see.
    """
    base = max_starters(roster, units)
    return {pos for pos in positions if max_starters([*roster, frozenset({pos})], units) > base}


def unfilled_starting_slots(roster: Sequence[frozenset[str]], units: Sequence[str]) -> int:
    """Non-util starting slots this roster cannot fill however it is arranged -
    the multi-position version of summing unmet roster minimums."""
    fixed = [u for u in units if u != UTIL]
    return len(fixed) - max_starters(roster, fixed)
