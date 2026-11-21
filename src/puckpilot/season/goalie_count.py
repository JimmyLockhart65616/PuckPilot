"""How a roster's goalie starts are counted on one night.

One rule, used everywhere a week is projected: the live plan
(`week.expected_starts`), the add gate's replay (`add_gate`) and the odds
calibration (`calibration.build_cases`). They were three copies of the same
sort; a change to one had to be made in all three or the gate and the
calibration would judge a model the live run does not use.

"top" (as first built): the G slots' worth of goalies with the most expected
value each count their whole P(start), the rest nothing - a lineup set in the
morning by P(start) and left. "min": every goalie counts for the chance that
he starts AND a slot is free of better goalies who start - the lineup of a
manager who plays whoever is confirmed, so a night's count is
E[min(slots, goalies starting)]. Measured live (`season g3`, weeks 1-7 of
2026-27, 84 team-weeks): "min" moved the league's saves bias from +32.6 to
+29.1 a team-week and the week-start Brier from .2179 to .2199 - it does not
earn a default.
"""

from __future__ import annotations

GOALIE_COUNTS = ("top", "min")


def slot_chances(goalies: list[tuple[float, float, int, str]], slots: int) -> dict[int, float]:
    """pid -> P(he starts and is in a G slot), for (value, P(start), pid, club).

    A goalie is in a slot when fewer than `slots` goalies worth more than him
    start. Two goalies of one club cannot both start, so a club's better
    goalies count as one chance, and his own club-mates not at all - if he
    starts, they do not. The values sum to E[min(slots, clubs starting)].
    """
    from puckpilot.season.odds import trials_pmf

    order = sorted(goalies, key=lambda g: (-g[0], g[2]))
    out: dict[int, float] = {}
    for i, (_v, p, pid, club) in enumerate(order):
        if p <= 0 or slots <= 0:
            continue
        ahead: dict[str, float] = {}
        for _v2, p2, _pid2, club2 in order[:i]:
            if club2 != club and p2 > 0:
                ahead[club2] = min(ahead.get(club2, 0.0) + p2, 1.0)
        free = float(trials_pmf(list(ahead.values()))[:slots].sum()) if ahead else 1.0
        if free > 0:
            out[pid] = p * free
    return out


def counted_starts(
    goalies: list[tuple[float, float, int, str]], slots: int, rule: str = "top"
) -> dict[int, float]:
    """pid -> the starts a night counts for him, for (value, P(start), pid, club).

    "top": the `slots` goalies with the most P(start) x value, ties to the
    higher P(start) then the higher pid, each counting his whole P(start); the
    dict is in that order. "min": `slot_chances`. A goalie left out counts 0.
    """
    if rule not in GOALIE_COUNTS:
        raise ValueError(f"goalie count must be one of {GOALIE_COUNTS}, not {rule!r}")
    if rule == "min":
        return slot_chances(goalies, slots)
    order = sorted(((p * v, p, pid) for v, p, pid, _club in goalies), reverse=True)
    return {pid: p for _, p, pid in order[: max(slots, 0)]}
