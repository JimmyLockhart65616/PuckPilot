"""Why the engine likes a player — and why you might not.

On draft night the engine does not pick. It offers a short list and the reasons
behind it, and a human decides. That only works if the reasons are honest: a
shortlist that only ever argues *for* its top choice is an autopicker with extra
steps.

So every candidate gets both sides. The cons are drawn from the same numbers as
the pros — bench overflow, positional caps, category weakness, and above all
whether the player will simply still be there next turn, which is the most
common reason to take someone else.

Nothing here is generated prose. Each reason is a fact about the board with a
number attached, so it can be checked and it cannot drift from what the engine
actually scored.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from puckpilot.draft.board import Candidate, DraftBoard

# Above this, waiting is usually right: the room is unlikely to take him.
LIKELY_TO_LAST = 0.65
# Below this, he is going before our next turn if we do not act.
WILL_NOT_LAST = 0.35
# Category z above/below which a player is notably strong/weak.
STRONG_CAT = 0.75
WEAK_CAT = -0.5


# How much a fact should weigh on the decision, lowest first. Timing leads
# because it is the only thing that cannot be recovered later: a player who
# will not last is a decision now, whereas a category edge keeps until the next
# pick. Grouping all pros then all cons - which is what this used to do - buries
# the deciding fact under agreement.
TIMING, NEED, MARKET, QUALITY, CATEGORY, GENERIC = range(6)
# How far ahead to look for a positional cliff, and how big a drop counts.
# Three deep because one player can be replaced; a gap that survives three
# is the position actually running out.
CLIFF_STEPS = 3
CLIFF_VORP = 1.5


@dataclass(frozen=True)
class Reason:
    kind: str  # "pro" | "con"
    text: str
    weight: int = GENERIC


def _cat_profile(board: DraftBoard, row: int, labels: dict[str, str]) -> list[tuple[str, float]]:
    """(label, z) per category for one player, strongest first."""
    out = []
    for key, arr in board.u.z_by_cat.items():
        label = labels.get(key, key)
        out.append((label, float(arr[row])))
    out.sort(key=lambda kv: -kv[1])
    return out


def roster_category_totals(board: DraftBoard, seat: int | None = None) -> dict[str, float]:
    """Summed category z across a seat's roster — where we are thin or deep."""
    seat = board.my_seat if seat is None else seat
    rows = [p.row for p in board.roster(seat)]
    totals: dict[str, float] = {}
    for key, arr in board.u.z_by_cat.items():
        totals[key] = float(np.sum(arr[rows])) if rows else 0.0
    return totals


def explain(
    board: DraftBoard,
    candidate: Candidate,
    labels: dict[str, str] | None = None,
    alternatives: list[Candidate] | None = None,
) -> list[Reason]:
    """Pros and cons for one candidate, most decision-relevant first."""
    labels = labels or {}
    pros: list[Reason] = []
    cons: list[Reason] = []
    seat = board.my_seat
    counts = board.counts[seat]
    pos = candidate.position

    # --- roster fit ------------------------------------------------------
    if candidate.fills_starter:
        pros.append(Reason("pro", f"Starts right away at {pos}", NEED))
    else:
        slot_count = dict(board.rules.shape.slots).get(pos, 0)
        cons.append(
            Reason("con", f"Rides the bench — {pos} starters already filled ({slot_count})", NEED)
        )

    need = board.needs(seat).get(pos)
    if need:
        pros.append(Reason("pro", f"You still must fill {need} more at {pos}", NEED))

    cap = board.rules.caps.get(pos)
    have = counts.get(pos, 0)
    if cap and have >= cap - 1:
        cons.append(Reason("con", f"Near the {pos} cap ({have}/{cap}) — limits later picks", NEED))

    # --- timing: the most common reason to take someone else -------------
    if candidate.p_survive >= LIKELY_TO_LAST:
        others = [
            c for c in (alternatives or []) if c.p_survive < WILL_NOT_LAST and c is not candidate
        ]
        tail = f"; {others[0].name} will not" if others else ""
        cons.append(
            Reason(
                "con",
                f"Likely still there next turn ({candidate.p_survive:.0%}) — you could wait{tail}",
                TIMING,
            )
        )
    elif candidate.p_survive <= WILL_NOT_LAST:
        pros.append(
            Reason(
                "pro",
                f"Will not last — {candidate.p_survive:.0%} chance he is there next turn",
                TIMING,
            )
        )

    # --- value vs the room ------------------------------------------------
    pick_now = board.made + 1
    if candidate.adp_rank and candidate.adp_rank > pick_now + 12:
        cons.append(
            Reason(
                "con",
                f"Ahead of the market: ADP {candidate.adp_rank:.0f} vs pick {pick_now}",
                MARKET,
            )
        )
    elif candidate.adp_rank and candidate.adp_rank + 12 < pick_now:
        pros.append(
            Reason(
                "pro", f"Value vs the room: ADP {candidate.adp_rank:.0f}, pick {pick_now}", MARKET
            )
        )

    # --- depth left at the position ---------------------------------------
    # Read against the real remaining pool, not against one player on an
    # already-truncated shortlist: the old version could only say "better than
    # the next name on this list", which is not the same question as "what will
    # still be here if I wait".
    left = board.supply().get(pos, 0)
    cliff = board.depth_after(candidate.row, steps=CLIFF_STEPS)
    if cliff >= CLIFF_VORP:
        pros.append(
            Reason(
                "pro",
                f"Last {pos} before a {cliff:.1f} VORP cliff - {left} left at the position",
                QUALITY,
            )
        )
    same_pos = [c for c in (alternatives or []) if c.position == pos and c is not candidate]
    if same_pos:
        gap = candidate.vorp - same_pos[0].vorp
        if cliff < CLIFF_VORP and abs(gap) < 0.4:
            cons.append(
                Reason(
                    "con",
                    f"{same_pos[0].name} is a near-equal {pos} ({gap:+.1f} VORP), "
                    f"{left} left at the position",
                    QUALITY,
                )
            )

    # --- categories -------------------------------------------------------
    profile = _cat_profile(board, candidate.row, labels)
    strong = [lbl for lbl, z in profile if z >= STRONG_CAT][:3]
    weak = [lbl for lbl, z in profile if z <= WEAK_CAT][-2:]
    if strong:
        pros.append(Reason("pro", f"Carries {', '.join(strong)}", CATEGORY))
    if weak:
        # Phrasing matters here: this is a category he does NOT help, and it sits
        # in the cons column. Anything softer reads as a positive at a glance.
        cons.append(Reason("con", f"Contributes almost nothing in {', '.join(weak)}", CATEGORY))

    # thinnest category on our roster, if he helps it
    totals = roster_category_totals(board, seat)
    if totals and board.roster(seat):
        thinnest = min(totals, key=lambda k: totals[k])
        z_here = float(board.u.z_by_cat[thinnest][candidate.row])
        if z_here >= STRONG_CAT:
            pros.append(
                Reason(
                    "pro",
                    f"Shores up {labels.get(thinnest, thinnest)}, your thinnest category",
                    CATEGORY,
                )
            )

    if pos == "G":
        cons.append(
            Reason("con", "Goalie — rate stats are near-noise year to year; workload is the signal")
        )

    # Most decision-relevant first, pros ahead of cons at equal weight so a card
    # still reads as a case rather than an alternating list.
    return _order_for_test(pros + cons)


def _order_for_test(reasons: list[Reason]) -> list[Reason]:
    """The ordering rule, exposed so a test can pin it without a whole board."""
    return sorted(reasons, key=lambda r: (r.weight, r.kind != "pro"))


def summarize(
    board: DraftBoard,
    candidates: list[Candidate],
    labels: dict[str, str] | None = None,
    top: int = 3,
) -> list[tuple[Candidate, list[Reason]]]:
    """The shortlist a human actually decides from."""
    shortlist = candidates[:top]
    return [(c, explain(board, c, labels, alternatives=candidates)) for c in shortlist]
