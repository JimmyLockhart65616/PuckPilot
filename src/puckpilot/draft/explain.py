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


@dataclass(frozen=True)
class Reason:
    kind: str  # "pro" | "con"
    text: str


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
        pros.append(Reason("pro", f"Starts right away at {pos}"))
    else:
        slot_count = dict(board.rules.shape.slots).get(pos, 0)
        cons.append(
            Reason("con", f"Rides the bench — {pos} starters already filled ({slot_count})")
        )

    need = board.needs(seat).get(pos)
    if need:
        pros.append(Reason("pro", f"You still must fill {need} more at {pos}"))

    cap = board.rules.caps.get(pos)
    have = counts.get(pos, 0)
    if cap and have >= cap - 1:
        cons.append(Reason("con", f"Near the {pos} cap ({have}/{cap}) — limits later picks"))

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
            )
        )
    elif candidate.p_survive <= WILL_NOT_LAST:
        pros.append(
            Reason("pro", f"Will not last — {candidate.p_survive:.0%} chance he is there next turn")
        )

    # --- value vs the room ------------------------------------------------
    pick_now = board.made + 1
    if candidate.adp_rank and candidate.adp_rank > pick_now + 12:
        cons.append(
            Reason(
                "con",
                f"Ahead of the market: ADP {candidate.adp_rank:.0f} vs pick {pick_now}",
            )
        )
    elif candidate.adp_rank and candidate.adp_rank + 12 < pick_now:
        pros.append(
            Reason("pro", f"Value vs the room: ADP {candidate.adp_rank:.0f}, pick {pick_now}")
        )

    # --- best left at the position ---------------------------------------
    same_pos = [c for c in (alternatives or []) if c.position == pos and c is not candidate]
    if same_pos:
        gap = candidate.vorp - same_pos[0].vorp
        if gap > 1.0:
            pros.append(Reason("pro", f"Clear best {pos} left (+{gap:.1f} VORP on the next one)"))
        elif abs(gap) < 0.4:
            cons.append(
                Reason("con", f"{same_pos[0].name} is a near-equal {pos} ({gap:+.1f} VORP)")
            )

    # --- categories -------------------------------------------------------
    profile = _cat_profile(board, candidate.row, labels)
    strong = [lbl for lbl, z in profile if z >= STRONG_CAT][:3]
    weak = [lbl for lbl, z in profile if z <= WEAK_CAT][-2:]
    if strong:
        pros.append(Reason("pro", f"Carries {', '.join(strong)}"))
    if weak:
        # Phrasing matters here: this is a category he does NOT help, and it sits
        # in the cons column. Anything softer reads as a positive at a glance.
        cons.append(Reason("con", f"Contributes almost nothing in {', '.join(weak)}"))

    # thinnest category on our roster, if he helps it
    totals = roster_category_totals(board, seat)
    if totals and board.roster(seat):
        thinnest = min(totals, key=lambda k: totals[k])
        z_here = float(board.u.z_by_cat[thinnest][candidate.row])
        if z_here >= STRONG_CAT:
            pros.append(
                Reason("pro", f"Shores up {labels.get(thinnest, thinnest)}, your thinnest category")
            )

    if pos == "G":
        cons.append(
            Reason("con", "Goalie — rate stats are near-noise year to year; workload is the signal")
        )

    return pros + cons


def summarize(
    board: DraftBoard,
    candidates: list[Candidate],
    labels: dict[str, str] | None = None,
    top: int = 3,
) -> list[tuple[Candidate, list[Reason]]]:
    """The shortlist a human actually decides from."""
    shortlist = candidates[:top]
    return [(c, explain(board, c, labels, alternatives=candidates)) for c in shortlist]
