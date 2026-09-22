"""What the phone gets, and what comes back.

The page is dumb on purpose - it renders what it is handed and computes
nothing - so this module owns every judgement about what is worth a person's
attention at 6:45pm. That ordering is the design: tonight's changes, then
anything waiting on a decision, then the week, then the roster. Everything
below the first card is reference.

Times are localised here rather than in the page. The browser knows its own
timezone but not the league's, and a lineup deadline shown in the wrong one is
worse than no deadline at all.
"""

from __future__ import annotations

import sqlite3
from typing import Any

from puckpilot.season import proposals as proposals_mod
from puckpilot.season import protocol as protocol_mod

DEFAULT_TZ = "America/Toronto"


def _move_kind(m) -> str:
    if m.is_bench:
        return "bench"
    return "start" if m.from_slot in ("BN", "?") else "move"


def _move_detail(m) -> str:
    if m.is_bench:
        return f"was {m.from_slot}"
    if m.from_slot in ("BN", "?"):
        return f"into {m.to_slot}"
    return f"{m.from_slot} to {m.to_slot}"


def build(
    conn: sqlite3.Connection,
    manager: str,
    league_key: str,
    team_name: str,
    plan=None,
    week_plan=None,
    roster=None,
    reasons: dict[str, str] | None = None,
    tz: str = DEFAULT_TZ,
) -> dict[str, Any]:
    """Assemble one manager's view. Every part is optional but the shape is not.

    A missing section renders as absent rather than as an error; a page that
    half-draws is worse than one that says a thing is not there yet, which is
    the lesson the draft relay's cold payload taught.
    """
    snap: dict[str, Any] = {
        "team": team_name,
        "date": "",
        "moves": [],
        "out": [],
        "lock_local": "",
        "playing": None,
        "rostered": None,
        "proposals": [],
        "protocol": None,
        "week": None,
        "roster": [],
    }

    if plan is not None:
        snap["date"] = plan.date
        snap["moves"] = [
            {
                "kind": _move_kind(m),
                "name": m.player.name,
                # The reason, where there is one - the slot change is the
                # instruction, but the reason is what makes it checkable.
                "detail": (reasons or {}).get(m.player.player_key) or _move_detail(m),
            }
            for m in plan.moves
        ]
        snap["out"] = [p.label() for p in plan.out]
        snap["lock_local"] = plan.deadline(tz)
        snap["playing"] = len(plan.playing)
        snap["rostered"] = len(plan.playing) + len(plan.idle) + len(plan.out)

    pending = proposals_mod.pending(conn, manager, league_key)
    snap["proposals"] = [
        {
            "id": p.id,
            "add": p.add_name,
            "drop": p.drop_name if p.drop_player_key else "",
            "why": _why(p),
            "timing": str(p.reason.get("timing", "")),
        }
        for p in pending
    ]

    if week_plan is not None:
        live = protocol_mod.load(conn, manager, league_key, week_plan.week)
        snap["protocol"] = _protocol(live)
        snap["week"] = {
            "week": week_plan.week,
            "opponent": week_plan.opponent,
            "cats": [
                {
                    "label": o.category.label,
                    "ours": _round(o.ours, o.category.key),
                    "theirs": _round(o.theirs, o.category.key),
                    "state": _state(o),
                }
                for o in week_plan.outlook
            ],
            "note": _week_note(week_plan),
        }

    if roster is not None:
        snap["roster"] = [
            {
                "slot": p.selected_slot,
                "name": p.name,
                "team": p.team,
                "opp": "",
                "status": p.status_full or p.status,
            }
            for p in roster.players
        ]
    return snap


def _why(p) -> str:
    helps = ", ".join(p.reason.get("helps", []))
    gain = p.reason.get("gain")
    bits = []
    if gain is not None:
        bits.append(f"+{float(gain):.1f} this week")
    if p.reason.get("games"):
        bits.append(f"{p.reason['games']} games")
    if helps:
        bits.append(helps)
    return " · ".join(bits)


def _protocol(live) -> dict | None:
    if live is None:
        return None
    return {
        "id": live.id,
        "week": live.week,
        "opponent": live.opponent,
        "status": live.status,
        "give_up": [f"{s.category.label} ({s.reason()})" for s in live.of(protocol_mod.CONCEDE)],
        "go_after": [f"{s.category.label} ({s.margin:+.1f})" for s in live.of(protocol_mod.CHASE)],
    }


def _state(o) -> str:
    if not o.reachable:
        return "gone"
    if o.in_play:
        return "close"
    return "ahead" if o.margin > 0 else "behind"


def _round(v: float, key: str) -> float:
    # A rate needs three places to be readable; a counting stat needs one.
    return round(v, 3) if v and abs(v) < 5 else round(v, 1)


def _week_note(wp) -> str:
    counted = [o for o in wp.outlook if o.measured]
    if counted and all(o.lineup_room == 0.0 for o in counted):
        return "Everyone with a game fits in a slot this week - only an add moves anything."
    return ""


def apply_decisions(conn: sqlite3.Connection, decisions: list[dict]) -> list[str]:
    """Record decisions collected from the page. Returns what happened, in words.

    A decision that cannot be applied - already decided, gone, unknown - is
    reported rather than raised. These arrive in batches from a public endpoint
    and one stale tap must not stop the rest.
    """
    out: list[str] = []
    for d in sorted(decisions, key=lambda x: x.get("seq", 0)):
        kind, ident, approve = d.get("kind"), d.get("id"), bool(d.get("approve"))
        verb = "approved" if approve else "rejected"
        try:
            if kind == "protocol":
                p = protocol_mod.decide(conn, int(ident), approve)
                out.append(f"protocol #{p.id} (week {p.week}) {verb}")
            elif kind == "proposal":
                p = proposals_mod.decide(conn, int(ident), approve)
                out.append(f"proposal #{p.id} {p.add_name} {verb}")
            else:
                out.append(f"ignored decision of unknown kind {kind!r}")
        except (protocol_mod.ProtocolError, proposals_mod.ProposalError, ValueError) as e:
            out.append(f"could not apply {kind} #{ident}: {e}")
    return out
