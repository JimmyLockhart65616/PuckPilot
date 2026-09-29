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
    if getattr(m, "is_ir", False):
        return "ir"
    if getattr(m, "is_activation", False):
        return "activate"
    if m.is_bench:
        return "bench"
    return "start" if m.from_slot in ("BN", "?") else "move"


def _move_detail(m) -> str:
    if getattr(m, "is_ir", False):
        return f"{m.from_slot} to {m.to_slot} - frees a roster spot"
    if getattr(m, "is_activation", False):
        return f"off {m.from_slot}, into {m.to_slot}"
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
    week_no: int | None = None,
    next_run=None,
    acted: dict | None = None,
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
        "alerts": [],
        # When the next run is due. The relay judges staleness against it: a
        # page with nothing to do until 11:00 is quiet, not dead.
        "next_run_utc": None,
        "next_local": "",
        # Whether tonight's changes were made in Yahoo, and when - so a plan
        # nobody carried out never reads as done.
        "acted": None,
    }
    if next_run is not None:
        from datetime import UTC
        from zoneinfo import ZoneInfo

        local = next_run.astimezone(ZoneInfo(tz))
        snap["next_run_utc"] = next_run.astimezone(UTC).isoformat(timespec="seconds")
        snap["next_local"] = local.strftime("%a %I:%M %p").replace(" 0", " ")

    if plan is not None:
        snap["date"] = plan.date
        # Roster moves first: they decide who is available to the rest.
        snap["moves"] = [
            {
                "kind": _move_kind(m),
                "name": m.player.name,
                # The reason, where there is one - the slot change is the
                # instruction, but the reason is what makes it checkable.
                "detail": (reasons or {}).get(m.player.player_key) or _move_detail(m),
            }
            for m in (*getattr(plan, "ir_moves", ()), *plan.moves)
        ]
        snap["alerts"] = list(getattr(plan, "ir_alerts", ()))
        snap["out"] = [p.label() for p in plan.out]
        snap["lock_local"] = plan.deadline(tz)
        snap["playing"] = len(plan.playing)
        snap["rostered"] = len(plan.playing) + len(plan.idle) + len(plan.out)

    if acted is not None:
        snap["acted"] = _acted(acted, tz)

    # In the order they were proposed: the search's best first, and a later
    # card's "assumes ... is made too" refers to one above it.
    pending = sorted(proposals_mod.pending(conn, manager, league_key), key=lambda p: p.id)
    snap["proposals"] = [
        {
            "id": p.id,
            "add": p.add_name,
            "drop": p.drop_name if p.drop_player_key else "",
            "why": _why(p),
            "timing": str(p.reason.get("timing", "")),
            "detail": _reasons(p.reason.get("detail")),
        }
        for p in pending
    ]

    # The protocol card shows on every run, not only the one that derived it -
    # a push replaces the whole page, and an Approve button that vanishes on
    # the next run is one that cannot be pressed.
    week_no = week_plan.week if week_plan is not None else week_no
    if week_no is not None:
        snap["protocol"] = _protocol(protocol_mod.load(conn, manager, league_key, week_no))
    if week_plan is not None:
        banked = getattr(week_plan, "banked", False)
        snap["week"] = {
            "week": week_plan.week,
            "opponent": week_plan.opponent,
            "status": getattr(week_plan, "status", ""),
            "days_left": getattr(week_plan, "days_left", 0),
            # Starts, not team games: what each side can still actually collect.
            "games_left": {"ours": week_plan.our_games, "theirs": week_plan.their_games},
            # Calibrated (gate G1) - the only reason a percentage is allowed here.
            "expected": _one(getattr(week_plan, "expected", None)),
            "of": len(week_plan.outlook),
            "cats": [
                {
                    "label": o.category.label,
                    "ours": _round(o.ours, o.category.key),
                    "theirs": _round(o.theirs, o.category.key),
                    "now_ours": _maybe(getattr(o, "banked_ours", None), o.category.key, banked),
                    "now_theirs": _maybe(getattr(o, "banked_theirs", None), o.category.key, banked),
                    "state": _state(o),
                    "chance": _pct(getattr(o, "expected", None)),
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


def _acted(acted: dict, tz: str) -> dict:
    from zoneinfo import ZoneInfo

    ok = bool(acted.get("ok"))
    at = acted.get("at")
    when = ""
    if at is not None:
        when = " at " + at.astimezone(ZoneInfo(tz)).strftime("%I:%M %p").lstrip("0")
    head = f"Made in Yahoo{when}" if ok else f"NOT made{when}"
    msg = str(acted.get("message", ""))
    return {"ok": ok, "text": f"{head} - {msg}" if msg else head}


def _why(p) -> str:
    """The net effect, in the league's own units - not an abstract score."""
    bits = []
    gain = p.reason.get("expected_gain")
    if gain is not None:
        bits.append(f"+{float(gain):.2f} categories expected")
    extra = p.reason.get("extra_starts")
    if extra is not None:
        bits.append(f"{float(extra):+g} starts this week")
    if p.reason.get("moved"):
        bits.append(str(p.reason["moved"]))
    return " · ".join(bits)


def _reasons(detail) -> list[dict]:
    """A proposal's reasons as titled sections, in reading order.

    Titles are resolved here so the page stays a renderer: a proposal made
    before reasons were recorded simply has none.
    """
    from puckpilot.season.add_story import SECTIONS

    if not isinstance(detail, dict):
        return []
    return [
        {"title": title, "lines": [str(x) for x in detail[key]]}
        for key, title in SECTIONS
        if isinstance(detail.get(key), list) and detail[key]
    ]


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
    """likely / in play / long shot by the odds bands; gone when even every
    lever left could not make it more than a long shot."""
    if not o.reachable:
        return "gone"
    band = getattr(o, "band", None)
    if band:
        return band
    if o.in_play:
        return "in play"
    return "likely" if o.margin > 0 else "long shot"


def _round(v: float, key: str) -> float:
    # A rate needs three places to be readable; a counting stat needs one.
    return round(v, 3) if v and abs(v) < 5 else round(v, 1)


def _pct(p) -> int | None:
    """A category's expected score as a whole percentage, or None."""
    return None if p is None else int(round(100 * p))


def _one(v) -> float | None:
    return None if v is None else round(v, 1)


def _maybe(v, key: str, banked: bool):
    """A banked total, or None before the week has anything in it."""
    return _round(v, key) if banked and v is not None else None


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
