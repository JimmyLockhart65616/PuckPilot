"""Standing approval: a rotation player's drop, approved in advance.

Asked for on 2026-10-31. Of the five cards since PuckPilot began making
approved moves, four were approved - each 11 to 12 hours after it went out,
so a move found in the morning was made after that night's games - and the
drop on almost every card of the season was one of the players the roster
rotates anyway. The tap there was a delay, not a decision.

So a manager can approve those in advance (`authority.StandingApproval`): a
card whose drop is one of his rotation players (`season.tiers`), of a kind he
chose to cover, is made once it has been on the page for his window - unless
he stopped it. It is still a card. It is shown first with the time it will be
made and pushed to the phone, and it is approved by
`proposals.approve_standing`, which refuses one that was not shown with that
time, or whose time has not come.

    offer        mark a waiting card as covered - its drop is a rotation
                 player, its kind is covered, PuckPilot makes it - or record
                 why it is not, for the card
    stamp        the time each offered card will be made, until it is shown
    shown        the page showed it: the window starts there
    approve_due  at or after that time, approve it - unless he stopped it, set
                 the drop to another tier since, or the grant is off
    due_times    when the scheduler has to run to make them

What re-checks a covered card at its time is only what the manager has said
since: a Stop, a pin, the grant. The drop's automatic tier moving with the
numbers does not withdraw it - he agreed to the card as it was shown - and
whatever makes a move impossible (the add taken, the drop gone) is the
executor's check, as for a tapped move.

`PUCKPILOT_STANDING_OFF=1` stops every one: a covered card is then an
ordinary card again, waiting for a tap.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

from puckpilot.season import proposals as proposals_mod
from puckpilot.season import tiers
from puckpilot.season.authority import COVERS
from puckpilot.season.moves import MoveKind

KILL_SWITCH = "PUCKPILOT_STANDING_OFF"
DEFAULT_TZ = "America/Toronto"
# A card whose time passed this long before a run came - the machine was off
# - is not made on numbers that old: it goes back, and the run's search offers
# it again with a new time if it still pays.
MISSED_HOURS = 3.0


def killed() -> bool:
    """The kill switch: every standing approval waits for a tap instead."""
    return os.environ.get(KILL_SWITCH, "").strip() not in ("", "0")


def grant_hash(grant) -> str:
    """The settings a card was offered under, short - kept on its approval."""
    return hashlib.sha256(json.dumps(asdict(grant), sort_keys=True).encode()).hexdigest()[:12]


def move_kind(p, drop, add_position: str | None) -> str | None:
    """Which of the grant's kinds `p` is (`authority.COVERS`), or None when it
    drops nobody. A swap whose add's position is not known is the wider kind."""
    if p.move_kind is MoveKind.ACTIVATE:
        return "make_room"
    if not p.drop_player_key:
        return None
    if not add_position or drop is None:
        return "cross_kind"
    return "same_kind" if (add_position == "G") == (drop.position == "G") else "cross_kind"


def refusal(p, grant, placed, roster, executes: bool, kinds=None, used: int = 0) -> str:
    """Why card `p` is not one the standing approval covers, or "" if it is.

    `placed` is the run's drop tiers (`tiers.board`), `roster` ours as read,
    `executes` whether PuckPilot makes approved moves at all and `kinds` the
    kinds its executor makes, `used` this week's moves without a tap so far.
    """
    if not grant.enabled:
        return "no standing approval is given"
    if killed():
        return f"{KILL_SWITCH} is set"
    if not executes or (kinds is not None and p.move_kind.value not in kinds):
        return "PuckPilot does not make this kind of move"
    if str(p.reason.get("timing", "")).startswith("on waivers"):
        return "a waiver claim is made by hand"
    drop = next((q for q in roster.players if q.player_key == p.drop_player_key), None)
    what = move_kind(p, drop, p.reason.get("add_position"))
    if what is None:
        return "it drops nobody - only a Rotation player's drop is covered"
    where = (placed or {}).get(p.drop_player_key)
    if drop is None or where is None:
        return f"{p.drop_name} is not on the roster"
    if where.tier is not tiers.Tier.ROTATION:
        return f"{p.drop_name} is {where.tier.label}, not Rotation"
    if not grant.covers(what):
        return f"{COVERS[what]} are not covered"
    if grant.max_per_week is not None and used >= grant.max_per_week:
        return f"this week's {grant.max_per_week} without a tap are made"
    return ""


def used_this_week(conn: sqlite3.Connection, manager: str, league_key: str, week: int) -> int:
    """Moves this week approved in advance, or offered and waiting."""
    n = 0
    for p in proposals_mod.listing(conn, manager, league_key, limit=500):
        if p.reason.get("week") != week:
            continue
        made = p.status in (proposals_mod.APPROVED, proposals_mod.EXECUTED)
        if (made and p.reason.get("approved_by") == "standing") or (
            p.is_live and p.reason.get("standing")
        ):
            n += 1
    return n


def offer(
    conn: sqlite3.Connection,
    manager: str,
    league_key: str,
    grant,
    placed,
    roster,
    executes: bool,
    kinds=None,
    week: int = 0,
    now: datetime | None = None,
) -> list:
    """Mark each waiting card the grant covers; on the rest, say why not.

    A card is offered once: its window starts when the page first shows it,
    and a later run does not move it. One not covered keeps no offer, and
    is re-checked every run - a pin to Rotation can make it covered.
    """
    at = (now or datetime.now(UTC)).isoformat(timespec="seconds")
    used = used_this_week(conn, manager, league_key, week)
    offered = []
    for p in sorted(proposals_mod.pending(conn, manager, league_key), key=lambda q: q.id):
        if p.reason.get("standing"):
            continue
        why = refusal(p, grant, placed, roster, executes, kinds, used)
        if why:
            if grant.enabled and p.reason.get("standing_not") != why:
                proposals_mod.annotate(conn, p.id, standing_not=why)
            continue
        drop = next(q for q in roster.players if q.player_key == p.drop_player_key)
        proposals_mod.annotate(
            conn,
            p.id,
            standing={
                "kind": move_kind(p, drop, p.reason.get("add_position")),
                "tier_source": placed[p.drop_player_key].source,
                "grant": grant_hash(grant),
                "window": grant.veto_minutes,
                "offered_at": at,
                "due_at": None,
                "shown_at": None,
            },
            standing_not=None,
        )
        used += 1
        offered.append(proposals_mod.get(conn, p.id))
    return offered


def _offers(conn, manager: str, league_key: str = ""):
    for p in proposals_mod.pending(conn, manager, league_key):
        st = p.reason.get("standing")
        if st:
            yield p, st


def stamp(conn: sqlite3.Connection, manager: str, league_key: str, now: datetime) -> dict:
    """The time each offered card not yet shown will be made, from `now`: what
    the page about to be pushed says. {id: when}."""
    out = {}
    for p, st in _offers(conn, manager, league_key):
        if st.get("shown_at"):
            continue
        due = now + timedelta(minutes=int(st.get("window") or 0))
        proposals_mod.annotate(
            conn, p.id, standing={**st, "due_at": due.isoformat(timespec="seconds")}
        )
        out[p.id] = due
    return out


def shown(conn: sqlite3.Connection, manager: str, league_key: str, at: datetime) -> list[int]:
    """The page now shows the stamped cards: their windows run from here."""
    out = []
    for p, st in _offers(conn, manager, league_key):
        if st.get("shown_at") or not st.get("due_at"):
            continue
        proposals_mod.annotate(
            conn, p.id, standing={**st, "shown_at": at.isoformat(timespec="seconds")}
        )
        out.append(p.id)
    return out


def _since(p, st: dict, grant, pins) -> str:
    """What the manager has said since the card was shown that stops it being
    made unasked, or "" for nothing."""
    if not grant.enabled:
        return "the standing approval was turned off"
    if killed():
        return f"{KILL_SWITCH} is set"
    if not grant.covers(str(st.get("kind", ""))):
        return f"{COVERS.get(str(st.get('kind', '')), 'these moves')} are no longer covered"
    pin = pins.get(p.drop_player_key)
    if pin is not None and pin is not tiers.Tier.ROTATION:
        return f"you set {p.drop_name} to {pin.label}"
    if pin is None and st.get("tier_source") == tiers.YOU:
        return f"you took Rotation off {p.drop_name}"
    return ""


def approve_due(
    conn: sqlite3.Connection,
    manager: str,
    league_key: str,
    grant,
    now: datetime | None = None,
    missed_hours: float = MISSED_HOURS,
) -> tuple[list, list]:
    """(approved, (card, why) put back to wait for a tap) - every offered card
    whose time has come. One whose time passed more than `missed_hours`
    before this run goes back too: the search re-prices it, and the offer
    that follows gives it a new time."""
    at = now or datetime.now(UTC)
    pins = tiers.pins(conn, manager, league_key)
    approved, back = [], []
    for p, st in sorted(_offers(conn, manager, league_key), key=lambda x: x[0].id):
        if not st.get("shown_at") or not st.get("due_at"):
            continue
        when = datetime.fromisoformat(str(st["due_at"]))
        if when > at:
            continue
        why = _since(p, st, grant, pins)
        if not why and at - when > timedelta(hours=missed_hours):
            why = "its time passed with no run to make it - offered again with a new one"
        if why:
            proposals_mod.annotate(conn, p.id, standing=None, standing_not=why)
            back.append((proposals_mod.get(conn, p.id), why))
            continue
        approved.append(proposals_mod.approve_standing(conn, p.id, grant_hash(grant), at))
    return approved, back


def due_at(p, shown: bool = True) -> datetime | None:
    """When an offered card will be made: once shown (`shown`), or as the page
    about to be pushed will say."""
    st = p.reason.get("standing") or {}
    if not st.get("due_at") or (shown and not st.get("shown_at")):
        return None
    return datetime.fromisoformat(str(st["due_at"]))


def due_local(p, tz: str = DEFAULT_TZ) -> str:
    """The card's time in the manager's own clock: "9:05 PM"."""
    st = p.reason.get("standing") or {}
    if not st.get("due_at"):
        return ""
    when = datetime.fromisoformat(str(st["due_at"])).astimezone(ZoneInfo(tz))
    return when.strftime("%I:%M %p").lstrip("0")


def due_times(conn: sqlite3.Connection, manager: str, day: str, tz: str = DEFAULT_TZ) -> list[str]:
    """HH:MM local of every shown card due later on `day` - a run each, so a
    card is made at its time and not at the next lock's. Rounded up to the
    minute: a task starts on the minute, and one a few seconds early would
    find the card not yet due."""
    zone = ZoneInfo(tz)
    now = datetime.now(zone)
    out: set[str] = set()
    for p, _ in _offers(conn, manager):
        when = due_at(p)
        if when is None:
            continue
        local = when.astimezone(zone)
        if local.second or local.microsecond:
            local = (local + timedelta(minutes=1)).replace(second=0, microsecond=0)
        if local.date().isoformat() == day and local > now:
            out.add(local.strftime("%H:%M"))
    return sorted(out)


def next_due(conn: sqlite3.Connection, manager: str, now: datetime | None = None):
    """The earliest card's time still ahead - one the page about to be pushed
    shows too - or None. The page's "next update" line."""
    at = now or datetime.now(UTC)
    ahead = [
        w for p, _ in _offers(conn, manager) if (w := due_at(p, shown=False)) is not None and w > at
    ]
    return min(ahead) if ahead else None
