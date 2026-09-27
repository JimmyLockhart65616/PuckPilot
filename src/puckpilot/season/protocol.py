"""A stance for the week, agreed in advance.

Some categories are decided before the week starts. Against a roster built on
hits you are not winning hits, and every roster decision made as though you
might is a decision made for nothing. The useful question is not "who is worth
most" but "worth most in the categories still open".

That is a strategic call, not an arithmetic one, so it is not made
autonomously. Each week the tool reads the matchup, says which categories look
out of reach and which are live, proposes a stance, and waits. Approving it is
what licenses the daily lineup to act on it for the rest of the week - the same
shape as the lineup authority itself: a person agrees the criteria up front,
and the routine decisions that follow are the tool's to make.

Three stances, and only one of them changes any number:

    chase     a close category - weight it up
    hold      everything else - left exactly alone
    concede   out of reach - weight it down

`hold` is deliberately inert, including for categories held by a mile. Fading
something you are winning looks like the same idea and is not: the lineup does
not spend a budget on a category, so weighting down a safe one only risks
losing it. And this codebase has been here before - draft-time category
weights were measured as harmful three separate times (docs/STATUS.md,
2026-07-20 and 2026-09-11), so the intervention is kept to the one case where
the reasoning is not "this category matters less" but "this category is
already over".
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime

from puckpilot.engine.categories import Category, resolve

CHASE = "chase"
HOLD = "hold"
CONCEDE = "concede"

PROPOSED = "proposed"
APPROVED = "approved"
REJECTED = "rejected"

# A category is conceded when the lineup cannot close it - measured, by
# re-slotting the week with that category weighted heavily and seeing how far
# it actually moves, rather than by a share of the total someone chose. The
# fallback below is used only when that headroom was never computed.
OUT_OF_REACH = 0.20

# What a stance does to a category's contribution. `chase` is a nudge rather
# than a thumb on the scale; `concede` is not zero, because a conceded
# category still pays if the projection is wrong.
WEIGHTS = {CHASE: 1.35, HOLD: 1.0, CONCEDE: 0.25}


class ProtocolError(RuntimeError):
    """A protocol cannot be built, stored or decided as asked."""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass(frozen=True)
class CategoryStance:
    category: Category
    stance: str
    margin: float
    relative: float

    @property
    def weight(self) -> float:
        return WEIGHTS[self.stance]

    def reason(self) -> str:
        if self.stance == CONCEDE:
            return (
                f"behind {abs(self.margin):.1f} ({abs(self.relative):.0%}) - "
                f"more than a lineup change and an add together could close"
            )
        if self.stance == CHASE:
            return f"{self.margin:+.1f} - live, a single move could take it"
        if self.relative >= OUT_OF_REACH:
            return f"{self.margin:+.1f} - comfortable, left alone"
        return f"{self.margin:+.1f}"


@dataclass(frozen=True)
class WeekProtocol:
    manager: str
    league_key: str
    team_key: str
    week: int
    opponent: str
    stances: tuple[CategoryStance, ...]
    status: str = PROPOSED
    id: int | None = None
    decided_at: str = ""

    def weights(self) -> dict[str, float]:
        """Category key -> multiplier. Only a live protocol changes anything."""
        if self.status != APPROVED:
            return {}
        return {s.category.key: s.weight for s in self.stances if s.stance != HOLD}

    def of(self, stance: str) -> tuple[CategoryStance, ...]:
        return tuple(s for s in self.stances if s.stance == stance)

    @property
    def is_active(self) -> bool:
        return self.status == APPROVED and bool(self.weights())

    def describe(self) -> str:
        head = f"Week {self.week} protocol vs {self.opponent or '?'}  [{self.status.upper()}]"
        lines = [head]
        for stance, title in (
            (CONCEDE, "Give up"),
            (CHASE, "Go after"),
        ):
            got = self.of(stance)
            if not got:
                continue
            lines.append(f"  {title}:")
            for s in got:
                lines.append(f"    {s.category.label:5} {s.reason()}")
        held = self.of(HOLD)
        if held:
            lines.append("  Leave alone: " + ", ".join(s.category.label for s in held))
        if self.status == PROPOSED:
            lines.append("")
            lines.append(
                "  This changes how the daily lineup values players for the rest "
                "of the week. Nothing happens until you approve it."
            )
        return "\n".join(lines)


def derive(outlook, manager: str, league_key: str, team_key: str, week: int, opponent: str):
    """Read a stance off the week's projected categories.

    By odds where the outlook models the spread - a long shot even with every
    lever pulled is conceded, a coin flip is chased - and by the relative
    margin only where it does not. The margin is taken in our favour (`edge`),
    so a category scored lower-is-better is not read backwards.
    """
    from puckpilot.season.week import Z_BAND

    stances = []
    for o in outlook:
        measured = getattr(o, "measured", False)
        edge = getattr(o, "edge", o.margin)
        z = getattr(o, "z", None)
        if measured:
            unreachable = not o.reachable
        elif z is not None:
            unreachable = z <= -Z_BAND
        else:
            unreachable = o.relative <= -OUT_OF_REACH
        if edge < 0 and unreachable:
            stance = CONCEDE
        elif o.in_play:
            stance = CHASE
        else:
            stance = HOLD
        stances.append(
            CategoryStance(category=o.category, stance=stance, margin=o.margin, relative=o.relative)
        )
    return WeekProtocol(
        manager=manager,
        league_key=league_key,
        team_key=team_key,
        week=week,
        opponent=opponent,
        stances=tuple(stances),
    )


# -- storage ----------------------------------------------------------------


def save(conn: sqlite3.Connection, p: WeekProtocol) -> WeekProtocol:
    """Store a proposed protocol, replacing any undecided one for that week.

    A week has one plan. Re-deriving it on Wednesday should refresh Monday's
    proposal rather than stack another beside it - but never overwrite one that
    has been approved, because the lineup has been acting on that.

    And while the decision is unchanged the row keeps its id, margins refreshed
    in place. A scheduled job re-derives this several times a day; replacing the
    row each time would leave whoever is looking at the page holding an Approve
    button for something that no longer exists.
    """
    live = load(conn, p.manager, p.league_key, p.week)
    if live and live.status == APPROVED:
        raise ProtocolError(
            f"week {p.week} already has an approved protocol; reject it first to replace it"
        )
    if live and live.status == PROPOSED and _same_decision(live, p):
        conn.execute(
            "UPDATE week_protocols SET stances_json = ?, opponent = ? WHERE id = ?",
            (_stances_json(p), p.opponent, live.id),
        )
        conn.commit()
        return load_by_id(conn, live.id)
    conn.execute(
        "DELETE FROM week_protocols WHERE manager = ? AND league_key = ? AND week = ? "
        "AND status = ?",
        (p.manager, p.league_key, p.week, PROPOSED),
    )
    cur = conn.execute(
        "INSERT INTO week_protocols "
        "(manager, league_key, team_key, week, opponent, stances_json, status) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            p.manager,
            p.league_key,
            p.team_key,
            p.week,
            p.opponent,
            _stances_json(p),
            p.status,
        ),
    )
    conn.commit()
    return load_by_id(conn, int(cur.lastrowid))


def _stances_json(p: WeekProtocol) -> str:
    return json.dumps([[s.category.key, s.stance, s.margin, s.relative] for s in p.stances])


def _same_decision(a: WeekProtocol, b: WeekProtocol) -> bool:
    """Same categories chased and conceded. Margins drift every run; the
    decision is what a person agreed to."""
    return {(s.category.key, s.stance) for s in a.stances} == {
        (s.category.key, s.stance) for s in b.stances
    }


def _row(r: sqlite3.Row) -> WeekProtocol:
    stances = tuple(
        CategoryStance(category=resolve_key(k), stance=st, margin=m, relative=rel)
        for k, st, m, rel in json.loads(r["stances_json"])
    )
    return WeekProtocol(
        id=int(r["id"]),
        manager=r["manager"],
        league_key=r["league_key"],
        team_key=r["team_key"],
        week=int(r["week"]),
        opponent=r["opponent"] or "",
        stances=stances,
        status=r["status"],
        decided_at=r["decided_at"] or "",
    )


def resolve_key(key: str) -> Category:
    """A category by its stored key rather than its display label."""
    from puckpilot.engine.categories import CATALOG

    for c in CATALOG.values():
        if c.key == key:
            return c
    return resolve(key)


def load_by_id(conn: sqlite3.Connection, pid: int) -> WeekProtocol:
    r = conn.execute("SELECT * FROM week_protocols WHERE id = ?", (pid,)).fetchone()
    if r is None:
        raise ProtocolError(f"no protocol #{pid}")
    return _row(r)


def load(conn: sqlite3.Connection, manager: str, league_key: str, week: int) -> WeekProtocol | None:
    """The protocol governing a week: an approved one, else the latest proposal."""
    r = conn.execute(
        "SELECT * FROM week_protocols WHERE manager = ? AND league_key = ? AND week = ? "
        "AND status <> ? ORDER BY CASE status WHEN ? THEN 0 ELSE 1 END, id DESC LIMIT 1",
        (manager, league_key, week, REJECTED, APPROVED),
    ).fetchone()
    return _row(r) if r else None


def active(
    conn: sqlite3.Connection, manager: str, league_key: str, week: int
) -> WeekProtocol | None:
    """Only an approved protocol governs anything."""
    p = load(conn, manager, league_key, week)
    return p if p and p.status == APPROVED else None


def decide(conn: sqlite3.Connection, pid: int, approve: bool) -> WeekProtocol:
    p = load_by_id(conn, pid)
    if p.status != PROPOSED:
        raise ProtocolError(f"protocol #{pid} is already {p.status}")
    conn.execute(
        "UPDATE week_protocols SET status = ?, decided_at = ? WHERE id = ?",
        (APPROVED if approve else REJECTED, _now(), pid),
    )
    conn.commit()
    return load_by_id(conn, pid)


def listing(
    conn: sqlite3.Connection, manager: str = "", league_key: str = "", limit: int = 30
) -> list[WeekProtocol]:
    sql = "SELECT * FROM week_protocols WHERE 1=1"
    args: list = []
    if manager:
        sql += " AND manager = ?"
        args.append(manager)
    if league_key:
        sql += " AND league_key = ?"
        args.append(league_key)
    sql += " ORDER BY week DESC, id DESC LIMIT ?"
    args.append(limit)
    return [_row(r) for r in conn.execute(sql, args)]
