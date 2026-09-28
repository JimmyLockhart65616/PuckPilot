"""The approval queue, and the only route from a suggestion to a transaction.

`waiver_proposals` has been in the schema since Phase 5 with a
pending/approved/rejected/executed state machine and no callers at all. This is
its consumer, and it is deliberately the narrow point: a lineup change runs
under standing authority, but an add or a drop spends one of a fixed number of
weekly acquisitions and may drop a player who does not come back, so it goes
through here.

The asymmetry is structural rather than configured. `take_for_execution` is the
only function that hands an executor anything, and it will only ever return a
row a person has approved. There is no argument that makes it return a pending
one, and nothing else in the package returns a proposal at all.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime

PENDING = "pending"
APPROVED = "approved"
REJECTED = "rejected"
EXECUTED = "executed"


class ProposalError(RuntimeError):
    """A proposal cannot be created, decided or executed as asked."""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass(frozen=True)
class Proposal:
    id: int
    manager: str
    league_key: str
    team_key: str
    kind: str
    status: str
    add_player_key: str
    drop_player_key: str
    add_pid: int
    drop_pid: int | None
    reason: dict
    created_at: str = ""
    decided_at: str = ""
    executed_at: str = ""
    # Set when a newer search withdrew it before anyone decided: no longer
    # shown, counted or approvable.
    superseded_at: str = ""

    @property
    def is_live(self) -> bool:
        return self.status == PENDING and not self.superseded_at

    @property
    def add_name(self) -> str:
        return str(self.reason.get("add_name", self.add_player_key))

    @property
    def drop_name(self) -> str:
        return str(self.reason.get("drop_name", self.drop_player_key or "-"))

    def describe(self) -> str:
        line = f"#{self.id} {self.status.upper():9} ADD {self.add_name}"
        if self.drop_player_key:
            line += f"  DROP {self.drop_name}"
        moved = self.reason.get("moved")
        extra = self.reason.get("extra_starts")
        bits = []
        if extra is not None:
            bits.append(f"{float(extra):+g} starts")
        if moved:
            bits.append(str(moved))
        if bits:
            line += "  (" + ", ".join(bits) + ")"
        return line


def _row_to_proposal(r: sqlite3.Row) -> Proposal:
    try:
        reason = json.loads(r["reason_json"])
    except (TypeError, ValueError):
        reason = {}
    return Proposal(
        id=int(r["id"]),
        manager=r["manager"] or "",
        league_key=r["league_key"] or "",
        team_key=r["team_key"] or "",
        kind=r["kind"] or "add_drop",
        status=r["status"],
        add_player_key=r["add_player_key"] or "",
        drop_player_key=r["drop_player_key"] or "",
        add_pid=int(r["add_pid"]),
        drop_pid=r["drop_pid"],
        reason=reason if isinstance(reason, dict) else {},
        created_at=r["created_at"] or "",
        decided_at=r["decided_at"] or "",
        executed_at=r["executed_at"] or "",
        # sqlite3.Row's `in` tests values, not column names - hence keys().
        superseded_at=(r["superseded_at"] if "superseded_at" in r.keys() else None)  # noqa: SIM118
        or "",
    )


def propose(
    conn: sqlite3.Connection,
    manager: str,
    league_key: str,
    team_key: str,
    targets,
    week: int,
    max_pending: int = 5,
    kind: str = "add_drop",
    supersede: bool = False,
) -> list[Proposal]:
    """Record add/drop targets as pending proposals.

    A job may run several times a day, so the same add has to stop coming back.
    It is suppressed when one is still awaiting a decision, when one is already
    approved and waiting to be executed, and when it was turned down earlier
    this week - asking again the same week about a player you said no to is
    exactly the notification that teaches someone to stop reading them. A new
    week reconsiders, because by then the schedule and the standings have moved.

    With `supersede`, the search's answer replaces the queue: anything still
    pending that it did not propose again is withdrawn. Without it, stale
    proposals accumulate - on 2026-09-28 five from the pre-fix engine (all
    dropping Tuch or Malkin) filled `max_pending` and would have left the new
    engine no room to propose anything on the season's first day.
    """
    if supersede:
        keep = {(t.player.player_key, t.drop.player_key if t.drop else "") for t in targets}
        now = _now()
        for p in pending(conn, manager, league_key):
            if (p.add_player_key, p.drop_player_key) not in keep:
                conn.execute(
                    "UPDATE waiver_proposals SET superseded_at = ? WHERE id = ?", (now, p.id)
                )
    existing = _already_asked(conn, manager, league_key, week)
    # The cap is on what is awaiting a decision. A refusal from earlier this
    # week stops that player coming back, but it is not clutter in the queue,
    # so it must not use up a slot.
    room = max_pending - len(pending(conn, manager, league_key))
    made: list[Proposal] = []
    for t in targets:
        if room <= 0:
            break
        if t.player.player_key in existing:
            continue
        if t.player.nhl_player_id is None:
            continue
        reason = {
            "week": week,
            # `gain` is a share of the live gap now, not an abstract value
            # number - adds are priced by re-slotting the week and subtracting.
            "gain": round(t.score, 3),
            # Under odds pricing: expected categories won this week, added.
            "expected_gain": None if getattr(t, "gain", None) is None else round(t.gain, 3),
            "starts": round(t.starts, 2),
            "drop_starts": round(t.drop_starts, 2),
            "extra_starts": round(t.extra_starts, 2),
            "moved": t.moved(),
            "helps": list(t.helps),
            "timing": t.timing,
            "add_name": t.player.name,
            "add_team": t.player.team,
            "drop_name": t.drop.name if t.drop else "",
        }
        cur = conn.execute(
            "INSERT INTO waiver_proposals "
            "(manager, league_key, team_key, kind, add_pid, drop_pid, "
            " add_player_key, drop_player_key, reason_json, status) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                manager,
                league_key,
                team_key,
                kind,
                t.player.nhl_player_id,
                t.drop.nhl_player_id if t.drop else None,
                t.player.player_key,
                t.drop.player_key if t.drop else "",
                json.dumps(reason),
                PENDING,
            ),
        )
        made.append(get(conn, int(cur.lastrowid)))
        existing.add(t.player.player_key)
        room -= 1
    conn.commit()
    return made


def _already_asked(conn: sqlite3.Connection, manager: str, league_key: str, week: int) -> set[str]:
    """Adds not to raise again: open ones, and ones refused this week."""
    out: set[str] = set()
    for p in listing(conn, manager, league_key, limit=200):
        open_already = p.is_live or p.status == APPROVED
        refused_this_week = p.status == REJECTED and p.reason.get("week") == week
        if open_already or refused_this_week:
            out.add(p.add_player_key)
    return out


def get(conn: sqlite3.Connection, proposal_id: int) -> Proposal:
    row = conn.execute("SELECT * FROM waiver_proposals WHERE id = ?", (proposal_id,)).fetchone()
    if row is None:
        raise ProposalError(f"no proposal #{proposal_id}")
    return _row_to_proposal(row)


def listing(
    conn: sqlite3.Connection,
    manager: str = "",
    league_key: str = "",
    status: str | None = None,
    limit: int = 50,
) -> list[Proposal]:
    sql = "SELECT * FROM waiver_proposals WHERE 1=1"
    args: list = []
    if manager:
        sql += " AND manager = ?"
        args.append(manager)
    if league_key:
        sql += " AND league_key = ?"
        args.append(league_key)
    if status:
        sql += " AND status = ?"
        args.append(status)
    if status == PENDING:
        sql += " AND superseded_at IS NULL"
    sql += " ORDER BY id DESC LIMIT ?"
    args.append(limit)
    return [_row_to_proposal(r) for r in conn.execute(sql, args)]


def pending(conn: sqlite3.Connection, manager: str = "", league_key: str = "") -> list[Proposal]:
    return listing(conn, manager, league_key, status=PENDING)


def decide(conn: sqlite3.Connection, proposal_id: int, approve: bool) -> Proposal:
    """Record a person's decision. The only way a proposal becomes actionable."""
    p = get(conn, proposal_id)
    if p.status != PENDING:
        raise ProposalError(f"proposal #{proposal_id} is already {p.status}")
    if p.superseded_at:
        raise ProposalError(
            f"proposal #{proposal_id} was withdrawn by a newer search - decide on the current ones"
        )
    conn.execute(
        "UPDATE waiver_proposals SET status = ?, decided_at = ? WHERE id = ?",
        (APPROVED if approve else REJECTED, _now(), proposal_id),
    )
    conn.commit()
    return get(conn, proposal_id)


def take_for_execution(conn: sqlite3.Connection, proposal_id: int) -> Proposal:
    """Hand an executor a proposal, if and only if a person approved it.

    This is the mechanism behind "transactions always require approval". There
    is no flag that relaxes it and no other function returns a proposal to an
    executor, so making transactions autonomous would mean deleting this.
    """
    p = get(conn, proposal_id)
    if p.status == PENDING:
        raise ProposalError(
            f"proposal #{proposal_id} has not been approved. Transactions are never "
            f"executed without a decision - approve it first."
        )
    if p.status != APPROVED:
        raise ProposalError(f"proposal #{proposal_id} is {p.status}, not approved")
    return p


def mark_executed(conn: sqlite3.Connection, proposal_id: int, message: str = "") -> Proposal:
    p = take_for_execution(conn, proposal_id)
    reason = dict(p.reason)
    if message:
        reason["result"] = message
    conn.execute(
        "UPDATE waiver_proposals SET status = ?, executed_at = ?, reason_json = ? WHERE id = ?",
        (EXECUTED, _now(), json.dumps(reason), proposal_id),
    )
    conn.commit()
    return get(conn, proposal_id)


def record_action(
    conn: sqlite3.Connection,
    manager: str,
    league_key: str,
    team_key: str,
    date: str,
    kind: str,
    detail: dict,
    outcome: str = "planned",
    message: str = "",
) -> int:
    """Append to the audit log.

    Every autonomous lineup change lands here as well as every executed
    transaction, because criteria granted in advance can only be argued with
    afterwards from a record of what they actually did.
    """
    cur = conn.execute(
        "INSERT INTO season_actions "
        "(manager, league_key, team_key, date, kind, detail_json, outcome, message) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (manager, league_key, team_key, date, kind, json.dumps(detail), outcome, message),
    )
    conn.commit()
    return int(cur.lastrowid)


def actions(
    conn: sqlite3.Connection, manager: str = "", since: str = "", limit: int = 50
) -> list[sqlite3.Row]:
    sql = "SELECT * FROM season_actions WHERE 1=1"
    args: list = []
    if manager:
        sql += " AND manager = ?"
        args.append(manager)
    if since:
        sql += " AND date >= ?"
        args.append(since)
    sql += " ORDER BY id DESC LIMIT ?"
    args.append(limit)
    return list(conn.execute(sql, args))
