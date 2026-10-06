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
    horizon: str = "",
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
        fresh = {(t.player.player_key, t.drop.player_key if t.drop else ""): t for t in targets}
        now = _now()
        for p in pending(conn, manager, league_key):
            t = fresh.get((p.add_player_key, p.drop_player_key))
            if t is None:
                _withdraw(conn, p, "a newer search no longer proposes it", now)
            else:
                # Kept, with today's reasons: the numbers move every day.
                conn.execute(
                    "UPDATE waiver_proposals SET reason_json = ? WHERE id = ?",
                    (json.dumps(_reason(t, week, horizon)), p.id),
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
        reason = _reason(t, week, horizon)
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


def refresh(conn: sqlite3.Connection, kept: dict[int, object], lapsed: dict[int, str]) -> None:
    """Apply a re-check of the queue: fresh reasons for what still pays, and
    withdrawal, with the reason, of what no longer does.

    Only proposals still awaiting a decision are touched - one approved or
    rejected while the run was working keeps its decision.
    """
    now = _now()
    for pid, t in kept.items():
        p = get(conn, pid)
        if not p.is_live:
            continue
        reason = _reason(t, int(p.reason.get("week", 0)), str(p.reason.get("horizon", "")))
        reason["checked_at"] = now
        conn.execute(
            "UPDATE waiver_proposals SET reason_json = ? WHERE id = ?", (json.dumps(reason), pid)
        )
    for pid, why in lapsed.items():
        p = get(conn, pid)
        if p.is_live:
            _withdraw(conn, p, why, now)
    conn.commit()


def _withdraw(conn: sqlite3.Connection, p: Proposal, why: str, now: str) -> None:
    reason = dict(p.reason)
    reason["withdrawn"] = why
    conn.execute(
        "UPDATE waiver_proposals SET superseded_at = ?, reason_json = ? WHERE id = ?",
        (now, json.dumps(reason), p.id),
    )


def withdrawn_since(
    conn: sqlite3.Connection, manager: str, league_key: str, since: str
) -> list[Proposal]:
    """Proposals withdrawn or cancelled since `since` (UTC ISO), newest first."""
    rows = conn.execute(
        "SELECT * FROM waiver_proposals WHERE manager = ? AND league_key = ? "
        "AND status IN (?, ?) AND superseded_at IS NOT NULL AND superseded_at >= ? "
        "ORDER BY id DESC",
        (manager, league_key, PENDING, REJECTED, since),
    )
    return [_row_to_proposal(r) for r in rows]


def _reason(t, week: int, horizon: str = "") -> dict:
    """What a proposal records about why it was made."""
    out = {
        "week": week,
        # `gain` is a share of the live gap under share pricing; under odds
        # pricing it is the change in expected categories (`expected_gain`).
        "gain": round(t.score, 3),
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
        # The reasons behind it, section -> lines (season/add_story.py).
        "detail": getattr(t, "detail", {}) or {},
    }
    if horizon:
        # Priced against another week than the one under way ("next week").
        out["horizon"] = horizon
    if getattr(t, "after_games_of", ""):
        # Its value is next week's alone: made after that day's games, not before.
        out["after_games_of"] = t.after_games_of
    return out


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


def decide(
    conn: sqlite3.Connection,
    proposal_id: int,
    approve: bool,
    to_execute: bool = False,
    tapped_at: float | None = None,
) -> Proposal:
    """Record a person's decision. The only way a proposal becomes actionable.

    `to_execute` is what the page said Approve would do when it was tapped:
    make the move in Yahoo (True), or record it for the person to make (False).
    Only an approval given on the first kind is ever handed to the executor -
    a "yes" to "make it yourself" is not a "yes" to "make it for me".
    """
    p = get(conn, proposal_id)
    if p.status != PENDING:
        raise ProposalError(f"proposal #{proposal_id} is already {p.status}")
    if p.superseded_at:
        raise ProposalError(
            f"proposal #{proposal_id} was withdrawn by a newer search - decide on the current ones"
        )
    reason = dict(p.reason)
    if approve:
        reason["approved_to_execute"] = bool(to_execute)
        if tapped_at is not None:
            reason["tapped_at"] = datetime.fromtimestamp(float(tapped_at), UTC).isoformat(
                timespec="seconds"
            )
    conn.execute(
        "UPDATE waiver_proposals SET status = ?, decided_at = ?, reason_json = ? WHERE id = ?",
        (APPROVED if approve else REJECTED, _now(), json.dumps(reason), proposal_id),
    )
    conn.commit()
    return get(conn, proposal_id)


def annotate(conn: sqlite3.Connection, proposal_id: int, **fields) -> Proposal:
    """Merge notes into a proposal's reason - when a move was submitted, which
    failure was last reported - without touching its status."""
    p = get(conn, proposal_id)
    reason = {**p.reason, **fields}
    conn.execute(
        "UPDATE waiver_proposals SET reason_json = ? WHERE id = ?",
        (json.dumps(reason), proposal_id),
    )
    conn.commit()
    return get(conn, proposal_id)


def cancel(conn: sqlite3.Connection, proposal_id: int, why: str) -> Proposal:
    """Call off a move that is waiting or approved but not yet made.

    A person's instruction, carried out: on 2026-10-04 two adds approved for
    the week's last two days were to be judged against the next week instead,
    and cancelled if they did not hold up there - both cost categories. It
    ends as rejected, so nothing can execute it, with the reason kept and shown
    on the page for a day like any withdrawal. A move already made cannot be
    cancelled here; only another move undoes it.
    """
    p = get(conn, proposal_id)
    if p.status not in (PENDING, APPROVED):
        raise ProposalError(f"proposal #{proposal_id} is {p.status} - nothing to cancel")
    now = _now()
    reason = {**p.reason, "cancelled": why, "withdrawn": f"cancelled - {why}"}
    conn.execute(
        "UPDATE waiver_proposals SET status = ?, decided_at = ?, superseded_at = ?, "
        "reason_json = ? WHERE id = ?",
        (REJECTED, now, now, json.dumps(reason), proposal_id),
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
