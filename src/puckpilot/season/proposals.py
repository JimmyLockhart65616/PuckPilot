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

A row becomes approved two ways, both one row at a time: `decide`, a tap, and
`approve_standing`, the manager's approval given in advance for a rotation
player's drop (season/standing.py) - which refuses any row that was not shown
on the page with the time it would be made, before that time comes.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime

from puckpilot.season.moves import MoveKind, kind_of

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
    def move_kind(self) -> MoveKind:
        return kind_of(self.kind)

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
    kind = kind_of(kind).value  # refuses a kind nothing knows how to make
    if supersede:
        fresh = {(t.player.player_key, t.drop.player_key if t.drop else ""): t for t in targets}
        now = _now()
        for p in pending(conn, manager, league_key):
            if p.kind != kind:
                continue  # another kind's queue: a make-room move is not this search's to drop
            t = fresh.get((p.add_player_key, p.drop_player_key))
            if t is None:
                _withdraw(conn, p, "a newer search no longer proposes it", now)
            else:
                # Kept, with today's reasons: the numbers move every day.
                conn.execute(
                    "UPDATE waiver_proposals SET reason_json = ? WHERE id = ?",
                    (json.dumps(_kept(p.reason, _reason(t, week, horizon))), p.id),
                )
    existing = _already_asked(conn, manager, league_key, week)
    # The cap is on what is awaiting a decision. A refusal from earlier this
    # week stops that player coming back, but it is not clutter in the queue,
    # so it must not use up a slot - and nor does a make-room move, which is
    # required rather than chosen.
    room = max_pending - sum(
        1 for p in pending(conn, manager, league_key) if p.move_kind.is_search_result
    )
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
        reason = _kept(
            p.reason, _reason(t, int(p.reason.get("week", 0)), str(p.reason.get("horizon", "")))
        )
        reason["checked_at"] = now
        conn.execute(
            "UPDATE waiver_proposals SET reason_json = ? WHERE id = ?", (json.dumps(reason), pid)
        )
    for pid, why in lapsed.items():
        p = get(conn, pid)
        if p.is_live:
            _withdraw(conn, p, why, now)
    conn.commit()


def propose_fix(
    conn: sqlite3.Connection,
    manager: str,
    league_key: str,
    team_key: str,
    fix,
    week: int,
    reason: dict | None = None,
) -> tuple[Proposal, bool]:
    """Queue one make-room move (`legality.Fix` with a drop). Returns (proposal, new).

    One per returning player: an open one (pending, or approved and not yet
    made) for the same player is kept - its reason refreshed while pending -
    unless the drop it names is no longer the right one, in which case a
    pending one is withdrawn with the reason and replaced. An approved one is
    never replaced: it was decided, and its drop is held for it.
    """
    if fix.player is None or fix.drop is None:
        raise ProposalError("a make-room proposal needs the returning player and a drop")
    if fix.player.nhl_player_id is None and fix.drop.nhl_player_id is None:
        raise ProposalError(f"{fix.player.name} is not in the player map")
    body = {
        "week": week,
        "add_name": fix.player.name,
        "add_team": fix.player.team,
        "drop_name": fix.drop.name,
        "slot": fix.slot,
        "releases": fix.releases,
        "why_drop": fix.reason,
        "timing": "",
        "helps": [],
        **(reason or {}),
    }
    now = _now()
    for p in listing(conn, manager, league_key, limit=200):
        if p.kind != MoveKind.ACTIVATE.value or p.add_player_key != fix.player.player_key:
            continue
        if p.status == APPROVED:
            return p, False
        if not p.is_live:
            continue
        if p.drop_player_key == fix.drop.player_key:
            merged = {**p.reason, **body}
            conn.execute(
                "UPDATE waiver_proposals SET reason_json = ? WHERE id = ?",
                (json.dumps(merged), p.id),
            )
            conn.commit()
            return get(conn, p.id), False
        _withdraw(conn, p, f"{fix.drop.name} is now the one to drop instead", now)
    cur = conn.execute(
        "INSERT INTO waiver_proposals "
        "(manager, league_key, team_key, kind, add_pid, drop_pid, "
        " add_player_key, drop_player_key, reason_json, status) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            manager,
            league_key,
            team_key,
            MoveKind.ACTIVATE.value,
            fix.player.nhl_player_id if fix.player.nhl_player_id is not None else 0,
            fix.drop.nhl_player_id,
            fix.player.player_key,
            fix.drop.player_key,
            json.dumps(body),
            PENDING,
        ),
    )
    conn.commit()
    return get(conn, int(cur.lastrowid)), True


def retire_fixes(
    conn: sqlite3.Connection, manager: str, league_key: str, still_stuck: set[str]
) -> list[Proposal]:
    """Withdraw pending make-room moves whose player no longer needs room.

    He came back by hand, a spot opened, or his status put him back in line
    for the IR slot: the drop would now cost a player for nothing. Approved
    ones are left to the executor's own check (`transactions.still_needed`),
    which cancels them with the reason.
    """
    now = _now()
    out = []
    for p in pending(conn, manager, league_key):
        if p.kind == MoveKind.ACTIVATE.value and p.add_player_key not in still_stuck:
            _withdraw(conn, p, f"{p.add_name} no longer needs a roster spot made", now)
            out.append(get(conn, p.id))
    if out:
        conn.commit()
    return out


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


# What a re-priced card keeps from before: what it was offered and shown under.
STICKY = ("standing", "standing_not")


def _kept(old: dict, new: dict) -> dict:
    """`new` reasons, with what the card was offered and shown under kept."""
    return {**new, **{k: old[k] for k in STICKY if k in old}}


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
        # Whether a goalie or a skater comes in - a standing approval can
        # cover same-kind swaps and not the other kind.
        "add_position": getattr(t.player, "position", ""),
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


def approved_unmade(
    conn: sqlite3.Connection, manager: str, league_key: str, week: int
) -> list[Proposal]:
    """This week's approved moves not yet made, oldest first.

    Approved is decided, not done: a move approved on "make it yourself" stays
    approved until it is made in Yahoo, and one approved for the executor
    until its next run. Whether the add is on the roster yet is the caller's to
    check - this only knows what was decided.
    """
    got = [
        p
        for p in listing(conn, manager, league_key, status=APPROVED, limit=200)
        if p.reason.get("week") == week
    ]
    return sorted(got, key=lambda p: p.id)


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


def approve_standing(
    conn: sqlite3.Connection, proposal_id: int, grant: str, now: datetime | None = None
) -> Proposal:
    """Approve a card under the manager's standing approval: the one way but a tap.

    Only a card offered under it (`reason.standing`, by season/standing.py),
    shown on the page with the time it would be made (`shown_at`), once that
    time (`due_at`) has come, and still waiting - not stopped, decided or
    withdrawn. There is no argument that skips any of those: an approval in
    advance is consent to a move the manager could see coming and stop.
    `grant` records the settings it was given under.
    """
    p = get(conn, proposal_id)
    if p.status != PENDING:
        raise ProposalError(f"proposal #{proposal_id} is already {p.status}")
    if p.superseded_at:
        raise ProposalError(f"proposal #{proposal_id} was withdrawn")
    st = p.reason.get("standing") or {}
    if not st:
        raise ProposalError(f"proposal #{proposal_id} was not offered under a standing approval")
    if not st.get("shown_at") or not st.get("due_at"):
        raise ProposalError(
            f"proposal #{proposal_id} has not been shown with the time it would be made"
        )
    at = now or datetime.now(UTC)
    if datetime.fromisoformat(str(st["due_at"])) > at:
        raise ProposalError(f"proposal #{proposal_id} is not due until {st['due_at']}")
    reason = {
        **p.reason,
        "approved_to_execute": True,
        "approved_by": "standing",
        "grant": grant,
    }
    conn.execute(
        "UPDATE waiver_proposals SET status = ?, decided_at = ?, reason_json = ? WHERE id = ?",
        (APPROVED, at.isoformat(timespec="seconds"), json.dumps(reason), proposal_id),
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


def reopen(conn: sqlite3.Connection, proposal_id: int, why: str) -> Proposal:
    """Ask again: an approval given on words that no longer say what Approve does.

    Back to pending with its decision cleared, so the next push shows the card
    as it reads now and a tap on it is consent to that. Only an approved move
    never submitted can be reopened - one submitted may have gone through.
    """
    p = get(conn, proposal_id)
    if p.status != APPROVED:
        raise ProposalError(f"proposal #{proposal_id} is {p.status}, not approved")
    if p.reason.get("submitted_at"):
        raise ProposalError(f"proposal #{proposal_id} was already submitted to Yahoo")
    gone = ("approved_to_execute", "tapped_at", "manual_noted")
    reason = {k: v for k, v in p.reason.items() if k not in gone}
    reason["reopened"] = why
    conn.execute(
        "UPDATE waiver_proposals SET status = ?, decided_at = NULL, superseded_at = NULL, "
        "reason_json = ? WHERE id = ?",
        (PENDING, json.dumps(reason), proposal_id),
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
