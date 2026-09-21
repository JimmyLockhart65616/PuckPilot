"""Tonight's lineup: who should start, and what to change to get there.

The measured prize is +14.9% of season value over setting a lineup once and
leaving it, and 92.3% of the hindsight ceiling is reachable with morning
knowledge alone. That is what this produces, once a day, for 185 game days.

The output is a *diff*, not a lineup. "Bench Cuylle, start Michkov" is what a
person acts on at 6:45pm; a thirteen-row table is something they have to read
and compare themselves, which is the work this is supposed to remove.

Four things decide whether a player is a candidate tonight, and only the first
comes from the engines:

    his team plays          the forward schedule
    he is not ruled out     Yahoo's own status, which the draft board never had
    his slot is unlocked    Yahoo's is_editable - a started game cannot be undone
    a goalie is starting    P(start), which the optimizer weights value by

The weekly goalie minimum is the one hard rule here. Yahoo enforces it and
falling short forfeits the categories, so when the days left in the week can no
longer satisfy it, starting a goalie stops being a preference.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field

from puckpilot.engine.lineup import optimize_lineup
from puckpilot.season import calendar
from puckpilot.season.authority import LineupAuthority
from puckpilot.season.roster import RosterPlayer, TeamRoster
from puckpilot.season.settings import YAHOO_TO_POS, LeagueRuntime
from puckpilot.season.values import ValueModel

# Yahoo's slot name for a benched player, and the engine's name for the flex.
BENCH = "BN"
UTIL = "UTIL"
YAHOO_UTIL = "Util"


@dataclass(frozen=True)
class Candidate:
    player: RosterPlayer
    value: float
    p_start: float | None = None
    note: str = ""
    questionable: bool = False

    @property
    def is_goalie(self) -> bool:
        return self.player.position == "G"


@dataclass(frozen=True)
class Move:
    """One change to make in Yahoo."""

    player: RosterPlayer
    to_slot: str
    from_slot: str

    @property
    def is_start(self) -> bool:
        return self.to_slot != BENCH

    @property
    def is_bench(self) -> bool:
        return self.to_slot == BENCH

    @property
    def kind_order(self) -> int:
        """Starts first: they are the reason for the message."""
        if self.is_bench:
            return 2
        return 0 if self.from_slot in (BENCH, "?") else 1

    def describe(self) -> str:
        if self.is_bench:
            return f"BENCH {self.player.name}  (was {self.from_slot})"
        if self.from_slot in (BENCH, "?"):
            return f"START {self.player.name} in {self.to_slot}"
        return f"MOVE  {self.player.name} {self.from_slot} -> {self.to_slot}"


@dataclass(frozen=True)
class LineupPlan:
    date: str
    team_key: str
    manager: str
    moves: tuple[Move, ...] = ()
    gain: float = 0.0
    playing: tuple[Candidate, ...] = ()
    idle: tuple[RosterPlayer, ...] = ()
    out: tuple[RosterPlayer, ...] = ()
    locked: tuple[RosterPlayer, ...] = ()
    notes: tuple[str, ...] = ()
    within_authority: bool = False
    authority_reason: str = ""
    empty_slots: tuple[str, ...] = field(default=())
    lock_utc: str = ""
    lock_team: str = ""

    @property
    def is_noop(self) -> bool:
        return not self.moves

    def deadline(self, tz: str = "America/Toronto") -> str:
        """When the first of tonight's games locks a slot, in local time.

        A daily league locks each player when his own game starts, so this is
        the earliest of them - the moment after which part of the lineup can no
        longer be changed at all.
        """
        if not self.lock_utc:
            return ""
        from datetime import datetime
        from zoneinfo import ZoneInfo

        try:
            when = datetime.fromisoformat(self.lock_utc.replace("Z", "+00:00"))
            local = when.astimezone(ZoneInfo(tz))
        except (ValueError, OSError, KeyError):
            return self.lock_utc
        # "%-I" strips the leading zero on Unix and is not supported on Windows,
        # so do it by hand rather than branch on the platform.
        return local.strftime("%I:%M %p").lstrip("0")

    def text(self) -> str:
        lines = [f"{self.date}  {self.manager}  ({self.team_key})"]
        if self.is_noop:
            lines.append("")
            lines.append("  Lineup is already optimal - nothing to change.")
        else:
            lines.append("")
            lines.append(f"  {len(self.moves)} change(s), worth {self.gain:+.2f} today:")
            for m in self.moves:
                lines.append(f"    {m.describe()}")
        if self.empty_slots:
            lines.append("")
            lines.append("  Scoring nothing tonight: " + ", ".join(self.empty_slots))
        if self.out:
            lines.append("")
            lines.append("  Out: " + ", ".join(p.label() for p in self.out))
        if self.locked:
            lines.append("  Locked (game started): " + ", ".join(p.name for p in self.locked))
        for n in self.notes:
            lines.append(f"  ! {n}")
        lines.append("")
        lines.append(
            f"  {len(self.playing)} of {len(self.playing) + len(self.idle) + len(self.out)} "
            f"rostered players have a game."
        )
        if self.lock_utc:
            lines.append(
                f"  First lock: {self.deadline()} ({self.lock_team}) - changes are free until then."
            )
        lines.append(f"  Authority: {self.authority_reason}")
        return "\n".join(lines)


def _slot_for(engine_slot: str) -> str:
    """The engine's slot name as Yahoo spells it."""
    return YAHOO_UTIL if engine_slot == UTIL else _POS_TO_YAHOO.get(engine_slot, engine_slot)


_POS_TO_YAHOO = {v: k for k, v in YAHOO_TO_POS.items()}


def _engine_slot(yahoo_slot: str) -> str | None:
    """Yahoo's slot name as the engine spells it, or None if it is not startable."""
    if yahoo_slot in (YAHOO_UTIL, UTIL):
        return UTIL
    return YAHOO_TO_POS.get(yahoo_slot)


def goalie_starts_this_week(
    conn: sqlite3.Connection,
    runtime: LeagueRuntime,
    manager: str,
    team_key: str,
    date: str,
) -> int:
    """Goalie starts already recorded this week, from our own snapshots.

    Yahoo reports the same number in the roster payload's `minimum_games`, and
    that is the authority; this is the fallback when the payload is not to hand.
    """
    try:
        week = runtime.week(runtime.week_of(date))
    except Exception:  # noqa: BLE001 - no calendar means no weekly claim
        return 0
    row = conn.execute(
        "SELECT COUNT(*) AS n FROM yahoo_roster_snapshots "
        "WHERE manager = ? AND team_key = ? AND date >= ? AND date < ? "
        "  AND selected_slot = 'G'",
        (manager, team_key, week.start, date),
    ).fetchone()
    return int(row["n"]) if row else 0


def build_plan(
    conn: sqlite3.Connection,
    runtime: LeagueRuntime,
    roster: TeamRoster,
    values: ValueModel,
    goalie_source,
    date: str,
    manager: str = "",
    authority: LineupAuthority | None = None,
    goalie_starts_so_far: int | None = None,
) -> LineupPlan:
    """Decide tonight's lineup and diff it against what Yahoo currently has."""
    auth = authority or LineupAuthority()
    shape = runtime.shape()
    season = runtime.nhl_season
    playing_teams = calendar.teams_playing(conn, date, season)
    p_starts = goalie_source.starts(date) if goalie_source else {}

    notes: list[str] = []
    candidates: list[Candidate] = []
    idle: list[RosterPlayer] = []
    out: list[RosterPlayer] = []
    locked: list[RosterPlayer] = []

    for p in roster.players:
        if p.is_out:
            out.append(p)
            continue
        if not p.is_editable:
            locked.append(p)
            continue
        if p.team not in playing_teams:
            idle.append(p)
            continue
        if p.nhl_player_id is None:
            notes.append(f"{p.name} is not in the player map - cannot value him, left as-is.")
            locked.append(p)
            continue

        value = values.per_game(p.nhl_player_id, date)
        p_start: float | None = None
        note = ""

        if p.position == "G":
            p_start = float(p_starts.get(p.nhl_player_id, 0.0))
            if p_start < auth.min_goalie_p_start:
                note = f"{p_start:.0%} to start"
                idle.append(p)
                continue
            value *= p_start

        questionable = False
        if p.is_questionable:
            if auth.start_questionable == "never":
                out.append(p)
                continue
            questionable = auth.start_questionable == "only_if_needed"
            if questionable:
                note = f"{p.status} - started only to fill a slot"

        candidates.append(
            Candidate(player=p, value=value, p_start=p_start, note=note, questionable=questionable)
        )

    _demote_questionable(candidates)

    # The weekly goalie minimum is a rule, not a preference.
    if auth.enforce_min_games and runtime.min_games_played:
        _force_goalie_if_required(
            conn, runtime, roster, candidates, date, manager, goalie_starts_so_far, notes
        )

    incumbent = {
        c.player.player_key: slot
        for c in candidates
        if (slot := _engine_slot(c.player.selected_slot)) is not None
    }
    assignment = optimize_lineup(
        [(c.player.player_key, c.player.eligible, c.value) for c in candidates],
        shape,
        incumbent=incumbent,
    )

    moves, gain = _diff(roster, candidates, assignment, shape)
    empty = _empty_slots(shape, assignment)

    # Leave a lineup alone when the change is not worth making. Churning for
    # 0.01 makes the audit log unreadable and trains you to ignore the message.
    if moves and gain < auth.min_gain:
        notes.append(
            f"{len(moves)} change(s) available but worth only {gain:+.2f}, "
            f"below the agreed {auth.min_gain:.2f} - left alone."
        )
        moves, gain = [], 0.0

    lock = calendar.first_lock(conn, [p.team for p in roster.players if p.team], date, season)
    within, reason = _check_authority(auth, moves, notes)
    return LineupPlan(
        date=date,
        team_key=roster.team_key,
        manager=manager,
        moves=tuple(moves),
        gain=gain,
        playing=tuple(candidates),
        idle=tuple(idle),
        out=tuple(out),
        locked=tuple(locked),
        notes=tuple(notes),
        within_authority=within,
        authority_reason=reason,
        empty_slots=tuple(empty),
        lock_utc=lock[1] if lock else "",
        lock_team=lock[0] if lock else "",
    )


def _demote_questionable(candidates: list[Candidate]) -> None:
    """Rank day-to-day players below every healthy one, without going negative.

    The obvious implementation - subtract enough to push them under - drives
    the value negative, and the optimizer prefers an empty slot to a negative
    player, so a questionable player would never start even when he is the only
    option. Instead their values are compressed into the gap below the cheapest
    healthy candidate: order among them is kept, and they stay startable.
    """
    q = [i for i, c in enumerate(candidates) if c.questionable and c.value > 0]
    if not q:
        return
    healthy = [c.value for i, c in enumerate(candidates) if i not in set(q) and c.value > 0]
    floor = min(healthy) if healthy else 1.0
    top = max(candidates[i].value for i in q)
    for i in q:
        c = candidates[i]
        candidates[i] = Candidate(
            player=c.player,
            value=floor * 0.5 * (c.value / top),
            p_start=c.p_start,
            note=c.note,
            questionable=True,
        )


def _force_goalie_if_required(
    conn, runtime, roster, candidates, date, manager, so_far, notes
) -> None:
    """Make starting a goalie non-negotiable when the week is running out."""
    try:
        week = runtime.week(runtime.week_of(date))
    except Exception:  # noqa: BLE001
        return
    started = (
        so_far
        if so_far is not None
        else goalie_starts_this_week(conn, runtime, manager, roster.team_key, date)
    )
    needed = runtime.min_games_played - started
    if needed <= 0:
        return
    days_left = sum(1 for d in week.dates() if d >= date)
    if needed < days_left:
        return
    for i, c in enumerate(candidates):
        if c.is_goalie:
            candidates[i] = Candidate(
                player=c.player,
                value=c.value + 1e6,
                p_start=c.p_start,
                note="forced: the week's goalie minimum can no longer be met otherwise",
            )
    notes.append(
        f"Weekly goalie minimum: {needed} start(s) needed with {days_left} day(s) left - "
        f"starting a goalie tonight is mandatory."
    )


def _diff(roster, candidates, assignment, shape) -> tuple[list[Move], float]:
    """What must change in Yahoo, and what today's lineup gains by it.

    Two things this is careful about.

    A player whose team is idle tonight costs nothing sitting in a starting
    slot - he simply scores nothing - so he is left alone unless somebody who
    IS playing needs that slot. Benching every idle starter would mean eight
    pointless changes on a three-game Friday, and a tool that asks for eight
    changes worth nothing is one you stop reading.

    The gain is the difference between the two lineups' totals, not the sum of
    what moves in. A player sliding from C to Util changes the shape of the
    lineup and nothing about its value.
    """
    from puckpilot.engine.lineup import slot_instances

    by_key = {c.player.player_key: c for c in candidates}
    current = roster.slotted()
    target = {key: _slot_for(slot) for key, slot in assignment.items()}

    # Slots the new lineup claims, against what the roster actually has.
    capacity: dict[str, int] = {}
    for slot in slot_instances(shape):
        y = _slot_for(slot)
        capacity[y] = capacity.get(y, 0) + 1
    for y in target.values():
        capacity[y] = capacity.get(y, 0) - 1

    moves: list[Move] = []
    for key, want in target.items():
        have = current.get(key, "?")
        if have != want:
            moves.append(Move(player=by_key[key].player, to_slot=want, from_slot=have))

    # Anyone still in a starting slot the new lineup needs has to step aside.
    for p in roster.players:
        key = p.player_key
        if key in target or p.on_bench or p.on_ir:
            continue
        have = current.get(key, "?")
        if capacity.get(have, 0) <= 0:
            moves.append(Move(player=p, to_slot=BENCH, from_slot=have))
        else:
            capacity[have] = capacity[have] - 1

    new_total = sum(by_key[k].value for k in target)
    old_total = sum(c.value for c in candidates if not c.player.on_bench and not c.player.on_ir)
    moves.sort(key=lambda m: (m.kind_order, m.player.name))
    return moves, new_total - old_total


def _empty_slots(shape, assignment) -> list[str]:
    """Slots no player with a game tonight can fill.

    Not "empty" in Yahoo - an idle player may well be sitting in one - but they
    will score nothing either way, which is the fact worth reporting.
    """
    from puckpilot.engine.lineup import slot_instances

    filled: dict[str, int] = {}
    for slot in assignment.values():
        filled[slot] = filled.get(slot, 0) + 1
    empty: list[str] = []
    for slot in slot_instances(shape):
        if filled.get(slot, 0) > 0:
            filled[slot] -= 1
        else:
            empty.append(_slot_for(slot))
    return empty


def _check_authority(auth: LineupAuthority, moves, notes) -> tuple[bool, str]:
    """Whether to act, and always a reason - the reason is what gets printed.

    "Nothing to do" is trivially within authority, but saying "will act
    automatically" to someone who has not granted any is a lie about what the
    tool is about to do the night it does have something to do.
    """
    granted = "standing authority granted" if auth.enabled else "recommend only"
    if not moves:
        return auth.enabled, f"nothing to do ({granted})."
    if not auth.enabled:
        return False, "recommend only - no standing authority granted."
    blocked = [m.player.name for m in moves if not m.is_start and m.player.name in auth.never_bench]
    if blocked:
        return False, f"would bench {', '.join(blocked)}, who you said never to bench."
    if notes:
        return True, "will act automatically (see the notes above)."
    return True, "will act automatically."
