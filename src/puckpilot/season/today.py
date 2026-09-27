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
falling short forfeits the categories, so when the goalie games left in the
week can no longer be relied on to satisfy it, starting a goalie stops being a
preference - including one below the agreed P(start) floor. The count comes
from Yahoo's own roster payload, never from our snapshots.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field

from puckpilot.engine.lineup import optimize_lineup
from puckpilot.season import calendar
from puckpilot.season.authority import LineupAuthority
from puckpilot.season.roster import RosterPlayer, TeamRoster
from puckpilot.season.settings import IR_SLOTS, YAHOO_TO_POS, LeagueRuntime
from puckpilot.season.values import ValueModel

# Yahoo's slot name for a benched player, and the engine's name for the flex.
BENCH = "BN"
UTIL = "UTIL"
YAHOO_UTIL = "Util"

# Force tonight's goalie start unless the rest of the week reaches the minimum
# with at least this probability. Starting a goalie who has a game costs almost
# nothing; missing the minimum forfeits the category.
GOALIE_CONFIDENCE = 0.9

# Large enough to beat any real value, small enough to stay finite.
_FORCED_START = 1e6


@dataclass(frozen=True)
class Candidate:
    player: RosterPlayer
    value: float
    p_start: float | None = None
    note: str = ""
    questionable: bool = False
    # Added for the optimizer only - a forced start - and never counted as value
    # gained, or one mandatory goalie reports a million-point night.
    bump: float = 0.0

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
    def is_ir(self) -> bool:
        """Onto an IR slot - off the active roster."""
        return self.to_slot in IR_SLOTS

    @property
    def is_activation(self) -> bool:
        """Off an IR slot and back onto the active roster."""
        return self.from_slot in IR_SLOTS and self.to_slot not in IR_SLOTS

    @property
    def is_start(self) -> bool:
        return self.to_slot != BENCH and not self.is_ir

    @property
    def is_bench(self) -> bool:
        return self.to_slot == BENCH and not self.is_activation

    @property
    def kind_order(self) -> int:
        """Roster moves first - they change who is available - then starts."""
        if self.is_ir or self.is_activation:
            return -1
        if self.is_bench:
            return 2
        return 0 if self.from_slot in (BENCH, "?") else 1

    def describe(self) -> str:
        if self.is_ir:
            return f"IR    {self.player.name} {self.from_slot} -> {self.to_slot}"
        if self.is_activation:
            return f"ACTIVATE {self.player.name} {self.from_slot} -> {self.to_slot}"
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
    # IR is kept apart from the lineup diff: it gains nothing tonight, so the
    # min_gain test cannot judge it, and its authority is `manage_ir`.
    ir_moves: tuple[Move, ...] = ()
    ir_alerts: tuple[str, ...] = ()
    ir_within_authority: bool = False

    @property
    def is_noop(self) -> bool:
        return not self.moves and not self.ir_moves

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
        for a in self.ir_alerts:
            lines.append(f"  !! {a}")
        if self.ir_moves:
            lines.append("")
            how = "will make" if self.ir_within_authority else "recommended"
            lines.append(f"  Roster ({how}):")
            for m in self.ir_moves:
                lines.append(f"    {m.describe()}")
        lines.append("")
        if self.moves:
            lines.append(f"  {len(self.moves)} change(s), worth {self.gain:+.2f} today:")
            for m in self.moves:
                lines.append(f"    {m.describe()}")
        else:
            lines.append("  Lineup is already optimal - nothing to change.")
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


def yahoo_goalie_games(roster: TeamRoster, runtime: LeagueRuntime, date: str) -> int | None:
    """Goalie games already counted toward this week's minimum, per Yahoo.

    Only Yahoo's number is used. The fallback this replaced counted goalie
    *slot*-days in our own snapshots - two a day whether anyone played - so the
    minimum read as met by the second day and the forced start never fired.
    A count for another week (a roster read for a different date) is refused.
    """
    if roster.goalie_games is None:
        return None
    if roster.goalie_games_week is not None:
        try:
            if runtime.week_of(date) != roster.goalie_games_week:
                return None
        except Exception:  # noqa: BLE001 - no calendar: trust the payload
            pass
    return roster.goalie_games


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
    weights: dict[str, float] | None = None,
    goalie_confidence: float = GOALIE_CONFIDENCE,
) -> LineupPlan:
    """Decide tonight's lineup and diff it against what Yahoo currently has.

    `weights` is an approved week protocol's category stance. Absent - which is
    the default, and the case whenever nobody has agreed one - every category
    counts the same and this is the plain value decision.

    `goalie_starts_so_far` is Yahoo's count toward the weekly goalie minimum
    (`yahoo_goalie_games`). Without it the minimum is reported as unchecked
    rather than guessed at.
    """
    auth = authority or LineupAuthority()
    shape = runtime.shape()
    season = runtime.nhl_season
    playing_teams = calendar.teams_playing(conn, date, season)
    p_starts = goalie_source.starts(date) if goalie_source else {}

    notes: list[str] = []
    ir_moves, ir_alerts = ir_changes(runtime, roster)
    vacating = frozenset(m.player.player_key for m in ir_moves if m.is_ir)
    activating = frozenset(m.player.player_key for m in ir_moves if m.is_activation)
    forced = False
    if auth.enforce_min_games and runtime.min_games_played:
        forced, why = _minimum_forces_a_start(
            conn, runtime, roster, goalie_source, date, goalie_starts_so_far, goalie_confidence
        )
        if why:
            notes.append(why)
    if weights:
        notes.append(
            "Acting on this week's approved protocol: "
            + ", ".join(f"{k} x{v:g}" for k, v in sorted(weights.items()))
        )
    candidates: list[Candidate] = []
    spare: list[tuple[float, float, RosterPlayer]] = []
    idle: list[RosterPlayer] = []
    out: list[RosterPlayer] = []
    locked: list[RosterPlayer] = []

    for p in roster.players:
        if p.is_out:
            out.append(p)
            continue
        if p.on_ir and p.player_key not in activating:
            # Cannot play from IR, and there is no room to bring him back.
            idle.append(p)
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
        if weights:
            value *= values.tilt(p.nhl_player_id, weights)
        p_start: float | None = None
        note = ""

        if p.position == "G":
            p_start = float(p_starts.get(p.nhl_player_id, 0.0))
            below = p_start < auth.min_goalie_p_start
            # The minimum is a rule and the floor a preference, so on a night
            # the rule needs a start, any goalie who might play is a candidate.
            if below and not (forced and p_start > 0.0):
                if auth.fill_empty_goalie_slot and p_start > 0.0:
                    # Held back: he starts only if a G slot is left empty.
                    spare.append((value * p_start, p_start, p))
                    continue
                note = f"{p_start:.0%} to start"
                idle.append(p)
                continue
            if below:
                note = f"{p_start:.0%} to start - in for the weekly minimum"
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

    # A goalie below the floor fills a G slot only if nobody above it will:
    # one who does not start then scores what the empty slot would have.
    g_slots = sum(n for pos, n in shape.slots if pos == "G")
    room = g_slots - sum(1 for c in candidates if c.is_goalie)
    for i, (v, p_start, g) in enumerate(sorted(spare, key=lambda x: -x[0])):
        if i < room:
            candidates.append(
                Candidate(
                    player=g,
                    value=v,
                    p_start=p_start,
                    note=f"{p_start:.0%} to start - in because the slot would otherwise be empty",
                )
            )
        else:
            idle.append(g)

    # The weekly goalie minimum is a rule, not a preference.
    if forced:
        goalies = [i for i, c in enumerate(candidates) if c.is_goalie]
        for i in goalies:
            c = candidates[i]
            candidates[i] = Candidate(
                player=c.player,
                value=c.value,
                p_start=c.p_start,
                note=c.note or "forced: the week's goalie minimum needs a start tonight",
                questionable=c.questionable,
                bump=_FORCED_START,
            )
        if not goalies:
            notes.append("None of your goalies plays tonight, so the minimum is at risk.")

    incumbent = {
        c.player.player_key: slot
        for c in candidates
        if (slot := _engine_slot(c.player.selected_slot)) is not None
    }
    assignment = optimize_lineup(
        [(c.player.player_key, c.player.eligible, c.value + c.bump) for c in candidates],
        shape,
        incumbent=incumbent,
    )

    moves, gain = _diff(roster, candidates, assignment, shape, vacating)
    empty = _empty_slots(shape, assignment)

    # Leave a lineup alone when the change is not worth making. Churning for
    # 0.01 makes the audit log unreadable and trains you to ignore the message.
    if moves and gain < auth.min_gain:
        notes.append(
            f"{len(moves)} change(s) available but worth only {gain:+.2f}, "
            f"below the agreed {auth.min_gain:.2f} - left alone."
        )
        moves, gain = [], 0.0

    # Activated straight into tonight's lineup: that move replaces "to BN".
    straight_in = {m.player.player_key for m in moves if m.is_activation}
    ir_moves = [m for m in ir_moves if m.player.player_key not in straight_in]

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
        ir_moves=tuple(ir_moves),
        ir_alerts=tuple(ir_alerts),
        ir_within_authority=auth.enabled and auth.manage_ir,
    )


def ir_changes(runtime: LeagueRuntime, roster: TeamRoster) -> tuple[list[Move], list[str]]:
    """Roster moves between IR and the active roster, and what blocks them.

    Out players go onto a free IR slot they are eligible for - Yahoo's own
    `eligible_positions` says which, so no status table is kept here. That
    frees an active roster spot, which an add can then fill without a drop.

    The reverse matters more. A player still in an IR slot once he is no longer
    eligible for it makes the roster illegal, and Yahoo refuses every add and
    drop until it is fixed: he is activated when an active spot is free, and
    otherwise the alert says a drop has to come first.
    """
    cap = {s.position: s.count for s in runtime.slots if s.position in IR_SLOTS}
    free = dict(cap)
    for p in roster.players:
        if p.selected_slot in free:
            free[p.selected_slot] -= 1
    active_cap = sum(s.count for s in runtime.slots if s.position not in IR_SLOTS)
    active = sum(1 for p in roster.players if not p.on_ir)

    moves: list[Move] = []
    # Most restrictive slot first, so the flexible ones stay open for later.
    order = [s for s in ("IR", "IR-LT", "NA", "IR+") if s in cap] + sorted(
        s for s in cap if s not in ("IR", "IR-LT", "NA", "IR+")
    )
    for p in roster.players:
        if not p.is_out or p.on_ir or not p.is_editable:
            continue
        slot = next((s for s in order if s in p.yahoo_eligible and free.get(s, 0) > 0), None)
        if slot is None:
            continue
        moves.append(Move(player=p, to_slot=slot, from_slot=p.selected_slot))
        free[slot] -= 1
        active -= 1

    alerts: list[str] = []
    for p in roster.players:
        if not p.on_ir or p.selected_slot in p.yahoo_eligible:
            continue
        if active < active_cap:
            moves.append(Move(player=p, to_slot=BENCH, from_slot=p.selected_slot))
            active += 1
        else:
            alerts.append(
                f"{p.name} is no longer eligible for {p.selected_slot}, which makes the "
                f"roster illegal - Yahoo blocks every add and drop until he is back on "
                f"the active roster, and there is no free spot: a drop has to come first."
            )
    return moves, alerts


def open_roster_spots(runtime: LeagueRuntime, roster: TeamRoster) -> int:
    """Active roster spots free once today's IR moves are made.

    Each one is an add that needs no drop - the spot an injured player was
    holding, once he is on IR.
    """
    moves, _ = ir_changes(runtime, roster)
    cap = sum(s.count for s in runtime.slots if s.position not in IR_SLOTS)
    active = sum(1 for p in roster.players if not p.on_ir)
    active += sum(1 for m in moves if m.is_activation) - sum(1 for m in moves if m.is_ir)
    return max(cap - active, 0)


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
            bump=c.bump,
        )


def _minimum_forces_a_start(
    conn, runtime, roster, goalie_source, date, so_far, confidence
) -> tuple[bool, str]:
    """Whether tonight's goalie start is mandatory, and the sentence saying so.

    Asks how likely the rest of the week is to supply what is still needed:
    each later day's goalies with a game, the likeliest first, only as many as
    there are G slots, each an independent chance at P(start). The rule this
    replaced counted calendar days, so a Saturday with one goalie game left
    looked as safe as one with six. (Two goalies of the same club are not
    independent - one starts - but a roster rarely carries both.)
    """
    try:
        week = runtime.week(runtime.week_of(date))
    except Exception:  # noqa: BLE001 - no calendar, no weekly rule to apply
        return False, ""
    required = runtime.min_goalie_games(week.number)
    if not required:
        return False, ""
    if so_far is None:
        return False, (
            f"Yahoo's goalie count for the week was not in the roster read, so the "
            f"{required}-game minimum is not being checked tonight."
        )
    needed = required - so_far
    if needed <= 0:
        return False, ""
    chances = _goalie_chances_after(conn, runtime, roster, goalie_source, week, date)
    p_reach = _p_at_least(chances, needed)
    if p_reach >= confidence:
        return False, ""
    return True, (
        f"Weekly goalie minimum: {so_far} of {required} played, and only a "
        f"{p_reach:.0%} chance the rest of the week covers it - a goalie start "
        f"tonight is mandatory."
    )


def _goalie_chances_after(conn, runtime, roster, goalie_source, week, date) -> list[float]:
    """P(start) of every goalie game this roster could still count after `date`."""
    slots = sum(n for pos, n in runtime.shape().slots if pos == "G")
    ours = [
        p
        for p in roster.players
        if p.position == "G" and not p.is_out and p.nhl_player_id is not None
    ]
    out: list[float] = []
    for day in week.dates():
        if day <= date:
            continue
        playing = calendar.teams_playing(conn, day, runtime.nhl_season)
        starts = goalie_source.starts(day) if goalie_source else {}
        ps = sorted(
            (float(starts.get(p.nhl_player_id, 0.0)) for p in ours if p.team in playing),
            reverse=True,
        )
        out.extend(ps[:slots])
    return out


def _p_at_least(chances: list[float], k: int) -> float:
    """P(at least k successes) over independent chances - exact, by counting."""
    dist = [1.0]
    for p in chances:
        nxt = [0.0] * (len(dist) + 1)
        for n, q in enumerate(dist):
            nxt[n] += q * (1.0 - p)
            nxt[n + 1] += q * p
        dist = nxt
    return sum(dist[k:])


def _diff(
    roster, candidates, assignment, shape, vacating: frozenset[str] = frozenset()
) -> tuple[list[Move], float]:
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

    # Anyone still in a starting slot the new lineup needs has to step aside -
    # except a player going to IR, whose slot is freed by that move instead.
    for p in roster.players:
        key = p.player_key
        if key in target or p.on_bench or p.on_ir or key in vacating:
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
