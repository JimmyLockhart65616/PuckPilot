"""Roster legality: the states Yahoo will not transact from, and the moves that fix them.

Yahoo checks a roster before every add. Two states fail that check and are
reachable in any league with IR slots:

- a player in an IR / IR+ / IR-LT / NA slot he is no longer eligible for (he
  was cleared to play, called up, or his designation changed), and
- an active roster over its cap (a player brought back by hand into a full one).

Either one stops every add until it is fixed, so it is not a preference to
weigh: nothing else the tool proposes can be made while it stands. On
2026-10-19 a defenceman on IR+ was cleared to play with sixteen of sixteen
active spots full; the run printed "a drop has to come first" on every one of
sixty runs over twelve days, proposed nothing, and nothing made the drop.

So this module owns the whole loop, from Yahoo's own data (`eligible_positions`
and the league's slot counts) and nothing else:

    ir_changes   the slot moves that need no transaction - onto a free IR slot
                 he is eligible for, and back onto the active roster when a
                 spot is free (lineup authority makes these)
    check        every violation, in words
    resolve      for each violation the lineup cannot fix, the drop that fixes
                 it: the least valuable legal drop over the rest of the season,
                 or the returning player himself when he is worth less than
                 anyone he could replace. Keepers being protected, players
                 listed out or NA, and drops already promised to an approved
                 move are never chosen.
    as_resolved  the roster as it will be once those are made - what the add
                 search prices against, so its adds are made on a legal roster.

A fix that needs a drop is a transaction (`moves.MoveKind.ACTIVATE`) and so a
proposal a person approves, like any add. It is never weighed against the
add bar: it is the precondition for every add, not a choice among them.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace

from puckpilot.season.moves import BENCH, Move
from puckpilot.season.roster import RosterPlayer, TeamRoster
from puckpilot.season.settings import IR_SLOTS, LeagueRuntime

# Violation kinds.
IR_INELIGIBLE = "ir_ineligible"
OVER_ACTIVE_CAP = "over_active_cap"

# Onto IR: the most restrictive slot first, so the flexible ones stay open.
_IR_ORDER = ("IR", "IR-LT", "NA", "IR+")


def active_cap(runtime: LeagueRuntime) -> int:
    """Active roster spots: every slot but the IR kind."""
    return sum(s.count for s in runtime.slots if s.position not in IR_SLOTS)


def _active(roster: TeamRoster) -> int:
    return sum(1 for p in roster.players if not p.on_ir)


def ir_changes(runtime: LeagueRuntime, roster: TeamRoster) -> tuple[list[Move], list[str]]:
    """Roster moves between IR and the active roster, and what blocks them.

    Out players go onto a free IR slot they are eligible for - Yahoo's own
    `eligible_positions` says which, so no status table is kept here. That
    frees an active roster spot, which an add can then fill without a drop.

    The reverse matters more. A player still in an IR slot once he is no longer
    eligible for it makes the roster illegal, and Yahoo refuses every add until
    it is fixed: he is activated when an active spot is free, and otherwise the
    alert says a drop has to come first (`resolve` chooses it).
    """
    cap = {s.position: s.count for s in runtime.slots if s.position in IR_SLOTS}
    free = dict(cap)
    for p in roster.players:
        if p.selected_slot in free:
            free[p.selected_slot] -= 1
    limit = active_cap(runtime)
    active = _active(roster)

    moves: list[Move] = []
    order = [s for s in _IR_ORDER if s in cap] + sorted(s for s in cap if s not in _IR_ORDER)
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
    for p in _stuck(roster):
        if active < limit and p.is_editable:
            moves.append(Move(player=p, to_slot=BENCH, from_slot=p.selected_slot))
            active += 1
        elif active < limit:
            # His game today has begun: Yahoo will not move him until tomorrow,
            # and a move it refuses fails the whole lineup.
            alerts.append(
                f"{p.name} can come off {p.selected_slot} into a free spot, but his game today "
                f"has begun - he comes back tomorrow morning."
            )
        else:
            alerts.append(_ir_text(p))
    return moves, alerts


def open_roster_spots(runtime: LeagueRuntime, roster: TeamRoster) -> int:
    """Active roster spots free once today's IR moves are made.

    Each one is an add that needs no drop - the spot an injured player was
    holding, once he is on IR.
    """
    moves, _ = ir_changes(runtime, roster)
    active = _active(roster)
    active += sum(1 for m in moves if m.is_activation) - sum(1 for m in moves if m.is_ir)
    return max(active_cap(runtime) - active, 0)


def _stuck(roster: TeamRoster) -> tuple[RosterPlayer, ...]:
    """In an IR slot he is no longer eligible for - Yahoo's own test, the same
    one `TeamRoster.illegal_ir` makes (read off `players`, so the replay's
    roster adapters answer it too)."""
    return tuple(p for p in roster.players if p.on_ir and p.selected_slot not in p.yahoo_eligible)


def _ir_text(p: RosterPlayer) -> str:
    return (
        f"{p.name} is no longer eligible for {p.selected_slot}, which makes the roster "
        f"illegal - Yahoo refuses every add until he is back on the active roster, and "
        f"there is no free spot: a drop has to come first."
    )


@dataclass(frozen=True)
class Violation:
    """One reason Yahoo would refuse an add today."""

    kind: str
    players: tuple[RosterPlayer, ...]
    message: str


def check(runtime: LeagueRuntime, roster: TeamRoster) -> tuple[Violation, ...]:
    """Every violation on `roster` as Yahoo has it now.

    An ineligible IR player is a violation even when today's IR moves will
    bring him back by themselves: until they are made, an add is refused.
    """
    out: list[Violation] = []
    limit = active_cap(runtime)
    for p in _stuck(roster):
        out.append(Violation(IR_INELIGIBLE, (p,), _ir_text(p)))
    over = _active(roster) - limit
    if over > 0:
        out.append(
            Violation(
                OVER_ACTIVE_CAP,
                (),
                f"{_active(roster)} players on the active roster against {limit} spots - "
                f"Yahoo refuses every add until {over} {'is' if over == 1 else 'are'} dropped "
                f"or moved to IR.",
            )
        )
    return tuple(out)


@dataclass(frozen=True)
class Fix:
    """How one violation is made good.

    `player` is the one coming back from IR (None for an over-cap roster);
    `drop` the player who leaves, None when a free spot means the lineup can
    bring him back with no transaction at all.
    """

    player: RosterPlayer | None
    drop: RosterPlayer | None
    slot: str
    reason: str

    @property
    def needs_transaction(self) -> bool:
        return self.drop is not None

    @property
    def releases(self) -> bool:
        """The returning player is himself the one let go."""
        return self.drop is not None and self.player is not None and self.drop == self.player


@dataclass(frozen=True)
class Resolution:
    violations: tuple[Violation, ...]
    fixes: tuple[Fix, ...]
    # Violations with no fix, in words: no legal drop exists.
    unresolved: tuple[str, ...] = ()

    @property
    def legal(self) -> bool:
        return not self.violations

    @property
    def transactions(self) -> tuple[Fix, ...]:
        """The fixes a person has to approve."""
        return tuple(f for f in self.fixes if f.needs_transaction)


def droppable(
    roster: TeamRoster, hold: Iterable[str] = (), to_ir: Iterable[str] = ()
) -> list[RosterPlayer]:
    """Players who may be proposed as a drop - the one filter every search uses.

    Never: one Yahoo marks undroppable, one not in the player map (he cannot be
    valued), one on IR or going there today (an injured regular is put on IR,
    not cut for a streamer), one listed out or NA (whether that is a night or a
    season is a person's call), one of next season's protected keepers, or the
    drop of a move already approved and waiting to be made.
    """
    hold, to_ir = set(hold), set(to_ir)
    return [
        p
        for p in roster.players
        if not p.is_undroppable
        and p.nhl_player_id is not None
        and not p.on_ir
        and not p.is_out
        and p.player_key not in to_ir
        and not p.keeper_protected
        and p.player_key not in hold
    ]


def position_counts(roster: TeamRoster, to_ir: Iterable[str] = ()) -> dict[str, int]:
    """Active players by primary position, once today's IR moves are made."""
    to_ir = set(to_ir)
    counts: dict[str, int] = {}
    for p in roster.players:
        if not p.on_ir and p.player_key not in to_ir:
            counts[p.position] = counts.get(p.position, 0) + 1
    return counts


def legal_drop(incoming: str | None, p: RosterPlayer, counts: Mapping[str, int], rules) -> bool:
    """May `p` be dropped to make room for a player of position `incoming`?

    Position arithmetic only: over a positional cap the drop must come from
    that position, and no drop may take a position below its minimum -
    counting the player coming in, so a centre for a centre is never refused
    for leaving the roster a centre short. `incoming` None: nobody comes in.
    """
    if rules is None:
        return True
    if incoming is not None:
        over_cap = counts.get(incoming, 0) + 1 > rules.caps.get(incoming, 99)
        if over_cap and p.position != incoming:
            return False
    after = counts.get(p.position, 0) - 1 + (1 if p.position == incoming else 0)
    return after >= rules.mins.get(p.position, 0)


def resolve(
    runtime: LeagueRuntime,
    roster: TeamRoster,
    keep: Mapping[str, float],
    rules=None,
    hold: Iterable[str] = (),
    promised: Mapping[str, str] | None = None,
) -> Resolution:
    """The moves that make `roster` legal, cheapest first.

    `keep` is each player's rest-of-season value by player key (what letting
    him go costs - `week.keep_values`); `rules` the roster's positional caps
    and minimums (`LeagueConfig.draft_rules()`); `hold` drops already promised
    to other moves; `promised` a make-room drop already approved, by returning
    player - kept while that drop is still on the roster, since a decision
    was given on it.

    Returning players are placed most valuable first, so a free spot goes to
    the one worth most. Each make-room drop is the cheapest legal one not
    already used; the returning player is released instead when he is worth
    no more than that drop and may himself be let go.
    """
    violations = check(runtime, roster)
    if not violations:
        return Resolution((), ())
    moves, _ = ir_changes(runtime, roster)
    to_ir = {m.player.player_key for m in moves if m.is_ir}
    limit = active_cap(runtime)
    active = _active(roster) - len(to_ir)
    counts = position_counts(roster, to_ir)
    pool = droppable(roster, hold, to_ir)
    used: set[str] = set()

    def worth(p: RosterPlayer) -> float:
        return float(keep.get(p.player_key, 0.0))

    fixes: list[Fix] = []
    unresolved: list[str] = []
    by_key = {q.player_key: q for q in roster.players}
    for p in sorted(_stuck(roster), key=worth, reverse=True):
        agreed = by_key.get((promised or {}).get(p.player_key, ""))
        if agreed is not None and active >= limit:
            used.add(agreed.player_key)
            counts[agreed.position] = counts.get(agreed.position, 0) - (agreed != p)
            counts[p.position] = counts.get(p.position, 0) + (agreed != p)
            fixes.append(Fix(p, agreed, p.selected_slot, "approved - waiting to be made"))
            continue
        if active < limit:
            fixes.append(Fix(p, None, p.selected_slot, "a free active spot - no drop needed"))
            active += 1
            counts[p.position] = counts.get(p.position, 0) + 1
            continue
        legal = [
            q
            for q in sorted(pool, key=worth)
            if q.player_key not in used and legal_drop(p.position, q, counts, rules)
        ]
        may_release = not p.keeper_protected and not p.is_undroppable
        if may_release and (not legal or worth(p) <= worth(legal[0])):
            fixes.append(
                Fix(
                    p,
                    p,
                    p.selected_slot,
                    "worth less over the rest of the season than anyone he could replace",
                )  # fmt: skip
            )
            continue
        if not legal:
            unresolved.append(
                f"{p.name} cannot come back from {p.selected_slot}: no player may be dropped "
                f"for him (keepers, players out and roster minimums rule them all out)"
            )
            continue
        q = legal[0]
        used.add(q.player_key)
        counts[q.position] = counts.get(q.position, 0) - 1
        counts[p.position] = counts.get(p.position, 0) + 1
        fixes.append(Fix(p, q, p.selected_slot, "the least valuable legal drop"))

    over = active - limit
    for _ in range(max(over, 0)):
        legal = [
            q
            for q in sorted(pool, key=worth)
            if q.player_key not in used and legal_drop(None, q, counts, rules)
        ]
        if not legal:
            unresolved.append("the active roster is over its limit and nobody may be dropped")
            break
        q = legal[0]
        used.add(q.player_key)
        counts[q.position] = counts.get(q.position, 0) - 1
        fixes.append(Fix(None, q, "", "the active roster is over its limit"))
    return Resolution(violations, tuple(fixes), tuple(unresolved))


def as_resolved(roster: TeamRoster, resolution: Resolution) -> TeamRoster:
    """`roster` as it will stand once every fix is made.

    The drops gone and each returning player on the bench - which slot he
    then plays in is the lineup's call. A fix nobody could make leaves the
    roster as it is.
    """
    gone = {f.drop.player_key for f in resolution.fixes if f.drop is not None}
    back = {f.player.player_key for f in resolution.fixes if f.player is not None}
    players = tuple(
        replace(p, selected_slot=BENCH) if p.player_key in back else p
        for p in roster.players
        if p.player_key not in gone
    )
    return replace(roster, players=players)
