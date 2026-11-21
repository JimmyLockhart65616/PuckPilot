"""What can change on a roster, as types rather than strings.

Two different things are called a move here, and both live in this module so
every layer - the lineup, roster legality, the approval queue, the page and an
executor - reads them from one place:

- `Move`: a slot change on the current roster (start, bench, onto IR, back off
  it). Lineup authority covers these; the actuator makes them.
- `MoveKind`: a *transaction* - something that changes who is on the roster and
  so always goes through a person's approval (`proposals.py`). Each kind says
  whether it spends one of the league's acquisitions, how its card reads, and
  what the roster must show for it to count as made.

A new kind of transaction is one more `MoveKind` member and the places that
dispatch on it (`transactions.carry_out`, an executor's `kinds`); nothing else
should test a kind by string.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from puckpilot.season.roster import RosterPlayer
from puckpilot.season.settings import IR_SLOTS

# Yahoo's slot name for a benched player.
BENCH = "BN"


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


class MoveKind(StrEnum):
    """A transaction a person approves. The value is what `waiver_proposals.kind` stores.

    ADD_DROP  a free agent in, and - optionally - one of ours out. The kind every
              proposal had before 2026-10-31, add-only moves included.
    ACTIVATE  a player in an IR slot he is no longer eligible for comes back to
              the active roster, which is full: one player is dropped to make
              room (or, when he is the cheapest himself, he is the one released).
              Until it is made the roster is illegal and Yahoo refuses every add.
    """

    ADD_DROP = "add_drop"
    ACTIVATE = "activate"

    @property
    def spends_acquisition(self) -> bool:
        """Whether Yahoo counts it against the weekly and season add limits.

        Only adds are acquisitions; a drop is not, and the league read on
        2026-10-12 has no drop limit.
        """
        return self is MoveKind.ADD_DROP

    @property
    def is_search_result(self) -> bool:
        """Raised by the add search, and so re-priced and withdrawn by it.

        A legality move is not: it is required, not chosen, and only the
        roster becoming legal another way retires it.
        """
        return self is MoveKind.ADD_DROP

    def head(self, add_name: str, drop_name: str, slot: str = "") -> str:
        """The card's first line, in the page's words."""
        if self is MoveKind.ACTIVATE:
            back = f" comes off {slot}" if slot else " comes back"
            if not drop_name or drop_name == add_name:
                return f"Release {add_name} from {slot or 'IR'} - the roster is illegal until then"
            return f"Make room: drop {drop_name} so {add_name}{back}"
        return f"Add {add_name}" + (f" – drop {drop_name}" if drop_name else "")

    def made_on(self, roster, add_key: str, drop_key: str) -> bool:
        """Whether `roster` (a fresh read) shows this move made.

        ADD_DROP: the add on it and the drop (if any) gone. ACTIVATE: the drop
        gone, and the returning player - unless he was the one released - still
        rostered. Whether he is already off the IR slot is the lineup's to do on
        the same run, so it is not part of "made".
        """
        keys = {p.player_key for p in roster.players}
        if self is MoveKind.ACTIVATE:
            released = not drop_key or drop_key == add_key
            if released:
                return add_key not in keys
            return drop_key not in keys and add_key in keys
        return add_key in keys and (not drop_key or drop_key not in keys)


def kind_of(value: str | None) -> MoveKind:
    """The stored kind as a `MoveKind`; rows from before kinds existed are ADD_DROP."""
    if not value:
        return MoveKind.ADD_DROP
    try:
        return MoveKind(value)
    except ValueError as e:
        known = ", ".join(k.value for k in MoveKind)
        raise ValueError(f"unknown move kind {value!r} (known: {known})") from e
