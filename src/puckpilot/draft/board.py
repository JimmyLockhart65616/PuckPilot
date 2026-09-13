"""Live draft state: what has gone, what is left, and whose turn it is.

`sim.run_draft` already models a draft, but it owns the whole event loop — it
decides every pick itself. A live draft is the inverse: picks arrive from
outside (a human typing, a feed polling) and the state has to be corrected,
undone, and inspected between them. This is that state, as a mutable object.

The pick sequence is the piece that repays care. A keeper league does not run a
clean snake: keepers occupy draft slots, so a team keeping fewer than the
maximum gets *more* live picks than one keeping the maximum. On the real
2026-27 board only 29 of a possible 36 keeper slots are eligible, so the board
is definitely uneven. That matters beyond bookkeeping, because
`RosterValuePolicy.survival_discount` — the knob the draft sim credits with the
engine's edge — is a function of how many picks pass before our next turn.
Getting it from a uniform-rounds assumption would quietly mis-tune every
recommendation, so the live slot sequence is modelled explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from puckpilot.draft.engine import DraftRules, Universe, eligible_positions
from puckpilot.draft.sim import effective_adp, snake_order
from puckpilot.keepers import _norm


class DraftBoardError(RuntimeError):
    pass


class UnknownPlayerError(DraftBoardError):
    """The room drafted someone our board has never heard of.

    Distinct from every other refusal because it is the only one where the pick
    REALLY HAPPENED. A duplicate poll must not advance the clock; a player we
    cannot rank must, or our count of picks made falls behind the room's.
    """


@dataclass(frozen=True)
class Pick:
    """One player off the board."""

    overall: int  # 0-based index into the live pick sequence; keepers are -1
    seat: int
    row: int  # universe row index
    player_id: int
    name: str
    position: str
    source: str  # keeper | manual | feed | sim


@dataclass
class Candidate:
    """A ranked recommendation, with the reasoning kept visible.

    The decomposition is not decoration: on the clock a human needs to see *why*
    a name is on top, and specifically whether it is on top because it is the
    best player or because the room is about to take him.
    """

    row: int
    player_id: int
    name: str
    position: str
    team: str
    vorp: float
    z_total: float
    score: float  # roster-aware, survival-discounted — what the engine ranks on
    adp_rank: float
    p_survive: float  # P(still there at our next pick)
    fills_starter: bool
    projected: dict[str, float] = field(default_factory=dict)


class DraftBoard:
    """Mutable draft state over a fixed `Universe`.

    Keepers come off the board and pre-fill position counts without consuming a
    live pick, mirroring `sim.run_draft`.
    """

    def __init__(
        self,
        universe: Universe,
        rules: DraftRules,
        my_seat: int,
        keepers: dict[int, list[int]] | None = None,
        keeper_rounds: dict[int, list[int]] | None = None,
        roster_rounds: int | None = None,
    ):
        self.u = universe
        self.rules = rules
        self.n_teams = rules.shape.n_teams
        if not 0 <= my_seat < self.n_teams:
            raise DraftBoardError(f"seat {my_seat} outside 0..{self.n_teams - 1}")
        self.my_seat = my_seat

        self._row_of: dict[int, int] = {int(pid): i for i, pid in enumerate(universe.ids)}
        self.avail = np.ones(len(universe), dtype=bool)
        self.counts: list[dict[str, int]] = [{} for _ in range(self.n_teams)]
        self.picks: list[Pick] = []
        self.keeper_picks: list[Pick] = []
        self.unmatched_keepers: list[int] = []

        keepers = keepers or {}
        for seat, pids in keepers.items():
            for pid in pids:
                row = self._row_of.get(int(pid))
                if row is None or not self.avail[row]:
                    # Never silent: an unplaced keeper leaves an elite player
                    # wrongly draftable, which is the worst board error there is.
                    self.unmatched_keepers.append(int(pid))
                    continue
                self.avail[row] = False
                self._bump(seat, row)
                self.keeper_picks.append(self._pick_at(-1, seat, row, "keeper"))

        # Same re-basing sim.run_draft does: with keepers gone, full-board ADP
        # ranks make everyone look later-going than they are, and every
        # comparison against a pick number is then wrong.
        if keepers:
            self.u = self.u.with_adp(effective_adp(self.u.adp_rank, self.avail))

        self.slots = self._live_slots(
            roster_rounds if roster_rounds is not None else rules.shape.roster_size,
            {s: len(p) for s, p in keepers.items()},
            keeper_rounds,
        )

    # ---- construction helpers -------------------------------------------------

    def _live_slots(
        self,
        roster_rounds: int,
        keeper_counts: dict[int, int],
        keeper_rounds: dict[int, list[int]] | None,
    ) -> list[tuple[int, int]]:
        """The (round, seat) slots that will actually be picked, in order.

        Yahoo assigns each keeper to a draft round, and that round is then not
        picked. Which round is a league-mechanics detail we do not know until the
        commissioner sets it, so the default assumption is stated rather than
        hidden: keepers consume a seat's *earliest* rounds, which is what Yahoo
        does when keepers are slotted by value. Pass `keeper_rounds` to override
        once the real assignment is known.
        """
        consumed: dict[int, set[int]] = {}
        for seat in range(self.n_teams):
            if keeper_rounds and seat in keeper_rounds:
                consumed[seat] = set(keeper_rounds[seat])
            else:
                consumed[seat] = set(range(keeper_counts.get(seat, 0)))

        order = snake_order(self.n_teams, roster_rounds)
        slots: list[tuple[int, int]] = []
        for i, seat in enumerate(order):
            rnd = i // self.n_teams
            if rnd not in consumed[seat]:
                slots.append((rnd, seat))
        return slots

    def _bump(self, seat: int, row: int) -> None:
        pos = str(self.u.pos[row])
        self.counts[seat][pos] = self.counts[seat].get(pos, 0) + 1

    def _pick_at(self, overall: int, seat: int, row: int, source: str) -> Pick:
        return Pick(
            overall=overall,
            seat=seat,
            row=row,
            player_id=int(self.u.ids[row]),
            name=str(self.u.names[row]),
            position=str(self.u.pos[row]),
            source=source,
        )

    # ---- live state -----------------------------------------------------------

    @property
    def made(self) -> int:
        return len(self.picks)

    @property
    def complete(self) -> bool:
        return self.made >= len(self.slots)

    def on_the_clock(self) -> int | None:
        """Seat picking now, or None once the draft is over."""
        return None if self.complete else self.slots[self.made][1]

    def current_round(self) -> int | None:
        return None if self.complete else self.slots[self.made][0] + 1

    def supply(self) -> dict[str, int]:
        """How many players remain at each position.

        The engine does not use this - dynamic replacement was built, measured,
        and left off because it helped in one target season and hurt in another
        (see `RosterValuePolicy._dynamic_vorp`). So the scarcity goes to the
        drafter as a fact instead of into the score as a weight.
        """
        # Counted above replacement, not raw. The pool is deliberately deeper
        # than the draft is long, so a raw count reads "D 340 left" and means
        # nothing; what a drafter wants is how many STARTABLE ones remain, and
        # that number is immune to how deep the pool happens to go.
        startable = self.avail & (self.u.vorp > 0)
        return {
            pos: int((startable & (self.u.pos == pos)).sum())
            for pos in sorted({str(p) for p in self.u.pos})
        }

    def depth_after(self, row: int, steps: int = 3) -> float:
        """VORP drop from this player to the `steps`-th next available at his
        position - a real read of the cliff.

        The previous signal compared against one player on an already-truncated
        shortlist, so it could only ever say "better than the next guy on this
        list". This looks at the actual remaining pool.
        """
        pos = self.u.pos[row]
        here = self.avail & (self.u.pos == pos)
        here[row] = False
        if not here.any():
            return 0.0
        rest = np.sort(self.u.vorp[here])[::-1]
        return float(self.u.vorp[row] - rest[min(steps, len(rest)) - 1])

    def pick_context(self, seat: int | None = None) -> dict:
        """The `ctx` a policy needs to score this board, built in ONE place.

        Every key here changes what the engine picks, so a caller that builds
        its own dict silently scores a different board. That already happened:
        adding `avail` to `recommend` made the console disagree with the same
        policy called directly, because the two constructed ctx separately.
        """
        s = self.my_seat if seat is None else seat
        return {
            "pick_no": self.made,
            "next_pick_no": self.next_pick_no(s),
            "avail": self.avail,
        }

    def next_pick_no(self, seat: int | None = None, after: int | None = None) -> int | None:
        """0-based index of `seat`'s next pick after `after` (default: now).

        This is what feeds `ctx["next_pick_no"]`, so it deliberately matches the
        semantics `sim.run_draft` tuned against: an absolute index into the pick
        sequence, not a round or a countdown.
        """
        seat = self.my_seat if seat is None else seat
        start = self.made if after is None else after + 1
        for i in range(start, len(self.slots)):
            if self.slots[i][1] == seat:
                return i
        return None

    def picks_left(self, seat: int | None = None) -> int:
        seat = self.my_seat if seat is None else seat
        return sum(1 for i in range(self.made, len(self.slots)) if self.slots[i][1] == seat)

    def roster(self, seat: int | None = None) -> list[Pick]:
        """Players on a seat's roster. Placeholders for picks we could not rank
        are excluded - they consumed a slot, but we do not know who they were,
        and callers index the universe by `row`."""
        seat = self.my_seat if seat is None else seat
        return [p for p in self.keeper_picks + self.picks if p.seat == seat and p.row >= 0]

    def needs(self, seat: int | None = None) -> dict[str, int]:
        """Unmet roster minimums — what still has to be filled to be legal."""
        seat = self.my_seat if seat is None else seat
        counts = self.counts[seat]
        return {
            pos: n
            for pos, m in self.rules.mins.items()
            if (n := max(0, m - counts.get(pos, 0))) > 0
        }

    # ---- mutation -------------------------------------------------------------

    def record(self, player_id: int, seat: int | None = None, source: str = "manual") -> Pick:
        """Mark a player drafted by the seat on the clock (or an explicit seat)."""
        if self.complete:
            raise DraftBoardError("draft is already complete")
        row = self._row_of.get(int(player_id))
        if row is None:
            raise UnknownPlayerError(f"player {player_id} is not in the ranked universe")
        if not self.avail[row]:
            raise DraftBoardError(f"{self.u.names[row]} is already off the board")

        overall = self.made
        seat = self.slots[overall][1] if seat is None else int(seat)
        if not 0 <= seat < self.n_teams:
            # A seat number we cannot place used to walk off the end of
            # `counts` with an IndexError, taking the console down. On draft
            # night the feed is the only pick source and nobody is at the
            # keyboard, so an unexpected seat must be a refusal `apply` can
            # swallow, not a crash.
            raise DraftBoardError(f"seat {seat} is outside this {self.n_teams}-team board")
        self.avail[row] = False
        self._bump(seat, row)
        pick = self._pick_at(overall, seat, row, source)
        self.picks.append(pick)
        return pick

    def record_unknown(self, seat: int | None = None, label: str = "") -> Pick:
        """Consume a pick for a player we cannot rank.

        The room took someone off a board that does not contain them, and that
        pick is real whether we can price him or not. Refusing it used to leave
        `made` behind the room, and `made` is the index into `slots` that
        produces `on_the_clock`, `current_round` and `next_pick_no` - and
        `survival()` is a logistic in `adp_rank - next_pick_no`. So a silently
        dropped pick does not merely lose a name: it makes every "lasts N%" on
        screen and every timing reason wrong, for the rest of the draft, and
        the error accumulates. With 11 of the top-163 Yahoo-priced players
        currently off our board, a full room drifts 15-20 picks.

        Availability and position counts are deliberately untouched: we do not
        know who he was, so we cannot claim to know what position was filled.
        """
        if self.complete:
            raise DraftBoardError("draft is already complete")
        overall = self.made
        seat = self.slots[overall][1] if seat is None else int(seat)
        if not 0 <= seat < self.n_teams:
            seat = self.slots[overall][1]
        pick = Pick(
            overall=overall,
            seat=seat,
            row=-1,
            player_id=0,
            name=label or "(not on our board)",
            position="?",
            source="unknown",
        )
        self.picks.append(pick)
        return pick

    def undo(self) -> Pick | None:
        """Take back the last pick. A mistyped name mid-draft is not a crisis."""
        if not self.picks:
            return None
        pick = self.picks.pop()
        if pick.row < 0:
            # A placeholder for a player we could not rank: it consumed a slot
            # and nothing else, so there is nothing to give back.
            return pick
        self.avail[pick.row] = True
        pos = pick.position
        self.counts[pick.seat][pos] = max(0, self.counts[pick.seat].get(pos, 0) - 1)
        return pick

    # ---- lookup ---------------------------------------------------------------

    def find(self, query: str, limit: int = 8, available_only: bool = True) -> list[int]:
        """Rows matching a typed/scraped name, best first.

        Accent- and punctuation-insensitive via the same normalizer the keeper
        list uses, so 'stutzle', 'Stützle' and 'Tim Stuetzle' are one player.
        Exact match wins outright; otherwise prefix beats substring.
        """
        q = _norm(query)
        if not q:
            return []
        exact: list[int] = []
        prefix: list[int] = []
        contains: list[int] = []
        for row, raw in enumerate(self.u.names):
            if available_only and not self.avail[row]:
                continue
            n = _norm(str(raw))
            if n == q:
                exact.append(row)
            elif n.startswith(q):
                prefix.append(row)
            elif q in n:
                contains.append(row)

        # Match quality first, then value — so a hurried partial name surfaces
        # the player actually meant rather than an obscure one that also matches.
        def by_value(rows: list[int]) -> list[int]:
            return sorted(rows, key=lambda r: -float(self.u.vorp[r]))

        return (by_value(exact) + by_value(prefix) + by_value(contains))[:limit]

    def eligible_now(self, seat: int | None = None) -> set[str]:
        seat = self.my_seat if seat is None else seat
        return eligible_positions(self.counts[seat], self.rules, self.picks_left(seat))
