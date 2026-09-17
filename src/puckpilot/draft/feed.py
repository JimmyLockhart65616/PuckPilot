"""Where picks come from.

The console does not care whether a pick was typed by a human, produced by a
bot, or scraped off Yahoo — it only needs to know that someone is off the board.
One protocol covers all three, so the draft-night feed can be swapped (or fall
back to manual mid-draft) without the state machine noticing.

This mirrors `data.goalies.GoalieStartSource`, which solved the same shape of
problem for lineups: one interface, a hindsight implementation for backtests and
a real one for live use.

The Yahoo feed is deliberately absent. It gets built against recordings from
`ppilot draft capture`, because guessing at a draft room's internals and finding
out on draft night is exactly the failure this ordering avoids.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from puckpilot.draft.board import DraftBoard, DraftBoardError, UnknownPlayerError
from puckpilot.keepers import _norm


@dataclass(frozen=True)
class PickEvent:
    """A pick observed from outside, before the board has accepted it."""

    # None -> the room took someone we cannot identify. The pick still
    # happened, so it still consumes a slot; see `apply`.
    player_id: int | None
    seat: int | None = None  # None -> whoever is on the clock
    source: str = "feed"
    # What to call an unidentified pick on screen ("Ivan Demidov (unmapped)").
    label: str = ""
    # The room's own 1-based pick number, when the source knows it. Lets a
    # recovered feed name a pick that was advanced by hand instead of counting
    # the same slot twice.
    pick_no: int | None = None


class PickFeed(Protocol):
    """A source of picks. `poll` returns what is new since the last call."""

    name: str

    def poll(self, board: DraftBoard) -> list[PickEvent]: ...


class ManualFeed:
    """Picks typed (or pasted) by a human.

    The guaranteed path. Every other feed can fail on draft night — the room
    changes its markup, an API stays unapproved, the network drops — and this one
    cannot, so it is always live alongside whatever else is running.

    Names are resolved leniently: `submit` takes a player id, a full name, or
    enough of a name to be unambiguous, and raises with the candidate list rather
    than guessing when it is not.
    """

    name = "manual"

    def __init__(self) -> None:
        self._queue: list[PickEvent] = []

    def submit(self, board: DraftBoard, text: str, seat: int | None = None) -> PickEvent:
        """Queue one pick from typed text. Raises if the name is ambiguous."""
        text = text.strip()
        if not text:
            raise DraftBoardError("nothing to look up")
        if text.isdigit() and int(text) in board._row_of:
            event = PickEvent(int(text), seat, self.name)
            self._queue.append(event)
            return event

        rows = board.find(text)
        if not rows:
            raise DraftBoardError(f"no available player matches {text!r}")
        # An exact name wins outright even when it prefixes others, so "Sebastian
        # Aho" is never ambiguous against a second Sebastian Aho-like match.
        exact = [r for r in rows if _norm(str(board.u.names[r])) == _norm(text)]
        if len(exact) == 1:
            rows = exact
        elif len(rows) > 1:
            names = ", ".join(str(board.u.names[r]) for r in rows[:5])
            raise DraftBoardError(f"{text!r} is ambiguous: {names}")
        event = PickEvent(int(board.u.ids[rows[0]]), seat, self.name)
        self._queue.append(event)
        return event

    def submit_many(self, board: DraftBoard, text: str) -> tuple[list[PickEvent], list[str]]:
        """Bulk entry from a pasted draft-results panel, one name per line.

        Returns (queued, unresolved). Unresolved lines are handed back rather
        than aborting the paste — a results panel carries headers and team names
        alongside the players, and a partial reconciliation is still useful.
        """
        queued: list[PickEvent] = []
        unresolved: list[str] = []
        for raw in text.splitlines():
            line = raw.strip()
            if not line:
                continue
            try:
                queued.append(self.submit(board, line))
            except DraftBoardError:
                unresolved.append(line)
        return queued, unresolved

    def poll(self, board: DraftBoard) -> list[PickEvent]:
        out, self._queue = self._queue, []
        return out


class SimFeed:
    """Bot picks, for offline mock drafts.

    Uses the same bot field the draft sim scores against
    (`sim._default_opponents`), so a mock draft is played against the opponents
    the engine's win rate was actually measured over — practice against the same
    field the evidence came from, not a softer one.
    """

    name = "sim"

    def __init__(self, bots: list, rng: np.random.Generator, skip_seats: set[int] | None = None):
        self.bots = bots
        self.rng = rng
        self.skip_seats = skip_seats or set()

    def poll(self, board: DraftBoard) -> list[PickEvent]:
        seat = board.on_the_clock()
        if seat is None or seat in self.skip_seats:
            return []
        if not board.avail.any():
            # An exhausted board would otherwise have _pick_best return an
            # already-taken row forever; say nothing instead of looping.
            return []
        ctx = {"pick_no": board.made, "next_pick_no": board.next_pick_no(seat)}
        row = self.bots[seat].pick(
            board.u,
            board.avail,
            board.counts[seat],
            board.rules,
            board.picks_left(seat),
            self.rng,
            ctx,
        )
        return [PickEvent(int(board.u.ids[row]), seat, self.name)]


class YahooDraftFeed:
    """Picks read from Yahoo's own `draftresults` endpoint.

    Yahoo returns the whole draft every call, so `poll` diffs against what it
    has already reported rather than trusting a cursor - which also means a
    dropped poll, a restart, or a mid-draft reconnect self-heals on the next
    call instead of losing picks.

    Every failure is soft. A network blip, an expired session, an unmappable
    player: all return what is usable and record the rest. On draft night the
    console must keep working with manual entry underneath, and an exception
    here would take the whole thing down at the worst moment.
    """

    name = "yahoo"

    def __init__(
        self,
        session,
        league_key: str,
        key_to_nhl: dict[str, int],
        key_names: dict[str, str] | None = None,
    ):
        self.session = session
        self.league_key = league_key
        self.key_to_nhl = key_to_nhl
        self.key_names = key_names or {}
        self.seen: set[str] = set()
        self.unmapped: list[str] = []
        self.last_error: str | None = None
        self.seat_of_team: dict[str, int] = {}

    def set_seats(self, team_keys: list[str]) -> None:
        """Fix the seat order, so a pick lands on the right roster.

        Without this every pick would be attributed to whoever the board thinks
        is on the clock, which is right only while nothing has gone wrong.
        """
        self.seat_of_team = {k: i for i, k in enumerate(team_keys)}

    def poll(self, board: DraftBoard) -> list[PickEvent]:
        try:
            results = self.session.draft_results(self.league_key)
            self.last_error = None
        except Exception as e:  # network, auth, shape - never fatal here
            self.last_error = f"{e.__class__.__name__}: {e}"
            return []

        events: list[PickEvent] = []
        for entry in results:
            player_key = entry.get("player_key")
            if not player_key or player_key in self.seen:
                continue
            self.seen.add(player_key)
            nhl_id = self.key_to_nhl.get(player_key)
            seat = self.seat_of_team.get(entry.get("team_key", ""))
            pick_no = _int_or_none(entry.get("pick"))
            if nhl_id is None:
                # A prospect outside our NHL data. The pick still happened, so
                # it is emitted unidentified and consumes the slot, rather than
                # being dropped and leaving the clock a pick behind the room.
                self.unmapped.append(player_key)
                name = self.key_names.get(player_key)
                label = f"{name} (not on our board)" if name else f"{player_key} (unmapped)"
                events.append(PickEvent(None, seat, self.name, label=label, pick_no=pick_no))
                continue
            events.append(PickEvent(nhl_id, seat, self.name, pick_no=pick_no))
        return events


def _int_or_none(v) -> int | None:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def apply(board: DraftBoard, events: list[PickEvent]) -> tuple[list, list[str]]:
    """Push observed picks into the board. Returns (accepted, rejected messages).

    A feed that reports a player already off the board is not an error worth
    stopping a draft for — it is a duplicate poll, or a reconciliation catching
    up — so rejections are reported and the draft continues.
    """
    accepted, rejected = [], []
    for event in events:
        slot = None if event.pick_no is None else board.slot_for_room_pick(event.pick_no)
        if event.pick_no is not None and slot is None and event.player_id is None:
            # A keeper round on this board. A known player is already off it
            # (a keeper); an unknown one must not consume a live slot that the
            # room has not reached.
            rejected.append(f"pick {event.pick_no} is a keeper slot (not consumed)")
            continue
        try:
            if (
                event.player_id is not None
                and slot is not None
                and slot < board.made
                and board.picks[slot].row < 0
            ):
                # The feed has recovered and is naming a pick that was advanced
                # by hand as "unknown". Fill it in; recording it fresh would
                # count one slot twice.
                accepted.append(board.fill_placeholder(slot, event.player_id, event.source))
                continue
            if event.player_id is None:
                if slot is not None and slot < board.made:
                    rejected.append(f"pick {event.pick_no} is already on the board")
                    continue
                board.record_unknown(event.seat, event.label)
                rejected.append(f"{event.label or 'unidentified player'} (slot consumed)")
                continue
            accepted.append(board.record(event.player_id, event.seat, event.source))
        except UnknownPlayerError as e:
            # The pick really happened - the room took someone off a board that
            # does not contain them. Consume the slot so our clock stays with
            # the room's; `next_pick_no` drives every survival probability on
            # screen. Still reported, so the console can say how blind it is.
            if slot is not None and slot < board.made:
                rejected.append(f"{e} (pick {event.pick_no} already on the board)")
                continue
            board.record_unknown(event.seat, event.label)
            rejected.append(f"{e} (slot consumed)")
        except DraftBoardError as e:
            rejected.append(str(e))
    return accepted, rejected


class ReplayFeed:
    """A draft that already happened, played back at whatever pace you like.

    The draft-night console has never been watched end to end, because doing so
    used to require a live Yahoo room: a lobby, a browser, eleven strangers and
    forty minutes. That is a bad way to discover the interface is wrong.

    This drives the same `poll(board)` the websocket does, from picks already on
    disk - a harvested mock (`data/mocks/*.json`) or the committed fixture - so
    the whole console can be exercised offline, deterministically, and as fast
    or as slow as is useful.

    Picks arrive in draft order regardless of which seat made them, exactly as
    the socket delivers them. A pick the board cannot place (a player outside
    our ranked pool) is skipped by `apply`, the same as live.
    """

    name = "replay"

    def __init__(
        self,
        picks: list[dict],
        yahoo_to_nhl: dict[str, int],
        interval: float = 0.0,
        clock=time.monotonic,
        n_teams: int | None = None,
        yahoo_names: dict[str, str] | None = None,
    ):
        # Rooms are whatever the lobby hands out - the harvested ones are
        # 14-team while this league is 12. Seat numbers from a differently
        # shaped room cannot be mapped onto our board, so when they disagree
        # the pick lands on whoever is on the clock. The pick ORDER is what
        # makes a replay useful; the seat attribution is not transferable.
        self.source_teams = n_teams
        self.yahoo_to_nhl = yahoo_to_nhl
        self.yahoo_names = yahoo_names or {}
        # Test hook for the failure drill: while paused, poll() delivers
        # nothing, exactly as a dead socket would.
        self.paused = False
        self.missed = 0
        self.interval = interval
        self._clock = clock
        self._picks = sorted(picks, key=lambda p: int(p.get("pick", 0)))
        self._next = 0
        self._last = None
        self.last_error: str | None = None
        self.unmapped: list[str] = []

    @property
    def exhausted(self) -> bool:
        return self._next >= len(self._picks)

    def poll(self, board: DraftBoard) -> list[PickEvent]:
        if self.exhausted:
            return []
        now = self._clock()
        if self.interval and self._last is not None and now - self._last < self.interval:
            return []
        self._last = now

        row = self._picks[self._next]
        self._next += 1
        if self.paused:
            # The room keeps drafting while the socket is dead: the pick
            # happens (`room_picks` moves) but this feed never delivers it,
            # which is exactly what the drill has to recover from by hand.
            self.missed += 1
            return []
        yahoo_id = str(row.get("yahoo_id"))
        nhl_id = self.yahoo_to_nhl.get(yahoo_id)
        seat = int(row.get("seat", 0))
        # Seat numbers from a differently shaped room mean nothing here, and
        # nor do its pick numbers, which count no keeper slots.
        same_shape = not self.source_teams or self.source_teams == board.n_teams
        board_seat = max(0, seat - 1) if same_shape else None
        if nhl_id is None:
            # Consumes the slot, exactly as live: dropping it here used to let
            # a replay drift one pick further behind the room with every
            # prospect, which is the failure the replay exists to expose.
            self.unmapped.append(yahoo_id)
            label = self.yahoo_names.get(yahoo_id)
            return [
                PickEvent(
                    None,
                    board_seat,
                    self.name,
                    label=f"{label} (not on our board)" if label else f"Yahoo player {yahoo_id}",
                )
            ]
        return [PickEvent(nhl_id, board_seat, self.name)]

    def status(self) -> dict:
        return {
            "chosen": "replay",
            "frames": len(self._picks),
            "picks_detected": self._next - self.missed,
            "highest_pick": self._next,
            "room_picks": self._next,
            "gaps": [],
            "unmapped": len(self.unmapped),
            "unmapped_names": [
                f"{self.yahoo_names[i]} (not on our board)"
                if i in self.yahoo_names
                else f"Yahoo player {i}"
                for i in self.unmapped[-10:]
            ],
            "on_the_clock": None,
            "error": self.last_error,
            "paused": self.paused,
        }
