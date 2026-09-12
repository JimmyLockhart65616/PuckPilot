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

from puckpilot.draft.board import DraftBoard, DraftBoardError
from puckpilot.keepers import _norm


@dataclass(frozen=True)
class PickEvent:
    """A pick observed from outside, before the board has accepted it."""

    player_id: int
    seat: int | None = None  # None -> whoever is on the clock
    source: str = "feed"


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

    def __init__(self, session, league_key: str, key_to_nhl: dict[str, int]):
        self.session = session
        self.league_key = league_key
        self.key_to_nhl = key_to_nhl
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
            if nhl_id is None:
                # A prospect outside our NHL data. Recorded so the console can
                # say "Yahoo took someone we cannot rank" instead of desyncing.
                self.unmapped.append(player_key)
                continue
            events.append(
                PickEvent(nhl_id, self.seat_of_team.get(entry.get("team_key", "")), self.name)
            )
        return events


def apply(board: DraftBoard, events: list[PickEvent]) -> tuple[list, list[str]]:
    """Push observed picks into the board. Returns (accepted, rejected messages).

    A feed that reports a player already off the board is not an error worth
    stopping a draft for — it is a duplicate poll, or a reconciliation catching
    up — so rejections are reported and the draft continues.
    """
    accepted, rejected = [], []
    for event in events:
        try:
            accepted.append(board.record(event.player_id, event.seat, event.source))
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
    ):
        # Rooms are whatever the lobby hands out - the harvested ones are
        # 14-team while this league is 12. Seat numbers from a differently
        # shaped room cannot be mapped onto our board, so when they disagree
        # the pick lands on whoever is on the clock. The pick ORDER is what
        # makes a replay useful; the seat attribution is not transferable.
        self.source_teams = n_teams
        self.yahoo_to_nhl = yahoo_to_nhl
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
        yahoo_id = str(row.get("yahoo_id"))
        nhl_id = self.yahoo_to_nhl.get(yahoo_id)
        if nhl_id is None:
            # Recorded rather than silently dropped: the console shows how far
            # the board is behind the room, and that has to stay honest here.
            self.unmapped.append(yahoo_id)
            return []
        seat = int(row.get("seat", 0))
        if self.source_teams and self.source_teams != board.n_teams:
            return [PickEvent(nhl_id, None, self.name)]
        return [PickEvent(nhl_id, max(0, seat - 1), self.name)]

    def status(self) -> dict:
        return {
            "chosen": "replay",
            "frames": len(self._picks),
            "picks_detected": self._next,
            "highest_pick": self._next,
            "gaps": [],
            "unmapped": len(self.unmapped),
            "on_the_clock": None,
            "error": self.last_error,
        }
