"""The draft-night pick feed.

Yahoo's draft room pushes pick events to the browser over a websocket. This
reads that stream from the page the user has open and marks players off the
board. It is a permanent part of the draft-night console rather than a
diagnostic: on 2026-09-18 this is how the console learns what has been taken.

It is passive. It opens no connection of its own - it attaches to the socket the
user's own session already has, reads what arrives, and never sends.

Measured against a full 12-team, 16-round mock (2026-09-08 capture) it reported
**192 of 192 picks, none wrong, none unmapped**. Nothing else came close: the
header ticker managed 137/192, and no HTTP endpoint carried picks at all.

`parse_frame` is the only description of what this accepts, and it is
deliberately strict: anything whose shape it does not recognise is ignored
rather than guessed at. Frames it does accept identify players by Yahoo player
id rather than by name, which is why this path is robust where every
name-matching attempt was not - no accents, no nicknames, no "Mitch" vs
"Mitchell". Ids resolve through `yahoo_player_map`.

Picks carry a sequence number, so a dropped frame is *detectable* rather than
silently absent - `FeedState.gaps` reports it instead of the board quietly
desyncing, which is the property the DOM readers lacked.
"""

from __future__ import annotations

import re
import sqlite3
import threading
import time
from dataclasses import dataclass, field

# C / LW / RW / D / G, and comma-joined variants if Yahoo ever sends them.
POSITION_RE = re.compile(r"^[A-Z]{1,3}(,[A-Z]{1,3})*$")


@dataclass(frozen=True)
class PickFrame:
    """A player came off the board."""

    pick: int  # 1-based overall pick number
    yahoo_id: str
    seat: int  # 1-based draft slot, as Yahoo numbers them
    position: str


@dataclass(frozen=True)
class ClockFrame:
    """A pick is now on the clock. Arrives before the pick that answers it."""

    pick: int
    seat: int
    seconds: int


@dataclass(frozen=True)
class SnapshotFrame:
    """Every pick made so far, sent when the socket (re)connects:
    `P|1=5425,12,0|2=7905,10,0|...` - pick=yahoo_id,team,flag. Found in a live
    league room on draft night; ignoring it lost every pick made while the
    socket was reconnecting."""

    picks: tuple[PickFrame, ...]


def _parse_snapshot(fields: list[str]) -> SnapshotFrame | None:
    picks = []
    for item in fields:
        pick, _, rest = item.partition("=")
        parts = rest.split(",")
        if not pick.isdigit() or len(parts) < 2 or not parts[0].isdigit() or not parts[1].isdigit():
            continue
        picks.append(PickFrame(pick=int(pick), yahoo_id=parts[0], seat=int(parts[1]), position=""))
    return SnapshotFrame(tuple(picks)) if picks else None


def parse_frame(payload: str) -> PickFrame | ClockFrame | SnapshotFrame | None:
    """Classify one websocket frame. Returns None for anything else.

    Heartbeats, chat, joins and Yahoo's advice payloads all share this socket,
    so unknown shapes are ignored rather than guessed at.
    """
    if not payload or "|" not in payload:
        return None
    f = payload.split("|")
    if f[0] == "0" and len(f) == 6 and f[1].isdigit() and f[2].isdigit():
        if not POSITION_RE.match(f[4]):
            return None  # e.g. the lobby's 0|2227628|3|3|32763
        return PickFrame(pick=int(f[1]), yahoo_id=f[2], seat=int(f[3]), position=f[4])
    if f[0] == "P" and len(f) > 1:
        return _parse_snapshot(f[1:])
    if f[0] == "D" and len(f) == 4 and all(x.isdigit() for x in f[1:4]):
        return ClockFrame(pick=int(f[1]), seat=int(f[2]), seconds=int(f[3]))
    return None


def load_yahoo_id_map(conn: sqlite3.Connection, league_key: str | None = None) -> dict[str, int]:
    """Bare Yahoo player id -> NHL player id.

    The map table stores full keys ("477.p.6743"); the socket sends "6743".
    """
    sql = "SELECT player_key, nhl_player_id FROM yahoo_player_map WHERE nhl_player_id IS NOT NULL"
    params: tuple = ()
    if league_key:
        sql += " AND league_key = ?"
        params = (league_key,)
    out: dict[str, int] = {}
    for key, nhl_id in conn.execute(sql, params):
        out[str(key).rsplit(".", 1)[-1]] = int(nhl_id)
    return out


def load_yahoo_names(conn: sqlite3.Connection) -> dict[str, str]:
    """Bare Yahoo player id -> the name Yahoo gave him, INCLUDING players who
    never resolved to an NHL id.

    Those are exactly the ones the console has to name: "unmapped: 1" is a
    counter nobody can act on at the clock, "unmapped: Ivan Demidov" is a
    player the drafter can see is gone.
    """
    return {
        str(key).rsplit(".", 1)[-1]: str(name)
        for key, name in conn.execute("SELECT player_key, full_name FROM yahoo_player_map")
    }


def unmapped_label(yahoo_id: str, names: dict[str, str] | None) -> str:
    name = (names or {}).get(str(yahoo_id))
    return f"{name} (not on our board)" if name else f"Yahoo player {yahoo_id} (unmapped)"


@dataclass
class FeedState:
    """What the feed has observed, for display and for gap detection."""

    picks: dict[int, PickFrame] = field(default_factory=dict)
    on_the_clock: ClockFrame | None = None
    unmapped: list[str] = field(default_factory=list)
    frames_seen: int = 0

    @property
    def highest_pick(self) -> int:
        return max(self.picks, default=0)

    @property
    def room_progress(self) -> int:
        """Picks the room has made. The clock frame says it directly (the pick
        on the clock, minus one); without one, the unbroken run of picks from
        #1. Never the highest pick seen: a league room's reconnect snapshot
        lists the keeper picks at the END of the draft too, which read as the
        room being 127 picks ahead on draft night."""
        n = 0
        while n + 1 in self.picks:
            n += 1
        # A pick frame can land a moment before the clock frame that follows
        # it, so whichever evidence is further along wins.
        clock = self.on_the_clock.pick - 1 if self.on_the_clock is not None else 0
        return max(n, clock)

    @property
    def gaps(self) -> list[int]:
        """Pick numbers we never saw that the room has already made.

        The socket numbers every pick, so a dropped frame is *detectable* rather
        than silently absent - which is the property the DOM scrapers lacked.
        Bounded by the clock when there is one, so keeper picks listed ahead
        of it do not turn every live pick in between into a "gap".
        """
        top = self.on_the_clock.pick - 1 if self.on_the_clock is not None else self.highest_pick
        return [n for n in range(1, top + 1) if n not in self.picks]


class WebsocketFeed:
    """PickFeed over the draft room's websocket.

    Attaches to a whole browser context: the draft opens in its own tab, so
    following a single page loses the draft the moment the user joins one.
    """

    name = "websocket"

    def __init__(
        self,
        context,
        yahoo_to_nhl: dict[str, int],
        yahoo_names: dict[str, str] | None = None,
        frame_log=None,
    ):
        # Every raw frame, appended to this file when given. Draft night found
        # a live room whose pick messages mostly did not match the mock rooms
        # this parser was built on; the log is how the next parser gets built.
        self.frame_log = frame_log
        self.context = context
        self.yahoo_to_nhl = yahoo_to_nhl
        self.yahoo_names = yahoo_names or {}
        self.state = FeedState()
        self.last_error: str | None = None
        self._emitted: set[int] = set()
        self._lock = threading.Lock()
        self._attached: set[int] = set()
        context.on("page", self._attach_page)
        for page in list(getattr(context, "pages", []) or []):
            self._attach_page(page)

    def _attach_page(self, page) -> None:
        if id(page) in self._attached:
            return
        self._attached.add(id(page))
        page.on("websocket", self._attach_socket)

    def _attach_socket(self, ws) -> None:
        def on_frame(payload) -> None:
            self.ingest(str(payload))

        ws.on("framereceived", on_frame)

    def ingest(self, payload: str) -> None:
        """Feed one raw frame in. Public so a capture can be replayed offline."""
        if self.frame_log is not None:
            try:
                with open(self.frame_log, "a", encoding="utf-8") as fh:
                    fh.write(f"{time.time():.3f}\t{payload}\n")
            except OSError:
                pass
        try:
            frame = parse_frame(payload)
        except Exception as e:  # a malformed frame must never stop a draft
            self.last_error = f"{e.__class__.__name__}: {e}"
            return
        if frame is None:
            return
        with self._lock:
            self.state.frames_seen += 1
            if isinstance(frame, ClockFrame):
                self.state.on_the_clock = frame
            elif isinstance(frame, SnapshotFrame):
                for p in frame.picks:
                    self.state.picks.setdefault(p.pick, p)
            else:
                self.state.picks[frame.pick] = frame

    def poll(self, board) -> list:
        """New picks since the last call, in pick order."""
        from puckpilot.draft.feed import PickEvent

        with self._lock:
            fresh = sorted(set(self.state.picks) - self._emitted)
            frames = [self.state.picks[n] for n in fresh]
            self._emitted |= set(fresh)

        events = []
        for fr in frames:
            nhl_id = self.yahoo_to_nhl.get(fr.yahoo_id)
            # Yahoo numbers seats from 1; the board from 0.
            seat = max(0, fr.seat - 1)
            if nhl_id is None:
                # A player outside our pool (deep prospect). The pick still
                # happened: it used to be dropped here with `continue`, so
                # `board.made` never advanced and every pick after it was
                # scored against a clock one slot behind the room. Emitted as
                # an unidentified pick instead, which consumes the slot.
                with self._lock:
                    self.state.unmapped.append(fr.yahoo_id)
                events.append(
                    PickEvent(
                        None,
                        seat,
                        self.name,
                        label=unmapped_label(fr.yahoo_id, self.yahoo_names),
                        pick_no=fr.pick,
                    )
                )
                continue
            events.append(PickEvent(nhl_id, seat, self.name, pick_no=fr.pick))
        return events

    def unmapped_names(self) -> list[str]:
        with self._lock:
            ids = list(self.state.unmapped)
        return [unmapped_label(i, self.yahoo_names) for i in ids]

    def room_picks_made(self) -> int:
        """How far the ROOM has got, from the best evidence on the socket: the
        highest pick number seen, or the pick now on the clock minus one if
        that is further along (a pick frame can be dropped; the clock frame
        that follows it still says where the room is)."""
        with self._lock:
            return self.state.room_progress

    def status(self) -> dict:
        with self._lock:
            return {
                "chosen": "websocket",
                "frames": self.state.frames_seen,
                "picks_detected": len(self.state.picks),
                "highest_pick": self.state.room_progress,
                "gaps": self.state.gaps[:10],
                "unmapped": len(self.state.unmapped),
                "unmapped_names": [
                    unmapped_label(i, self.yahoo_names) for i in self.state.unmapped[-10:]
                ],
                "room_picks": self.state.room_progress,
                "on_the_clock": (
                    None
                    if self.state.on_the_clock is None
                    # No "seat": in a real league room the frame carries the
                    # Yahoo TEAM id, not the draft slot (it matched the slot
                    # only in mock rooms), so comparing it with the board's
                    # seat raised a false "draft order differs" alarm.
                    else {"pick": self.state.on_the_clock.pick}
                ),
                "error": self.last_error,
            }


def pump(context, seconds: float) -> None:
    """Let Playwright's sync driver service the browser for `seconds`.

    The sync API only processes incoming events while the caller is inside a
    Playwright call. `WebsocketFeed.poll()` reads in-memory state and makes no
    such call, so a loop that merely sleeps never dispatches anything — and the
    browser stalls with a blank tab when the draft room opens, because the
    client never attaches to the new target.

    Sleeping is the fallback for when no page is available; it keeps the loop
    alive but cannot pump, which is exactly the situation to avoid.
    """
    try:
        for page in context.pages:
            if not page.is_closed():
                page.wait_for_timeout(seconds * 1000)
                return
    except Exception:
        pass
    time.sleep(seconds)


def replay(
    payloads, yahoo_to_nhl: dict[str, int], yahoo_names: dict[str, str] | None = None
) -> WebsocketFeed:
    """Drive a feed from recorded frames, with no browser involved.

    This is what turns one captured draft into a permanent regression test.
    """

    class _NullContext:
        pages: list = []

        def on(self, *_a, **_k):
            return None

    feed = WebsocketFeed(_NullContext(), yahoo_to_nhl, yahoo_names)
    for payload in payloads:
        feed.ingest(str(payload))
    return feed
