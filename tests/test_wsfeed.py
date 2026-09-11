"""Yahoo draft-room websocket protocol — the pick feed.

The fixture is a real 12-team, 16-round Yahoo mock draft captured 2026-09-08:
385 protocol frames, replayed here in full. It contains only pick and clock
frames (player ids, seat numbers) — no session data, no participant names — so a
live draft never has to be run again to test this path.

Scored against Yahoo's own results email, this source got 190/190 mappable picks
right with nothing invented and nothing out of order. The two it cannot resolve
are prospects absent from our NHL data entirely, which is a pool-coverage gap
rather than a feed defect.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from puckpilot.data import store
from puckpilot.draft.wsfeed import (
    ClockFrame,
    PickFrame,
    load_yahoo_id_map,
    parse_frame,
    replay,
)

FIXTURE = Path(__file__).parent / "fixtures" / "yahoo_draft_ws.json"


@pytest.fixture(scope="module")
def capture():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


# ---- frame grammar --------------------------------------------------------


def test_pick_frame_is_parsed():
    """0 | pick | yahooPlayerId | seat | position | flag"""
    f = parse_frame("0|1|6743|1|C|0")
    assert isinstance(f, PickFrame)
    assert (f.pick, f.yahoo_id, f.seat, f.position) == (1, "6743", 1, "C")


def test_clock_frame_is_parsed():
    f = parse_frame("D|13|12|30")
    assert isinstance(f, ClockFrame)
    assert (f.pick, f.seat, f.seconds) == (13, 12, 30)


def test_lobby_frame_is_rejected_on_shape():
    """`0|2227628|3|3|32763` shares the type byte but has five fields and a
    number where the position belongs. Accepting it would inject a phantom
    pick numbered 2,227,628."""
    assert parse_frame("0|2227628|3|3|32763") is None


def test_unrelated_traffic_is_ignored():
    """Heartbeats, joins, chat and Yahoo's own advice share this socket."""
    for payload in ("C|24", "J|3", "L|5", "H|S|30|0|0|0", "", "S", "O|other|118|[{}]"):
        assert parse_frame(payload) is None


def test_multi_position_codes_are_accepted():
    f = parse_frame("0|7|7554|7|LW,RW|0")
    assert isinstance(f, PickFrame) and f.position == "LW,RW"


# ---- full replay of a real draft ------------------------------------------


def test_replays_every_pick_of_a_real_draft_in_order(capture, db):
    """The regression test the live capture bought us: 192 picks, exact order."""
    expected = capture["expected_picks"]
    # map Yahoo ids straight to synthetic NHL ids so the test needs no real DB
    id_map = {}
    for frame in (parse_frame(p) for p in capture["frames"]):
        if isinstance(frame, PickFrame):
            id_map[frame.yahoo_id] = int(frame.yahoo_id)

    feed = replay(capture["frames"], id_map)
    state = feed.state
    assert len(state.picks) == 192
    assert state.highest_pick == 192
    assert state.gaps == [], "a numbered pick sequence must have no holes"
    assert sorted(state.picks) == list(range(1, 193))
    assert len(expected) == 192


def test_seat_attribution_follows_the_snake(capture, db):
    """Seat 3 in a 12-team snake picks 3, 22, 27, 46, 51... Getting this wrong
    would put opponents' players on our roster."""
    feed = replay(capture["frames"], {})
    seat_of = {p: f.seat for p, f in feed.state.picks.items()}
    seat3 = sorted(p for p, s in seat_of.items() if s == 3)
    assert seat3[:5] == [3, 22, 27, 46, 51]
    assert len(seat3) == 16  # one per round


def test_poll_emits_each_pick_once_and_maps_seats_to_zero_based(capture, db):
    class Board:
        pass

    id_map = {
        f.yahoo_id: int(f.yahoo_id)
        for f in (parse_frame(p) for p in capture["frames"])
        if isinstance(f, PickFrame)
    }
    feed = replay(capture["frames"], id_map)
    events = feed.poll(Board())
    assert len(events) == 192
    assert events[0].seat == 0, "Yahoo numbers seats from 1, the board from 0"
    assert feed.poll(Board()) == [], "a pick must never be reported twice"


def test_unmappable_players_are_recorded_not_silently_dropped(capture, db):
    """Deep prospects have no NHL data. The board must know it is behind."""

    class Board:
        pass

    feed = replay(capture["frames"], {})  # nothing maps
    assert feed.poll(Board()) == []
    assert len(feed.state.unmapped) == 192
    assert feed.status()["unmapped"] == 192


# ---- gap detection --------------------------------------------------------


def test_a_dropped_frame_shows_up_as_a_gap():
    """The socket numbers every pick, so a lost frame is detectable — the
    property the DOM scrapers could never provide."""
    feed = replay(["0|1|100|1|C|0", "0|3|300|3|C|0"], {})
    assert feed.state.highest_pick == 3
    assert feed.state.gaps == [2]


def test_no_gaps_reported_before_the_draft_starts():
    feed = replay(["C|24", "J|3"], {})
    assert feed.state.gaps == [] and feed.state.highest_pick == 0


def test_on_the_clock_tracks_the_latest_clock_frame():
    feed = replay(["D|1|1|30", "0|1|100|1|C|0", "D|2|2|30"], {})
    assert feed.state.on_the_clock.pick == 2
    assert feed.status()["on_the_clock"] == {"pick": 2, "seat": 2}


def test_malformed_frames_never_raise():
    """A draft must not stop because one frame was garbage."""
    feed = replay(["0|x|y|z|C|0", None, 12345, "0|1|", "0|1|100|1|C|0"], {})
    assert len(feed.state.picks) == 1


# ---- id map ---------------------------------------------------------------


def test_id_map_strips_the_game_key_prefix(db):
    """The map stores '477.p.6743'; the socket sends '6743'."""
    store.upsert_player(db, 8478402, "Connor McDavid", "C", "EDM")
    db.execute(
        "INSERT INTO yahoo_player_map"
        " (player_key, league_key, full_name, nhl_player_id, adp_rank)"
        " VALUES ('477.p.6743', '477.l.1', 'Connor McDavid', 8478402, 1)"
    )
    assert load_yahoo_id_map(db) == {"6743": 8478402}


def test_id_map_omits_players_we_could_not_resolve(db):
    db.execute(
        "INSERT INTO yahoo_player_map"
        " (player_key, league_key, full_name, nhl_player_id, adp_rank)"
        " VALUES ('477.p.33811', '477.l.1', 'Gavin McKenna', NULL, 92)"
    )
    assert load_yahoo_id_map(db) == {}


# ---- driver pumping -------------------------------------------------------


class _Page:
    """Minimal stand-in for a Playwright page."""

    def __init__(self, closed=False, raises=False):
        self._closed, self._raises, self.waits = closed, raises, []

    def is_closed(self):
        return self._closed

    def wait_for_timeout(self, ms):
        if self._raises:
            raise RuntimeError("Target page, context or browser has been closed")
        self.waits.append(ms)


class _Ctx:
    def __init__(self, pages):
        self.pages = pages


def test_pump_waits_inside_playwright_not_python():
    """The sync driver only dispatches events from inside a Playwright call.
    A loop that merely sleeps never sees the draft room's new tab open, and the
    browser hangs on a blank page - the exact failure this prevents."""
    from puckpilot.draft.wsfeed import pump

    page = _Page()
    pump(_Ctx([page]), 0.01)
    assert page.waits == [10.0]


def test_pump_skips_closed_tabs():
    from puckpilot.draft.wsfeed import pump

    dead, live = _Page(closed=True), _Page()
    pump(_Ctx([dead, live]), 0.01)
    assert live.waits == [10.0] and dead.waits == []


def test_pump_falls_back_to_sleeping_when_no_page_is_usable():
    """Still has to return so the console keeps serving, even though it cannot
    pump - losing the browser must not spin the loop or crash it."""
    import time

    from puckpilot.draft.wsfeed import pump

    for ctx in (_Ctx([]), _Ctx([_Page(closed=True)]), _Ctx([_Page(raises=True)])):
        start = time.perf_counter()
        pump(ctx, 0.02)
        assert time.perf_counter() - start >= 0.015
