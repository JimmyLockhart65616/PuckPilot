"""The end-to-end harness, and the push path it drives.

Two things are tested here. First that a real room's frame stream survives
every hop - websocket parser, board, page, push, relay, guest - offline, from
the committed capture. Second that the harness can fail: a relay that alters
what it was given, rejects the console's key, lets a guest write, or runs a
different build must each turn a run red. A harness that has only ever passed
has not shown it checks anything.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from puckpilot.cli import _push_snapshots
from puckpilot.draft import e2e
from puckpilot.draft.board import DraftBoard
from puckpilot.draft.engine import DraftRules, Universe
from puckpilot.draft.wsfeed import PickFrame, WebsocketFeed, parse_frame
from puckpilot.engine.valuation import LeagueShape
from puckpilot.web import wire
from puckpilot.web.access import Access
from puckpilot.web.relay import RelayState, build_id
from puckpilot.web.relay import serve as serve_relay
from puckpilot.web.server import LiveState
from tests import draftkit

FIXTURE = Path(__file__).parent / "fixtures" / "yahoo_draft_ws.json"
YAHOO_POS = {"C": "C", "LW": "L", "RW": "R", "D": "D", "G": "G"}


@pytest.fixture
def relay():
    r = e2e.start_local_relay()
    yield r
    r.close()


# ---- a real room, every hop -------------------------------------------------------


def _room_board():
    """A universe made of the players the 2026-09-08 room actually drafted.

    Ids are the Yahoo ids and positions come off the frames, so the stream maps
    one-to-one without the private database. Value falls with draft order plus
    a deep bench of undrafted players, so the engine has a real board to rank.
    """
    frames = json.loads(FIXTURE.read_text(encoding="utf-8"))["frames"]
    picks = [f for f in map(parse_frame, frames) if isinstance(f, PickFrame)]
    rows = {}
    for f in picks:
        pos = YAHOO_POS[f.position.split(",")[0]]
        rows[int(f.yahoo_id)] = {
            "name": f"Drafted {f.yahoo_id}",
            "position": pos,
            "team": "AAA",
            "vorp": 30.0 - f.pick * 0.15,
            "z_total": 30.0 - f.pick * 0.15,
            "adp_rank": float(f.pick),
            "goals": 40.0,
        }
    for i, pos in enumerate("CLRDG" * 20):
        rows[900000 + i] = {
            "name": f"Undrafted {pos}{i}",
            "position": pos,
            "team": "BBB",
            "vorp": -2.0 - i * 0.05,
            "z_total": -2.0 - i * 0.05,
            "adp_rank": 200.0 + i,
            "goals": 5.0,
        }
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "player_id"
    shape = LeagueShape(
        n_teams=12,
        slots=(("C", 2), ("L", 2), ("R", 2), ("D", 4), ("G", 2)),
        util_slots=1,
        bench_slots=3,
    )
    board = DraftBoard(Universe(df.sort_values("vorp", ascending=False)), DraftRules(shape, 16), 3)
    idmap = {f.yahoo_id: int(f.yahoo_id) for f in picks}
    names = {int(f.yahoo_id): e2e.YahooRow(f"Drafted {f.yahoo_id}", frozenset()) for f in picks}
    return board, frames, idmap, names


def test_a_real_rooms_frame_stream_survives_every_hop(relay):
    board, frames, idmap, names = _room_board()
    feed = WebsocketFeed(e2e._NoContext(), idmap)
    harness = e2e.Harness(board, relay, seats=(3, 8), every=4, board_rows=60, names=names)
    result = harness.run(
        e2e.frame_polls(frames, feed),
        "fixture 2026-09-08",
        "capture",
        room_picks=192,
        unmapped=lambda: feed.state.unmapped,
    )
    assert result.passed, result.summary()
    assert result.picks == 192 and result.names_checked == 192
    assert result.drift == [] and result.unmapped == 0
    assert result.checks == 1 + 192 // 4
    assert feed.status()["gaps"] == []


def test_a_synthetic_draft_with_keepers_and_market_rows_passes(relay):
    import numpy as np

    board = draftkit.board()
    harness = e2e.Harness(board, relay, cats=draftkit.CATS, seats=(0, 3, 5), board_rows=40)
    polls = draftkit_polls(board, seed=2)
    result = harness.run(polls, "synthetic", "sim")
    assert result.passed, result.summary()
    assert board.complete and result.picks == len(board.slots)
    assert result.checks == len(board.slots) + 1
    assert np.isfinite(result.timings["push"]).all()


def draftkit_polls(board, seed):
    feed = draftkit.sim_feed(seed)
    for _ in range(len(board.slots) * 2):
        yield feed.poll


def test_unmapped_room_picks_are_drift_not_silence(relay):
    board, frames, idmap, names = _room_board()
    for yahoo_id in list(idmap)[40:43]:  # three prospects the map does not know
        del idmap[yahoo_id]
    feed = WebsocketFeed(e2e._NoContext(), idmap)
    harness = e2e.Harness(board, relay, seats=(3,), every=50, board_rows=30, names=names)
    result = harness.run(
        e2e.frame_polls(frames, feed),
        "fixture",
        "capture",
        room_picks=192,
        unmapped=lambda: feed.state.unmapped,
    )
    assert result.unmapped == 3
    assert result.drift and "3 pick(s) behind the room" in result.drift[0]


# ---- the harness can fail ---------------------------------------------------------


class _LyingState(RelayState):
    """Hands back a board one pick behind what it was given."""

    def get(self, seat):
        out = super().get(seat)
        if "made" in out and not out.get("waiting"):
            out["made"] -= 1
        return out


def _relay_with(state, owner="owner-k", guest="guest-k"):
    srv = serve_relay(state, Access(owner=owner, guest=guest), port=0, host="127.0.0.1")
    return e2e.Relay(f"http://127.0.0.1:{srv.server_address[1]}", owner, guest, local=srv)


def _short_run(relay, **kw):
    board = draftkit.board()
    harness = e2e.Harness(board, relay, cats=draftkit.CATS, seats=(0,), board_rows=20, **kw)

    def polls():
        feed = draftkit.sim_feed(4)
        for _ in range(6):
            yield feed.poll

    return harness.run(polls(), "short", "sim", room_picks=6)


def test_a_relay_that_alters_the_board_fails_the_run():
    relay = _relay_with(_LyingState())
    try:
        result = _short_run(relay)
    finally:
        relay.close()
    assert not result.passed
    assert any("relay differs on 'made'" in v for v in result.violations)


def test_a_rejected_push_fails_the_run(relay):
    wrong = e2e.Relay(relay.url, "not-the-owner-key", relay.guest)
    result = _short_run(wrong)
    assert any("push at pick" in v and "403" in v for v in result.violations)


def test_a_relay_that_lets_a_guest_write_fails_the_run():
    relay = _relay_with(RelayState(), owner="same-key", guest="same-key")
    try:
        result = _short_run(relay)
    finally:
        relay.close()
    assert any("let a guest push" in v for v in result.violations)


def test_a_relay_running_other_code_fails_when_the_build_is_expected(relay):
    ok = _short_run(relay, expect_build=build_id())
    assert ok.passed, ok.summary()
    drifted = _short_run(relay, expect_build="000000000000")
    assert any("the deployment is not this code" in v for v in drifted.violations)


def test_a_mismapped_player_is_caught_by_name():
    import numpy as np

    board = draftkit.board()
    relay = e2e.start_local_relay()
    victim = int(board.u.ids[int(np.flatnonzero(board.avail & (board.u.source != "market"))[5])])
    names = {victim: e2e.YahooRow("Somebody Else Entirely", frozenset())}
    try:
        harness = e2e.Harness(board, relay, cats=draftkit.CATS, seats=(0,), every=99, names=names)

        def polls():
            from puckpilot.draft.feed import PickEvent

            yield lambda b: [PickEvent(victim, None, "websocket")]

        result = harness.run(polls(), "mismap", "capture", room_picks=1)
    finally:
        relay.close()
    assert any("the room drafted 'Somebody Else Entirely'" in v for v in result.violations)


@pytest.mark.parametrize(
    "a, b, same",
    [
        ("Tim Stützle", "Tim Sttzle", True),  # MoneyPuck's accent loss
        ("Mitch Marner", "Mitchell Marner", True),
        ("Egor Chinakhov", "Yegor Chinakhov", True),
        ("Nick Paul", "Nicholas Paul", True),
        ("J.T. Miller", "JT Miller", True),
        ("Brady Tkachuk", "Matthew Tkachuk", False),  # brothers
        ("Quinn Hughes", "Jack Hughes", False),
        ("Elias Pettersson", "Elias Lindholm", False),
        ("Sebastian Aho", "Sebastian Cossa", False),
    ],
)
def test_same_player_tolerates_spelling_not_substitution(a, b, same):
    assert e2e.same_player(a, b) is same


# ---- the console's push, directly -------------------------------------------------


def _state():
    return LiveState(board=draftkit.board(), feed=None, top=3, board_rows=20, cats=draftkit.CATS)


def test_push_round_trips_exactly_what_the_console_shows(relay):
    state = _state()
    assert _push_snapshots(relay.url, relay.owner, state, [0, 3]) == ""
    from tests.test_relay import _req  # noqa: F401  (shared helper style)

    for seat in (0, 3):
        text = e2e._http(f"{relay.url}/state?seat={seat}&k={relay.guest}")[1]
        assert e2e.integrity.relay_violations(state.snapshot(seat), wire.loads(text)) == []


def test_push_reports_rather_than_raises_when_the_snapshot_fails(relay):
    class Broken(LiveState):
        def snapshot(self, seat=None):
            raise KeyError("boom")

    msg = _push_snapshots(relay.url, relay.owner, Broken(board=draftkit.board()), [0])
    assert "snapshot for push failed" in msg and "boom" in msg


def test_push_reports_a_dead_relay_a_bad_key_and_a_bad_url():
    state = _state()
    assert _push_snapshots("http://127.0.0.1:9", "k", state, [0])  # nothing listens on :9
    assert _push_snapshots("not a url", "k", state, [0])
    r = e2e.start_local_relay()
    try:
        assert "403" in _push_snapshots(r.url, "wrong", state, [0])
    finally:
        r.close()


def test_push_survives_numpy_and_nan_in_the_snapshot(relay):
    import numpy as np

    class Odd(LiveState):
        def snapshot(self, seat=None):
            snap = super().snapshot(seat)
            snap["made"] = np.int64(snap["made"])
            snap["board"][0]["vorp"] = float("nan")
            return snap

    assert _push_snapshots(relay.url, relay.owner, Odd(board=draftkit.board()), [0]) == ""
    text = e2e._http(f"{relay.url}/state?seat=0&k={relay.guest}")[1]
    assert wire.loads(text)["board"][0]["vorp"] is None
