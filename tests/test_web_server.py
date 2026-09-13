"""The draft-night web view.

It runs on a fixed, guessable loopback port while a draft is live, and it has no
authentication - so the thing worth testing is that a page the user happens to
have open in another tab cannot reach in and change the board.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from urllib.parse import quote

import pytest

from puckpilot.web.server import LiveState, serve


class _Pick:
    name = "Connor McDavid"


class _Board:
    """Just enough board for the routes under test."""

    made = 3
    total = 84
    complete = False

    def __init__(self):
        self.undone = 0

    def undo(self):
        self.undone += 1
        return _Pick()


@pytest.fixture
def server():
    state = LiveState(board=_Board(), feed=None, top=3)
    # port 0 lets the OS pick a free one, so the suite cannot collide with a
    # real draft console the user has open
    srv = serve(state, port=0)
    srv.state = state
    yield srv
    srv.shutdown()
    srv.server_close()


def _url(srv, path):
    return f"http://127.0.0.1:{srv.server_address[1]}{path}"


def _request(srv, path, method="GET", headers=None):
    req = urllib.request.Request(_url(srv, path), method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


# ---- the board must not be mutable by a cross-site request ----------------


def test_undo_is_not_reachable_by_a_bare_get(server):
    """A GET is what an <img> tag or a typed URL produces. If GET mutated the
    board, any page open during the draft could rewind it."""
    status, _ = _request(server, "/undo")
    assert status == 405
    assert server.state.board.undone == 0


def test_undo_is_not_reachable_by_a_prefix_dodge(server):
    """`startswith('/undo')` also matches '/undo.png', which is exactly the
    shape that fits in an <img src>. Routes are matched exactly."""
    for path in ("/undo.png", "/undoX", "/undo/../undo"):
        status, _ = _request(server, path)
        assert status != 200 or server.state.board.undone == 0
    assert server.state.board.undone == 0


def test_a_cross_site_post_is_refused(server):
    status, body = _request(
        server,
        "/undo",
        method="POST",
        headers={"Sec-Fetch-Site": "cross-site", "Origin": "https://evil.example"},
    )
    assert status == 403
    assert "cross-origin" in body
    assert server.state.board.undone == 0


def test_a_same_origin_post_still_works(server):
    """The recovery hatch has to keep working - the feed is the only pick
    source, so a bad frame needs a way back mid-draft."""
    status, body = _request(
        server, "/undo", method="POST", headers={"Sec-Fetch-Site": "same-origin"}
    )
    assert status == 200
    assert "Connor McDavid" in json.loads(body)["result"]
    assert server.state.board.undone == 1


# ---- read routes -----------------------------------------------------------


def test_the_page_is_html(server):
    status, body = _request(server, "/")
    assert status == 200 and body.lstrip().startswith("<!doctype html")


def test_state_serves_a_real_board():
    """`/state` is what the page polls once a second, so it has to render a
    genuine board rather than merely return 200."""
    from tests.test_board import _board

    state = LiveState(board=_board(), feed=None, top=3)
    srv = serve(state, port=0)
    try:
        status, body = _request(srv, "/state")
        payload = json.loads(body)
        assert status == 200
        assert payload["shortlist"] and payload["board"]
        assert "needs" in payload and "roster" in payload
    finally:
        srv.shutdown()
        srv.server_close()


def test_it_binds_loopback_only(server):
    """The board and roster are private and there is no auth, so this must not
    be reachable from the network."""
    assert server.server_address[0] == "127.0.0.1"


def test_a_rebound_host_header_is_refused(server):
    """Sec-Fetch-Site alone does not survive DNS rebinding: the attacker's own
    name resolves to 127.0.0.1, so the browser truthfully says same-origin while
    the page is theirs. Host is the header that still distinguishes them."""
    for route, method in (("/undo", "POST"), ("/state", "GET")):
        status, _ = _request(
            server,
            route,
            method=method,
            headers={"Host": "evil.example.com", "Sec-Fetch-Site": "same-origin"},
        )
        assert status == 403, f"{method} {route} accepted a rebound Host"
    assert server.state.board.undone == 0


def test_the_page_can_still_undo(server):
    """A regression guard: making /undo POST-only left `draft live --web` with
    no reachable undo at all until the page got a button."""
    from puckpilot.web.server import PAGE

    assert 'id="undo"' in PAGE
    assert "q('/undo'), {method:'POST'}" in PAGE


# ---- one board, two seats --------------------------------------------------
#
# Two managers in the same room share a board: the picks are universal, but
# "what do I still need" is not. These pin that the seat actually reaches the
# advice, because a view that silently answered for the wrong roster would look
# entirely plausible on screen.


def _live_board():
    """A small real board, built here rather than imported.

    Deliberately self-contained: these tests are about the HTTP surface, and
    borrowing another module's fixture couples them to a refactor happening in
    the draft package.
    """
    import pandas as pd

    from puckpilot.draft.board import DraftBoard
    from puckpilot.draft.engine import DraftRules, Universe
    from puckpilot.engine.valuation import LeagueShape

    shape = LeagueShape(
        n_teams=4,
        slots=(("C", 1), ("L", 1), ("R", 1), ("D", 2), ("G", 1)),
        util_slots=1,
        bench_slots=1,
    )
    rules = DraftRules(
        shape=shape,
        rounds=7,
        caps={"C": 3, "L": 3, "R": 3, "D": 4, "G": 2},
        mins={"C": 1, "L": 1, "R": 1, "D": 2, "G": 1},
    )
    rows, pid = {}, 1
    for pos in ("C", "L", "R", "D", "G"):
        for i in range(10):
            rows[pid] = {
                "name": f"Player {pos}{chr(65 + i)}",
                "position": pos,
                "team": "AAA",
                "vorp": 20.0 - i,
                "z_total": 20.0 - i,
                "adp_rank": float(pid),
                "goals": 30.0 - i,
            }
            pid += 1
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "player_id"
    return DraftBoard(
        Universe(df.sort_values("vorp", ascending=False)), rules, my_seat=0, roster_rounds=7
    )


def _real_server(access=None):
    state = LiveState(board=_live_board(), feed=None, top=3)
    srv = serve(state, port=0, access=access)
    srv.state = state
    return srv


def _seated_server():
    """A board with picks on it, so the two seats have different rosters."""
    srv = _real_server()
    board = srv.state.board
    for _ in range(6):  # seats 0,1,2,3,3,2 under a 4-team snake
        cand = int(board.u.ids[int(board.avail.argmax())])
        board.record(cand)
    return srv


def test_state_answers_for_the_requested_seat():
    srv = _seated_server()
    try:
        mine = json.loads(_request(srv, "/state?seat=0")[1])
        theirs = json.loads(_request(srv, "/state?seat=2")[1])
        assert mine["seat"] == 0 and theirs["seat"] == 2
        assert mine["roster"] != theirs["roster"]
        # The board is the same object; only the advisory half moves.
        assert mine["made"] == theirs["made"]
        assert mine["n_left"] == theirs["n_left"]
    finally:
        srv.shutdown()
        srv.server_close()


def test_state_without_a_seat_still_answers_for_the_board(server=None):
    srv = _seated_server()
    try:
        bare = json.loads(_request(srv, "/state")[1])
        assert bare["seat"] == srv.state.board.my_seat
    finally:
        srv.shutdown()
        srv.server_close()


def test_a_bad_seat_is_a_400_not_a_traceback():
    srv = _real_server()
    try:
        for path in ("/state?seat=99", "/state?seat=-1", "/state?seat=abc", "/state?seat="):
            status, body = _request(srv, path)
            assert status == 400, f"{path} -> {status}"
            assert "error" in json.loads(body)
    finally:
        srv.shutdown()
        srv.server_close()


# ---- sharing: the board leaves the machine, so tokens replace the bind -----


def _access():
    from puckpilot.web.access import Access

    return Access(owner="owner-key", guest="guest-key")


def test_sharing_off_needs_no_key():
    """The default path must be untouched: no key, still served."""
    srv = _real_server()
    try:
        assert _request(srv, "/state")[0] == 200
        assert _request(srv, "/")[0] == 200
    finally:
        srv.shutdown()
        srv.server_close()


def test_shared_refuses_every_route_without_a_key():
    srv = _real_server(access=_access())
    try:
        for path in ("/", "/state"):
            assert _request(srv, path)[0] == 403, path
        assert _request(srv, "/undo", method="POST")[0] == 403
        assert srv.state.board.made == 0
    finally:
        srv.shutdown()
        srv.server_close()


def test_a_guest_reads_any_seat_but_cannot_undo():
    """The one destructive control on the view. A guest mid-draft must not
    reach it, but must still be able to read the board."""
    srv = _seated_server()
    srv.shutdown()
    srv.server_close()
    srv = _real_server(access=_access())
    try:
        status, body = _request(srv, "/state?seat=2&k=guest-key")
        assert status == 200
        payload = json.loads(body)
        assert payload["seat"] == 2
        assert payload["can_undo"] is False

        status, body = _request(srv, "/undo?k=guest-key", method="POST")
        assert status == 403
        assert "owner-only" in body
    finally:
        srv.shutdown()
        srv.server_close()


def test_the_owner_key_still_undoes():
    srv = _real_server(access=_access())
    board = srv.state.board
    board.record(int(board.u.ids[int(board.avail.argmax())]))
    try:
        assert json.loads(_request(srv, "/state?k=owner-key")[1])["can_undo"] is True
        status, body = _request(srv, "/undo?k=owner-key", method="POST")
        assert status == 200
        assert "undid" in json.loads(body)["result"]
        assert board.made == 0
    finally:
        srv.shutdown()
        srv.server_close()


def test_the_key_may_ride_in_a_header():
    srv = _real_server(access=_access())
    try:
        status, _ = _request(srv, "/state", headers={"X-PuckPilot-Key": "guest-key"})
        assert status == 200
    finally:
        srv.shutdown()
        srv.server_close()


def test_a_near_miss_key_is_refused():
    srv = _real_server(access=_access())
    try:
        for key in ("owner-ke", "owner-keyx", "OWNER-KEY", "guest-ke", ""):
            assert _request(srv, f"/state?k={key}")[0] == 403, key
    finally:
        srv.shutdown()
        srv.server_close()


# ---- the board panel shows the whole board, including what we cannot take ----


def _ids_at(board, pos, k):
    return [int(pid) for pid, p in zip(board.u.ids, board.u.pos, strict=True) if p == pos][:k]


def test_the_board_panel_is_not_masked_by_our_own_roster_rules():
    """The one panel whose job is 'what is left' must not answer 'what may I
    take'. Capped and non-needed positions vanished from it entirely, which
    hides the best remaining player at a position regardless of his value."""
    srv = _real_server()
    board = srv.state.board
    for pid in _ids_at(board, "C", 3):  # caps={"C": 3}
        board.record(pid, seat=0)

    snap = srv.state.snapshot(0)
    assert board.blocked(0).get("C") == "cap"
    on_board = [r for r in snap["board"] if r["position"] == "C"]
    assert on_board, "centres are still on the board even though we cannot take one"
    assert all(r["blocked"] == "cap" for r in on_board)
    # ...and the shortlist, which answers the other question, still obeys them.
    assert all(r["position"] != "C" for r in snap["shortlist"])


def test_a_position_closed_by_roster_minimums_says_so():
    """Not at the cap - just out of picks to spare. Different fact, different
    tag, and the difference decides whether waiting can ever reopen it."""
    srv = _real_server()
    board = srv.state.board
    # Seat 0 picks at 0, 7, 8, 15 under a 4-team snake; run the draft to 16 so
    # he has three left and four minimums still unmet.
    wanted = {0: "C", 7: "C", 8: "L", 15: "D"}
    while board.made < 16:
        pos = wanted.get(board.made)
        avail = [
            int(pid)
            for pid, p_, ok in zip(board.u.ids, board.u.pos, board.avail, strict=True)
            if ok and (pos is None or p_ == pos)
        ]
        board.record(avail[0])
    assert board.picks_left(0) == 3

    blocked = board.blocked(0)
    assert blocked.get("C") == "min", blocked
    snap = srv.state.snapshot(0)
    assert any(r["position"] == "C" and r["blocked"] == "min" for r in snap["board"])


def test_an_open_board_tags_nobody():
    srv = _real_server()
    snap = srv.state.snapshot(0)
    assert snap["board"] and all(r["blocked"] == "" for r in snap["board"])


def test_a_non_ascii_key_is_a_refusal_not_a_traceback():
    """`secrets.compare_digest` raises on non-ASCII str, and the key arrives in
    a query string anyone can type."""
    access = _access()
    srv = _real_server(access)
    code, _body = _request(srv, "/state?k=" + quote("ké" * 8))
    assert code == 403
