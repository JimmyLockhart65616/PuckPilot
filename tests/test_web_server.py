"""The draft-night web view.

It runs on a fixed, guessable loopback port while a draft is live, and it has no
authentication - so the thing worth testing is that a page the user happens to
have open in another tab cannot reach in and change the board.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

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
