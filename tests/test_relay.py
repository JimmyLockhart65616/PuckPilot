"""The relay: the draft view, served from somewhere that is not the drafter's PC.

It holds no board and ranks nothing - it hands back whatever the console last
pushed. So what is worth testing is the boundary: that a stranger with the URL
gets nothing, that only the owner can write, that a seat reaches the right
snapshot, and that a cold URL says "waiting" rather than failing in a way that
looks like the draft is over.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

import pytest

from puckpilot.web.access import Access
from puckpilot.web.relay import RelayState, serve

OWNER, GUEST = "owner-key", "guest-key"


@pytest.fixture
def relay():
    state = RelayState()
    srv = serve(state, Access(owner=OWNER, guest=GUEST), port=0, host="127.0.0.1")
    srv.state = state
    yield srv
    srv.shutdown()
    srv.server_close()


def _req(srv, path, method="GET", body=None, headers=None):
    url = f"http://127.0.0.1:{srv.server_address[1]}{path}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def _push(srv, seats, key=OWNER):
    return _req(srv, "/push", "POST", {"seats": seats}, {"X-PuckPilot-Key": key})


# ---- the boundary ----------------------------------------------------------


def test_no_key_reads_nothing(relay):
    for path in ("/", "/state"):
        assert _req(relay, path)[0] == 403, path


def test_a_guest_cannot_push(relay):
    """The relay is a mirror of one machine's board. A guest who could write to
    it could show the other manager a draft that never happened."""
    status, body = _push(relay, {"0": {"made": 99}}, key=GUEST)
    assert status == 403
    assert "owner-only" in body
    assert relay.state.seats == {}


def test_the_owner_pushes_and_the_guest_reads(relay):
    assert _push(relay, {"0": {"made": 3, "who": "me"}, "2": {"made": 3, "who": "him"}})[0] == 200
    status, body = _req(relay, f"/state?seat=2&k={GUEST}")
    assert status == 200
    payload = json.loads(body)
    assert payload["who"] == "him"
    assert payload["made"] == 3


def test_a_cold_url_says_waiting_rather_than_failing(relay):
    """Opened before the console connects, the page must not look like a draft
    that has ended - empty board, zero picks, no explanation."""
    payload = json.loads(_req(relay, f"/state?k={GUEST}")[1])
    assert payload["waiting"] is True
    assert payload["shortlist"] == [] and payload["board"] == []
    assert "Waiting" in payload["note"]


def test_undo_is_not_offered_on_the_relay(relay):
    """The board lives on the console. A relay that accepted undo would silently
    diverge from it."""
    _push(relay, {"0": {"made": 1}})
    assert json.loads(_req(relay, f"/state?k={GUEST}")[1])["can_undo"] is False
    assert _req(relay, f"/undo?k={OWNER}", "POST")[0] == 404
    assert _req(relay, f"/undo?k={OWNER}")[0] == 405


def test_an_unknown_seat_is_named_not_guessed(relay):
    _push(relay, {"0": {"made": 1}})
    payload = json.loads(_req(relay, f"/state?seat=9&k={GUEST}")[1])
    assert "no snapshot for seat 9" in payload["error"]
    assert payload["seats"] == ["0"]


def test_a_later_push_replaces_the_earlier_one(relay):
    _push(relay, {"0": {"made": 1}})
    _push(relay, {"0": {"made": 2}})
    assert json.loads(_req(relay, f"/state?seat=0&k={GUEST}")[1])["made"] == 2
    assert relay.state.pushes == 2


def test_health_needs_no_key(relay):
    """The platform's probe has no credentials, and must not be able to read
    the board either."""
    status, body = _req(relay, "/healthz")
    assert status == 200
    payload = json.loads(body)
    assert payload["ok"] is True
    assert "shortlist" not in payload


def test_a_malformed_push_is_refused(relay):
    for body in ({"nope": 1}, {"seats": "not-an-object"}, {"seats": None}):
        status, _ = _req(relay, "/push", "POST", body, {"X-PuckPilot-Key": OWNER})
        assert status == 400, body
    assert relay.state.seats == {}


PROBE = """
import sys
BANNED = {'numpy', 'pandas', 'scipy'}


class Block:
    def find_module(self, name, path=None):
        return self if name.split('.')[0] in BANNED else None

    def load_module(self, name):
        raise ImportError('blocked: ' + name)


sys.meta_path.insert(0, Block())
import puckpilot.web.relay  # noqa: E402,F401
print('ok')
"""


def test_the_relay_starts_without_the_engine_installed():
    """The whole reason the page and the access rules live in their own modules.

    Checked by actually importing the relay with the engine's dependencies
    blocked, rather than by grepping its import lines - a transitive import
    through some future helper would slip straight past a grep, and would
    surface only as a container that will not start.
    """
    import subprocess
    import sys

    r = subprocess.run([sys.executable, "-c", PROBE], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "ok" in r.stdout
