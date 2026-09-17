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


# ---- freshness and drift -------------------------------------------------------


def test_health_names_the_build_so_drift_is_visible_without_a_key(relay):
    """The deployed relay ran a page two revisions behind the checkout and nothing
    could tell. A keyless health check that names the build closes that."""
    from puckpilot.web.relay import build_id

    before = json.loads(_req(relay, "/healthz")[1])
    assert before["build"] == build_id()
    assert len(before["build"]) == 12
    assert before["last_push_age"] is None

    _push(relay, {"0": {"made": 1}})
    after = json.loads(_req(relay, "/healthz")[1])
    assert 0 <= after["last_push_age"] < 5
    assert "shortlist" not in after and "made" not in after


def test_the_build_id_ignores_line_endings_but_not_content(tmp_path):
    from pathlib import Path

    from puckpilot.web import relay as relay_mod

    src = Path(relay_mod.__file__).resolve().parent
    lf, crlf, edited = tmp_path / "lf", tmp_path / "crlf", tmp_path / "edited"
    for d in (lf, crlf, edited):
        d.mkdir()
        for name in relay_mod.BUILD_FILES:
            text = (src / name).read_bytes().replace(b"\r\n", b"\n")
            (d / name).write_bytes(text.replace(b"\n", b"\r\n") if d is crlf else text)
    (edited / "page.py").write_bytes((edited / "page.py").read_bytes() + b"\n# changed\n")

    assert relay_mod.build_id(lf) == relay_mod.build_id(crlf)
    assert relay_mod.build_id(lf) != relay_mod.build_id(edited)


def test_a_relay_that_stops_hearing_from_the_console_says_so(relay):
    """A frozen board looks exactly like a slow room. After three missed
    heartbeats the guest's view must say it is not live."""
    import time

    from puckpilot.web.relay import STALE_AFTER_S

    _push(relay, {"0": {"made": 5, "seconds_since_pick": 4.0}})
    fresh = json.loads(_req(relay, f"/state?k={GUEST}")[1])
    assert fresh["stale"] is False

    relay.state.pushed_at = time.time() - (STALE_AFTER_S + 10)
    old = json.loads(_req(relay, f"/state?k={GUEST}")[1])
    assert old["stale"] is True
    assert old["relay_age"] > STALE_AFTER_S


def test_the_pick_clock_keeps_running_between_pushes(relay):
    import time

    _push(relay, {"0": {"made": 5, "seconds_since_pick": 4.0}})
    relay.state.pushed_at = time.time() - 20
    payload = json.loads(_req(relay, f"/state?k={GUEST}")[1])
    assert 23.5 < payload["seconds_since_pick"] < 30


def test_a_nan_pushed_by_an_older_console_is_served_as_null(relay):
    """Python's json reads NaN; the guest's browser does not. The relay must not
    store it and hand it on."""
    from puckpilot.web import wire

    url = f"http://127.0.0.1:{relay.server_address[1]}/push"
    body = b'{"seats": {"0": {"made": 2, "board": [{"vorp": NaN}]}}}'
    req = urllib.request.Request(url, data=body, method="POST", headers={"X-PuckPilot-Key": OWNER})
    with urllib.request.urlopen(req, timeout=5) as r:
        assert r.status == 200
    text = _req(relay, f"/state?k={GUEST}")[1]
    assert "NaN" not in text
    assert wire.loads(text)["board"] == [{"vorp": None}]


def test_the_waiting_view_carries_every_key_a_real_snapshot_has(relay):
    """Otherwise the page renders a cold URL as "seat undefined"."""
    from puckpilot.web.server import LiveState
    from tests.test_board import _board

    real = LiveState(board=_board(), feed=None, top=3).snapshot(0)
    waiting = json.loads(_req(relay, f"/state?k={GUEST}")[1])
    assert set(real) - set(waiting) == set()


IMPORTS_PROBE = """
import sys
import puckpilot.web.relay  # noqa: F401
for name, mod in sorted(sys.modules.items()):
    f = getattr(mod, '__file__', None)
    if name.startswith('puckpilot') and f:
        print(f)
"""


def test_the_image_copies_every_module_the_relay_imports():
    """The Dockerfile copies files one by one. A new import the COPY list misses
    builds fine and then dies on start, which on draft night is a blank page."""
    import re
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    r = subprocess.run(
        [sys.executable, "-c", IMPORTS_PROBE], capture_output=True, text=True, cwd=root
    )
    assert r.returncode == 0, r.stderr
    imported = {
        Path(line.strip()).resolve().relative_to(root / "src").as_posix()
        for line in r.stdout.splitlines()
        if line.strip()
    }
    dockerfile = (root / "deploy/relay/Dockerfile").read_text(encoding="utf-8")
    copied = {
        m.group(1).removeprefix("src/") for m in re.finditer(r"^COPY\s+(\S+)", dockerfile, re.M)
    }
    assert imported, "the probe found no puckpilot modules"
    assert imported <= copied, f"imported but not copied: {sorted(imported - copied)}"


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


def test_the_build_context_denies_by_default():
    """`.dockerignore` is a security control here, not tidiness.

    `az acr build` uploads the build context to a cloud registry, and this repo
    root holds `secrets/chrome-profile/` - a logged-in Yahoo session, hundreds of
    megabytes of cookies and tokens - alongside `data/captures/`, over a gigabyte
    of real draft-room recordings. Without an allow-list all of it ships to ACR.

    Pinned as deny-by-default rather than as a list of excluded paths: a new
    sensitive directory then needs no change here to stay out.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    ignore = root / ".dockerignore"
    assert ignore.is_file(), ".dockerignore is missing; the build context would carry secrets"

    lines = [
        line.strip()
        for line in ignore.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert lines and lines[0] == "*", "the first rule must exclude everything"

    allowed = {line[1:].rstrip("/") for line in lines if line.startswith("!")}
    assert allowed == {"src"}, f"only src/ may be re-included, got {sorted(allowed)}"
