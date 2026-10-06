"""The in-season relay: who may see what, and how decisions get home."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from puckpilot.web.season_relay import (
    MAX_PENDING_DECISIONS,
    SeasonState,
    build_id,
    parse_keys,
    serve,
)

KEYS = {"KEYJ": "jimmy", "KEYD": "sam"}


@pytest.fixture
def relay():
    state = SeasonState()
    # port 0 so the suite can never collide with a relay the user is running
    server = serve(state, KEYS, 0, host="127.0.0.1")
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        yield state, base
    finally:
        server.shutdown()


def call(base, path, key=None, body=None):
    url = base + path + (f"?k={key}" if key else "")
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"} if data else {}
    )
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


# -- keys and isolation -----------------------------------------------------


def test_keys_parse_from_one_environment_string():
    assert parse_keys("jimmy:abc,sam:def") == {"abc": "jimmy", "def": "sam"}
    assert parse_keys("") == {}
    assert parse_keys("garbage,also:") == {}


def test_no_key_and_wrong_key_are_both_refused(relay):
    _, base = relay
    assert call(base, "/state")[0] == 403
    assert call(base, "/state", "nope")[0] == 403


def test_a_non_ascii_key_is_a_refusal_not_a_crash(relay):
    """compare_digest raises on non-ASCII. A crash is a 500 with a traceback in
    the log; a wrong key has to be a 403."""
    _, base = relay
    req = urllib.request.Request(base + "/state", headers={"X-PuckPilot-Key": "ééé"})
    try:
        urllib.request.urlopen(req)
        raise AssertionError("expected a refusal")
    except urllib.error.HTTPError as e:
        assert e.code == 403


def test_one_manager_cannot_see_another(relay):
    """They are rivals in the same league."""
    _, base = relay
    call(base, "/push", "KEYJ", {"snapshot": {"team": "Home Team"}})
    assert json.loads(call(base, "/state", "KEYJ")[1])["team"] == "Home Team"
    assert json.loads(call(base, "/state", "KEYD")[1])["empty"] is True


def test_health_needs_no_key_and_names_the_build(relay):
    _, base = relay
    code, body = call(base, "/healthz")
    assert code == 200
    assert json.loads(body)["build"] == build_id()


# -- the view ---------------------------------------------------------------


def test_a_cold_relay_returns_a_drawable_payload(relay):
    """A fresh URL must render a page that says so, not one missing half its
    keys - the draft relay drew "seat undefined" that way."""
    _, base = relay
    s = json.loads(call(base, "/state", "KEYJ")[1])
    assert s["empty"] is True and s["stale"] is True
    for key in ("moves", "proposals", "roster", "week", "protocol", "age_seconds"):
        assert key in s


def test_a_fresh_push_is_not_stale_and_an_old_one_is():
    state = SeasonState(stale_after=0.0)
    state.push("jimmy", {"team": "X"})
    assert state.get("jimmy")["stale"] is True
    fresh = SeasonState(stale_after=3600.0)
    fresh.push("jimmy", {"team": "X"})
    assert fresh.get("jimmy")["stale"] is False


def test_nan_never_reaches_the_page(relay):
    """JSON.parse rejects bare NaN and the page freezes on it."""
    from puckpilot.web import wire

    state, base = relay
    state.push("jimmy", wire.clean({"x": float("nan")}))
    body = call(base, "/state", "KEYJ")[1]
    assert "NaN" not in body
    json.loads(body)


# -- decisions --------------------------------------------------------------


def test_a_decision_is_handed_over_exactly_once(relay):
    _, base = relay
    call(base, "/decide", "KEYJ", {"kind": "proposal", "id": 7, "approve": True})
    got = json.loads(call(base, "/decisions", "KEYJ")[1])["decisions"]
    assert [(d["kind"], d["id"], d["approve"]) for d in got] == [("proposal", 7, True)]
    assert json.loads(call(base, "/decisions", "KEYJ")[1])["decisions"] == []


def test_a_decision_shows_immediately_rather_than_after_the_next_push(relay):
    _, base = relay
    call(base, "/push", "KEYJ", {"snapshot": {"proposals": [{"id": 7, "add": "X"}]}})
    call(base, "/decide", "KEYJ", {"kind": "proposal", "id": 7, "approve": True})
    assert json.loads(call(base, "/state", "KEYJ")[1])["proposals"] == []


def test_deciding_a_protocol_marks_it_rather_than_removing_it(relay):
    _, base = relay
    call(base, "/push", "KEYJ", {"snapshot": {"protocol": {"id": 3, "status": "proposed"}}})
    call(base, "/decide", "KEYJ", {"kind": "protocol", "id": 3, "approve": True})
    assert json.loads(call(base, "/state", "KEYJ")[1])["protocol"]["status"] == "approved"


def test_decisions_do_not_cross_managers(relay):
    state, base = relay
    call(base, "/decide", "KEYJ", {"kind": "proposal", "id": 1, "approve": True})
    assert json.loads(call(base, "/decisions", "KEYD")[1])["decisions"] == []
    assert len(state.drain("jimmy")) == 1


def test_an_unknown_decision_kind_is_refused(relay):
    _, base = relay
    assert call(base, "/decide", "KEYJ", {"kind": "trade", "id": 1, "approve": True})[0] == 400


def test_undelivered_decisions_do_not_grow_without_bound():
    state = SeasonState()
    for i in range(MAX_PENDING_DECISIONS):
        state.decide("jimmy", "proposal", i, True)
    with pytest.raises(RuntimeError, match="local job running"):
        state.decide("jimmy", "proposal", 999, True)


# -- the page ---------------------------------------------------------------


def test_the_page_is_served_even_without_a_key(relay):
    """It renders its own "wrong key" message, which beats a bare 403 on a
    phone."""
    _, base = relay
    code, body = call(base, "/")
    assert code == 200 and "<!doctype" in body


def test_an_unknown_post_route_is_a_404(relay):
    _, base = relay
    assert call(base, "/nonsense", "KEYJ", {"x": 1})[0] == 404


# -- the image --------------------------------------------------------------

IMPORTS_PROBE = """
import sys
sys.path.insert(0, 'src')
import puckpilot.web.season_relay  # noqa
for m in list(sys.modules.values()):
    f = getattr(m, '__file__', None)
    if f and 'puckpilot' in f:
        print(f)
"""


def test_the_image_copies_every_module_the_relay_imports():
    """The Dockerfile copies files one by one. A missed import builds fine and
    then dies on start, which is a blank page on somebody's phone."""
    root = Path(__file__).resolve().parents[1]
    r = subprocess.run(
        [sys.executable, "-c", IMPORTS_PROBE], capture_output=True, text=True, cwd=root
    )
    assert r.returncode == 0, r.stderr
    src = (root / "src").resolve()
    imported = {
        p.relative_to(src).as_posix()
        for line in r.stdout.splitlines()
        if line.strip()
        for p in [Path(line.strip()).resolve()]
        if p.is_relative_to(src)
    }
    dockerfile = (root / "deploy/season/Dockerfile").read_text(encoding="utf-8")
    copied = {
        m.group(1).removeprefix("src/") for m in re.finditer(r"^COPY\s+(\S+)", dockerfile, re.M)
    }
    assert imported, "the probe found no puckpilot modules"
    assert imported <= copied, f"imported but not copied: {sorted(imported - copied)}"


def test_the_relay_never_drags_the_engine_into_its_image():
    """numpy, pandas and scipy would be ~400 MB for a process that ranks
    nothing, and a slow cold start on a phone."""
    probe = """
import sys
BANNED = {'numpy', 'pandas', 'scipy'}

class Block:
    def find_module(self, name, path=None):
        return self if name.split('.')[0] in BANNED else None
    def load_module(self, name):
        raise ImportError('blocked: ' + name)

sys.meta_path.insert(0, Block())
sys.path.insert(0, 'src')
import puckpilot.web.season_relay  # noqa
print('ok')
"""
    root = Path(__file__).resolve().parents[1]
    r = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, cwd=root)
    assert r.returncode == 0, r.stdout + r.stderr


def test_the_deploy_script_refuses_a_permissive_dockerignore():
    """`az acr build` uploads the repo root. Without deny-by-default that is a
    logged-in Yahoo session and a gigabyte of captures, sent to a registry."""
    root = Path(__file__).resolve().parents[1]
    script = (root / "deploy/season/deploy.sh").read_text(encoding="utf-8")
    assert "refusing to build" in script
    head = (root / ".dockerignore").read_text(encoding="utf-8").splitlines()[:20]
    assert "*" in [ln.strip() for ln in head]


# -- stale means overdue, not merely old -------------------------------------


def test_an_old_push_is_not_stale_while_its_next_run_is_still_ahead():
    """A quiet night: last pushed at 11:00, nothing to run until tomorrow."""
    from datetime import UTC, datetime, timedelta

    state = SeasonState(stale_after=0.0)  # the flat rule would call it stale
    later = (datetime.now(UTC) + timedelta(hours=10)).isoformat()
    state.push("jimmy", {"team": "X", "next_run_utc": later})
    assert state.get("jimmy")["stale"] is False


def test_a_run_that_was_due_and_did_not_come_makes_it_stale():
    from datetime import UTC, datetime, timedelta

    state = SeasonState()
    missed = (datetime.now(UTC) - timedelta(hours=2)).isoformat()
    state.push("jimmy", {"team": "X", "next_run_utc": missed})
    assert state.get("jimmy")["stale"] is True


def test_within_the_grace_a_late_run_is_not_yet_stale():
    from datetime import UTC, datetime, timedelta

    state = SeasonState()
    just = (datetime.now(UTC) - timedelta(minutes=10)).isoformat()
    state.push("jimmy", {"team": "X", "next_run_utc": just})
    assert state.get("jimmy")["stale"] is False


def test_every_state_carries_the_build_the_page_checks_against():
    """Cold or warm, so a page opened before the first push still notices a
    redeploy."""
    from puckpilot.web.season_relay import SeasonState, build_id

    state = SeasonState()
    assert state.get("jimmy")["build"] == build_id()
    state.push("jimmy", {"team": "Home Team", "moves": []})
    assert state.get("jimmy")["build"] == build_id()
