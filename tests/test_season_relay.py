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
    BlobStore,
    SeasonState,
    StoreConflict,
    StoreUnavailable,
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
    for key in ("moves", "proposals", "roster", "week", "protocol", "age_seconds", "blockers"):
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


ROSTER = {
    "roster": [
        {"player_key": "p.1", "name": "Tyler Kleven", "tier": "upgrade",
         "tier_label": "Upgrade only", "tier_source": "auto",
         "tier_why": "goes only for a player worth more"},
        {"player_key": "p.2", "name": "Locked", "tier": "never", "tier_label": "Never",
         "tier_source": "yahoo", "tier_why": "Yahoo does not let him be dropped"},
    ]
}  # fmt: skip


def test_a_tier_set_on_the_roster_goes_home_once_and_shows_at_once(relay):
    _, base = relay
    call(base, "/push", "KEYJ", {"snapshot": ROSTER})
    code, _ = call(base, "/tier", "KEYJ", {"player": "p.1", "tier": "rotation"})
    assert code == 200
    row = json.loads(call(base, "/state", "KEYJ")[1])["roster"][0]
    assert (row["tier"], row["tier_label"], row["tier_source"]) == ("rotation", "Rotation", "you")
    got = json.loads(call(base, "/decisions", "KEYJ")[1])["decisions"]
    assert [(d["kind"], d["player"], d["tier"]) for d in got] == [("tier", "p.1", "rotation")]
    assert json.loads(call(base, "/decisions", "KEYJ")[1])["decisions"] == []


def test_a_tier_is_refused_for_a_locked_or_unknown_player_or_an_unknown_tier(relay):
    """Yahoo's lock beats any pin; a player not shown, or a tier nothing knows,
    is a 400 rather than a decision the run would have to throw away."""
    state, base = relay
    call(base, "/push", "KEYJ", {"snapshot": ROSTER})
    for body in (
        {"player": "p.2", "tier": "rotation"},
        {"player": "p.9", "tier": "rotation"},
        {"player": "p.1", "tier": "sometimes"},
    ):
        assert call(base, "/tier", "KEYJ", body)[0] == 400, body
    assert call(base, "/tier", "KEYD", {"player": "p.1", "tier": "never"})[0] == 400  # not his
    assert state.drain("jimmy") == []
    assert call(base, "/tier", "KEYJ", {"player": "p.1", "tier": "auto"})[0] == 200


def test_the_relay_tiers_are_the_engine_tiers():
    """The relay imports no engine, so it carries its own copy of the names."""
    from puckpilot.season import tiers
    from puckpilot.web.season_relay import AUTO_TIER, TIER_LABELS

    assert {t.value: t.label for t in tiers.Tier} == TIER_LABELS
    assert AUTO_TIER == tiers.AUTO


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


def test_the_image_build_refuses_a_permissive_dockerignore():
    """The build context is the repo root. Without deny-by-default, a logged-in
    Yahoo session and a gigabyte of captures are one careless COPY away from a
    public image."""
    root = Path(__file__).resolve().parents[1]
    workflow = (root / ".github/workflows/publish-season-image.yml").read_text(encoding="utf-8")
    assert "refusing to build" in workflow
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


# -- kept in a blob, so it survives scaling to zero ----------------------------


class MemoryStore:
    """BlobStore's contract without the network: writes conditional on an ETag."""

    def __init__(self):
        self.data: dict | None = None
        self.etag = 0
        self.fail_next_save = False
        self.before_next_save = None

    def load(self):
        if self.data is None:
            return {}, None
        return json.loads(json.dumps(self.data)), str(self.etag)

    def save(self, data, etag):
        hook, self.before_next_save = self.before_next_save, None
        if hook:
            hook()
        if self.fail_next_save:
            self.fail_next_save = False
            raise StoreUnavailable("storage is down")
        if etag != (str(self.etag) if self.data is not None else None):
            raise StoreConflict
        self.data = json.loads(json.dumps(data))
        self.etag += 1
        return str(self.etag)


def test_a_restart_keeps_the_view_and_what_is_waiting():
    store = MemoryStore()
    before = SeasonState(store=store)
    before.push("jimmy", {"team": "X", "proposals": [{"id": 7}]})
    before.decide("jimmy", "proposal", 7, True)

    after = SeasonState(store=store)  # the next replica, after scaling to zero
    assert after.get("jimmy")["empty"] is False
    assert after.get("jimmy")["proposals"] == []  # still shown as decided
    assert [d["id"] for d in after.drain("jimmy")] == [7]


def test_two_processes_never_hand_over_the_same_decision():
    """A deploy briefly runs the old replica beside the new one."""
    store = MemoryStore()
    old, new = SeasonState(store=store), SeasonState(store=store)
    old.push("jimmy", {"team": "X", "proposals": [{"id": 7}]})
    old.decide("jimmy", "proposal", 7, True)
    assert [d["id"] for d in new.drain("jimmy")] == [7]
    assert old.drain("jimmy") == []
    assert new.drain("jimmy") == []


def test_a_drain_that_cannot_be_stored_hands_over_nothing():
    store = MemoryStore()
    state = SeasonState(store=store)
    state.push("jimmy", {"team": "X", "proposals": [{"id": 7}]})
    state.decide("jimmy", "proposal", 7, True)
    store.fail_next_save = True
    with pytest.raises(StoreUnavailable):
        state.drain("jimmy")
    assert [d["id"] for d in state.drain("jimmy")] == [7]  # still waiting, handed over once


def test_a_write_that_loses_a_race_starts_again_from_the_winner():
    store = MemoryStore()
    a, b = SeasonState(store=store), SeasonState(store=store)
    a.push("jimmy", {"team": "X"})
    # b writes between a's read and a's write
    store.before_next_save = lambda: b.push("sam", {"team": "Y"})
    a.decide("jimmy", "protocol", 1, True)

    view = SeasonState(store=store)
    assert view.get("sam")["team"] == "Y"
    assert [d["id"] for d in view.drain("jimmy")] == [1]


def test_a_storage_fault_is_a_503_the_local_job_retries():
    store = MemoryStore()
    server = serve(SeasonState(store=store), KEYS, 0, host="127.0.0.1")
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        store.fail_next_save = True
        assert call(base, "/push", "KEYJ", {"snapshot": {"team": "X"}})[0] == 503
        assert call(base, "/push", "KEYJ", {"snapshot": {"team": "X"}})[0] == 200
        assert json.loads(call(base, "/healthz")[1])["stored"] is True
    finally:
        server.shutdown()


def test_the_blob_store_only_writes_over_what_it_read():
    """BlobStore against a stand-in for Azure's conditional PUT."""
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    blob = {"body": None, "etag": 0}

    class FakeBlob(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _status(self, code):
            self.send_response(code)
            if blob["body"] is not None:
                self.send_header("ETag", f'"{blob["etag"]}"')
            self.send_header("Content-Length", "0" if code != 200 else str(len(blob["body"])))
            self.end_headers()

        def do_GET(self):
            if blob["body"] is None:
                self._status(404)
                return
            self._status(200)
            self.wfile.write(blob["body"])

        def do_PUT(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            current = f'"{blob["etag"]}"' if blob["body"] is not None else None
            if self.headers.get("If-None-Match") == "*" and current is not None:
                self._status(409)
            elif "If-Match" in self.headers and self.headers["If-Match"] != current:
                self._status(412)
            else:
                blob["body"], blob["etag"] = body, blob["etag"] + 1
                self._status(201)

    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeBlob)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        store = BlobStore(f"http://127.0.0.1:{server.server_address[1]}/season/state.json?sig=x")
        assert store.load() == ({}, None)
        first = store.save({"seq": 1}, None)
        with pytest.raises(StoreConflict):
            store.save({"seq": 2}, None)  # somebody created it already
        assert store.load() == ({"seq": 1}, first)
        second = store.save({"seq": 2}, first)
        with pytest.raises(StoreConflict):
            store.save({"seq": 3}, first)  # written over since
        assert store.load() == ({"seq": 2}, second)
    finally:
        server.shutdown()
