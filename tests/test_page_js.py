"""The page's own JavaScript, run against the payloads it will really receive.

The page and the payloads are written in two languages by two code paths - the
console's snapshot, the relay's copy of it, the relay's cold and error replies -
and nothing checked that one could draw the other. A key renamed on one side
does not fail; it renders "undefined", or throws inside the poll, which the page
used to report as "server unreachable".

So the script is extracted from PAGE and executed in Node against a minimal DOM,
once per payload shape, and the rendered text is scanned for the tell-tale
"undefined" / "NaN" / "null". Skipped when Node is not installed.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from puckpilot.draft.feed import apply
from puckpilot.web import wire
from puckpilot.web.page import PAGE
from puckpilot.web.relay import RelayState
from puckpilot.web.server import LiveState
from tests import draftkit

NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="node is not installed")

HARNESS = r"""
const fs = require('fs');
const vm = require('vm');
const input = JSON.parse(fs.readFileSync(0, 'utf8'));

function makeEl(id){
  return {id, textContent:'', innerHTML:'', hidden:false, className:'', children:[],
    listeners:{},
    appendChild(c){ this.children.push(c); },
    addEventListener(t, f){ this.listeners[t] = f; }};
}
let els = {};
globalThis.document = {
  getElementById: id => els[id] || (els[id] = makeEl(id)),
  createElement: tag => makeEl(tag),
};
globalThis.window = globalThis;
globalThis.location = {search: input.search};
let nextFetch = null;
globalThis.fetch = async () => {
  if(!nextFetch) throw new Error('offline');
  const f = nextFetch;
  return {ok: f.status < 400, status: f.status,
          json: async () => JSON.parse(f.body)};
};
globalThis.setInterval = () => 0;
vm.runInThisContext(input.script);

function dump(){
  const out = {};
  for(const [id, e] of Object.entries(els)){
    const kids = e.children.map(c => c.innerHTML + c.textContent).join('\n');
    out[id] = {text: e.textContent + '\n' + e.innerHTML + '\n' + kids,
               hidden: e.hidden, className: e.className, children: e.children.length};
  }
  return out;
}

(async () => {
  const results = [];
  for(const c of input.cases){
    els = {};
    let error = null;
    try {
      if(c.mode === 'tick'){
        nextFetch = {status: c.status, body: c.body};
        await tick();
      } else {
        render(c.payload);
      }
    } catch(e){ error = String(e && e.stack || e); }
    results.push({name: c.name, error, dom: dump()});
  }
  process.stdout.write(JSON.stringify(results));
})();
"""

SCRIPT = re.search(r"<script>(.*)</script>", PAGE, re.S).group(1)
TELLTALE = re.compile(r"\b(undefined|NaN|null)\b")


def _run(cases, search="?seat=0", tmp_path: Path | None = None):
    harness = (tmp_path or Path.cwd()) / "page_harness.js"
    harness.write_text(HARNESS, encoding="utf-8")
    r = subprocess.run(
        [NODE, str(harness)],
        input=json.dumps({"script": SCRIPT, "search": search, "cases": cases}),
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
    )
    assert r.returncode == 0, r.stderr
    return {res["name"]: res for res in json.loads(r.stdout)}


def _visible(dom, skip=("diag",)):
    return "\n".join(v["text"] for k, v in dom.items() if k not in skip and not v["hidden"])


def _as_browser_sees(payload):
    return wire.loads(wire.dumps(payload))


@pytest.fixture(scope="module")
def payloads():
    board = draftkit.board()
    state = LiveState(board=board, feed=None, top=3, board_rows=60, cats=draftkit.CATS)
    feed = draftkit.sim_feed(21)
    for _ in range(9):
        apply(board, feed.poll(board))
    state.last_pick_at = 1.0  # a pick has happened, so the clock is a number
    mid = {**state.snapshot(0), "can_undo": True}
    other = {**state.snapshot(3), "can_undo": True}

    relay = RelayState()
    relay.push(wire.clean({"0": state.snapshot(0)}))
    fresh = {**relay.get("0"), "can_undo": False}
    relay.pushed_at -= 45
    stale = {**relay.get("0"), "can_undo": False}
    waiting = {**RelayState().get(None), "can_undo": False}
    unknown = {**relay.get("9"), "can_undo": False}

    while not board.complete:
        apply(board, feed.poll(board))
    done = {**state.snapshot(0), "can_undo": True}
    return {
        "mid": _as_browser_sees(mid),
        "other_seat": _as_browser_sees(other),
        "relay_fresh": _as_browser_sees(fresh),
        "relay_stale": _as_browser_sees(stale),
        "relay_waiting": _as_browser_sees(waiting),
        "relay_unknown_seat": _as_browser_sees(unknown),
        "done": _as_browser_sees(done),
        "forbidden": {"error": "a valid access key is required"},
    }


@pytest.fixture(scope="module")
def rendered(payloads, tmp_path_factory):
    cases = [{"name": k, "payload": v} for k, v in payloads.items()]
    return _run(cases, tmp_path=tmp_path_factory.mktemp("page"))


def test_every_payload_shape_renders_without_throwing(rendered):
    errors = {k: v["error"] for k, v in rendered.items() if v["error"]}
    assert errors == {}


def test_nothing_renders_as_undefined_nan_or_null(rendered):
    leaks = {}
    for name, res in rendered.items():
        hits = sorted({m.group(1) for m in TELLTALE.finditer(_visible(res["dom"]))})
        if hits:
            leaks[name] = hits
    assert leaks == {}


def test_a_live_board_draws_every_panel(rendered, payloads):
    dom = rendered["mid"]["dom"]
    assert dom["banner"]["hidden"] is True
    assert dom["short"]["children"] == len(payloads["mid"]["shortlist"]) == 3
    assert dom["board"]["children"] == len(payloads["mid"]["board"])
    assert dom["undo"]["hidden"] is False
    assert f"{payloads['mid']['made'] + 1}/{payloads['mid']['total']}" in dom["pick"]["text"]


def test_the_relay_copy_hides_undo_and_shows_its_age(rendered):
    dom = rendered["relay_fresh"]["dom"]
    assert dom["undo"]["hidden"] is True
    assert "relay" in dom["relay"]["text"] and "live" in dom["relay"]["text"]
    assert dom["banner"]["hidden"] is True


def test_a_stale_relay_says_not_live_at_the_top(rendered):
    dom = rendered["relay_stale"]["dom"]
    assert dom["banner"]["hidden"] is False
    assert "NOT LIVE" in dom["banner"]["text"]
    assert "bad" in dom["relay"]["text"]


def test_a_cold_relay_says_waiting_not_undefined(rendered):
    dom = rendered["relay_waiting"]["dom"]
    assert "Waiting for the draft console" in dom["banner"]["text"]
    assert "waiting" in dom["turn"]["text"]
    assert "roster minimums met" not in dom["needs"]["text"]


def test_an_unknown_seat_names_the_seats_that_exist(rendered):
    dom = rendered["relay_unknown_seat"]["dom"]
    assert "no snapshot for seat 9" in dom["banner"]["text"]
    assert "seats with a view: 0" in dom["banner"]["text"]


def test_a_refused_key_is_said_not_swallowed(rendered):
    assert "a valid access key is required" in rendered["forbidden"]["dom"]["banner"]["text"]


def test_the_finished_draft_reads_as_finished(rendered, payloads):
    dom = rendered["done"]["dom"]
    total = payloads["done"]["total"]
    assert f"{total}/{total}" in dom["pick"]["text"]
    assert "draft complete" in dom["turn"]["text"]
    assert dom["short"]["children"] == 0


def test_the_poll_tells_a_bad_reply_from_a_dead_server(tmp_path, payloads):
    broken = dict(payloads["mid"])
    broken["shortlist"] = None  # a renamed/missing key on the snapshot side
    res = _run(
        [
            {"name": "html", "mode": "tick", "status": 502, "body": "<html>Bad Gateway</html>"},
            {"name": "undrawable", "mode": "tick", "status": 200, "body": json.dumps(broken)},
            {"name": "ok", "mode": "tick", "status": 200, "body": json.dumps(payloads["mid"])},
        ],
        tmp_path=tmp_path,
    )
    assert "unreadable reply from the server (HTTP 502)" in res["html"]["dom"]["banner"]["text"]
    assert "could not draw the board" in res["undrawable"]["dom"]["banner"]["text"]
    assert res["ok"]["dom"]["banner"]["hidden"] is True
    for r in res.values():
        assert r["error"] is None
