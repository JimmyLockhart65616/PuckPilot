"""The in-season page's own JavaScript, run against the payloads it will get.

Same reasoning as the draft page's version: the page and the payloads are
written in two languages by two code paths, and a key renamed on one side does
not fail loudly - it renders "undefined", or throws inside the poll, which the
draft page used to report as "server unreachable".

So the script is pulled out of PAGE, run in Node against a small DOM, and the
rendered text is scanned for the tell-tales. Skipped when Node is absent.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess

import pytest

from puckpilot.web import wire
from puckpilot.web.season_page import PAGE
from puckpilot.web.season_relay import SeasonState

NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="node is not installed")

HARNESS = r"""
const vm = require('vm');
const fs = require('fs');
const input = JSON.parse(fs.readFileSync(0, 'utf8'));
const out = [];

function node(tag) {
  return {
    tagName: tag, className: '', _text: '', children: [], style: {}, disabled: false,
    set textContent(v) { this._text = String(v); this.children = []; },
    get textContent() {
      return this._text + this.children.map(c => ' ' + c.textContent).join('');
    },
    appendChild(c) { this.children.push(c); return c; },
    replaceChild(a, b) {
      const i = this.children.indexOf(b);
      if (i >= 0) this.children[i] = a; else this.children.push(a);
    },
    querySelectorAll() { return []; },
    get parentNode() { return this._parent || node('div'); },
    set parentNode(v) { this._parent = v; },
  };
}

const fresh = node('div');
const app = node('div');
const doc = {
  createElement: node,
  getElementById: (id) => (id === 'fresh' ? fresh : app),
  addEventListener: () => {},
  hidden: false,
};

const sandbox = {
  document: doc,
  window: {},
  location: { search: '?k=test' },
  URLSearchParams: class { constructor(s) { this.s = s; } get() { return 'test'; } },
  setInterval: () => {},
  fetch: () => Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve(input) }),
  console,
};
sandbox.globalThis = sandbox;
vm.createContext(sandbox);
vm.runInContext(SCRIPT, sandbox);

// The script calls tick() itself, which resolves on a microtask.
setTimeout(() => {
  console.log(JSON.stringify({ fresh: fresh.textContent, app: app.textContent }));
}, 20);
"""


def run_page(payload: dict) -> dict:
    script = re.search(r"<script>(.*?)</script>", PAGE, re.S).group(1)
    harness = "const SCRIPT = " + json.dumps(script) + ";\n" + HARNESS
    proc = subprocess.run(
        [NODE, "-e", harness],
        input=wire.dumps(payload),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])


def assert_clean(rendered: dict) -> None:
    whole = rendered["fresh"] + " " + rendered["app"]
    for tell in ("undefined", "NaN", "[object Object]"):
        assert tell not in whole, f"{tell!r} in rendered page: {whole[:400]}"


def test_a_cold_relay_draws_a_page_that_says_so():
    """A fresh URL must not draw "undefined" anywhere - the draft page did
    exactly that on a cold relay."""
    out = run_page(SeasonState().get("jimmy"))
    assert_clean(out)
    assert "Nothing pushed yet" in out["app"]
    assert "NOT LIVE" in out["fresh"]


def test_a_full_payload_renders_every_section():
    state = SeasonState()
    state.push(
        "jimmy",
        {
            "team": "Home Team",
            "date": "2026-10-07",
            "moves": [
                {"kind": "start", "name": "Alex Tuch", "detail": "into LW"},
                {"kind": "bench", "name": "Will Cuylle", "detail": "was LW"},
                {"kind": "move", "name": "J.T. Miller", "detail": "C to Util"},
            ],
            "out": ["Mathew Barzal (NYI C) [O]"],
            "lock_local": "7:00 PM",
            "playing": 9,
            "rostered": 17,
            "proposals": [
                {
                    "id": 4,
                    "add": "Shane Pinto",
                    "drop": "Alex Laferriere",
                    "why": "+9.6 this week · 4 games · PIM +1.9",
                    "timing": "free agent - this one is a race",
                }
            ],
            "protocol": {
                "id": 2,
                "week": 2,
                "opponent": "Rival FC",
                "status": "proposed",
                "give_up": ["HIT (behind 10.9)"],
                "go_after": ["PPP (+0.7)"],
            },
            "week": {
                "week": 2,
                "opponent": "Rival FC",
                "cats": [
                    {"label": "G", "ours": 11.0, "theirs": 10.0, "state": "close"},
                    {"label": "HIT", "ours": 41.2, "theirs": 52.2, "state": "gone"},
                    {"label": "SV%", "ours": 0.898, "theirs": 0.902, "state": "close"},
                ],
                "note": "Everyone with a game fits in a slot this week.",
            },
            "roster": [
                {"slot": "C", "name": "Evgeni Malkin", "team": "PIT", "opp": "", "status": ""},
                {"slot": "IR+", "name": "Mathew Barzal", "team": "NYI", "opp": "", "status": "Out"},
            ],
        },
    )
    out = run_page(state.get("jimmy"))
    assert_clean(out)
    for expected in (
        "Home Team",
        "Alex Tuch",
        "Will Cuylle",
        "Shane Pinto",
        "HIT",
        "Rival FC",
        "7:00 PM",
        "Evgeni Malkin",
    ):
        assert expected in out["app"], f"{expected!r} missing"
    assert "updated just now" in out["fresh"]


def test_a_payload_missing_optional_sections_still_draws():
    """The lineup runs before the weekly plan exists, so half a payload is the
    normal case rather than an error."""
    state = SeasonState()
    state.push("jimmy", {"team": "Home Team", "moves": [], "proposals": []})
    out = run_page(state.get("jimmy"))
    assert_clean(out)
    assert "Home Team" in out["app"]


def test_an_old_push_renders_as_not_live():
    """A lineup that was right this morning is wrong by evening, so a stale
    page has to say so rather than look current."""
    state = SeasonState(stale_after=0.0)
    state.push("jimmy", {"team": "Home Team", "moves": []})
    out = run_page(state.get("jimmy"))
    assert "NOT LIVE" in out["fresh"]


def test_a_zero_value_category_does_not_render_as_undefined():
    """0 is falsy in JavaScript and a table full of blanks is a real bug."""
    state = SeasonState()
    state.push(
        "jimmy",
        {
            "team": "T",
            "moves": [],
            "week": {
                "week": 1,
                "opponent": "X",
                "cats": [{"label": "PPP", "ours": 0, "theirs": 0, "state": "close"}],
                "note": "",
            },
        },
    )
    out = run_page(state.get("jimmy"))
    assert_clean(out)
    assert "PPP" in out["app"]
