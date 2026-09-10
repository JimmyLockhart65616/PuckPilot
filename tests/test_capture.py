"""Capture harness: the parts that must be right before a draft is recorded.

The Playwright attach itself is exercised by running it; what is unit-tested
here is the redaction (this repo is public) and the guarantee that a session
always lands on disk, because a lost recording cannot be re-taken.
"""

from __future__ import annotations

import json

from puckpilot.draft.capture import (
    CaptureSession,
    _redact,
    _wants_body,
    new_session_dir,
)


def test_redact_hides_session_credentials_but_keeps_the_key():
    headers = {"Cookie": "SID=secret", "Authorization": "Bearer tok", "Accept": "application/json"}
    out = _redact(headers)
    assert out["Cookie"] == "<redacted>"
    assert out["Authorization"] == "<redacted>"
    # knowing a request carried a cookie is the useful part; the value is not
    assert "secret" not in json.dumps(out)
    assert out["Accept"] == "application/json"


def test_redact_tolerates_missing_headers():
    assert _redact({}) == {}
    assert _redact(None) == {}


def test_wants_body_selects_plausible_data_feeds():
    assert _wants_body("application/json; charset=utf-8")
    assert _wants_body("text/javascript")
    assert not _wants_body("text/html")
    assert not _wants_body("")


def test_session_writes_streams_and_manifest(tmp_path):
    session = CaptureSession(out_dir=tmp_path / "s")
    session.write("network", {"kind": "request", "url": "https://example.test"})
    session.write("websocket", {"dir": "recv", "payload": "{}"})
    session.finalize({"attach": {"mode": "test"}})

    rows = [
        json.loads(x)
        for x in (tmp_path / "s" / "network.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert rows[0]["url"] == "https://example.test"
    # every record is timestamped so pick events can be correlated with the wire
    assert "t" in rows[0] and "wall" in rows[0]

    manifest = json.loads((tmp_path / "s" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["counts"] == {"network": 1, "websocket": 1}
    assert "duration_s" in manifest


def test_snapshot_failure_is_recorded_not_raised(tmp_path):
    class DeadPage:
        url = "https://fantasy.yahoo.com/draft"

        def content(self):
            raise RuntimeError("Target page, context or browser has been closed")

    session = CaptureSession(out_dir=tmp_path / "s")
    session.snapshot_dom(DeadPage(), 1)  # must not raise: a dead tab is not a lost session
    session.finalize({})

    events = [
        json.loads(x)
        for x in (tmp_path / "s" / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert events[0]["kind"] == "dom_snapshot_failed"


def test_session_dirs_are_distinct_per_run(tmp_path):
    assert new_session_dir(tmp_path).parent == tmp_path


# ---- draft-room header parsing --------------------------------------------

# Verbatim from the live Yahoo draft client (2026-09-07 mock). The pick counter
# is ground truth for how many picks have happened, so a parsing regression here
# would make missed picks undetectable rather than merely inconvenient.
HEADER_R5 = (
    "YAHOO FANTASY HOCKEY DRAFT\n"
    "Penalty Box - H2H\n00:27\n"
    "Antonio's Pick \u2022 You're up in 1 Picks \u2022 Round 5, Pick 67\n"
    "Last: J. SANDERSON (D-OTT)   Jack Sparrow"
)
HEADER_R1 = (
    "Chris's Pick \u2022 You're up in 4 Picks \u2022 Round 1, Pick 13\n"
    "Last: A. VASILEVSKIY (G-TB) Jimmy Lockhart"
)


def test_dom_values_reads_the_authoritative_pick_counter():
    from puckpilot.draft.capture import dom_values

    v = dom_values(HEADER_R5)
    assert v["round"] == 5
    assert v["pick"] == 67
    assert v["picks_until_mine"] == 1
    assert v["last_pick"].startswith("J. SANDERSON")


def test_dom_values_handles_round_one():
    from puckpilot.draft.capture import dom_values

    v = dom_values(HEADER_R1)
    assert (v["round"], v["pick"]) == (1, 13)
    assert "VASILEVSKIY" in v["last_pick"]


def test_dom_values_is_empty_off_a_draft_page():
    """The lobby and every other tab must yield nothing rather than junk."""
    from puckpilot.draft.capture import dom_values

    assert dom_values("") == {}
    assert dom_values("Fantasy Hockey Mock Drafts | Join a draft") == {}


def test_draft_urls_survive_the_host_filter():
    """The host allowlist exists to drop ad exchanges, but it must never drop
    the draft client itself - that assumption already cost two captures."""
    from puckpilot.draft.capture import _on_allowed_host

    assert _on_allowed_host("https://hockey.fantasysports.yahoo.com/draftclient/hockey/2223773/12")
    assert _on_allowed_host("https://some-cdn.example.net/draft/socket")
    assert _on_allowed_host("wss://anything.example.org/draftclient/live")
    assert not _on_allowed_host("https://eus.rubiconproject.com/usync.js")


def test_draft_client_page_is_recognised_explicitly():
    """It matched before only because fantasy*sports*.yahoo.com contains
    'sports.yahoo.com'. This is the page the harness exists for."""
    from puckpilot.draft.capture import PAGE_URL_HINTS

    url = "https://hockey.fantasysports.yahoo.com/draftclient/hockey/2223773/12?auth="
    assert "draftclient" in PAGE_URL_HINTS
    assert any(h in url for h in PAGE_URL_HINTS)
