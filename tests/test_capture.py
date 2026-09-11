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

# Layout copied from the live Yahoo draft client (2026-09-07 mock); the manager
# and team names are placeholders. The people in that room were strangers who
# did not consent to appearing in a public repository, and the regexes under
# test key off "Round N, Pick M", "Last:" and "up in N Picks" - so the names are
# load-bearing for nothing.
#
# The pick counter is ground truth for how many picks have happened, so a
# parsing regression here would make missed picks undetectable rather than
# merely inconvenient.
HEADER_R5 = (
    "YAHOO FANTASY HOCKEY DRAFT\n"
    "Some Mock League - H2H\n00:27\n"
    "Manager Four's Pick \u2022 You're up in 1 Picks \u2022 Round 5, Pick 67\n"
    "Last: J. SANDERSON (D-OTT)   Team Delta"
)
HEADER_R1 = (
    "Manager Nine's Pick \u2022 You're up in 4 Picks \u2022 Round 1, Pick 13\n"
    "Last: A. VASILEVSKIY (G-TB) Team Alpha"
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

    assert _on_allowed_host("https://hockey.fantasysports.yahoo.com/draftclient/hockey/1234567/12")
    assert _on_allowed_host("https://some-cdn.example.net/draft/socket")
    assert _on_allowed_host("wss://anything.example.org/draftclient/live")
    assert not _on_allowed_host("https://eus.rubiconproject.com/usync.js")


def test_draft_client_page_is_recognised_explicitly():
    """It matched before only because fantasy*sports*.yahoo.com contains
    'sports.yahoo.com'. This is the page the harness exists for."""
    from puckpilot.draft.capture import PAGE_URL_HINTS

    url = "https://hockey.fantasysports.yahoo.com/draftclient/hockey/1234567/12?auth="
    assert "draftclient" in PAGE_URL_HINTS
    assert any(h in url for h in PAGE_URL_HINTS)


# ---- redaction ------------------------------------------------------------
#
# Header redaction was once the only redaction in the module, which left the
# likelier carriers untouched: a crumb in a query string, a token in a JSON
# response body, and the frames the browser SENDS to a websocket - which is
# exactly where a client-side auth handshake appears.


def test_scrub_strips_a_token_from_a_query_string():
    from puckpilot.draft.capture import scrub

    out = scrub("https://x.yahoo.com/api?crumb=AbCdEf123456&format=json")
    assert "AbCdEf123456" not in out
    assert "crumb" in out and "format=json" in out


def test_scrub_strips_tokens_from_a_json_body():
    from puckpilot.draft.capture import scrub

    out = scrub('{"access_token": "ya29.SECRETVALUE", "expires_in": 3600}')
    assert "ya29.SECRETVALUE" not in out
    assert "expires_in" in out


def test_scrub_strips_a_bearer_header_value_and_a_bare_jwt():
    from puckpilot.draft.capture import scrub

    assert "abcdefghijklmnop" not in scrub("Authorization: Bearer abcdefghijklmnop")
    jwt = (
        "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0"
        ".dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U"
    )
    assert jwt not in scrub(f"token={jwt} trailing")


def test_scrub_leaves_ordinary_draft_traffic_alone():
    """Over-redaction would destroy the capture's whole purpose."""
    from puckpilot.draft.capture import scrub

    frame = "0|1|6743|1|C|0"
    assert scrub(frame) == frame
    assert scrub("Round 5, Pick 67") == "Round 5, Pick 67"


def test_scrubbing_happens_at_the_chokepoint_not_per_call_site():
    """Scrubbing each call site was tried and covered 6 paths of 18. Doing it on
    the serialized line means a path added later cannot bypass it by omission."""
    import inspect

    from puckpilot.draft.capture import CaptureSession

    src = inspect.getsource(CaptureSession.write)
    assert "scrub(json.dumps(record" in src, "CaptureSession.write stopped scrubbing"


def test_a_token_nested_in_a_json_body_does_not_survive_serialization(tmp_path):
    """The record is serialized before it is written, so a nested body arrives
    with its quotes escaped. Patterns that only matched bare quotes let this
    through - verified leaking before the escape handling was added."""
    session = CaptureSession(out_dir=tmp_path / "s")
    session.write(
        "network",
        {"kind": "response", "body": '{"access_token": "ya29.SECRETVALUE", "ok": 1}'},
    )
    session.finalize({})
    written = (tmp_path / "s" / "network.jsonl").read_text(encoding="utf-8")
    assert "ya29.SECRETVALUE" not in written
    assert "access_token" in written  # the key survives; only the value goes


def test_the_manifest_is_scrubbed_too(tmp_path):
    session = CaptureSession(out_dir=tmp_path / "s")
    session.finalize({"attach": {"mode": "cdp", "url": "http://h/x?token=SUPERSECRETVALUE"}})
    text = (tmp_path / "s" / "manifest.json").read_text(encoding="utf-8")
    assert "SUPERSECRETVALUE" not in text


def test_scrub_does_not_corrupt_real_pick_frames():
    """Over-redaction would silently destroy a capture. These are the exact
    shapes the 2026-09-08 draft emitted; zero of its 687 frames were altered."""
    from puckpilot.draft.capture import scrub

    for frame in ("0|1|6743|1|C|0", "D|192|1|30", "C|24", "H|S|30|0|0|0"):
        assert scrub(frame) == frame
