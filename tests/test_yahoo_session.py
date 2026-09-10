"""Yahoo session-API parsing.

The network side needs a logged-in browser and is exercised by running it. What
is unit-tested here is `flatten`, because Yahoo's response shape is genuinely
hostile: objects arrive as lists of single-key dicts, interleaved with real
dicts, with some fields nested a level deeper. Reading a field with plain
subscripting works until it silently does not.
"""

from __future__ import annotations

import inspect

import pytest

from puckpilot.yahoo.session import flatten

# Shape taken from a real /players response (2026-09-07).
PLAYER_ENTRY = [
    [
        {"player_key": "477.p.6743"},
        {"player_id": "6743"},
        {
            "name": {
                "full": "Connor McDavid",
                "first": "Connor",
                "last": "McDavid",
                "ascii_first": "Connor",
                "ascii_last": "McDavid",
            }
        },
        {"editorial_team_abbr": "EDM"},
        {"eligible_positions": [{"position": "C"}, {"position": "Util"}]},
    ],
    {"draft_analysis": [{"average_pick": "1.2"}]},
]


def test_flatten_lifts_fields_out_of_list_of_single_key_dicts():
    out = flatten(PLAYER_ENTRY)
    assert out["player_key"] == "477.p.6743"
    assert out["editorial_team_abbr"] == "EDM"


def test_flatten_reaches_nested_name():
    """`name` arrives one level deeper than everything around it."""
    assert flatten(PLAYER_ENTRY)["full"] == "Connor McDavid"


def test_flatten_normalizes_eligible_positions():
    """Multi-position eligibility is what the lineup optimizer was built for,
    so it has to survive parsing as a plain list."""
    assert flatten(PLAYER_ENTRY)["eligible_positions"] == ["C", "Util"]


def test_flatten_descends_through_nested_containers():
    assert flatten(PLAYER_ENTRY)["average_pick"] == "1.2"


def test_first_value_wins_so_league_meta_is_not_clobbered():
    """Yahoo repeats keys across sections; settings are merged under metadata,
    and a later generic 'name' must not overwrite the league's own."""
    out = flatten([{"name": "Ajaxians"}, {"team": [{"name": "Covie-19"}]}])
    assert out["name"] == "Ajaxians"


def test_structured_settings_are_kept_whole():
    settings = [{"roster_positions": [{"roster_position": {"position": "C", "count": 2}}]}]
    assert flatten(settings)["roster_positions"][0]["roster_position"]["count"] == 2


def test_flatten_tolerates_empty_and_scalar_input():
    assert flatten([]) == {}
    assert flatten({}) == {}
    assert flatten("nonsense") == {}


# ---- the fallback has to retire itself ------------------------------------
#
# This module reads Yahoo through a browser session cookie because the
# documented OAuth path answers 401 while our Fantasy API application is
# pending. That is a fallback, and the failure mode for a fallback is that it
# quietly outlives its reason. These pin the two properties that stop it:
# it refuses to run once OAuth works, and it never learned to write.


class _Probe:
    def __init__(self, granted):
        self.scope_granted = granted


def test_the_cookie_fallback_refuses_to_run_once_oauth_is_approved(monkeypatch):
    import puckpilot.yahoo.probe as probe_mod
    from puckpilot.yahoo.session import FallbackNoLongerNeeded, require_fallback_still_needed

    monkeypatch.setattr(probe_mod, "probe", lambda *_a, **_k: _Probe(True))
    with pytest.raises(FallbackNoLongerNeeded, match="approved"):
        require_fallback_still_needed()


def test_it_still_runs_while_the_application_is_pending(monkeypatch):
    import puckpilot.yahoo.probe as probe_mod
    from puckpilot.yahoo.session import require_fallback_still_needed

    monkeypatch.setattr(probe_mod, "probe", lambda *_a, **_k: _Probe(False))
    require_fallback_still_needed()  # must not raise


def test_a_broken_probe_is_not_read_as_approval(monkeypatch):
    """No network, no token, Yahoo down: none of those prove OAuth works, and
    treating them as approval would disable the only path we have."""
    import puckpilot.yahoo.probe as probe_mod
    from puckpilot.yahoo.session import require_fallback_still_needed

    def boom(*_a, **_k):
        raise ConnectionError("down")

    monkeypatch.setattr(probe_mod, "probe", boom)
    require_fallback_still_needed()  # must not raise


def test_the_session_client_exposes_no_way_to_write():
    """Read-only is the compliance claim; this is what makes it checkable
    rather than a promise in a docstring."""
    from puckpilot.yahoo import session as mod

    src = inspect.getsource(mod)
    for verb in ("method:", "method=", "'POST'", '"POST"', "'PUT'", "'DELETE'"):
        assert verb not in src, f"session.py gained a write path: {verb}"
    public = [n for n in dir(mod.YahooSession) if not n.startswith("_")]
    assert set(public) <= {
        "get",
        "league_keys",
        "league_meta",
        "teams",
        "draft_results",
        "players",
        "user_data_dir",
        "headless",
        "check_oauth",
    }, f"unexpected public method on YahooSession: {public}"


def test_requests_are_throttled():
    """A paging loop against an undocumented endpoint must not become a burst."""
    from puckpilot.yahoo.session import MIN_INTERVAL_S

    assert MIN_INTERVAL_S >= 0.5
