"""Keeper contracts reconstructed from draft history.

Synthetic two-team, four-round league: keepers go in the final round. The cases
are the ones that went wrong by hand - a contract that has run out, one that
moved teams mid-season, and a live late-round re-draft that is NOT a keep.
"""

from __future__ import annotations

from puckpilot.yahoo.keeperhistory import (
    SeasonHistory,
    bare,
    derive_contracts,
    keeper_picks,
    keeper_window,
    renewed_from,
)

TEAMS = {
    "t.1": {"guid": "G1", "nickname": "Ann", "name": "Ann FC"},
    "t.2": {"guid": "G2", "nickname": "Bob", "name": "Bob FC"},
}


def _season(key, picks, rosters, names=None):
    return SeasonHistory(
        league_key=key,
        season=key,
        teams={f"{key}.{k}": v for k, v in TEAMS.items()},
        picks=[
            {"round": r, "pick": i + 1, "team_key": f"{key}.{t}", "player_key": f"{key}.p.{p}"}
            for i, (r, t, p) in enumerate(picks)
        ],
        rosters={f"{key}.{t}": [str(p) for p in ids] for t, ids in rosters.items()},
        names=names or {},
    )


def _history():
    # 4 rounds, 2 teams, 1 keeper each in round 4.
    s1 = _season(
        "s1",
        [
            (1, "t.1", 10),
            (1, "t.2", 20),
            (2, "t.2", 21),
            (2, "t.1", 11),
            (3, "t.1", 12),
            (3, "t.2", 22),
            (4, "t.2", 23),
            (4, "t.1", 13),
        ],
        {"t.1": [10, 11, 12, 13], "t.2": [20, 21, 22, 23]},
    )
    # Ann keeps 10 (round 4); Bob keeps 20. Bob re-drafts 21 LIVE in round 1.
    s2 = _season(
        "s2",
        [
            (1, "t.1", 30),
            (1, "t.2", 21),
            (2, "t.2", 31),
            (2, "t.1", 32),
            (3, "t.1", 33),
            (3, "t.2", 34),
            (4, "t.2", 20),
            (4, "t.1", 10),
        ],
        # 10 moves to Bob mid-season
        {"t.1": [30, 32, 33], "t.2": [21, 31, 34, 20, 10]},
    )
    # Bob keeps 10 (acquired) and... only one keeper per team; keeps 20 again.
    s3 = _season(
        "s3",
        [
            (1, "t.1", 40),
            (1, "t.2", 41),
            (2, "t.2", 42),
            (2, "t.1", 43),
            (3, "t.1", 30),
            (3, "t.2", 44),
            (4, "t.2", 20),
            (4, "t.1", 32),
        ],
        {"t.1": [40, 43, 30, 32], "t.2": [41, 42, 44, 20]},
        names={"20": "Veteran", "32": "Second-year", "30": "Redrafted", "40": "Rookie"},
    )
    return [s1, s2, s3]


def test_bare_and_renew_helpers():
    assert bare("465.p.6743") == "6743"
    assert renewed_from({"renew": "465_12345"}) == "465.l.12345"
    assert renewed_from({"renew": ""}) is None


def test_the_keeper_window_is_measured_not_assumed():
    s1, s2, s3 = _history()
    assert keeper_window(s2, s1, default=3) == 1
    assert keeper_window(s1, None, default=3) == 3  # nothing before it to measure


def test_a_live_redraft_of_last_years_player_is_not_a_keep():
    s1, s2, _ = _history()
    kept = keeper_picks(s2, s1, roster_rounds=4, window=1)
    assert kept == {"20": "G2", "10": "G1"}
    assert "21" not in kept  # re-drafted in round 1


def test_contracts_count_consecutive_keeps_and_expire():
    report = derive_contracts(
        _history(), current_teams={}, n_keepers=1, keeper_years=2, roster_rounds=4
    )
    bob = next(m for m in report.managers if m.nickname == "Bob")
    ann = next(m for m in report.managers if m.nickname == "Ann")
    # 20 was kept in s2 and s3: two keeps against a two-keep contract.
    assert bob.expired == ["20"]
    assert bob.continuing == []
    # 32 was kept once (s3) and is still on Ann's roster.
    assert ann.continuing == [("32", 1)]
    # 30 was drafted live in s3 - a first-year candidate, never a contract.
    assert "30" in ann.candidates and "40" in ann.candidates
    assert ann.open_slots(1) == 0 and bob.open_slots(1) == 1


def test_managers_are_matched_to_this_season_by_guid_not_team_number():
    current = {"x.t.9": {"guid": "G1", "nickname": "Ann", "name": "Renamed FC"}}
    report = derive_contracts(_history(), current, n_keepers=1, keeper_years=2, roster_rounds=4)
    ann = next(m for m in report.managers if m.nickname == "Ann")
    assert ann.current_team_key == "x.t.9" and ann.team_name == "Renamed FC"
