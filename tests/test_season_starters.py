"""Announced starting goalies: parsing, keeping, answering, and scoring them."""

from __future__ import annotations

import pytest

from puckpilot.data import store
from puckpilot.season import starters
from puckpilot.season.goalies import ChainedGoalieSource, StaticGoalieSource
from tests.conftest import add_goalie_game

DAY = "2026-11-21"


def entry(key, team, call=None, status="", date=DAY):
    """A `player` entry shaped as Yahoo answers `out=starting_status`."""
    core = [
        {"player_key": key},
        {"name": {"full": f"Goalie {key}"}},
        {"editorial_team_abbr": team},
        {"display_position": "G"},
        {"primary_position": "G"},
        {"eligible_positions": [{"position": "G"}]},
    ]
    if status:
        core.append({"status": status})
    out = [core]
    if call is not None:
        out.append(
            {"starting_status": [{"coverage_type": "date", "date": date}, {"is_starting": call}]}
        )
    return out


def body(entries):
    players = {str(i): {"player": e} for i, e in enumerate(entries)}
    players["count"] = len(entries)
    return {"fantasy_content": {"league": [{"league_key": "999.l.1"}, {"players": players}]}}


PMAP = {"g1": 1, "g2": 2, "g3": 3, "g4": 4, "g5": 5, "g6": 6}


def test_a_call_is_one_zero_or_none_and_a_team_is_the_nhl_abbreviation():
    on = starters.parse_call(entry("g1", "LA", 1), DAY, PMAP)
    off = starters.parse_call(entry("g2", "LA", 0), DAY, PMAP)
    none = starters.parse_call(entry("g3", "NJ", status="IR"), DAY, PMAP)
    assert (on.is_starting, off.is_starting, none.is_starting) == (True, False, None)
    assert (on.team, none.team, none.status) == ("LAK", "NJD", "IR")
    assert (on.player_id, on.game_date, none.game_date) == (1, DAY, DAY)


class FakeSession:
    def __init__(self, entries):
        self.entries = {e[0][0]["player_key"]: e for e in entries}
        self.paths: list[str] = []

    def get(self, path):
        self.paths.append(path)
        keys = path.split("player_keys=")[1].split(";")[0].split(",")
        return body([self.entries[k] for k in keys if k in self.entries])


def test_calls_are_read_by_key_25_to_a_request():
    entries = [entry(f"k{i}", "AAA", i % 2) for i in range(30)]
    s = FakeSession(entries)
    got = starters.fetch_calls(s, "999.l.1", [f"k{i}" for i in range(30)], DAY)
    assert len(got) == 30 and len(s.paths) == 2
    assert all("out=starting_status" in p for p in s.paths)


def _calls(*rows):
    return [starters.parse_call(entry(*r), DAY, PMAP) for r in rows]


def test_a_read_with_no_call_does_not_erase_an_earlier_call(db):
    starters.save_calls(db, _calls(("g1", "AAA", 1), ("g2", "AAA", 0)), "2026-11-21T16:00:00+00:00")
    starters.save_calls(db, _calls(("g1", "AAA"), ("g2", "AAA")), "2026-11-21T19:00:00+00:00")
    got = starters.latest_calls(db, DAY)
    assert (got[1][1], got[2][1]) == (True, False)
    early = starters.latest_calls(db, DAY, as_of="2026-11-21T15:00:00+00:00")
    assert early == {}


def test_called_clubs_are_answered_and_the_rest_fall_through(db):
    starters.save_calls(
        db,
        _calls(
            ("g1", "AAA", 1),
            ("g2", "AAA", 0),
            ("g3", "AAA"),  # a third goalie on a called club
            ("g4", "BBB"),  # a club with no call yet
            ("g5", "CCC", 0),  # both listed goalies sit: a call-up starts
            ("g6", "CCC", 0),
        ),
    )
    src = starters.AnnouncedStartSource(db, miss=0.05, surprise=0.02)
    got = src.starts(DAY)
    assert got == {1: 0.95, 2: 0.02, 3: 0.02, 5: 0.02, 6: 0.02}
    assert src.starts("2026-11-22") == {}


def test_in_front_of_the_model_a_call_wins_only_for_its_club(db):
    starters.save_calls(db, _calls(("g1", "AAA", 0), ("g2", "AAA", 1)))
    model = StaticGoalieSource({DAY: {1: 0.7, 2: 0.3, 4: 0.6}})
    chain = ChainedGoalieSource(starters.AnnouncedStartSource(db), model)
    got = chain.starts(DAY)
    assert got[1] == pytest.approx(starters.CALL_SURPRISE)
    assert got[2] == pytest.approx(1 - starters.CALL_MISS)
    assert got[4] == 0.6  # not called: the model's


def test_check_scores_calls_made_before_puck_drop_only(db):
    store.upsert_schedule_game(
        db,
        game_id=1,
        season="20262027",
        game_type=2,
        game_date=DAY,
        start_time_utc="2026-11-22T00:00:00Z",
        home_team="AAA",
        away_team="BBB",
    )
    for pid in (1, 3):  # 1 started for AAA, 3 for BBB
        add_goalie_game(db, pid, "20262027", 1000 + pid, date=DAY)
    add_goalie_game(db, 2, "20262027", 1002, date=DAY, started=0)
    starters.save_calls(db, _calls(("g1", "AAA", 1), ("g2", "AAA", 0)), "2026-11-21T17:00:00+00:00")
    # BBB called only after puck drop - not a forecast
    starters.save_calls(db, _calls(("g3", "BBB", 1)), "2026-11-22T00:30:00+00:00")
    got = starters.check(db)
    assert (got.called_start, got.started_when_called, got.called_sit) == (1, 1, 1)
    assert (got.miss, got.surprise) == (0.0, 0.0)
    assert (got.starts, got.starts_called, got.coverage) == (2, 1, 0.5)
    assert got.lead_hours == [7.0]
    assert "called to start" in got.text()
    scored = starters.check(db, model_on=lambda d: {1: 0.5, 2: 0.5}, miss=0.1, surprise=0.1)
    assert scored.brier_calls == pytest.approx(0.01)  # (0.9-1)^2, (0.1-0)^2
    assert scored.brier_model == pytest.approx(0.25)
    assert "Brier" in scored.text()


def test_goalie_keys_are_the_mapped_goalies_who_play(db):
    db.executemany(
        "INSERT INTO yahoo_player_map (player_key, league_key, full_name, positions, nhl_player_id)"
        " VALUES (?, '999.l.1', ?, ?, ?)",
        [("g1", "A", "G", 1), ("g2", "B", "G", 2), ("s1", "S", "C,LW", 9), ("g9", "X", "G", None)],
    )
    add_goalie_game(db, 1, "20262027", 1, date=DAY)
    add_goalie_game(db, 9, "20262027", 2, date=DAY)
    assert starters.goalie_keys(db, "999.l.1", ("20262027", "20252026")) == ["g1"]


def test_the_authority_knob_takes_only_its_modes():
    from puckpilot.season.authority import AuthorityError, LineupAuthority

    assert LineupAuthority().announced_starters == "record"
    assert not LineupAuthority().league_statuses
    with pytest.raises(AuthorityError):
        LineupAuthority(announced_starters="trust")


def test_league_statuses_add_every_tagged_goalie_to_the_models_out_list(db):
    from types import SimpleNamespace

    from puckpilot.season.authority import LineupAuthority
    from puckpilot.season.run import _goalie_tags

    starters.save_calls(db, _calls(("g1", "AAA", None, "IR"), ("g2", "AAA"), ("g4", "BBB", 1)))

    def manager(on):
        lineup = LineupAuthority(league_statuses=on)
        return SimpleNamespace(authority=SimpleNamespace(lineup=lineup))

    assert _goalie_tags(db, manager(False), DAY, None) == (set(), set())
    out, fit = _goalie_tags(db, manager(True), DAY, None)
    assert out == {1} and fit == {2, 4}
