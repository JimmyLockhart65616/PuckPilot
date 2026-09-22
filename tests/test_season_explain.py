"""Saying why, in words a person can argue with.

The output was correct and unreadable, which for a tool somebody consults for
thirty seconds is the same as being wrong.
"""

from __future__ import annotations

import pytest

from puckpilot.data import store
from puckpilot.season import explain
from puckpilot.season.roster import RosterPlayer
from puckpilot.season.today import BENCH, LineupPlan, Move
from tests.test_season_today import Values, player, roster, runtime  # noqa: F401

DATE = "2026-10-07"
SEASON = "20262027"


@pytest.fixture
def db_games(db):
    for gid, (home, away) in enumerate([("TOR", "MTL"), ("WSH", "TBL")], start=1):
        store.upsert_schedule_game(
            db,
            game_id=gid,
            season=SEASON,
            game_type=2,
            game_date=DATE,
            start_time_utc=f"{DATE}T23:00:00Z",
            home_team=home,
            away_team=away,
        )
    db.commit()
    return db


def make_plan(moves=(), playing=(), idle=(), out=(), empty=(), locked=()):
    return LineupPlan(
        date=DATE,
        team_key="999.l.1.t.5",
        manager="jimmy",
        moves=tuple(moves),
        playing=tuple(playing),
        idle=tuple(idle),
        out=tuple(out),
        locked=tuple(locked),
        empty_slots=tuple(empty),
        lock_utc=f"{DATE}T23:00:00Z",
        lock_team="TOR",
        authority_reason="recommend only.",
    )


def test_a_start_says_who_they_play(db_games):
    p = player("p.1", "Alex Tuch", 1, "WSH", "RW", "BN")
    plan = make_plan(moves=[Move(player=p, to_slot="RW", from_slot=BENCH)])
    why = explain.move_reasons(db_games, runtime(), plan)
    assert why["p.1"] == "plays tonight (vs TBL)."


def test_a_bench_says_the_team_is_idle(db_games):
    """This is the real reason nine nights in ten, and it was never stated."""
    p = player("p.1", "Will Cuylle", 1, "NYR", "LW", "LW")
    plan = make_plan(moves=[Move(player=p, to_slot=BENCH, from_slot="LW")])
    why = explain.move_reasons(db_games, runtime(), plan)
    assert why["p.1"] == "NYR are not playing tonight."


def test_a_bench_of_someone_who_does_play_says_so_differently(db_games):
    p = player("p.1", "Someone", 1, "TOR", "C", "C")
    plan = make_plan(moves=[Move(player=p, to_slot=BENCH, from_slot="C")])
    why = explain.move_reasons(db_games, runtime(), plan)
    assert "someone worth more needs the slot" in why["p.1"]


def test_a_slot_shuffle_explains_what_it_frees(db_games):
    p = player("p.1", "Flex", 1, "TOR", "C", "C")
    plan = make_plan(moves=[Move(player=p, to_slot="Util", from_slot="C")])
    why = explain.move_reasons(db_games, runtime(), plan)
    assert "frees C" in why["p.1"]


def test_a_goalie_start_is_stated_as_a_probability_not_a_fact(db_games):
    """A goalie at 63% is a guess and the schedule is a fact; the wording has
    to keep them apart."""
    from puckpilot.season.today import Candidate

    p = player("p.1", "Dustin Wolf", 1, "TOR", "G", "BN")
    cand = Candidate(player=p, value=5.0, p_start=0.63)
    plan = make_plan(moves=[Move(player=p, to_slot="G", from_slot=BENCH)], playing=[cand])
    why = explain.move_reasons(db_games, runtime(), plan)
    assert "63% to start" in why["p.1"]


def test_a_day_off_says_so_plainly(db_games):
    story = " ".join(explain.plan_story(db_games, runtime(), make_plan()))
    assert "Nobody you own plays tonight" in story


def test_a_quiet_day_explains_why_there_is_nothing_to_do(db_games):
    from puckpilot.season.today import Candidate

    p = player("p.1", "Playing", 1, "TOR", "C", "C")
    story = " ".join(
        explain.plan_story(db_games, runtime(), make_plan(playing=[Candidate(player=p, value=1.0)]))
    )
    assert "already in a starting slot" in story


def test_empty_slots_are_explained_rather_than_listed(db_games):
    story = " ".join(explain.plan_story(db_games, runtime(), make_plan(empty=("C", "RW"))))
    assert "nobody who plays is eligible for them" in story
    assert "that is normal" in story


def test_the_deadline_is_explained_as_a_per_player_lock(db_games):
    story = " ".join(explain.plan_story(db_games, runtime(), make_plan()))
    assert "each player locks at his own game" in story


def test_an_injury_names_the_injury(db_games):
    hurt = RosterPlayer(
        player_key="p.9",
        yahoo_id="9",
        name="Mathew Barzal",
        team="NYI",
        primary_position="C",
        yahoo_eligible=frozenset({"C"}),
        selected_slot="IR+",
        nhl_player_id=9,
        status="O",
        status_full="Out",
        injury_note="Knee",
    )
    story = " ".join(explain.plan_story(db_games, runtime(), make_plan(out=[hurt])))
    assert "Mathew Barzal (Knee)" in story


# -- the protocol, in plain words -------------------------------------------


def test_the_protocol_says_what_approving_it_does_and_does_not_do():
    from puckpilot.engine.categories import resolve
    from puckpilot.season import protocol as protocol_mod
    from puckpilot.season.week import CategoryOutlook

    outlook = [
        CategoryOutlook(
            category=resolve("HIT"), ours=41.0, theirs=52.0, lineup_room=0.0, add_room=1.0
        ),
        CategoryOutlook(
            category=resolve("PPP"), ours=8.5, theirs=7.7, lineup_room=0.0, add_room=5.0
        ),
    ]
    proto = protocol_mod.derive(outlook, "jimmy", "l", "t", 2, "Them")
    text = " ".join(explain.protocol_story(proto, adds_left=3))
    assert "Stop spending on: HIT" in text
    assert "Go after: PPP" in text
    assert "steers which free agents get proposed" in text
    assert "does NOT do: change your daily lineup" in text
    assert "3 acquisition(s) left" in text


def test_a_protocol_with_nothing_to_say_says_that():
    from puckpilot.engine.categories import resolve
    from puckpilot.season import protocol as protocol_mod
    from puckpilot.season.week import CategoryOutlook

    outlook = [
        CategoryOutlook(
            category=resolve("SOG"), ours=94.0, theirs=80.0, lineup_room=0.0, add_room=5.0
        )
    ]
    proto = protocol_mod.derive(outlook, "jimmy", "l", "t", 2, "Them")
    assert "play it straight" in " ".join(explain.protocol_story(proto))
