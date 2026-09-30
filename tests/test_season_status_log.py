"""Injury tags as the runs see them, and what followed."""

from __future__ import annotations

from types import SimpleNamespace

from puckpilot.data import store
from puckpilot.season import status_log
from tests.conftest import add_skater_game


def _roster(*players):
    return SimpleNamespace(league_key="l", team_key="l.t.5", players=list(players))


def _p(status="", key="p.1", name="Jake Sanderson", team="OTT", pid=1, note=""):
    return SimpleNamespace(
        player_key=key, name=name, team=team, status=status, injury_note=note, nhl_player_id=pid
    )


def test_healthy_players_seen_first_are_the_baseline_not_news(db):
    assert status_log.record(db, _roster(_p()), "2026-09-30T15:00:00+00:00") == []


def test_a_player_first_seen_tagged_is_logged(db):
    [c] = status_log.record(db, _roster(_p("IR-NR", name="Mathew Barzal")), "2026-09-30T15:00:00Z")
    assert c.old is None and c.new == "IR-NR"
    assert c.describe() == "Mathew Barzal (OTT): first seen -> IR-NR"


def test_a_change_is_bracketed_by_the_runs_either_side(db):
    status_log.record(db, _roster(_p()), "2026-09-30T15:00:00+00:00")
    assert status_log.record(db, _roster(_p()), "2026-09-30T16:00:00+00:00") == []
    [c] = status_log.record(db, _roster(_p("DTD", note="Undisclosed")), "2026-09-30T23:10:00+00:00")
    assert (c.old, c.new) == ("", "DTD")
    # Last seen healthy at the second run, not the first.
    assert c.old_seen_at == "2026-09-30T16:00:00+00:00"
    assert c.describe() == "Jake Sanderson (OTT): healthy -> DTD (Undisclosed)"


def _game(db, gid, date, start):
    store.upsert_schedule_game(
        db,
        game_id=gid,
        season="20262027",
        game_type=2,
        game_date=date,
        start_time_utc=start,
        home_team="TOR",
        away_team="OTT",
    )
    db.commit()


def test_whether_a_tagged_player_played_his_next_game(db):
    _game(db, 1, "2026-10-03", "2026-10-03T23:00:00Z")
    _game(db, 2, "2026-10-08", "2026-10-08T23:00:00Z")
    status_log.record(db, _roster(_p()), "2026-09-30T15:00:00+00:00")
    status_log.record(db, _roster(_p("DTD")), "2026-09-30T23:10:00+00:00")

    [o] = status_log.outcomes(db, "l", "20262027", today="2026-10-01")
    assert (o.game_date, o.result) == ("2026-10-03", "upcoming")
    [o] = status_log.outcomes(db, "l", "20262027", today="2026-10-04")
    assert o.result == "did not play"
    add_skater_game(db, 1, "20262027", 1, date="2026-10-03")
    [o] = status_log.outcomes(db, "l", "20262027", today="2026-10-04")
    assert o.result == "played"
    assert status_log.summary([o]) == ["DTD: 1 of 1 played their next game (100%)"]


def test_a_tag_that_arrived_after_a_game_started_is_flagged(db):
    """Tagged between a 6:40 run and a 7:40 run, around a 7:00 game: no run
    could have acted on it."""
    _game(db, 1, "2026-10-03", "2026-10-03T23:00:00Z")
    status_log.record(db, _roster(_p()), "2026-10-03T22:40:00+00:00")
    status_log.record(db, _roster(_p("O")), "2026-10-03T23:40:00+00:00")
    [o] = status_log.outcomes(db, "l", "20262027", today="2026-10-04")
    assert o.missed_game == "2026-10-03"
    assert "arrived after a game had started" in status_log.summary([o])[-1]


def test_going_back_to_healthy_is_logged_but_not_followed_up(db):
    status_log.record(db, _roster(_p("DTD")), "2026-09-30T15:00:00+00:00")
    [c] = status_log.record(db, _roster(_p()), "2026-10-01T15:00:00+00:00")
    assert (c.old, c.new) == ("DTD", "")
    assert [o.change.new for o in status_log.outcomes(db, "l", "20262027", "2026-10-02")] == ["DTD"]
