"""The forward date axis over nhl_schedule.

The distinction that matters: the replay harnesses index into dates derived from
game *logs*, so they can only look backwards. These answer questions about days
that have not been played yet, which is the whole point in-season.
"""

from __future__ import annotations

import pytest

from puckpilot.data import store
from puckpilot.season import calendar


def game(conn, gid, date, home, away, season="20262027", game_type=2):
    store.upsert_schedule_game(
        conn,
        game_id=gid,
        season=season,
        game_type=game_type,
        game_date=date,
        start_time_utc=f"{date}T23:00:00Z",
        home_team=home,
        away_team=away,
    )


@pytest.fixture
def sched(db):
    # Mon 2026-10-05 through Sat 2026-10-10.
    game(db, 1, "2026-10-05", "TOR", "MTL")
    game(db, 2, "2026-10-06", "EDM", "CGY")
    game(db, 3, "2026-10-06", "TOR", "OTT")
    game(db, 4, "2026-10-07", "MTL", "BOS")
    game(db, 5, "2026-10-10", "TOR", "BOS")
    # a preseason game on a day TOR is otherwise idle: must never count
    game(db, 6, "2026-10-08", "TOR", "BUF", game_type=1)
    # another season entirely
    game(db, 7, "2026-10-06", "VAN", "SEA", season="20252026")
    db.commit()
    return db


def test_teams_playing_on_a_date(sched):
    assert calendar.teams_playing(sched, "2026-10-06", "20262027") == {"EDM", "CGY", "TOR", "OTT"}


def test_an_idle_day_is_empty_not_an_error(sched):
    assert calendar.teams_playing(sched, "2026-10-09", "20262027") == set()


def test_preseason_games_never_count(sched):
    """game_type 1 is preseason; starting a player for one scores nothing."""
    assert calendar.teams_playing(sched, "2026-10-08", "20262027") == set()


def test_another_season_is_not_visible(sched):
    assert "VAN" not in calendar.teams_playing(sched, "2026-10-06", "20262027")


def test_opponents_maps_both_directions(sched):
    opp = calendar.opponents_on(sched, "2026-10-05", "20262027")
    assert opp == {"TOR": "MTL", "MTL": "TOR"}


def test_team_game_dates_are_inclusive_and_ordered(sched):
    assert calendar.team_game_dates(sched, "TOR", "2026-10-05", "2026-10-10", "20262027") == [
        "2026-10-05",
        "2026-10-06",
        "2026-10-10",
    ]


def test_games_by_team_counts_every_club_in_one_pass(sched):
    counts = calendar.games_by_team(sched, "2026-10-05", "2026-10-11", "20262027")
    assert counts["TOR"] == 3
    assert counts["MTL"] == 2
    assert counts["EDM"] == 1
    assert "BUF" not in counts  # preseason only
    assert "VAN" not in counts  # other season


def test_games_by_team_is_the_week_lever(sched):
    """A 4-game week is double a 2-game week of counting stats; this is the
    number that decides a streaming add."""
    wk = calendar.games_by_team(sched, "2026-10-05", "2026-10-11", "20262027")
    assert wk["TOR"] > wk["EDM"]


def test_back_to_back_looks_both_ways(sched):
    # TOR plays the 5th and the 6th.
    assert calendar.back_to_back(sched, "TOR", "2026-10-06", "20262027") == (True, False)
    assert calendar.back_to_back(sched, "TOR", "2026-10-05", "20262027") == (False, True)
    assert calendar.back_to_back(sched, "TOR", "2026-10-10", "20262027") == (False, False)


def test_season_dates_span_the_schedule(sched):
    assert calendar.season_dates(sched, "20262027") == ("2026-10-05", "2026-10-10")


def test_an_unsynced_season_says_so(db):
    with pytest.raises(calendar.CalendarError, match="data sync"):
        calendar.season_dates(db, "20992100")


def test_dates_may_be_passed_as_date_objects(sched):
    from datetime import date

    assert calendar.teams_playing(sched, date(2026, 10, 5), "20262027") == {"TOR", "MTL"}
