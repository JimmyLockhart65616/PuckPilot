"""The starting-goalie ladder.

Measured accuracy of `TrailingStartShareSource` on real seasons, same protocol
in all three (60 sampled dates, first 21 days skipped so history exists):

    2025-26   61.5%   Brier 0.2174
    2024-25   64.8%   Brier 0.2061
    2023-24   62.4%   Brier 0.2103

That is the floor, and it is roughly 30 points below the 90% "noisy
announcement" the bench-regret replay assumed. These tests pin the model's
behaviour, not that number; `test_local_data.py` re-measures it against the
real database.
"""

from __future__ import annotations

from puckpilot.data import store
from puckpilot.season.goalies import (
    ChainedGoalieSource,
    StaticGoalieSource,
    TrailingStartShareSource,
)
from tests.conftest import add_goalie_game, add_player

SEASON = "20262027"


def sched(conn, gid, date, home, away, season=SEASON):
    store.upsert_schedule_game(
        conn,
        game_id=gid,
        season=season,
        game_type=2,
        game_date=date,
        start_time_utc=None,
        home_team=home,
        away_team=away,
    )


def started(conn, pid, gid, date, team, season=SEASON):
    add_goalie_game(conn, pid, season, gid, date=date, started=1)
    conn.execute(
        "UPDATE nhl_game_logs SET team_abbrev = ? WHERE player_id = ? AND game_id = ?",
        (team, pid, gid),
    )


def test_a_team_that_does_not_play_has_no_starter(db):
    add_player(db, 1, "Starter", "G")
    sched(db, 100, "2026-10-05", "TOR", "MTL")
    started(db, 1, 100, "2026-10-05", "TOR")
    db.commit()
    src = TrailingStartShareSource(db, SEASON)
    assert src.starts("2026-10-09") == {}


def test_a_lone_starter_is_near_certain(db):
    add_player(db, 1, "Workhorse", "G")
    for i, d in enumerate(["2026-10-01", "2026-10-03", "2026-10-05"]):
        sched(db, 100 + i, d, "TOR", "MTL")
        started(db, 1, 100 + i, d, "TOR")
    sched(db, 200, "2026-10-08", "TOR", "OTT")
    db.commit()
    p = TrailingStartShareSource(db, SEASON).starts("2026-10-08")
    assert p[1] == 1.0


def test_the_previous_starter_is_demoted_not_excluded(db):
    """Tandems do not strictly alternate - starters ride - so the demotion is a
    weight, not a veto. Strict alternation measured 53% against this model's 61%."""
    add_player(db, 1, "A", "G")
    add_player(db, 2, "B", "G")
    days = ["2026-10-01", "2026-10-03", "2026-10-05", "2026-10-07"]
    for i, d in enumerate(days):
        sched(db, 100 + i, d, "TOR", "MTL")
        started(db, 1 if i % 2 == 0 else 2, 100 + i, d, "TOR")
    sched(db, 300, "2026-10-09", "TOR", "OTT")
    db.commit()
    p = TrailingStartShareSource(db, SEASON).starts("2026-10-09")
    # 2 started the previous game, so 1 is favoured - but 2 keeps real weight
    assert p[1] > p[2] > 0.0


def test_a_backup_with_one_start_in_ten_falls_below_the_floor(db):
    add_player(db, 1, "Starter", "G")
    add_player(db, 2, "Backup", "G")
    days = [f"2026-10-{d:02d}" for d in range(1, 22, 2)]
    for i, d in enumerate(days):
        sched(db, 100 + i, d, "TOR", "MTL")
        started(db, 2 if i == 0 else 1, 100 + i, d, "TOR")
    sched(db, 400, "2026-10-25", "TOR", "OTT")
    db.commit()
    p = TrailingStartShareSource(db, SEASON).starts("2026-10-25")
    assert p.get(1, 0) > 0.8
    assert 2 not in p  # one start in the window is not a lineup decision


def test_only_games_before_the_date_are_used(db):
    """No lookahead: a start tonight must not inform tonight's prediction."""
    add_player(db, 1, "A", "G")
    add_player(db, 2, "B", "G")
    sched(db, 100, "2026-10-05", "TOR", "MTL")
    started(db, 1, 100, "2026-10-05", "TOR")
    sched(db, 101, "2026-10-07", "TOR", "OTT")
    started(db, 2, 101, "2026-10-07", "TOR")  # tonight's actual starter
    db.commit()
    p = TrailingStartShareSource(db, SEASON).starts("2026-10-07")
    assert p == {1: 1.0}


def test_last_season_carries_the_opening_weeks(db):
    """In October the current season has no history; last season's share is a
    better prior than nothing."""
    add_player(db, 1, "Last Year Starter", "G")
    sched(db, 50, "2026-04-01", "TOR", "MTL", season="20252026")
    started(db, 1, 50, "2026-04-01", "TOR", season="20252026")
    sched(db, 100, "2026-09-29", "TOR", "OTT")
    db.commit()
    assert TrailingStartShareSource(db, SEASON).starts("2026-09-29") == {}
    with_fb = TrailingStartShareSource(db, SEASON, fallback_season="20252026")
    assert with_fb.starts("2026-09-29") == {1: 1.0}


def test_relief_appearances_are_not_starts(db):
    add_player(db, 1, "Reliever", "G")
    sched(db, 100, "2026-10-05", "TOR", "MTL")
    add_goalie_game(db, 1, SEASON, 100, date="2026-10-05", started=0)
    sched(db, 101, "2026-10-07", "TOR", "OTT")
    db.commit()
    assert TrailingStartShareSource(db, SEASON).starts("2026-10-07") == {}


# -- the chain --------------------------------------------------------------


def test_an_earlier_source_overrides_a_later_one():
    announced = StaticGoalieSource({"2026-10-07": {1: 1.0}})
    model = StaticGoalieSource({"2026-10-07": {1: 0.55, 2: 0.45}})
    chained = ChainedGoalieSource(announced, model)
    p = chained.starts("2026-10-07")
    assert p[1] == 1.0  # the announcement wins
    assert p[2] == 0.45  # the model still answers for everyone else


def test_a_source_that_knows_nothing_today_costs_nothing():
    chained = ChainedGoalieSource(StaticGoalieSource({}), StaticGoalieSource({"d": {1: 0.6}}))
    assert chained.starts("d") == {1: 0.6}


def test_a_dead_source_degrades_to_the_floor_not_to_an_empty_lineup():
    class Broken:
        def starts(self, date):
            raise RuntimeError("feed is down")

    chained = ChainedGoalieSource(Broken(), StaticGoalieSource({"d": {1: 0.6}}))
    assert chained.starts("d") == {1: 0.6}


def test_a_frozen_source_does_not_see_games_after_its_cutoff(db):
    """Replay only: asked on Wednesday about Saturday, the trailing model would
    count Thursday's start. Frozen at Wednesday, it cannot."""
    from puckpilot.season.goalies import AsOfGoalieSource

    add_player(db, 1, "A", "G")
    add_player(db, 2, "B", "G")
    sched(db, 100, "2026-10-05", "TOR", "MTL")
    started(db, 1, 100, "2026-10-05", "TOR")
    sched(db, 101, "2026-10-08", "TOR", "OTT")
    started(db, 2, 101, "2026-10-08", "TOR")  # Thursday: after the cutoff
    sched(db, 102, "2026-10-10", "TOR", "BOS")
    db.commit()
    src = TrailingStartShareSource(db, SEASON)
    assert src.starts("2026-10-10") != {1: 1.0}  # unfrozen: Thursday counts
    frozen = AsOfGoalieSource(src, "2026-10-07")
    assert frozen.starts("2026-10-10") == {1: 1.0}
    assert frozen.starts("2026-10-09") == {}  # TOR do not play that day
