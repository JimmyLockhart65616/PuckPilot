"""The harness that scores the live daily path against the validated one.

The measurement itself needs a real season and lives behind
`ppilot lineup verify`; these pin the pieces that make the comparison fair,
because a harness that quietly measures the wrong thing is worse than none.
"""

from __future__ import annotations

from puckpilot.data import store
from puckpilot.league import LeagueConfig
from puckpilot.season.replay import runtime_for_replay, team_by_day
from tests.conftest import add_player, add_skater_game


def test_replay_runtime_reproduces_the_league_shape():
    league = LeagueConfig()
    rt = runtime_for_replay(league, "20252026", ["2025-10-06", "2026-04-01"])
    assert rt.shape().slots == league.shape.slots
    assert rt.shape().util_slots == league.shape.util_slots
    assert rt.shape().bench_slots == league.shape.bench_slots
    assert rt.nhl_season == "20252026"


def test_replay_runtime_covers_every_date_with_a_week():
    rt = runtime_for_replay(LeagueConfig(), "20252026", ["2025-10-06", "2026-04-01"])
    assert rt.week_of("2025-10-06") == 1
    assert rt.week_of("2026-04-01") == rt.weeks[-1].number
    # Monday-Sunday here, which the docstring says is wrong for a live league
    # and harmless for a replay.
    assert rt.weeks[0].start == "2025-10-06"


def test_replay_runtime_marks_util_startable_and_bench_not():
    rt = runtime_for_replay(LeagueConfig(), "20252026", ["2025-10-06"])
    by_pos = {s.position: s for s in rt.slots}
    assert by_pos["Util"].starting is True
    assert by_pos["BN"].starting is False


def test_team_by_day_follows_a_trade_rather_than_the_current_club(db):
    """`nhl_players.team_abbrev` holds current clubs, so for a past season it
    would put a traded player on the wrong team for the whole year."""
    add_player(db, 1, "Traded", "C", team="ZZZ")  # 'current' club, deliberately wrong
    add_skater_game(db, 1, "20252026", 10, date="2025-10-06")
    add_skater_game(db, 1, "20252026", 11, date="2025-12-20")
    db.execute("UPDATE nhl_game_logs SET team_abbrev='TOR' WHERE game_id=10")
    db.execute("UPDATE nhl_game_logs SET team_abbrev='MTL' WHERE game_id=11")
    db.commit()

    class Data:
        dates = ["2025-10-06", "2025-11-01", "2025-12-20", "2026-01-05"]

    seq = team_by_day(db, "20252026", Data(), {1})[1]
    assert seq == ["TOR", "TOR", "MTL", "MTL"]
    assert "ZZZ" not in seq


def test_a_player_with_no_logs_gets_no_team_rather_than_a_guess(db):
    add_player(db, 2, "Never Played", "C", team="TOR")
    db.commit()

    class Data:
        dates = ["2025-10-06", "2025-10-07"]

    assert team_by_day(db, "20252026", Data(), {2})[2] == ["", ""]


def test_pre_debut_days_use_the_first_known_team(db):
    """Same convention as `skater_availability`: before his first game, a
    player is treated as being on the team he debuts for."""
    add_player(db, 3, "Rookie", "C")
    add_skater_game(db, 3, "20252026", 20, date="2025-12-01")
    db.execute("UPDATE nhl_game_logs SET team_abbrev='OTT' WHERE game_id=20")
    db.commit()

    class Data:
        dates = ["2025-10-06", "2025-12-01"]

    assert team_by_day(db, "20252026", Data(), {3})[3] == ["OTT", "OTT"]


def test_non_scoring_slots_are_the_ones_that_bank_nothing():
    from puckpilot.season.replay import NON_SCORING

    assert {"BN", "IR", "IR+"} <= NON_SCORING


def test_store_has_the_tables_the_harness_writes_to(db):
    assert {"yahoo_roster_snapshots", "season_actions"} <= store.table_names(db)
