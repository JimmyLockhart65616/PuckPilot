"""The in-season readiness gate, and the incremental rule the daily sync uses.

The failure this is built around is staleness rather than breakage: a roster
read last week, a map built before the call-ups, a season nobody synced. Each
produces confident, wrong advice instead of an error, which is why they are
checks rather than exceptions.
"""

from __future__ import annotations

import json

from puckpilot.data import store
from puckpilot.data.sync import players_behind_their_boxscores
from puckpilot.season import preflight as pf
from puckpilot.season.preflight import FAIL, PASS, WARN
from puckpilot.season.roster import RosterPlayer, TeamRoster
from puckpilot.season.settings import LeagueRuntime, Week
from tests.conftest import add_player, add_skater_game
from tests.test_season_settings import payload

SEASON = "20262027"


def runtime(**over):
    weeks = over.pop("weeks", (Week(1, "2026-09-29", "2026-10-04"),))
    return LeagueRuntime.from_payload(payload(**over), weeks=weeks, fetched_at="2026-09-28")


# -- the daily sync's incremental rule --------------------------------------


def box(conn, pid, game_id, season, toi="18:00"):
    conn.execute(
        "INSERT OR REPLACE INTO nhl_boxscore_stats "
        "(game_id, player_id, season, team_abbrev, stats_json) VALUES (?, ?, ?, ?, ?)",
        (game_id, pid, season, "TOR", json.dumps({"playerId": pid, "toi": toi})),
    )
    conn.commit()


def test_a_player_whose_boxscore_is_ahead_needs_his_log(db):
    add_player(db, 1, "Played", "C")
    box(db, 1, 100, SEASON)
    assert players_behind_their_boxscores(db, SEASON) == [1]


def test_a_player_whose_log_caught_up_is_not_refetched(db):
    add_player(db, 1, "Played", "C")
    add_skater_game(db, 1, SEASON, 100, date="2026-10-01")
    box(db, 1, 100, SEASON)
    assert players_behind_their_boxscores(db, SEASON) == []


def test_a_backup_who_dressed_without_playing_is_not_chased_forever(db):
    """He gets a boxscore row every night he sits and a game-log entry only
    when he appears. Without the ice-time filter every backup in the league
    reads as permanently behind - 122 of them on a fully synced season."""
    add_player(db, 2, "Backup", "G")
    box(db, 2, 100, SEASON, toi="00:00")
    assert players_behind_their_boxscores(db, SEASON) == []


def test_only_the_season_asked_for_is_considered(db):
    add_player(db, 1, "Played", "C")
    box(db, 1, 100, "20252026")
    assert players_behind_their_boxscores(db, SEASON) == []


def test_the_fetch_list_can_be_capped(db):
    for pid in range(1, 6):
        add_player(db, pid, f"P{pid}", "C")
        box(db, pid, 100 + pid, SEASON)
    assert len(players_behind_their_boxscores(db, SEASON, limit=2)) == 2


# -- the checks -------------------------------------------------------------


def test_missing_settings_is_a_failure_not_a_warning():
    """Without them there is no week calendar, and every weekly number is wrong."""
    c = pf.check_runtime(None)
    assert c.status == FAIL
    assert "season settings --refresh" in c.detail


def test_a_calendar_that_never_got_fetched_fails():
    c = pf.check_calendar(runtime(weeks=()), "2026-09-29")
    assert c.status == FAIL
    assert "never computed" in c.detail


def test_a_date_outside_the_known_weeks_fails_rather_than_guesses():
    c = pf.check_calendar(runtime(), "2027-03-20")
    assert c.status == FAIL
    assert "not covered" in c.detail


def test_a_short_or_long_week_is_called_out():
    c = pf.check_calendar(runtime(), "2026-09-29")
    assert c.status == PASS
    assert any("other than seven days" in ln for ln in c.lines)


def test_an_unfetched_playoff_calendar_is_reported_but_not_fatal():
    c = pf.check_calendar(runtime(), "2026-09-29")
    assert c.status == PASS
    assert any("playoff matchups are not scheduled yet" in ln for ln in c.lines)


def test_a_missing_player_map_fails_with_the_command_to_fix_it(db):
    c = pf.check_player_map(db, "999.l.1")
    assert c.status == FAIL
    assert "yahoo playermap" in c.detail


def test_a_week_old_player_map_warns(db):
    db.execute(
        "INSERT INTO yahoo_player_map (player_key, league_key, full_name, nhl_player_id, "
        "updated_at) VALUES ('p.1','999.l.1','A',1,'2026-09-01 00:00:00')"
    )
    db.commit()
    c = pf.check_player_map(db, "999.l.1")
    assert c.status == WARN
    assert "call-ups" in c.detail


def test_an_unmapped_rostered_player_fails(db):
    r = TeamRoster(
        league_key="999.l.1",
        team_key="999.l.1.t.5",
        date="2026-09-29",
        players=(),
        unmapped=("Some Rookie",),
    )
    c = pf.check_roster(r)
    assert c.status == FAIL
    assert "Some Rookie" in c.detail


def test_an_unsynced_schedule_fails(db):
    assert pf.check_schedule(db, SEASON, "2026-09-29").status == FAIL


def test_last_nights_missing_games_fail(db):
    store.upsert_schedule_game(
        db,
        game_id=1,
        season=SEASON,
        game_type=2,
        game_date="2026-09-29",
        start_time_utc="2026-09-29T23:00:00Z",
        home_team="TOR",
        away_team="MTL",
    )
    db.commit()
    c = pf.check_data_freshness(db, SEASON, "2026-09-30")
    assert c.status == FAIL
    assert "data daily" in c.detail


def test_a_fully_synced_yesterday_passes(db):
    store.upsert_schedule_game(
        db,
        game_id=1,
        season=SEASON,
        game_type=2,
        game_date="2026-09-29",
        start_time_utc=None,
        home_team="TOR",
        away_team="MTL",
    )
    add_player(db, 1, "P", "C")
    add_skater_game(db, 1, SEASON, 1, date="2026-09-29")
    box(db, 1, 1, SEASON)
    c = pf.check_data_freshness(db, SEASON, "2026-09-30")
    assert c.status == PASS


def test_an_unpriced_rostered_player_warns():
    class Values:
        scale_season = "20252026"
        proj_pg = {1: 1.0}

        def knows(self, pid):
            return pid in self.proj_pg

    p = RosterPlayer(
        player_key="p.2",
        yahoo_id="2",
        name="Call Up",
        team="TOR",
        primary_position="C",
        yahoo_eligible=frozenset({"C"}),
        selected_slot="BN",
        nhl_player_id=2,
    )
    r = TeamRoster(league_key="l", team_key="t", date="d", players=(p,))
    c = pf.check_projections(Values(), r)
    assert c.status == WARN
    assert "Call Up" in c.detail


def test_the_report_is_not_ready_when_anything_fails():
    rep = pf.SeasonPreflightReport(checks=[pf.Check("x", FAIL, "broken")])
    assert rep.failed
    assert "NOT READY" in rep.text


def test_the_report_is_ready_on_warnings_alone():
    rep = pf.SeasonPreflightReport(checks=[pf.Check("x", WARN, "meh"), pf.Check("y", PASS, "ok")])
    assert not rep.failed
    assert "READY" in rep.text


# -- how old is that, really ------------------------------------------------


def test_a_naive_timestamp_is_read_as_utc_not_local():
    """The schema defaults to SQLite's `datetime('now')`, which is UTC. Reading
    it as local time made a map written minutes ago report "-0.2 days old"."""
    from datetime import UTC, datetime, timedelta

    from puckpilot.season.preflight import age_days

    just_now = datetime.now(UTC).replace(tzinfo=None).isoformat(sep=" ", timespec="seconds")
    assert age_days(just_now) is not None
    assert -0.01 < age_days(just_now) < 0.01

    a_week = (datetime.now(UTC) - timedelta(days=7)).replace(tzinfo=None).isoformat(sep=" ")
    assert 6.9 < age_days(a_week) < 7.1


def test_an_offset_aware_timestamp_is_respected():
    from datetime import UTC, datetime, timedelta

    from puckpilot.season.preflight import age_days

    stamp = (datetime.now(UTC) - timedelta(days=2)).isoformat()
    assert 1.9 < age_days(stamp) < 2.1


def test_an_unreadable_timestamp_is_none_rather_than_a_crash():
    from puckpilot.season.preflight import age_days

    assert age_days("not a date") is None
    assert age_days("") is None
    assert age_days(None) is None
