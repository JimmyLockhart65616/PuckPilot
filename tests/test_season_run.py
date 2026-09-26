"""The one-command day, and the scheduling that runs it.

Built to be scheduled, which is what most of these pin: it reports instead of
raising, and every step is safe to repeat.
"""

from __future__ import annotations

from pathlib import Path

from puckpilot.season import schedule
from puckpilot.season.run import RunReport, _guard, starts_a_week
from puckpilot.season.settings import LeagueRuntime, Week
from tests.test_season_settings import payload


def runtime():
    weeks = (Week(1, "2026-09-29", "2026-10-04"), Week(2, "2026-10-05", "2026-10-11"))
    return LeagueRuntime.from_payload(payload(), weeks=weeks, fetched_at="now")


# -- failures are reported, not raised --------------------------------------


def test_a_step_that_throws_becomes_a_line_not_an_exit():
    """A dead Yahoo session must not also cost the lineup we could have
    computed from data already on disk."""
    report = RunReport(date="2026-10-07", manager="jimmy")

    def boom():
        raise RuntimeError("yahoo is down")

    assert _guard(report, "roster", boom) is None
    assert report.failed
    assert "yahoo is down" in report.steps[0].detail


def test_a_failure_is_visible_in_the_written_record():
    report = RunReport(date="d", manager="jimmy")
    report.add("sync", True, "fine")
    report.add("roster", False, "broken")
    assert "[FAIL] roster" in report.text
    assert "finished with failures" in report.text


def test_a_clean_run_says_done():
    report = RunReport(date="d", manager="jimmy")
    report.add("sync", True, "fine")
    assert not report.failed
    assert report.text.rstrip().endswith("done")


# -- when the weekly plan runs ----------------------------------------------


def test_the_weekly_plan_runs_on_the_day_the_week_turns_over():
    assert starts_a_week(runtime(), "2026-09-29") is True
    assert starts_a_week(runtime(), "2026-10-05") is True


def test_it_does_not_run_on_any_other_day():
    assert starts_a_week(runtime(), "2026-10-01") is False


def test_a_date_outside_the_calendar_is_not_a_week_start():
    assert starts_a_week(runtime(), "2027-06-01") is False


# -- scheduling -------------------------------------------------------------


def test_every_task_runs_the_same_idempotent_command():
    items = schedule.tasks("jimmy", Path("C:/repo/puckpilot"))
    assert len({t.command for t in items}) == 1
    assert "season run" in items[0].command
    assert "--manager jimmy" in items[0].command


def test_the_fixed_schedule_is_only_an_anchor():
    """Coverage comes from lock-timed one-shots, not from guessing times. The
    anchor exists to start that chain: exactly one game day in 185 begins
    before 11:00."""
    assert schedule.DEFAULT_TIMES == ("11:00",)
    assert len(schedule.tasks("jimmy", Path("."))) == 1


def test_lock_runs_are_one_shot_and_separately_named():
    """Daily tasks would fire at yesterday's game times forever."""
    items = schedule.lock_tasks("jimmy", Path("."), ["12:40", "19:40"])
    assert [t.time for t in items] == ["12:40", "19:40"]
    assert all(schedule.LOCK_TAG in t.name for t in items)
    assert "ONCE" in items[0].create_args(once=True)
    assert "DAILY" in items[0].create_args()


def test_a_lock_run_does_the_same_work_as_the_anchor():
    anchor = schedule.tasks("jimmy", Path("C:/repo/puckpilot"))[0]
    lock = schedule.lock_tasks("jimmy", Path("C:/repo/puckpilot"), ["12:40"])[0]
    assert anchor.command == lock.command


def test_task_names_are_recognisable_and_unique():
    items = schedule.tasks("jimmy", Path("."))
    assert all(t.name.startswith(schedule.PREFIX) for t in items)
    assert len({t.name for t in items}) == len(items)


def test_the_command_runs_from_the_repo_so_relative_paths_resolve():
    t = schedule.tasks("jimmy", Path("C:/repo/puckpilot"))[0]
    assert 'cd /d "C:' in t.command


def test_the_preview_warns_when_the_page_key_is_not_set():
    """Without it a scheduled run computes the lineup and cannot publish it."""
    items = schedule.tasks("jimmy", Path("."))
    assert "WARNING" in schedule.describe(items, key_set=False)
    assert "setx" in schedule.describe(items, key_set=False)
    assert "WARNING" not in schedule.describe(items, key_set=True)


def test_the_preview_says_nothing_is_written_to_yahoo():
    text = schedule.describe(schedule.tasks("jimmy", Path(".")), key_set=True)
    assert "Nothing is written to Yahoo" in text


def test_custom_times_are_honoured():
    items = schedule.tasks("jimmy", Path("."), times=("07:30",))
    assert len(items) == 1 and items[0].time == "07:30"


# -- staleness, which is how a scheduled job goes wrong quietly -------------


def test_a_fresh_settings_cache_is_left_alone():
    from datetime import datetime, timedelta

    from puckpilot.season.run import RUNTIME_REFRESH_DAYS, _age_days

    recent = (datetime.now() - timedelta(hours=6)).isoformat()
    assert _age_days(recent) < RUNTIME_REFRESH_DAYS


def test_an_aged_settings_cache_is_due_a_refresh():
    """`current_week` simply stops advancing while everything looks healthy."""
    from datetime import datetime, timedelta

    from puckpilot.season.run import RUNTIME_REFRESH_DAYS, _age_days

    old = (datetime.now() - timedelta(days=9)).isoformat()
    assert _age_days(old) > RUNTIME_REFRESH_DAYS


# -- when the slots actually close ------------------------------------------


def _sched(db, day="2026-10-04"):
    from puckpilot.data import store

    games = [
        (1, "2026-10-04T17:00:00Z", "DET", "TOR"),  # 1pm local
        (2, "2026-10-04T22:00:00Z", "NYR", "MTL"),  # 6pm local
        (3, "2026-10-05T00:00:00Z", "ANA", "CGY"),  # 8pm local
        (4, "2026-10-04T23:00:00Z", "BOS", "BUF"),  # 7pm, nobody of ours
    ]
    for gid, utc, home, away in games:
        store.upsert_schedule_game(
            db,
            game_id=gid,
            season="20262027",
            game_type=2,
            game_date=day,
            start_time_utc=utc,
            home_team=home,
            away_team=away,
        )
    for name, team in (("Raymond", "DET"), ("Miller", "NYR"), ("Dostal", "ANA")):
        db.execute(
            "INSERT INTO yahoo_roster_snapshots (manager, league_key, team_key, date, "
            "player_key, name, team_abbrev, selected_slot) VALUES "
            "('jimmy','l','t',?,?,?,?,'BN')",
            (day, f"p.{name}", name, team),
        )
    db.commit()
    return db


def test_each_distinct_game_time_is_its_own_lock(db):
    from puckpilot.season import locks

    _sched(db)
    teams = locks.roster_teams(db, "jimmy")
    got = locks.locks_for(db, "20262027", "2026-10-04", teams)
    assert [x.hhmm for x in got] == ["13:00", "18:00", "20:00"]


def test_a_game_with_none_of_your_players_is_not_a_lock(db):
    """A 7pm Boston game is irrelevant if you own nobody in it, and the
    difference is how many times a browser launches on a Saturday."""
    from puckpilot.season import locks

    _sched(db)
    teams = locks.roster_teams(db, "jimmy")
    got = locks.locks_for(db, "20262027", "2026-10-04", teams)
    assert all("BOS" not in x.teams and "BUF" not in x.teams for x in got)


def test_a_lock_names_the_players_it_closes(db):
    from puckpilot.season import locks

    _sched(db)
    teams = locks.roster_teams(db, "jimmy")
    first = locks.locks_for(db, "20262027", "2026-10-04", teams)[0]
    assert first.players == ("Raymond",)
    assert "1:00 PM" in first.describe()


def test_runs_are_scheduled_before_the_lock_not_on_it(db):
    from puckpilot.season import locks

    _sched(db)
    teams = locks.roster_teams(db, "jimmy")
    got = locks.locks_for(db, "20262027", "2026-10-04", teams)
    assert locks.run_times(got) == ["12:40", "17:40", "19:40"]


def test_locks_already_past_are_not_worth_running_before(db):
    from datetime import datetime
    from zoneinfo import ZoneInfo

    from puckpilot.season import locks

    _sched(db)
    teams = locks.roster_teams(db, "jimmy")
    got = locks.locks_for(db, "20262027", "2026-10-04", teams)
    evening = datetime(2026, 10, 4, 19, 0, tzinfo=ZoneInfo("America/Toronto"))
    assert [x.hhmm for x in locks.upcoming(got, now=evening)] == ["20:00"]


def test_a_day_with_none_of_your_players_playing_says_so(db):
    from puckpilot.season import locks

    _sched(db)
    teams = locks.roster_teams(db, "jimmy")
    got = locks.locks_for(db, "20262027", "2026-12-25", teams)
    assert got == []
    assert "none of your players have a game" in locks.describe("2026-12-25", got)


def test_planning_with_no_roster_snapshot_is_empty_not_a_crash(db):
    from puckpilot.season import locks

    assert locks.roster_teams(db, "nobody") == {}
