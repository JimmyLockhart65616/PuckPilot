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


def test_there_is_more_than_one_run_a_day():
    """A player locks when his own game starts, so a Saturday matinee locks at
    one o'clock. A single evening run is too late for every afternoon game."""
    assert len(schedule.tasks("jimmy", Path("."))) >= 3


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
