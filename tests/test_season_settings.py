"""The league's own rules, as read from Yahoo rather than transcribed.

The payload shapes here mirror a real `/league/{key}/settings` response, but the
values are invented: the repo is public and a real league key is not test data.
Assertions against the actual league live in `test_local_data.py`.
"""

from __future__ import annotations

import pytest

from puckpilot.season.settings import LeagueRuntime, SettingsError, Week


def _slots(*spec):
    return [
        {"roster_position": {"position": p, "count": c, "is_starting_position": s}}
        for p, c, s in spec
    ]


def payload(**over):
    base = {
        "league_key": "999.l.1",
        "name": "Test League",
        "num_teams": 10,
        "scoring_type": "head",
        "season": "2026",
        "start_date": "2026-09-29",
        "end_date": "2027-03-28",
        "start_week": "1",
        "end_week": "25",
        "current_week": 1,
        "current_date": "2026-09-29",
        "playoff_start_week": "23",
        "num_playoff_teams": "8",
        "weekly_deadline": "intraday",
        "roster_type": "date",
        "waiver_type": "R",
        "waiver_rule": "all",
        "waiver_time": "1",
        "uses_faab": "0",
        "max_adds": "65",
        "max_weekly_adds": "3",
        "min_games_played": "3",
        "trade_end_date": "2027-03-03",
        "roster_positions": _slots(
            ("C", 2, 1),
            ("LW", 2, 1),
            ("RW", 2, 1),
            ("D", 4, 1),
            ("Util", 1, 1),
            ("G", 2, 1),
            ("BN", 3, 0),
            ("IR", 1, 0),
            ("IR+", 3, 0),
        ),
    }
    base.update(over)
    return base


def test_reads_the_calendar_and_caps_off_yahoo():
    r = LeagueRuntime.from_payload(payload())
    assert (r.start_date, r.end_date) == ("2026-09-29", "2027-03-28")
    assert (r.start_week, r.end_week, r.current_week) == (1, 25, 1)
    assert (r.max_weekly_adds, r.max_adds, r.min_games_played) == (3, 65, 3)
    assert r.waiver_days == 1 and r.uses_faab is False


def test_yahoo_season_year_becomes_the_eight_digit_form():
    assert LeagueRuntime.from_payload(payload()).nhl_season == "20262027"


def test_regular_weeks_is_derived_not_configured():
    """22, the value the league file had to be corrected to by hand in Sept."""
    assert LeagueRuntime.from_payload(payload()).regular_weeks == 22


def test_shape_matches_the_slots_the_engines_expect():
    shape = LeagueRuntime.from_payload(payload()).shape()
    assert shape.slots == (("C", 2), ("L", 2), ("R", 2), ("D", 4), ("G", 2))
    assert (shape.util_slots, shape.bench_slots, shape.n_teams) == (1, 3, 10)


def test_ir_slots_are_counted_but_never_startable():
    r = LeagueRuntime.from_payload(payload())
    assert r.ir_slots == 4  # IR + IR+ x3
    assert "IR" not in dict(r.shape().slots)


def test_daily_lock_is_read_not_assumed():
    assert LeagueRuntime.from_payload(payload()).is_daily_lineup is True
    weekly = LeagueRuntime.from_payload(payload(weekly_deadline="1", roster_type="week"))
    assert weekly.is_daily_lineup is False


def test_an_uncapped_league_keeps_none_rather_than_zero():
    """None means 'no cap' to waivers.budget_threshold; 0 would mean 'spend nothing'."""
    r = LeagueRuntime.from_payload(payload(max_adds="", max_weekly_adds=""))
    assert r.max_adds is None and r.max_weekly_adds is None


def test_missing_required_settings_raise_rather_than_default():
    bad = payload()
    del bad["start_date"]
    with pytest.raises(SettingsError, match="start_date"):
        LeagueRuntime.from_payload(bad)


def test_a_payload_without_roster_positions_is_refused():
    with pytest.raises(SettingsError, match="roster_positions"):
        LeagueRuntime.from_payload(payload(roster_positions=[]))


# -- the week calendar ------------------------------------------------------
#
# Yahoo says 2026-09-29 -> 2027-03-28 over 25 weeks. Monday-Sunday arithmetic
# from the start date ends on 2027-03-21, a week short, so at least one week is
# longer than seven days. These pin that the calendar is never invented.


def test_week_lookup_refuses_to_guess_when_the_calendar_is_absent():
    r = LeagueRuntime.from_payload(payload())
    assert r.weeks == ()
    with pytest.raises(SettingsError, match="must not be computed"):
        r.week_of("2026-10-07")


def test_week_lookup_uses_the_fetched_boundaries():
    weeks = (
        Week(1, "2026-09-29", "2026-10-04"),
        Week(2, "2026-10-05", "2026-10-11"),
        Week(3, "2026-10-12", "2026-10-25"),  # a long week, as the real season has
    )
    r = LeagueRuntime.from_payload(payload(), weeks=weeks)
    assert r.week_of("2026-09-29") == 1
    assert r.week_of("2026-10-04") == 1
    assert r.week_of("2026-10-05") == 2
    assert r.week_of("2026-10-20") == 3
    assert len(r.week(3).dates()) == 14


def test_a_date_outside_the_season_is_an_error_not_a_clamp():
    r = LeagueRuntime.from_payload(payload(), weeks=(Week(1, "2026-09-29", "2026-10-04"),))
    with pytest.raises(SettingsError, match="outside weeks"):
        r.week_of("2027-06-01")
