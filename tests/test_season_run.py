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
    anchor starts that chain - before every game day's first puck drop - and is
    the morning read, done before a manager checks the scores at breakfast."""
    assert schedule.DEFAULT_TIMES == ("07:00",)
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
    repo = Path("C:/repo/puckpilot")
    t = schedule.tasks("jimmy", repo)[0]
    assert f'cd /d "{repo}"' in t.command


def test_the_preview_warns_when_the_page_key_is_not_set():
    """Without it a scheduled run computes the lineup and cannot publish it."""
    items = schedule.tasks("jimmy", Path("."))
    assert "WARNING" in schedule.describe(items, key_set=False)
    assert "setx" in schedule.describe(items, key_set=False)
    assert "WARNING" not in schedule.describe(items, key_set=True)


def test_the_preview_says_what_is_and_is_not_written_to_yahoo():
    text = schedule.describe(schedule.tasks("jimmy", Path(".")), key_set=True)
    assert "No add, drop or claim is ever made" in text
    assert "only under" in text and "standing authority" in text


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


def _snap(db, team_key, date, *names):
    for name, club in names:
        db.execute(
            "INSERT INTO yahoo_roster_snapshots (manager, league_key, team_key, date, "
            "player_key, name, team_abbrev, selected_slot) VALUES "
            "('jimmy','l',?,?,?,?,?,'BN')",
            (team_key, date, f"p.{team_key}.{name}", name, club),
        )
    db.commit()


def test_a_roster_read_for_a_future_date_does_not_plan_today(db):
    """The real case: a 10-07 read taken on 09-21 was MAX(date) for days, so
    every lock was planned from a stale roster."""
    from puckpilot.season import locks

    _snap(db, "t", "2026-09-26", ("Current", "DET"))
    _snap(db, "t", "2026-10-07", ("Stale", "NYR"))
    assert locks.roster_teams(db, "jimmy", "t", "2026-09-26") == {"DET": ["Current"]}


def test_the_opponents_roster_never_leaks_into_our_locks(db):
    """Every run now saves the opponent's roster under the same manager."""
    from puckpilot.season import locks

    _snap(db, "t.5", "2026-09-29", ("Ours", "DET"))
    _snap(db, "t.5", "2026-09-30", ("Ours", "DET"))
    _snap(db, "t.11", "2026-09-30", ("Theirs", "BOS"))
    assert locks.roster_teams(db, "jimmy", "t.5", "2026-09-30") == {"DET": ["Ours"]}
    # Unnamed, our team is the one with the most snapshots.
    assert locks.roster_teams(db, "jimmy", day="2026-09-30") == {"DET": ["Ours"]}


# -- the week, on every run -------------------------------------------------


def _rt(weeks):
    from puckpilot.season.settings import LeagueRuntime, Week
    from tests.test_season_settings import payload

    return LeagueRuntime.from_payload(
        payload(), weeks=tuple(Week(*w) for w in weeks), fetched_at="now"
    )


def test_before_the_season_this_week_is_the_first_one():
    from puckpilot.season.run import week_for

    rt = _rt([(1, "2026-09-29", "2026-10-04"), (2, "2026-10-05", "2026-10-11")])
    assert week_for(rt, "2026-09-26").number == 1
    assert week_for(rt, "2026-10-06").number == 2
    assert week_for(rt, "2027-06-01") is None


def test_adds_used_come_from_yahoo_and_only_for_this_week():
    from puckpilot.season.roster import TeamRoster
    from puckpilot.season.run import adds_used
    from puckpilot.season.settings import Week

    r = TeamRoster(
        league_key="l",
        team_key="t",
        date="d",
        players=(),
        adds_this_week=2,
        adds_week=3,
        moves_season=11,
    )
    assert adds_used(r, Week(3, "a", "b")) == (2, 11)
    assert adds_used(r, Week(4, "a", "b")) == (0, 11)
    assert adds_used(None, Week(3, "a", "b")) == (0, 0)


def test_a_game_already_under_way_is_banked_not_projected(db):
    from datetime import UTC, datetime

    from puckpilot.data import store
    from puckpilot.season.run import started_clubs

    for gid, (utc, home, away) in enumerate(
        (("2026-10-04T17:00:00Z", "DET", "NYR"), ("2026-10-04T23:00:00Z", "ANA", "BOS")), 1
    ):
        store.upsert_schedule_game(
            db,
            game_id=gid,
            season="20262027",
            game_type=2,
            game_date="2026-10-04",
            start_time_utc=utc,
            home_team=home,
            away_team=away,
        )
    db.commit()
    at_six = datetime(2026, 10, 4, 22, 0, tzinfo=UTC)
    assert started_clubs(db, "20262027", "2026-10-04", now=at_six) == {"DET", "NYR"}


# -- when the next run is due, for the page's freshness line ------------------


def test_the_next_run_is_the_next_lock_run_or_the_next_anchor(db):
    from datetime import datetime
    from zoneinfo import ZoneInfo

    from puckpilot.season import locks

    _sched(db)  # 2026-10-04 locks at 13:00, 18:00, 20:00 for this roster
    tz = ZoneInfo("America/Toronto")

    def at(h, m, day=4):
        return datetime(2026, 10, day, h, m, tzinfo=tz)

    def nxt(now, day="2026-10-04"):
        return locks.next_run(db, "jimmy", "20262027", day, now=now)

    assert nxt(at(6, 0)) == at(7, 0)  # today's anchor comes first
    assert nxt(at(9, 0)) == at(12, 40)  # then the 13:00 lock, 20 min early
    assert nxt(at(19, 30)) == at(19, 40)  # the 20:00 lock's run
    assert nxt(at(19, 45)) == at(7, 0, day=5)  # just missed it: tomorrow
    assert nxt(at(21, 0)) == at(7, 0, day=5)  # nothing left today: tomorrow 07:00
    # A day with no games for this roster: just the anchors.
    assert nxt(at(12, 0, day=6), "2026-10-06") == at(7, 0, day=7)


# -- making the changes ---------------------------------------------------------


def _act_plan(within=True, ir_within=True, moves=("START A in LW",), ir_moves=()):
    from types import SimpleNamespace

    def mv(text):
        return SimpleNamespace(describe=lambda: text)

    return SimpleNamespace(
        date="2026-10-10",
        within_authority=within,
        ir_within_authority=ir_within,
        moves=[mv(t) for t in moves],
        ir_moves=[mv(t) for t in ir_moves],
    )


def _act(db, plan, apply=None):
    from types import SimpleNamespace

    from puckpilot.season.run import RunReport, act

    report = RunReport(date=plan.date, manager="jimmy")
    manager = SimpleNamespace(name="jimmy")
    roster = SimpleNamespace(team_key="999.l.1.t.5")
    return act(db, manager, "999.l.1", roster, plan, report, apply=apply), report


class _Applied:
    def __init__(self, ok=True, message="made 1 change(s)", lines=()):
        self.calls = []
        self.ok, self.message, self.lines = ok, message, list(lines)

    def __call__(self, manager, team_key, date, phases):
        self.calls.append([[m.describe() for m in phase] for phase in phases])
        return self


def test_changes_are_made_ir_first_and_recorded(db):
    from puckpilot.season import proposals

    fake = _Applied(lines=["A BN -> LW, B LW -> BN"])
    got, report = _act(db, _act_plan(ir_moves=("IR    S D -> IR+",)), apply=fake)
    assert fake.calls == [[["IR    S D -> IR+"], ["START A in LW"]]]
    assert got["ok"] and "made 1 change" in got["message"]
    [row] = proposals.actions(db, "jimmy")
    assert row["outcome"] == "executed" and row["kind"] == "lineup"
    assert "B LW -> BN" in row["detail_json"]
    assert not report.failed


def test_nothing_is_made_without_standing_authority(db):
    fake = _Applied()
    got, _ = _act(db, _act_plan(within=False, ir_within=False), apply=fake)
    assert got is None and fake.calls == []


def test_a_failed_change_is_loud_and_recorded(db):
    from puckpilot.season import proposals

    fake = _Applied(ok=False, message="the lineup changed since it was read - left alone")
    got, report = _act(db, _act_plan(), apply=fake)
    assert got["ok"] is False
    assert report.failed
    assert proposals.actions(db, "jimmy")[0]["outcome"] == "failed"


def test_without_an_actuator_the_page_says_the_changes_were_not_made(db):
    """A published clone has none: "will act automatically" must not stand in
    for a change nobody made."""
    got, report = _act(db, _act_plan())
    assert got == {"ok": False, "message": "Not made automatically - make these in Yahoo."}
    assert not report.failed


# -- re-checking the queue ------------------------------------------------------


def test_the_recheck_keeps_what_pays_and_says_why_the_rest_went(db):
    from dataclasses import replace
    from types import SimpleNamespace

    from puckpilot.season import proposals
    from puckpilot.season.run import _recheck
    from tests.test_season_proposals import make, target

    [a] = make(db, target(name="Rossi", key="p.a", pid=9101))
    [b] = make(db, target(name="Durzi", key="p.b", pid=9102))
    keep = replace(target(name="Rossi", key="p.a", pid=9101), gain=0.41, score=0.41)
    low = replace(target(name="Durzi", key="p.b", pid=9102), gain=0.03, score=0.03)
    plan = SimpleNamespace(targets=(keep,), lapsed=(low,), adds_left_week=2)
    terms = SimpleNamespace(add_scoring="odds", min_expected_gain=0.1, min_weekly_gain=0.25)
    lines = _recheck(db, [(a, None, None), (b, None, None)], {}, plan, terms)
    assert [p.id for p in proposals.pending(db, "jimmy")] == [a.id]
    why = proposals.get(db, b.id).reason["withdrawn"]
    assert why == "now worth +0.03 categories expected, below the 0.10 floor"
    assert any("still +0.41" in ln for ln in lines)


def test_with_no_acquisitions_left_this_week_everything_waiting_is_withdrawn(db):
    from types import SimpleNamespace

    from puckpilot.season import proposals
    from puckpilot.season.run import _recheck
    from tests.test_season_proposals import make, target

    [a] = make(db, target())
    plan = SimpleNamespace(targets=(target(),), lapsed=(), adds_left_week=0)
    terms = SimpleNamespace(add_scoring="odds", min_expected_gain=0.1, min_weekly_gain=0.25)
    _recheck(db, [(a, None, None)], {}, plan, terms)
    assert proposals.get(db, a.id).reason["withdrawn"] == "no acquisitions left this week"


def test_a_player_taken_or_a_drop_ruled_out_ends_the_proposal(db, monkeypatch):
    from types import SimpleNamespace

    from puckpilot.season import cli_support
    from puckpilot.season import pool as pool_mod
    from puckpilot.season.pool import FREE_AGENT, PoolPlayer
    from puckpilot.season.run import _workable
    from tests.test_season_proposals import make, target
    from tests.test_season_week import _rp

    [taken] = make(db, target(name="Taken", key="p.t", pid=9101))
    [hurt] = make(db, target(name="Fine", key="p.f", pid=9102))
    [ok] = make(db, target(name="Good", key="p.g", pid=9103))

    def fa(key, name, own):
        return PoolPlayer(key, name, "TOR", "C", frozenset({"C"}), 1, ownership_type=own)

    now = [fa("p.t", "Taken", "team"), fa("p.f", "Fine", FREE_AGENT), fa("p.g", "Good", FREE_AGENT)]
    monkeypatch.setattr(cli_support, "run_session", lambda m, fn: fn(None))
    monkeypatch.setattr(pool_mod, "fetch_players", lambda s, lk, keys, player_map=None: now)
    # "Drop Me" is on the roster, and out - never a drop on that.
    roster = SimpleNamespace(players=[_rp("p.drop", "Drop Me", 8001, "MTL", "C", status="NA")])
    workable, lapsed = _workable(
        db, SimpleNamespace(), "999.l.1", SimpleNamespace(roster=roster), [taken, hurt, ok]
    )
    assert lapsed[taken.id] == "Taken is no longer available"
    assert lapsed[hurt.id].startswith("Drop Me is now NA")
    assert workable == []


# -- the week's last day looks at the next one ---------------------------------------


def test_the_last_day_of_a_week_prices_adds_against_the_next():
    from puckpilot.season.run import preloads

    rt = _rt([(1, "2026-09-29", "2026-10-04"), (2, "2026-10-05", "2026-10-11")])
    assert preloads(rt, "2026-10-04", 1)  # Sunday
    assert not preloads(rt, "2026-10-03", 1)  # Saturday, by default
    assert preloads(rt, "2026-10-03", 2)  # if asked for two days
    assert not preloads(rt, "2026-10-04", 0)  # switched off


def test_the_last_week_of_the_calendar_has_nothing_to_prepare_for():
    from puckpilot.season.run import preloads

    rt = _rt([(1, "2026-09-29", "2026-10-04"), (2, "2026-10-05", "2026-10-11")])
    assert not preloads(rt, "2026-10-11", 1)


def test_the_look_ahead_is_on_and_the_measured_worse_search_is_off():
    import pytest

    from puckpilot.season.authority import AuthorityError, TransactionAuthority

    terms = TransactionAuthority()
    assert terms.look_ahead_days == 1 and terms.preload_days == 0
    with pytest.raises(AuthorityError, match="preload_days"):
        TransactionAuthority(preload_days=7)
    with pytest.raises(AuthorityError, match="look_ahead_days"):
        TransactionAuthority(look_ahead_days=-1)


def _both(db, now_gain, next_gain, playing_today=frozenset()):
    from dataclasses import replace
    from types import SimpleNamespace

    from puckpilot.season.run import _recheck_both
    from tests.test_season_proposals import make, target

    [p] = make(db, target(name="Matty Beniers", key="p.b", pid=9102))
    t = target(name="Matty Beniers", key="p.b", pid=9102)
    now = SimpleNamespace(
        targets=(replace(t, gain=now_gain, score=now_gain),), lapsed=(), adds_left_week=3
    )
    nxt = SimpleNamespace(
        targets=(replace(t, gain=next_gain, score=next_gain),), lapsed=(), adds_left_week=3
    )
    terms = SimpleNamespace(add_scoring="odds", min_expected_gain=0.1, min_weekly_gain=0.25)
    lines = _recheck_both(
        db, [(p, None, t.drop)], {}, now, nxt, terms, playing_today=frozenset(playing_today)
    )
    return p, lines


def test_a_move_whose_drop_plays_today_is_queued_and_judged_on_next_week(db):
    """Yahoo will not drop a player who has played that day (Error #174), and
    dropping him before his game gives it away: the move waits for the unlock,
    so today's value does not count - only next week's."""
    from puckpilot.season import proposals

    # "Drop Me" plays for MTL, and MTL plays today.
    p, _ = _both(db, 0.40, 0.05, playing_today={"MTL"})
    assert (
        proposals.get(db, p.id)
        .reason["withdrawn"]
        .startswith("+0.05 next week (queued - Drop Me plays today)")
    )
    p2, _ = _both(db, 0.0, 0.30, playing_today={"MTL"})
    [live] = proposals.pending(db, "jimmy")
    assert "queued - make it after tonight's games, once rosters unlock" in live.reason["timing"]


def test_a_late_add_that_hurts_next_week_is_withdrawn_with_both_numbers(db):
    """#10 on 2026-10-04: +0.25 for the day left, -0.32 the week after."""
    from puckpilot.season import proposals

    p, _ = _both(db, 0.25, -0.32)
    assert proposals.pending(db, "jimmy") == []
    assert proposals.get(db, p.id).reason["withdrawn"] == (
        "+0.25 for the rest of this week, -0.32 next week - -0.07 together, below the 0.10 floor"
    )


def test_a_late_add_that_pays_over_both_weeks_is_kept_and_says_so(db):
    from puckpilot.season import proposals

    p, lines = _both(db, 0.05, 0.20)
    [live] = proposals.pending(db, "jimmy")
    assert live.reason["detail"]["week"][0].startswith("Next week too: +0.20 categories expected")
    assert any("kept" in ln for ln in lines)


# -- mid-week swaps -----------------------------------------------------------------------


def test_mid_week_swaps_are_off_until_a_manager_turns_them_on():
    import pytest

    from puckpilot.season.authority import AuthorityError, TransactionAuthority

    terms = TransactionAuthority()
    assert terms.mid_week_floor is None and terms.mid_week_horizon
    with pytest.raises(AuthorityError, match="mid_week_floor"):
        TransactionAuthority(mid_week_floor=-0.1)


def test_a_mid_week_swap_must_pay_over_this_week_and_next(monkeypatch):
    """Priced over the days left it may clear the bar; priced over next week
    too it may not - a pickup who helps until Sunday and costs the week after."""
    from dataclasses import replace
    from types import SimpleNamespace

    import puckpilot.draft.sim as sim
    import puckpilot.season.week as weekmod
    from puckpilot.season.run import _pays_both
    from tests.test_season_proposals import target

    good = replace(target(name="Lasts", key="p.1", pid=1), gain=0.20, score=0.20)
    brief = replace(target(name="Brief", key="p.2", pid=2), gain=0.30, score=0.30)
    later = {
        "p.1": replace(good, gain=0.10, score=0.10),
        "p.2": replace(brief, gain=-0.20, score=-0.20),
    }
    monkeypatch.setattr(sim, "build_universe", lambda *a, **k: SimpleNamespace(frame=None))
    monkeypatch.setattr(
        weekmod,
        "build_week_plan",
        lambda *a, reprice=(), **k: SimpleNamespace(
            targets=tuple(later[p.player_key] for p, _ in reprice), lapsed=()
        ),
    )
    manager = SimpleNamespace(
        league=None,
        authority=SimpleNamespace(
            transactions=SimpleNamespace(add_scoring="odds", playoff_reserve=6, stream_spots=2)
        ),
    )
    runtime = SimpleNamespace(nhl_season="20262027")
    kept, notes = _pays_both(
        None, runtime, manager, None, (None, "Rival FC", None), [good, brief], 0.25, (None, None)
    )
    assert [t.player.name for t in kept] == ["Lasts"]
    assert kept[0].detail["week"][0].startswith("Next week too: +0.10 categories expected")
    assert notes == [
        "not proposed: Brief for Drop Me - +0.30 this week but -0.20 next week, "
        "+0.10 together, below 0.25"
    ]
