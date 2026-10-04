"""One command for a day, so nobody has to remember the order.

The pieces were right and the sequence was homework: sync last night, collect
whatever was decided on the phone, work out tonight's lineup, push it, and on
the first day of a fantasy week work out the week as well. Getting that wrong
is quiet - a stale sync just makes the advice slightly worse - which is exactly
the kind of thing a person should not be holding in their head at 6pm.

Built to be scheduled, which sets the rules:

Nothing raises. A step that fails is reported and the rest still run, because a
dead Yahoo session must not also cost you the lineup you could have computed
from data already on disk. The exit code tells a scheduler whether to care.

Everything is logged. An unattended run that leaves no trace is indistinguishable
from one that never happened.

It is safe to run repeatedly. Every step is idempotent, so running hourly costs
a little time and changes nothing twice.
"""

from __future__ import annotations

import sqlite3
import traceback
from dataclasses import dataclass, field
from datetime import UTC, datetime

from puckpilot.season import proposals as proposals_mod
from puckpilot.season.preflight import age_days as _age_days

# The league's own settings carry the week calendar and which week it is now.
# They change rarely, but a cache that never refreshes goes wrong silently:
# `current_week` simply stops advancing while everything still looks healthy.
RUNTIME_REFRESH_DAYS = 3.0

# The player map is the bridge between a Yahoo free agent and a projection, and
# the players it will be missing are exactly the ones an in-season add comes
# from: call-ups who did not exist when it was built. Refreshed weekly.
MAP_REFRESH_DAYS = 7.0


@dataclass
class Step:
    name: str
    ok: bool
    detail: str
    lines: list[str] = field(default_factory=list)


@dataclass
class RunReport:
    date: str
    manager: str
    steps: list[Step] = field(default_factory=list)

    @property
    def failed(self) -> bool:
        return any(not s.ok for s in self.steps)

    def add(self, name: str, ok: bool, detail: str, lines: list[str] | None = None) -> None:
        self.steps.append(Step(name, ok, detail, lines or []))

    @property
    def text(self) -> str:
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        out = [f"=== {self.manager} {self.date} (run {stamp}) ==="]
        for s in self.steps:
            out.append(f"[{'ok ' if s.ok else 'FAIL'}] {s.name}: {s.detail}")
            out += [f"       {ln}" for ln in s.lines]
        out.append("done" if not self.failed else "finished with failures")
        return "\n".join(out)


def _guard(report: RunReport, name: str, fn):
    """Run a step, and turn any failure into a line rather than an exit."""
    try:
        return fn()
    except Exception as e:  # noqa: BLE001 - a scheduled run reports, it does not crash
        tail = traceback.format_exc().splitlines()[-3:]
        report.add(name, False, f"{type(e).__name__}: {e}", tail)
        return None


def starts_a_week(runtime, day: str) -> bool:
    """Is `day` the first day of its fantasy week?"""
    try:
        return runtime.week(runtime.week_of(day)).start == day
    except Exception:  # noqa: BLE001
        return False


def preloads(runtime, day: str, last_days: int) -> bool:
    """Whether `day` is one of its week's last `last_days`, with a week after it.

    Then what is left of this week's acquisitions is better spent on the next
    week: they expire with this one, and a player added today plays all of the
    next.
    """
    from puckpilot.season.settings import SettingsError

    if last_days <= 0:
        return False
    try:
        w = runtime.week(runtime.week_of(day))
        runtime.week(w.number + 1)
    except SettingsError:
        return False
    return day in w.dates()[-last_days:]


def week_for(runtime, day: str):
    """The week `day` falls in, or the next one to start.

    Before the season - and between a week's end and the next's start, should
    the calendar ever have such a gap - "this week" means the one coming.
    """
    try:
        return runtime.week(runtime.week_of(day))
    except Exception:  # noqa: BLE001 - outside every known week
        later = [w for w in runtime.weeks if w.start > day]
        return min(later, key=lambda w: w.start) if later else None


@dataclass
class WeekContext:
    """What one run learned about the week, shared by every step after the read."""

    week: object | None = None
    roster: object | None = None
    theirs: object | None = None
    live: object | None = None
    raw: dict | None = None
    error: str = ""


def started_clubs(conn, season: str, day: str, now=None) -> set[str]:
    """Clubs whose game on `day` has already begun.

    Their stats are already in Yahoo's banked totals, so projecting the game
    as well would count it twice.
    """
    from datetime import UTC
    from datetime import datetime as _dt

    now = now or _dt.now(UTC)
    out: set[str] = set()
    for r in conn.execute(
        "SELECT start_time_utc, home_team, away_team FROM nhl_schedule "
        "WHERE season = ? AND game_type = 2 AND game_date = ? AND start_time_utc IS NOT NULL",
        (season, day),
    ):
        try:
            begins = _dt.fromisoformat(r["start_time_utc"].replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            continue
        if begins <= now:
            out.update((r["home_team"], r["away_team"]))
    return out


def live_inputs(conn, runtime, week, live, day: str, now=None) -> dict:
    """The banked-plus-remaining arguments for `build_week_plan`.

    Before the week starts (or with no labels to read Yahoo's stats by) nothing
    is banked and the whole week, or what is left of it, is projected.
    """
    from_day = max(day, week.start)
    started = started_clubs(conn, runtime.nhl_season, day, now) if from_day == day else set()
    out: dict = {"from_day": from_day, "started": started or None}
    out["status"] = live.status if live is not None else ""
    labels = runtime.stat_labels()
    if live is not None and live.started and labels:
        out["banked_ours"] = live.banked(labels, "ours")
        out["banked_theirs"] = live.banked(labels, "theirs")
    return out


def adds_used(roster, week) -> tuple[int, int]:
    """(this week, this season) - Yahoo's own counters, zero only when unknown."""
    wk = roster.adds_this_week if roster is not None else None
    if wk is not None and roster.adds_week not in (None, getattr(week, "number", None)):
        wk = 0  # a count for another week
    season = roster.moves_season if roster is not None else None
    return wk or 0, season or 0


def run_day(
    conn: sqlite3.Connection,
    manager,
    league_key: str,
    runtime,
    day: str,
    *,
    weekly: bool | None = None,
    do_sync: bool = True,
    propose: bool = True,
    reschedule: bool = True,
    say=print,
) -> RunReport:
    """Everything a day needs, in the order it needs it."""
    from puckpilot.season import cli_support, explain, publish, snapshot
    from puckpilot.season.fetch import discover_team_key, fetch_live, fetch_roster, save_roster
    from puckpilot.season.goalies import ChainedGoalieSource, TrailingStartShareSource
    from puckpilot.season.today import build_plan, open_roster_spots, yahoo_goalie_games
    from puckpilot.season.values import build_value_model
    from puckpilot.yahoo import playermap

    report = RunReport(date=day, manager=manager.name)
    page_key = manager.page.owner_key or _env_key()

    # 0. Keep the league's own rules current. Cheap, and the alternative is a
    # week calendar that quietly stops advancing.
    refreshed = _guard(
        report, "settings", lambda: _refresh_runtime(conn, manager, league_key, runtime, report)
    )
    runtime = refreshed or runtime
    season = runtime.nhl_season

    # 1. Last night's games, so recent form and the goalie model are current.
    if do_sync:

        def _sync():
            from puckpilot.data.nhl import NhlClient
            from puckpilot.data.sync import sync_day

            out = sync_day(conn, NhlClient(), season, progress=lambda _m: None)
            report.add(
                "sync",
                True,
                f"{out['boxscores']} boxscore(s), {out['players_synced']} player log(s)",
                [f"{out['still_behind']} still behind"] if out["still_behind"] else [],
            )

        _guard(report, "sync", _sync)

    # 2. Anything decided on the phone, before anything is proposed again.
    if manager.page.publishes and page_key:

        def _collect():
            got = publish.collect(manager.page.url, page_key)
            lines = snapshot.apply_decisions(conn, got)
            report.add("decisions", True, f"{len(got)} from your phone", lines)

        _guard(report, "decisions", _collect)

    # 3. Tonight - and the week's live score and the opponent's roster, read in
    # the same browser session. Intra-week state cannot be fetched afterwards,
    # so every run logs it.
    pmap = playermap.load_map(conn, league_key)
    week = week_for(runtime, day)
    ctx = WeekContext(week=week)

    def _read(session):
        key = manager.team_key or discover_team_key(session, league_key)
        roster = fetch_roster(session, key, day, player_map=pmap)
        if week is not None:
            try:
                ctx.live, ctx.raw = fetch_live(session, key, week.number)
                opp = ctx.live.theirs.team_key if ctx.live is not None else ""
                if opp:
                    when = day if week.contains(day) else week.start
                    ctx.theirs = fetch_roster(session, opp, when, player_map=pmap)
            except Exception as e:  # noqa: BLE001 - the lineup must not die with the score
                ctx.error = f"{type(e).__name__}: {e}"
        return roster

    roster = _guard(report, "roster", lambda: cli_support.run_session(manager, _read))
    ctx.roster = roster
    _guard(report, "score", lambda: _log_week(conn, manager, league_key, ctx, report))
    train = _train_seasons(season)
    models = _guard(
        report,
        "model",
        lambda: (
            build_value_model(conn, season, train, manager.league),
            ChainedGoalieSource(TrailingStartShareSource(conn, season, fallback_season=train[0])),
        ),
    )
    plan = None
    acted = None
    if roster is not None:
        save_roster(conn, manager.name, roster)
        report.add("roster", True, f"{len(roster)} players", list(roster.unmapped))
        _guard(report, "statuses", lambda: _statuses(conn, roster, report, "yours"))

        def _plan():
            values, goalies = models
            got = build_plan(
                conn,
                runtime,
                roster,
                values,
                goalies,
                day,
                manager=manager.name,
                authority=manager.authority.lineup,
                goalie_starts_so_far=yahoo_goalie_games(roster, runtime, day),
            )
            reasons = explain.move_reasons(conn, runtime, got)
            report.add(
                "lineup",
                True,
                f"{len(got.moves)} change(s), {got.gain:+.2f}"
                + (f", {len(got.ir_moves)} roster move(s)" if got.ir_moves else "")
                + (f", locks {got.deadline()}" if got.lock_utc else ""),
                [f"!! {a}" for a in got.ir_alerts]
                + [
                    f"{m.describe()} - {reasons.get(m.player.player_key, '')}"
                    for m in (*got.ir_moves, *got.moves)
                ],
            )
            proposals_mod.record_action(
                conn,
                manager.name,
                league_key,
                roster.team_key,
                day,
                "lineup",
                {
                    "moves": [m.describe() for m in got.moves],
                    "gain": round(got.gain, 3),
                    "reasons": reasons,
                },
                outcome="planned",
            )
            return got, reasons

        got = _guard(report, "lineup", _plan) if models else None
        plan, reasons = got if got else (None, {})
        if plan is not None:
            acted = _guard(
                report, "act", lambda: act(conn, manager, league_key, roster, plan, report)
            )

    # 4. The week: the full plan, with adds, on the day it turns over; where it
    # stands - banked plus what is left - on every other run.
    week_plan = None
    if weekly is None:
        weekly = starts_a_week(runtime, day)
    ahead = (
        not weekly
        and week is not None
        and preloads(runtime, day, manager.authority.transactions.preload_days)
    )
    if weekly and models:
        week_plan = _guard(
            report,
            "week",
            lambda: _weekly(conn, manager, league_key, runtime, day, propose, report, ctx, models),
        )
    elif (
        models
        and not ahead
        and week is not None
        and roster is not None
        and not pool_read_on(conn, league_key, day)
        and open_roster_spots(runtime, roster) > 0
    ):
        # Mid-week, the search runs again only when a roster spot has opened -
        # a player gone to IR, a drop. Searching every morning regardless was
        # measured (gate G2, 12 teams x 22 weeks x two seasons) and won nothing
        # over once a week: -0.06 +/- 0.06 and -0.06 +/- 0.08 categories a week,
        # with more adds spent. The replay never frees a spot mid-week, so this
        # one trigger is judgement, not measurement: an open spot is an add that
        # costs no drop, and it should not wait for Monday.
        week_plan = _guard(
            report,
            "week",
            lambda: _weekly(
                conn,
                manager,
                league_key,
                runtime,
                day,
                propose,
                report,
                ctx,
                models,
                start_of_week=False,
            ),
        )
    elif models and ctx.roster is not None and ctx.theirs is not None and week is not None:
        week_plan = _guard(
            report,
            "week",
            lambda: _outlook(
                conn,
                manager,
                league_key,
                runtime,
                day,
                report,
                ctx,
                models,
                # On the week's last day the queue is judged against next week.
                propose=propose and not ahead,
            ),
        )
    if ahead and models and ctx.roster is not None:
        _guard(
            report,
            "next week",
            lambda: _next_week(
                conn, manager, league_key, runtime, day, propose, report, ctx, models
            ),
        )

    # 5. Publish whatever we managed to work out.
    if manager.page.publishes and page_key:

        def _push():
            snap = snapshot.build(
                conn,
                manager.name,
                league_key,
                team_name=getattr(roster, "team_name", "") or manager.name,
                plan=plan,
                week_plan=week_plan,
                roster=roster,
                reasons=reasons if plan else None,
                week_no=week.number if week is not None else None,
                next_run=_next_run(conn, manager, runtime, day, roster),
                acted=acted,
            )
            publish.push(manager.page.url, page_key, snap)
            report.add("page", True, manager.page.url)

        _guard(report, "page", _push)

    # 6. Line up the rest of today against the real game times.
    if reschedule:
        own = manager.team_key or getattr(roster, "team_key", "")
        _guard(
            report,
            "schedule",
            lambda: _plan_rest_of_day(conn, manager, runtime, day, report, own),
        )

    say(report.text)
    return report


def act(conn, manager, league_key, roster, plan, report, apply=None) -> dict | None:
    """Make tonight's changes in Yahoo, where standing authority covers them.

    The published tool stops at the plan: making a change needs an actuator,
    and none ships with it. Without one this says so - on the page too, so
    "will act automatically" never stands in for a change nobody made. With
    one, every attempt lands in `season_actions` with what happened, because
    criteria granted in advance can only be argued with from a record.

    IR moves go first, as their own phase: tonight's lineup was planned on the
    roster they leave, so the lineup is not attempted if they do not finish.
    """
    phases = []
    if plan.ir_within_authority and plan.ir_moves:
        phases.append(list(plan.ir_moves))
    if plan.within_authority and plan.moves:
        phases.append(list(plan.moves))
    if not phases:
        return None
    if apply is None:
        try:  # Optional local actuator; a clone without one gets recommendations.
            from puckpilot.local.act import apply_lineup as apply
        except ImportError:
            report.add("act", True, "not made - no actuator installed")
            return {"ok": False, "message": "Not made automatically - make these in Yahoo."}
    result = apply(manager, roster.team_key, plan.date, phases)
    moves = [m.describe() for phase in phases for m in phase]
    proposals_mod.record_action(
        conn,
        manager.name,
        league_key,
        roster.team_key,
        plan.date,
        "lineup",
        {"moves": moves, "steps": list(result.lines)},
        outcome="executed" if result.ok else "failed",
        message=result.message,
    )
    report.add("act", result.ok, result.message, list(result.lines))
    return {"ok": bool(result.ok), "message": result.message, "at": datetime.now(UTC)}


def _next_run(conn, manager, runtime, day, roster):
    """When the scheduler will next run, for the page's freshness line."""
    from puckpilot.season import locks, schedule

    try:
        own = manager.team_key or getattr(roster, "team_key", "")
        return locks.next_run(
            conn, manager.name, runtime.nhl_season, day, own, anchor=schedule.ANCHOR_TIMES[0]
        )
    except Exception:  # noqa: BLE001 - freshness is advisory; never fail the push
        return None


def _plan_rest_of_day(conn, manager, runtime, day, report, team_key: str = ""):
    """Register a run shortly before each lock still to come today.

    Done on every run rather than once in the morning, so a game that moves, a
    roster that changes, or a missed run all self-correct at the next one.
    """
    from datetime import date as _date

    from puckpilot.config import REPO_ROOT
    from puckpilot.season import locks, schedule

    if day != _date.today().isoformat():
        report.add("schedule", True, f"not planning {day}; only today is schedulable")
        return
    teams = locks.roster_teams(conn, manager.name, team_key, day)
    if not teams:
        report.add("schedule", True, "no roster snapshot yet, so nothing to plan against")
        return
    todays = locks.locks_for(conn, runtime.nhl_season, day, teams)
    ahead = locks.upcoming(todays)
    times = locks.run_times(ahead)
    lines = schedule.plan_day(manager.name, REPO_ROOT, times)
    detail = (
        f"{len(ahead)} lock(s) left today: " + ", ".join(x.pretty for x in ahead)
        if ahead
        else "no locks left today"
    )
    report.add("schedule", True, detail, lines)


def _upkeep(conn, manager, league_key, season, report):
    """Weekly housekeeping, in the order each step makes the next one useful.

    A call-up has to cross three gaps before the tool can value him: onto an
    NHL roster, into our player table, and into the Yahoo map. Doing them in
    that order means one week's lag rather than three.
    """
    from puckpilot.data.nhl import NhlClient
    from puckpilot.data.sync import sync_current_rosters
    from puckpilot.season import cli_support
    from puckpilot.yahoo import playermap

    lines = []
    rosters = sync_current_rosters(conn, NhlClient(), season, progress=lambda _m: None)
    lines.append(
        f"NHL rosters: {rosters.get('new', 0)} new player(s), {rosters.get('changed', 0)} re-teamed"
    )

    again = playermap.reresolve_unmatched(conn)
    lines.append(f"re-resolved {again.matched}/{again.total} previously unmapped")

    age = _map_age(conn, league_key)
    if age is None or age >= MAP_REFRESH_DAYS:
        built = cli_support.run_session(
            manager, lambda s: playermap.build_map(conn, s, league_key, limit=900)
        )
        lines.append(f"player map rebuilt: {built.matched}/{built.total} matched")
    else:
        lines.append(f"player map {age:.1f} days old, still fresh")
    report.add("upkeep", True, "player map and rosters", lines)


def _map_age(conn, league_key: str) -> float | None:
    row = conn.execute(
        "SELECT MAX(updated_at) FROM yahoo_player_map WHERE league_key = ?", (league_key,)
    ).fetchone()
    return _age_days(row[0]) if row and row[0] else None


def _refresh_runtime(conn, manager, league_key, runtime, report):
    """Re-read the league settings when the cache has aged."""
    from puckpilot.season import cli_support
    from puckpilot.season.fetch import fetch_runtime, save_runtime

    age = _age_days(runtime.fetched_at)
    # A cache written before the scored categories were read has none, and
    # without them Yahoo's live category totals cannot be labelled.
    if age is not None and age < RUNTIME_REFRESH_DAYS and runtime.stat_categories:
        report.add("settings", True, f"cached {age:.1f} days ago, still fresh")
        return runtime
    got = cli_support.run_session(manager, lambda s: fetch_runtime(s, league_key, manager.team_key))
    save_runtime(conn, got)
    report.add(
        "settings",
        True,
        f"refreshed - week {got.current_week}, {len(got.weeks)} week(s) known",
    )
    return got


def _log_week(conn, manager, league_key, ctx, report):
    """Record this run's reading of the live score, and the opponent's roster."""
    from puckpilot.season.fetch import save_live, save_roster

    if ctx.error:
        report.add("score", False, f"live score not read: {ctx.error}")
        return
    if ctx.live is None:
        if ctx.week is not None:
            report.add("score", True, f"week {ctx.week.number}: no live score in the response")
        return
    save_live(conn, manager.name, league_key, ctx.live, ctx.raw)
    if ctx.theirs is not None:
        save_roster(conn, manager.name, ctx.theirs)
        _statuses(conn, ctx.theirs, report, "theirs")
    t = ctx.live
    left = ""
    if t.ours.remaining_games is not None:
        left = f", Yahoo counts {t.ours.remaining_games} games left vs {t.theirs.remaining_games}"
    report.add("score", True, f"week {t.week} vs {t.theirs.name} ({t.status}){left}")


def _statuses(conn, roster, report, whose: str) -> None:
    """Log injury and availability tags as this run saw them (season/status_log.py)."""
    from puckpilot.season import status_log

    changes = status_log.record(conn, roster)
    if changes:
        report.add(
            f"tags ({whose})",
            True,
            f"{len(changes)} change(s)",
            [c.describe() for c in changes],
        )


def _outlook(conn, manager, league_key, runtime, day, report, ctx, models, propose=True):
    """Where the week stands on a run that is not the week's first - and
    whether what is waiting for a decision still pays.

    Banked plus what is left, both sides, from the rosters just read - no add
    search, which is the weekly job's, and no new protocol, which Monday's.

    The queue is re-checked, though. A proposal is priced once, by the search,
    and midweek the search runs again only when a roster spot opens; on
    2026-09-30 the cards still argued from Tuesday morning - before J.T.
    Miller's NA cleared and before a night was banked - with "Tue 29: fills an
    empty C" on a day already played. So every such run re-prices each pending
    swap as the search would today, in the order proposed, and withdraws, with
    the reason, any whose player is gone, whose drop is no longer droppable, or
    that no longer clears the floor. Nothing new is searched for.
    """
    from puckpilot.draft.sim import build_universe
    from puckpilot.season import pool as pool_mod
    from puckpilot.season import week as weekmod
    from puckpilot.season.odds import OddsModel, log_week

    values, goalies = models
    season = runtime.nhl_season
    universe = build_universe(conn, season, _train_seasons(season), manager.league)
    used_week, used_season = adds_used(ctx.roster, ctx.week)
    waiting = (
        sorted(proposals_mod.pending(conn, manager.name, league_key), key=lambda p: p.id)
        if propose
        else []
    )
    workable, lapsed = _workable(conn, manager, league_key, ctx, waiting) if waiting else ([], {})
    terms = manager.authority.transactions
    plan = weekmod.build_week_plan(
        conn,
        runtime,
        manager.league,
        ctx.week,
        ctx.live.theirs.name if ctx.live is not None else "",
        ctx.roster,
        ctx.theirs,
        pool_mod.load_pool(conn, league_key, day),
        universe.frame,
        goalies,
        values,
        adds_used_week=used_week,
        adds_used_season=used_season,
        find_targets=False,
        odds_model=OddsModel(),
        min_gain=terms.min_weekly_gain,
        add_scoring=terms.add_scoring,
        min_expected_gain=terms.min_expected_gain,
        playoff_reserve=terms.playoff_reserve,
        stream_spots=terms.stream_spots,
        reprice=[(add, drop) for _, add, drop in workable] if waiting else None,
        **live_inputs(conn, runtime, ctx.week, ctx.live, day),
    )
    log_week(conn, manager.name, league_key, ctx.roster.team_key, plan, day)
    lines = _recheck(conn, workable, lapsed, plan, terms) if waiting else []
    report.add("week", True, _week_line(plan), lines)
    return plan


def _next_week(conn, manager, league_key, runtime, day, propose, report, ctx, models):
    """The week's last day: judge adds against the week to come.

    Acquisitions are counted per week and this week's expire tonight, while a
    player added today plays all of the next. On 2026-10-03 and 04 two adds
    were chosen - and approved - for the last two days of a lost week; against
    the next week both cost categories (-0.32 and -0.12), because Gibson and
    Malkin each had more games coming than the players replacing them.

    So on the last `preload_days` the search prices against next week - its
    opponent, its schedule, nothing banked - spending only what is left of
    this week's acquisitions; the first run of the day searches, later ones
    re-check whatever is waiting against next week the same way.
    """
    from puckpilot.draft.sim import build_universe
    from puckpilot.season import cli_support, pool
    from puckpilot.season import week as weekmod
    from puckpilot.season.fetch import fetch_matchups, fetch_roster
    from puckpilot.season.matchups import current_or_next
    from puckpilot.season.odds import OddsModel
    from puckpilot.yahoo import playermap

    nxt = runtime.week(ctx.week.number + 1)
    terms = manager.authority.transactions
    used_week, used_season = adds_used(ctx.roster, ctx.week)
    left = None if runtime.max_weekly_adds is None else runtime.max_weekly_adds - used_week
    search = propose and not pool_read_on(conn, league_key, day) and (left is None or left > 0)
    waiting = (
        sorted(proposals_mod.pending(conn, manager.name, league_key), key=lambda p: p.id)
        if propose and not search
        else []
    )
    if not search and not waiting:
        return None
    pmap = playermap.load_map(conn, league_key)
    team_key = ctx.roster.team_key

    def _read(session):
        m = current_or_next(fetch_matchups(session, team_key), nxt.start)
        if m is None:
            return None
        theirs = fetch_roster(session, m.opponent_key, nxt.start, player_map=pmap)
        fa = (
            pool.fetch_pool(session, league_key, "FA", limit=150, player_map=pmap) if search else []
        )
        return m, theirs, fa

    got = cli_support.run_session(manager, _read)
    if got is None:
        report.add("next week", False, f"no matchup for week {nxt.number}")
        return None
    m, theirs, fa = got
    if search:
        pool.save_pool(conn, league_key, day, fa)
    workable, lapsed = _workable(conn, manager, league_key, ctx, waiting) if waiting else ([], {})
    values, goalies = models
    universe = build_universe(
        conn, runtime.nhl_season, _train_seasons(runtime.nhl_season), manager.league
    )
    plan = weekmod.build_week_plan(
        conn,
        runtime,
        manager.league,
        nxt,
        m.opponent_name,
        ctx.roster,
        theirs,
        fa,
        universe.frame,
        goalies,
        values,
        adds_used_week=used_week,
        adds_used_season=used_season,
        min_gain=terms.min_weekly_gain,
        odds_model=OddsModel(),
        add_scoring=terms.add_scoring,
        min_expected_gain=terms.min_expected_gain,
        playoff_reserve=terms.playoff_reserve,
        stream_spots=terms.stream_spots,
        find_targets=search,
        reprice=[(add, drop) for _, add, drop in workable] if waiting else None,
        horizon="next week",
    )
    expect = f" - expect {plan.expected:.1f} of {len(plan.outlook)}" if plan.expected else ""
    head = f"week {nxt.number} vs {m.opponent_name}{expect} as the roster stands"
    lines: list[str] = []
    if search:
        made = proposals_mod.propose(
            conn,
            manager.name,
            league_key,
            team_key,
            plan.targets,
            nxt.number,
            max_pending=terms.max_pending,
            supersede=True,
            horizon="next week",
        )
        spare = "" if left is None else f" with this week's {left} leftover acquisition(s)"
        if made:
            lines += [p.describe() for p in made]
        elif plan.targets:
            lines.append("nothing new to propose")
        else:
            lines.append(f"nothing worth making{spare} - Monday's search starts fresh")
    else:
        lines += _recheck(conn, workable, lapsed, plan, terms)
    report.add("next week", True, head, lines)
    return plan


def _workable(conn, manager, league_key, ctx, waiting):
    """(proposal, add, drop) for those still possible, and {id: why} for the rest.

    One read of the pending players' ownership - a pool read is several pages
    and might not reach a player proposed days ago.
    """
    from dataclasses import replace

    from puckpilot.season import cli_support
    from puckpilot.season import pool as pool_mod
    from puckpilot.yahoo import playermap

    pmap = playermap.load_map(conn, league_key)
    keys = [p.add_player_key for p in waiting]
    found = cli_support.run_session(
        manager, lambda s: pool_mod.fetch_players(s, league_key, keys, player_map=pmap)
    )
    now = {a.player_key: a for a in found}
    mine = {p.player_key: p for p in ctx.roster.players}
    workable, lapsed = [], {}
    for p in waiting:
        add = now.get(p.add_player_key)
        drop = mine.get(p.drop_player_key) if p.drop_player_key else None
        if add is None or not add.is_available:
            lapsed[p.id] = f"{p.add_name} is no longer available"
        elif add.is_out:
            lapsed[p.id] = f"{p.add_name} is now listed {add.status}"
        elif p.drop_player_key and drop is None:
            lapsed[p.id] = f"{p.drop_name} is no longer on your roster"
        elif drop is not None and (drop.is_out or drop.on_ir):
            lapsed[p.id] = (
                f"{p.drop_name} is now {drop.status or drop.selected_slot} - "
                f"not a player to drop on that"
            )
        else:
            if add.nhl_player_id is None:
                add = replace(add, nhl_player_id=p.add_pid)
            workable.append((p, add, drop))
    return workable, lapsed


def _recheck(conn, workable, lapsed, plan, terms) -> list[str]:
    """Write a re-check back to the queue, and say what it did."""
    by_pair = {(t.player.player_key, t.drop.player_key if t.drop else ""): t for t in plan.targets}
    low = {(t.player.player_key, t.drop.player_key if t.drop else ""): t for t in plan.lapsed}
    odds = terms.add_scoring == "odds"
    floor = terms.min_expected_gain if odds else terms.min_weekly_gain
    kept: dict[int, object] = {}
    lapsed = dict(lapsed)
    if plan.adds_left_week == 0:
        lapsed.update({p.id: "no acquisitions left this week" for p, _, _ in workable})
        workable = []
    lines = []
    for p, _, _ in workable:
        pair = (p.add_player_key, p.drop_player_key)
        t = by_pair.get(pair)
        was = p.reason.get("expected_gain") if odds else p.reason.get("gain")
        if t is not None:
            kept[p.id] = t
            now = t.gain if odds and t.gain is not None else t.score
            before = f" (was {float(was):+.2f})" if was is not None else ""
            lines.append(f"#{p.id} {p.add_name} for {p.drop_name}: still {now:+.2f}{before}")
            continue
        t = low.get(pair)
        now = (t.gain if odds and t.gain is not None else t.score) if t is not None else 0.0
        unit = " categories expected" if odds else ""
        lapsed[p.id] = f"now worth {now:+.2f}{unit}, below the {floor:.2f} floor"
    proposals_mod.refresh(conn, kept, lapsed)
    lines += [f"#{pid} withdrawn: {why}" for pid, why in lapsed.items()]
    return lines


def _week_line(plan) -> str:
    bands: dict[str, list[str]] = {}
    for o in plan.outlook:
        bands.setdefault(o.band, []).append(o.category.label)
    bits = [f"{k}: {' '.join(v)}" for k, v in sorted(bands.items())]
    exp = f" - expect {plan.expected:.1f} of {len(plan.outlook)}" if plan.expected else ""
    return (
        f"week {plan.week} vs {plan.opponent or '?'} - starts left {plan.our_games} v "
        f"{plan.their_games}{exp} - " + "; ".join(bits)
    )


def pool_read_on(conn, league_key: str, day: str) -> bool:
    """Has today's free-agent pool been read - i.e. has today's add search run?"""
    row = conn.execute(
        "SELECT 1 FROM yahoo_fa_snapshots WHERE league_key = ? AND date = ? LIMIT 1",
        (league_key, day),
    ).fetchone()
    return row is not None


def _weekly(
    conn, manager, league_key, runtime, day, propose, report, ctx, models, start_of_week=True
):
    """The full plan with the add search - daily - and on a week's first day,
    the weekly upkeep and the protocol too.

    Reuses what this run already read - our roster, theirs, the live score -
    and falls back to reading the matchups itself when the score read failed.
    """
    from puckpilot.draft.sim import build_universe
    from puckpilot.season import cli_support, explain, pool
    from puckpilot.season import protocol as protocol_mod
    from puckpilot.season import week as weekmod
    from puckpilot.season.fetch import fetch_matchups, fetch_roster
    from puckpilot.season.matchups import current_or_next
    from puckpilot.season.odds import OddsModel, log_week
    from puckpilot.yahoo import playermap

    season = runtime.nhl_season
    pmap = playermap.load_map(conn, league_key)

    def _read(session):
        team_key = manager.team_key or (ctx.roster.team_key if ctx.roster else "")
        week = ctx.week
        opp_key = ctx.live.theirs.team_key if ctx.live is not None else ""
        opp_name = ctx.live.theirs.name if ctx.live is not None else ""
        if week is None or not opp_key:
            m = current_or_next(fetch_matchups(session, team_key), day)
            if m is None:
                return None
            week, opp_key, opp_name = m.as_week(), m.opponent_key, m.opponent_name
        when = day if week.contains(day) else week.start
        ours = ctx.roster
        if ours is None or ours.date != when:
            ours = fetch_roster(session, team_key, when, player_map=pmap)
        theirs = ctx.theirs
        if theirs is None:
            theirs = fetch_roster(session, opp_key, when, player_map=pmap) if opp_key else ours
        fa = pool.fetch_pool(session, league_key, "FA", limit=150, player_map=pmap)
        return week, opp_name, ours, theirs, fa

    if start_of_week:
        _guard(report, "upkeep", lambda: _upkeep(conn, manager, league_key, season, report))

    got = cli_support.run_session(manager, _read)
    if got is None:
        report.add("week", False, f"no matchup covering {day} and none after it")
        return None
    week, opp_name, ours, theirs, fa = got
    pool.save_pool(conn, league_key, day, fa)

    values, goalies = models
    universe = build_universe(conn, season, _train_seasons(season), manager.league)
    used_week, used_season = adds_used(ours, week)
    live = ctx.live if ctx.week is not None and ctx.week.number == week.number else None
    plan = weekmod.build_week_plan(
        conn,
        runtime,
        manager.league,
        week,
        opp_name,
        ours,
        theirs,
        fa,
        universe.frame,
        goalies,
        values,
        adds_used_week=used_week,
        adds_used_season=used_season,
        min_gain=manager.authority.transactions.min_weekly_gain,
        odds_model=OddsModel(),
        add_scoring=manager.authority.transactions.add_scoring,
        min_expected_gain=manager.authority.transactions.min_expected_gain,
        playoff_reserve=manager.authority.transactions.playoff_reserve,
        stream_spots=manager.authority.transactions.stream_spots,
        **live_inputs(conn, runtime, week, live, day),
    )
    log_week(conn, manager.name, league_key, ours.team_key, plan, day)
    lines = [ln for ln in explain.week_story(plan, runtime) if ln]

    if start_of_week:
        stance = protocol_mod.derive(
            plan.outlook, manager.name, league_key, ours.team_key, plan.week, opp_name
        )
        existing = protocol_mod.load(conn, manager.name, league_key, plan.week)
        if not (existing and existing.status == protocol_mod.APPROVED):
            stance = protocol_mod.save(conn, stance)
            lines.append(f"protocol #{stance.id} proposed - approve it on the page")

    stuck = ours.illegal_ir()
    if propose and plan.targets and stuck:
        lines.append(
            "no proposals: " + ", ".join(p.name for p in stuck) + " must leave IR first - "
            "Yahoo refuses every add and drop until then"
        )
    elif propose and plan.targets:
        made = proposals_mod.propose(
            conn,
            manager.name,
            league_key,
            ours.team_key,
            plan.targets,
            plan.week,
            max_pending=manager.authority.transactions.max_pending,
            supersede=True,
        )
        lines += [p.describe() for p in made] or ["nothing new to propose"]
    head = _week_line(plan) if not start_of_week else f"week {plan.week} vs {opp_name}"
    report.add("week", True, head, lines)
    return plan


def _train_seasons(season: str) -> tuple[str, ...]:
    y = int(season[:4])
    return tuple(f"{y - i}{y - i + 1}" for i in range(1, 4))


def _env_key() -> str:
    import os

    return os.environ.get("PUCKPILOT_MANAGER_KEY", "")
