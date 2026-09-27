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
from datetime import datetime

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
    from puckpilot.season.today import build_plan, yahoo_goalie_games
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
    if roster is not None:
        save_roster(conn, manager.name, roster)
        report.add("roster", True, f"{len(roster)} players", list(roster.unmapped))

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
                + (f", locks {got.deadline()}" if got.lock_utc else ""),
                [f"{m.describe()} - {reasons.get(m.player.player_key, '')}" for m in got.moves],
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

    # 4. The week: the full plan, with adds, on the day it turns over; where it
    # stands - banked plus what is left - on every other run.
    week_plan = None
    if weekly is None:
        weekly = starts_a_week(runtime, day)
    if weekly and models:
        week_plan = _guard(
            report,
            "week",
            lambda: _weekly(conn, manager, league_key, runtime, day, propose, report, ctx, models),
        )
    elif models and ctx.roster is not None and ctx.theirs is not None and week is not None:
        week_plan = _guard(
            report,
            "week",
            lambda: _outlook(conn, manager, league_key, runtime, day, report, ctx, models),
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
    t = ctx.live
    left = ""
    if t.ours.remaining_games is not None:
        left = f", Yahoo counts {t.ours.remaining_games} games left vs {t.theirs.remaining_games}"
    report.add("score", True, f"week {t.week} vs {t.theirs.name} ({t.status}){left}")


def _outlook(conn, manager, league_key, runtime, day, report, ctx, models):
    """Where the week stands on a run that is not the week's first.

    Banked plus what is left, both sides, from the rosters just read - no add
    search, which is the weekly job's, and no new protocol, which Monday's.
    """
    from puckpilot.draft.sim import build_universe
    from puckpilot.season import pool as pool_mod
    from puckpilot.season import week as weekmod

    values, goalies = models
    season = runtime.nhl_season
    universe = build_universe(conn, season, _train_seasons(season), manager.league)
    used_week, used_season = adds_used(ctx.roster, ctx.week)
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
        **live_inputs(conn, runtime, ctx.week, ctx.live, day),
    )
    report.add("week", True, _week_line(plan))
    return plan


def _week_line(plan) -> str:
    bands: dict[str, list[str]] = {}
    for o in plan.outlook:
        bands.setdefault(o.band, []).append(o.category.label)
    bits = [f"{k}: {' '.join(v)}" for k, v in sorted(bands.items())]
    return (
        f"week {plan.week} vs {plan.opponent or '?'} - starts left {plan.our_games} v "
        f"{plan.their_games} - " + "; ".join(bits)
    )


def _weekly(conn, manager, league_key, runtime, day, propose, report, ctx, models):
    """The week's first run: the full plan, the add search, and the protocol.

    Reuses what this run already read - our roster, theirs, the live score -
    and falls back to reading the matchups itself when the score read failed.
    """
    from puckpilot.draft.sim import build_universe
    from puckpilot.season import cli_support, explain, pool
    from puckpilot.season import protocol as protocol_mod
    from puckpilot.season import week as weekmod
    from puckpilot.season.fetch import fetch_matchups, fetch_roster
    from puckpilot.season.matchups import current_or_next
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
        **live_inputs(conn, runtime, week, live, day),
    )
    lines = [ln for ln in explain.week_story(plan, runtime) if ln]

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
        )
        lines += [p.describe() for p in made] or ["nothing new to propose"]
    report.add("week", True, f"week {plan.week} vs {opp_name}", lines)
    return plan


def _train_seasons(season: str) -> tuple[str, ...]:
    y = int(season[:4])
    return tuple(f"{y - i}{y - i + 1}" for i in range(1, 4))


def _env_key() -> str:
    import os

    return os.environ.get("PUCKPILOT_MANAGER_KEY", "")
