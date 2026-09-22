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
    say=print,
) -> RunReport:
    """Everything a day needs, in the order it needs it."""
    from puckpilot.season import cli_support, explain, publish, snapshot
    from puckpilot.season.fetch import discover_team_key, fetch_roster, save_roster
    from puckpilot.season.goalies import ChainedGoalieSource, TrailingStartShareSource
    from puckpilot.season.today import build_plan
    from puckpilot.season.values import build_value_model
    from puckpilot.yahoo import playermap

    report = RunReport(date=day, manager=manager.name)
    season = runtime.nhl_season
    page_key = manager.page.owner_key or _env_key()

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

    # 3. Tonight.
    pmap = playermap.load_map(conn, league_key)

    def _read(session):
        key = manager.team_key or discover_team_key(session, league_key)
        return fetch_roster(session, key, day, player_map=pmap)

    roster = _guard(report, "roster", lambda: cli_support.run_session(manager, _read))
    plan = None
    if roster is not None:
        save_roster(conn, manager.name, roster)
        report.add("roster", True, f"{len(roster)} players", list(roster.unmapped))

        def _plan():
            train = _train_seasons(season)
            values = build_value_model(conn, season, train, manager.league)
            goalies = ChainedGoalieSource(
                TrailingStartShareSource(conn, season, fallback_season=train[0])
            )
            got = build_plan(
                conn,
                runtime,
                roster,
                values,
                goalies,
                day,
                manager=manager.name,
                authority=manager.authority.lineup,
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

        got = _guard(report, "lineup", _plan)
        plan, reasons = got if got else (None, {})

    # 4. The week, on the day it turns over.
    week_plan = None
    if weekly is None:
        weekly = starts_a_week(runtime, day)
    if weekly:
        week_plan = _guard(
            report,
            "week",
            lambda: _weekly(conn, manager, league_key, runtime, day, propose, report),
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
            )
            publish.push(manager.page.url, page_key, snap)
            report.add("page", True, manager.page.url)

        _guard(report, "page", _push)

    say(report.text)
    return report


def _weekly(conn, manager, league_key, runtime, day, propose, report):
    from puckpilot.draft.sim import build_universe
    from puckpilot.season import cli_support, explain, pool
    from puckpilot.season import protocol as protocol_mod
    from puckpilot.season import week as weekmod
    from puckpilot.season.fetch import fetch_matchups, fetch_roster
    from puckpilot.season.goalies import ChainedGoalieSource, TrailingStartShareSource
    from puckpilot.season.matchups import for_date
    from puckpilot.season.values import build_value_model
    from puckpilot.yahoo import playermap

    season = runtime.nhl_season
    pmap = playermap.load_map(conn, league_key)

    def _read(session):
        team_key = manager.team_key
        m = for_date(fetch_matchups(session, team_key), day)
        if m is None:
            return None
        ours = fetch_roster(session, team_key, m.start, player_map=pmap)
        theirs = (
            fetch_roster(session, m.opponent_key, m.start, player_map=pmap)
            if m.opponent_key
            else ours
        )
        fa = pool.fetch_pool(session, league_key, "FA", limit=150, player_map=pmap)
        return m, ours, theirs, fa

    got = cli_support.run_session(manager, _read)
    if got is None:
        report.add("week", False, f"no matchup covering {day}")
        return None
    m, ours, theirs, fa = got
    pool.save_pool(conn, league_key, m.start, fa)

    train = _train_seasons(season)
    universe = build_universe(conn, season, train, manager.league)
    values = build_value_model(conn, season, train, manager.league)
    goalies = ChainedGoalieSource(TrailingStartShareSource(conn, season, fallback_season=train[0]))
    plan = weekmod.build_week_plan(
        conn,
        runtime,
        manager.league,
        m.as_week(),
        m.opponent_name,
        ours,
        theirs,
        fa,
        universe.frame,
        goalies,
        values,
        min_gain=manager.authority.transactions.min_weekly_gain,
    )
    lines = [ln for ln in explain.week_story(plan, runtime) if ln]

    stance = protocol_mod.derive(
        plan.outlook, manager.name, league_key, ours.team_key, plan.week, m.opponent_name
    )
    existing = protocol_mod.load(conn, manager.name, league_key, plan.week)
    if not (existing and existing.status == protocol_mod.APPROVED):
        stance = protocol_mod.save(conn, stance)
        lines.append(f"protocol #{stance.id} proposed - approve it on the page")

    if propose and plan.targets:
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
    report.add("week", True, f"week {plan.week} vs {m.opponent_name}", lines)
    return plan


def _train_seasons(season: str) -> tuple[str, ...]:
    y = int(season[:4])
    return tuple(f"{y - i}{y - i + 1}" for i in range(1, 4))


def _env_key() -> str:
    import os

    return os.environ.get("PUCKPILOT_MANAGER_KEY", "")
