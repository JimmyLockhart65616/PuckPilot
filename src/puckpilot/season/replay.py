"""Score the code that will actually run, against the policy that was validated.

`bench_regret_report` measured a lineup policy written for the backtest: take a
roster of player ids, look up a static per-game projection, solve the slotting.
The code that runs in-season is not that code. It reads a Yahoo roster, gates
candidates on injuries and lock state, blends value toward recent form, weights
goalies by a probability model, breaks ties toward not moving, and declines
changes below a threshold. Every one of those is a chance to lose value that
the original measurement cannot see.

So this replays `season.today.build_plan` itself, day by day, over a real
season, and scores it in the same units as the validated policy. The arms
isolate two questions:

    live path vs the validated optimizer, same goalie info   what the gating,
                                                             blending and
                                                             inertia cost
    our goalie model vs hindsight starts                     what ~62% accuracy
                                                             costs against 100%

Availability is derived exactly as `lineup_replay.skater_availability` derives
it - a player's team as of each date, from his game logs - rather than from
`nhl_players.team_abbrev`, which holds current clubs and would put half the
league on the wrong team for a past season.

Single-position eligibility throughout, deliberately: the validated arm has
only ever had one position per player, and an arm that differs in two ways
measures neither.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date as _date
from datetime import timedelta

import numpy as np

from puckpilot.data.goalies import GoalieStartSource, HindsightGoalieSource
from puckpilot.draft.replay import ReplayData, build_replay_data
from puckpilot.draft.sim import _default_opponents, build_universe, keepers_for, run_draft
from puckpilot.engine.lineup_replay import (
    GameValueModel,
    _daily_optimizer_total,
    _hindsight_total,
    _set_and_forget_total,
    projected_pg_values,
    skater_availability,
)
from puckpilot.league import DEFAULT_LEAGUE, LeagueConfig
from puckpilot.season.authority import LineupAuthority
from puckpilot.season.goalies import TrailingStartShareSource
from puckpilot.season.roster import RosterPlayer, TeamRoster
from puckpilot.season.settings import LeagueRuntime, RosterSlot, Week
from puckpilot.season.today import BENCH, build_plan
from puckpilot.season.values import ValueModel

POS_TO_YAHOO = {"C": "C", "L": "LW", "R": "RW", "D": "D", "G": "G"}
NON_SCORING = {BENCH, "IR", "IR+", "?"}


def runtime_for_replay(league: LeagueConfig, season: str, dates: list[str]) -> LeagueRuntime:
    """A `LeagueRuntime` for a season Yahoo will not answer for any more.

    Weeks are Monday-Sunday here, which is wrong for a live league - the real
    2026-27 calendar has a fourteen-day week - and harmless for a replay, where
    the only thing week boundaries decide is when the goalie minimum starts
    forcing a start.
    """
    slots = [RosterSlot(POS_TO_YAHOO.get(pos, pos), n, True) for pos, n in league.shape.slots]
    if league.shape.util_slots:
        slots.append(RosterSlot("Util", league.shape.util_slots, True))
    if league.shape.bench_slots:
        slots.append(RosterSlot("BN", league.shape.bench_slots, False))

    weeks: list[Week] = []
    if dates:
        start = _date.fromisoformat(dates[0])
        end = _date.fromisoformat(dates[-1])
        cur, n = start, 1
        while cur <= end:
            stop = min(cur + timedelta(days=6 - cur.weekday()), end)
            weeks.append(Week(n, cur.isoformat(), stop.isoformat()))
            cur, n = stop + timedelta(days=1), n + 1

    y = int(season[:4])
    return LeagueRuntime(
        league_key=f"replay.{season}",
        name=league.name,
        num_teams=league.shape.n_teams,
        scoring_type=league.scoring,
        yahoo_season=str(y),
        start_date=dates[0] if dates else "",
        end_date=dates[-1] if dates else "",
        start_week=1,
        end_week=len(weeks),
        current_week=1,
        current_date=dates[0] if dates else "",
        playoff_start_week=len(weeks) + 1,
        num_playoff_teams=league.playoff_teams,
        weekly_deadline="intraday",
        roster_type="date",
        waiver_type="R",
        waiver_rule="all",
        waiver_days=1,
        uses_faab=False,
        max_adds=league.season_acquisitions,
        max_weekly_adds=league.weekly_acquisitions,
        min_games_played=league.min_goalie_appearances,
        trade_end_date="",
        slots=tuple(slots),
        weeks=tuple(weeks),
        fetched_at="replay",
    )


def team_by_day(
    conn: sqlite3.Connection, season: str, data: ReplayData, pids: set[int]
) -> dict[int, list[str]]:
    """Each player's club as of each date index, from his own game logs.

    The same walk `skater_availability` does, kept as the team rather than
    collapsed to a yes/no, because the live path asks the calendar whether that
    club plays instead of being handed availability.
    """
    logs: dict[int, list[tuple[str, str]]] = {pid: [] for pid in pids}
    for pid, d, team in conn.execute(
        "SELECT player_id, game_date, team_abbrev FROM nhl_game_logs WHERE season = ?"
        " ORDER BY game_date",
        (season,),
    ):
        if pid in logs:
            logs[pid].append((d, team))

    out: dict[int, list[str]] = {}
    for pid, entries in logs.items():
        if not entries:
            out[pid] = [""] * len(data.dates)
            continue
        seq, ei, team = [], 0, entries[0][1]
        for date in data.dates:
            while ei < len(entries) and entries[ei][0] <= date:
                team = entries[ei][1]
                ei += 1
            seq.append(team)
        out[pid] = seq
    return out


def _live_total(
    conn: sqlite3.Connection,
    runtime: LeagueRuntime,
    roster: list[int],
    positions: dict[int, str],
    teams: dict[int, list[str]],
    values: ValueModel,
    goalie_src: GoalieStartSource,
    data: ReplayData,
    vm: GameValueModel,
    authority: LineupAuthority,
) -> tuple[float, int]:
    """Season value captured by the live decision path, and changes made."""
    keys = {pid: str(pid) for pid in roster}
    slot: dict[int, str] = dict.fromkeys(roster, BENCH)
    total, changes = 0.0, 0

    for i, date in enumerate(data.dates):
        players = tuple(
            RosterPlayer(
                player_key=keys[pid],
                yahoo_id=keys[pid],
                name=keys[pid],
                team=teams.get(pid, [""] * len(data.dates))[i],
                primary_position=POS_TO_YAHOO.get(positions[pid], positions[pid]),
                yahoo_eligible=frozenset({POS_TO_YAHOO.get(positions[pid], positions[pid])}),
                selected_slot=slot[pid],
                nhl_player_id=pid,
            )
            for pid in roster
        )
        team_roster = TeamRoster(
            league_key=runtime.league_key,
            team_key="replay",
            date=date,
            players=players,
        )
        plan = build_plan(
            conn,
            runtime,
            team_roster,
            values,
            goalie_src,
            date,
            manager="replay",
            authority=authority,
            goalie_starts_so_far=0,
        )
        for m in plan.moves:
            slot[int(m.player.player_key)] = m.to_slot
            changes += 1
        for pid, s in slot.items():
            if s not in NON_SCORING:
                total += vm.actual(data, pid, i)
    return total, changes


@dataclass
class LivePolicyReport:
    n_rosters: int
    season: str
    totals: dict[str, float]
    changes_per_roster: float
    text: str


def live_policy_report(
    conn: sqlite3.Connection,
    season: str = "20252026",
    train_seasons: tuple[str, ...] = ("20242025", "20232024", "20222023"),
    n_drafts: int = 1,
    seed: int | None = 123,
    league: LeagueConfig = DEFAULT_LEAGUE,
    min_gain: float = 0.0,
    progress: Callable[[str], None] | None = None,
) -> LivePolicyReport:
    """Replay drafted rosters under the live path and the validated policies."""
    from puckpilot.draft.engine import RosterValuePolicy

    rules = league.draft_rules()
    shape = rules.shape
    rng = np.random.default_rng(seed)
    say = progress or (lambda _m: None)

    skater_keys = [c.key for c in league.skater_cats]
    u = build_universe(conn, season, train_seasons, league)
    data = build_replay_data(conn, season, skater_keys)
    vm = GameValueModel(data, set(u.ids.tolist()), league.goalie_cats)
    pg_value = projected_pg_values(u.frame, vm, skater_keys)
    positions = dict(zip(u.ids.tolist(), u.pos.tolist(), strict=True))
    runtime = runtime_for_replay(league, season, data.dates)

    rosters: list[list[int]] = []
    for _ in range(n_drafts):
        opponents = _default_opponents(rng, league)
        order = rng.permutation(len(opponents))
        engine_seat = int(rng.integers(0, shape.n_teams))
        bots = []
        oi = 0
        for seat in range(shape.n_teams):
            if seat == engine_seat:
                bots.append(RosterValuePolicy())
            else:
                bots.append(opponents[order[oi]])
                oi += 1
        keepers = keepers_for(conn, u, season, league, rng)
        for ridx in run_draft(u, bots, rules, rng, keepers):
            rosters.append([int(u.ids[i]) for i in ridx])

    all_pids = {pid for r in rosters for pid in r}
    avail = skater_availability(conn, season, data, all_pids)
    teams = team_by_day(conn, season, data, all_pids)
    hind_g = HindsightGoalieSource(conn, season)
    model_g = TrailingStartShareSource(conn, season, fallback_season=train_seasons[0])
    values = ValueModel(vm=vm, data=data, proj_pg=pg_value, season=season, scale_season=season)
    auth = LineupAuthority(enabled=True, min_gain=min_gain, min_goalie_p_start=0.5)

    arms = [
        "hindsight",
        "validated_perfect",
        "validated_model",
        "live_perfect",
        "live_model",
        "baseline",
    ]
    sums = dict.fromkeys(arms, 0.0)
    changes = 0
    for k, roster in enumerate(rosters):
        sums["hindsight"] += _hindsight_total(roster, positions, data, shape, vm)
        sums["baseline"] += _set_and_forget_total(roster, positions, pg_value, data, shape, vm)
        for name, src in (("perfect", hind_g), ("model", model_g)):
            sums[f"validated_{name}"] += _daily_optimizer_total(
                roster,
                positions,
                pg_value,
                data,
                shape,
                avail,
                src,
                vm,
                min_goalie_appearances=league.min_goalie_appearances,
            )
            t, c = _live_total(conn, runtime, roster, positions, teams, values, src, data, vm, auth)
            sums[f"live_{name}"] += t
            if name == "model":
                changes += c
        say(f"  {k + 1}/{len(rosters)} rosters replayed")

    n = len(rosters)
    per = {k: v / n for k, v in sums.items()}
    ceiling = per["hindsight"] or 1.0
    rows = [
        ("hindsight-optimal", per["hindsight"]),
        ("validated optimizer, perfect G", per["validated_perfect"]),
        ("LIVE path, perfect G", per["live_perfect"]),
        ("validated optimizer, our G model", per["validated_model"]),
        ("LIVE path, our G model", per["live_model"]),
        ("set-and-forget baseline", per["baseline"]),
    ]
    lines = [
        f"Live-policy replay: {n} drafted rosters x {season}, min_gain {min_gain}",
        "",
        f"{'policy':36}{'value/roster':>14}{'% of hindsight':>16}",
    ]
    for label, v in rows:
        lines.append(f"{label:36}{v:>14.1f}{v / ceiling * 100:>15.1f}%")
    lines += [
        "",
        f"Live vs validated (perfect G):  {_pct(per['live_perfect'], per['validated_perfect'])}",
        f"Live vs validated (our G model): {_pct(per['live_model'], per['validated_model'])}",
        f"Our goalie model vs hindsight:   {_pct(per['live_model'], per['live_perfect'])}",
        f"Live vs set-and-forget:          {_pct(per['live_model'], per['baseline'])}",
        f"Lineup changes: {changes / n:.0f}/roster/season",
    ]
    return LivePolicyReport(
        n_rosters=n,
        season=season,
        totals=per,
        changes_per_roster=changes / n,
        text="\n".join(lines),
    )


def _pct(a: float, b: float) -> str:
    if not b:
        return "n/a"
    return f"{(a - b) / b * 100:+.1f}%  ({a - b:+.1f}/roster)"
