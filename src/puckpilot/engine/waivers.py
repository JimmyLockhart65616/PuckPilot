from __future__ import annotations

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from puckpilot.data.goalies import HindsightGoalieSource
from puckpilot.draft.engine import DraftRules, RosterValuePolicy
from puckpilot.draft.replay import ReplayData, build_replay_data
from puckpilot.draft.sim import _default_opponents, build_universe, run_draft
from puckpilot.engine.lineup_replay import (
    GameValueModel,
    _daily_optimizer_total,
    projected_pg_values,
    skater_availability,
)

FORM_WINDOW_DAYS = 14
FORM_SHRINK_K = 10.0  # games of trailing form worth as much as the prior
MIN_WEEKLY_GAIN = 0.5  # don't churn the roster for crumbs
MAX_CANDIDATES = 40  # FA pool screened per week, by blended value
FORWARD_DAYS = 14  # realized-outcome horizon per move


def blended_pg_value(
    pid: int,
    today: int,
    data: ReplayData,
    vm: GameValueModel,
    proj_pg: dict[int, float],
) -> float:
    """Projection shrunk toward trailing actual form. Uses ONLY games before
    `today` (date index) — the backtest must never peek forward."""
    window_start = max(0, today - FORM_WINDOW_DAYS)
    vals = [vm.actual(data, pid, i) for i in data.player_dates(pid) if window_start <= i < today]
    prior = proj_pg.get(pid, 0.0)
    if not vals:
        return prior
    w = len(vals) / (len(vals) + FORM_SHRINK_K)
    return w * float(np.mean(vals)) + (1 - w) * prior


def expected_games(
    pid: int,
    pos: str,
    week: range,
    today: int,
    avail: dict[int, set[int]],
    goalie_starts_by_day: dict[int, set[int]],
) -> float:
    """Games the player can be expected to play during `week`.

    Skaters: team games on the schedule. Goalies: team games x trailing start
    share (announcements aren't knowable a week out)."""
    team_days = avail.get(pid, set())
    n_team = len(team_days & set(week))
    if pos != "G":
        return float(n_team)
    lookback = range(max(0, today - FORM_WINDOW_DAYS), today)
    recent_team = team_days & set(lookback)
    if not recent_team:
        return 0.0
    started = sum(1 for i in recent_team if pid in goalie_starts_by_day.get(i, set()))
    return n_team * (started / len(recent_team))


@dataclass
class Move:
    week_start: int
    add_pid: int
    drop_pid: int
    expected_gain: float
    realized_gain: float = 0.0


def best_move(
    roster: list[int],
    fa_pool: set[int],
    week: range,
    today: int,
    positions: dict[int, str],
    data: ReplayData,
    vm: GameValueModel,
    proj_pg: dict[int, float],
    avail: dict[int, set[int]],
    goalie_starts_by_day: dict[int, set[int]],
    rules: DraftRules,
    min_gain: float = MIN_WEEKLY_GAIN,
) -> tuple[int, int, float] | None:
    """Highest expected-week-value add/drop pair that keeps the roster legal."""

    def week_value(pid: int) -> float:
        return blended_pg_value(pid, today, data, vm, proj_pg) * expected_games(
            pid, positions.get(pid, "C"), week, today, avail, goalie_starts_by_day
        )

    counts: dict[str, int] = {}
    for pid in roster:
        p = positions.get(pid, "C")
        counts[p] = counts.get(p, 0) + 1

    cand = sorted(fa_pool, key=week_value, reverse=True)[:MAX_CANDIDATES]
    roster_wv = {pid: week_value(pid) for pid in roster}
    best: tuple[int, int, float] | None = None
    for c in cand:
        cv = week_value(c)
        pc = positions.get(c, "C")
        over_cap = counts.get(pc, 0) + 1 > rules.caps.get(pc, 99)
        for d in sorted(roster, key=lambda p: roster_wv[p]):
            pd_ = positions.get(d, "C")
            if over_cap and pd_ != pc:
                continue
            if pd_ != pc and counts.get(pd_, 0) - 1 < rules.mins.get(pd_, 0):
                continue
            gain = cv - roster_wv[d]
            if gain > min_gain and (best is None or gain > best[2]):
                best = (c, d, gain)
            break  # droppables sorted ascending: first legal is the cheapest
    return best


@dataclass
class WaiverBacktestReport:
    n_teams: int
    season: str
    moves_per_team: float
    hit_rate: float
    mean_realized_gain: float
    improvement_pct: float
    moves: list[Move] = field(repr=False, default_factory=list)
    text: str = ""


def waiver_backtest(
    conn: sqlite3.Connection,
    season: str = "20252026",
    train_seasons: tuple[str, ...] = ("20242025", "20232024"),
    n_teams_tested: int = 6,
    seed: int | None = 7,
    progress: Callable[[str], None] | None = None,
) -> WaiverBacktestReport:
    """Weekly waiver moves vs standing pat over the real season.

    One drafted league; tested teams make at most one recommended add/drop per
    week while the other 11 rosters stay frozen (documented approximation — a
    live league's FA pool churns more). Scoring is the daily-optimizer captured
    value with perfect goalie info for both arms, so the delta isolates the
    waiver engine itself.
    """
    rules = DraftRules()
    shape = rules.shape
    rng = np.random.default_rng(seed)

    u = build_universe(conn, season, train_seasons)
    data = build_replay_data(conn, season)
    vm = GameValueModel(data, set(u.ids.tolist()))
    proj_pg = projected_pg_values(u.frame, vm)
    positions = dict(zip(u.ids.tolist(), u.pos.tolist(), strict=True))
    avail = skater_availability(conn, season, data, set(u.ids.tolist()))

    hind = HindsightGoalieSource(conn, season)
    didx = {d: i for i, d in enumerate(data.dates)}
    goalie_starts_by_day = {didx[d]: set(hind.starts(d)) for d in data.dates if hind.starts(d)}

    opponents = _default_opponents(rng)
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
    rosters = [[int(u.ids[i]) for i in r] for r in run_draft(u, bots, rules, rng)]
    drafted = {pid for r in rosters for pid in r}

    n_days = len(data.dates)
    week_starts = list(range(7, n_days, 7))  # first week: settle, no moves
    all_moves: list[Move] = []
    improvements: list[float] = []
    hits = 0

    for t in range(min(n_teams_tested, shape.n_teams)):
        frozen = list(rosters[t])
        roster = list(rosters[t])
        fa = set(u.ids.tolist()) - drafted
        team_moves: list[Move] = []

        segments: list[tuple[int, list[int]]] = [(0, list(roster))]
        for w0 in week_starts:
            week = range(w0, min(w0 + 7, n_days))
            mv = best_move(
                roster,
                fa,
                week,
                w0,
                positions,
                data,
                vm,
                proj_pg,
                avail,
                goalie_starts_by_day,
                rules,
            )
            if mv:
                add, drop, gain = mv
                roster.remove(drop)
                roster.append(add)
                fa.discard(add)
                fa.add(drop)
                fwd = range(w0, min(w0 + FORWARD_DAYS, n_days))
                realized = sum(vm.actual(data, add, i) for i in fwd) - sum(
                    vm.actual(data, drop, i) for i in fwd
                )
                team_moves.append(Move(w0, add, drop, gain, realized))
                segments.append((w0, list(roster)))

        moving_total = 0.0
        for si, (start, seg_roster) in enumerate(segments):
            end = segments[si + 1][0] if si + 1 < len(segments) else n_days
            moving_total += _daily_optimizer_total(
                seg_roster,
                positions,
                proj_pg,
                data,
                shape,
                avail,
                hind,
                vm,
                day_range=range(start, end),
            )
        frozen_total = _daily_optimizer_total(
            frozen, positions, proj_pg, data, shape, avail, hind, vm
        )
        improvements.append((moving_total - frozen_total) / frozen_total)
        hits += sum(1 for m in team_moves if m.realized_gain > 0)
        all_moves.extend(team_moves)
        if progress:
            progress(
                f"  team {t + 1}: {len(team_moves)} moves, "
                f"{(moving_total - frozen_total) / frozen_total:+.1%} vs frozen"
            )

    n_moves = len(all_moves)
    report = WaiverBacktestReport(
        n_teams=min(n_teams_tested, shape.n_teams),
        season=season,
        moves_per_team=n_moves / max(1, min(n_teams_tested, shape.n_teams)),
        hit_rate=hits / n_moves if n_moves else 0.0,
        mean_realized_gain=(
            float(np.mean([m.realized_gain for m in all_moves])) if n_moves else 0.0
        ),
        improvement_pct=float(np.mean(improvements)),
        moves=all_moves,
    )
    report.text = "\n".join(
        [
            f"Waiver backtest: {report.n_teams} teams x {season}, weekly single add/drop, "
            f"other 11 rosters frozen",
            "",
            f"moves/team/season:        {report.moves_per_team:.1f}",
            f"hit rate ({FORWARD_DAYS}d fwd value): {report.hit_rate:.1%}",
            f"mean realized gain/move:  {report.mean_realized_gain:+.2f}",
            f"season value vs standing pat: {report.improvement_pct:+.1%}",
        ]
    )
    return report
