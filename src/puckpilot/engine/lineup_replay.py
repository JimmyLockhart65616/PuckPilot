from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import numpy as np

from puckpilot.data.goalies import (
    GoalieStartSource,
    HindsightGoalieSource,
    NoisyGoalieSource,
)
from puckpilot.draft.replay import (
    G_GA,
    G_HOURS,
    G_SA,
    G_SHO,
    G_STARTS,
    G_WIDTH,
    G_WINS,
    ReplayData,
    build_replay_data,
)
from puckpilot.draft.sim import (
    _default_opponents,
    build_universe,
    keepers_for,
    run_draft,
)
from puckpilot.engine.categories import Category
from puckpilot.engine.lineup import optimize_lineup
from puckpilot.engine.valuation import LeagueShape
from puckpilot.league import DEFAULT_LEAGUE, LeagueConfig

# value assigned to a goalie start that the weekly minimum forces; large enough
# to outrank any real skater's expected value for that slot
_FORCED_START_VALUE = 1e6


class GameValueModel:
    """Scalar per-game fantasy value in z-like units, from the pool's real lines.

    Skater line: sum(stat_c / per-game SD of stat_c) over the LEAGUE'S skater
    categories. Goalie line: the league's goalie categories over their SDs, with
    rate cats (SV%, GAA) entering as impact — deviation from the pool average,
    volume-weighted — so one number per game keeps bench regret additive.
    """

    def __init__(
        self,
        data: ReplayData,
        pool_ids: set[int],
        goalie_cats: tuple[Category, ...] = DEFAULT_LEAGUE.goalie_cats,
    ):
        self.goalie_cats = goalie_cats
        sk = np.array(
            [v for pid in pool_ids for v in data.skater.get(pid, {}).values()], dtype=float
        )
        self.sk_sd = (
            np.where(sk.std(axis=0) > 0, sk.std(axis=0), 1.0)
            if len(sk)
            else np.ones(len(data.skater_keys))
        )
        g = np.array(
            [v for pid in pool_ids for v in data.goalie.get(pid, {}).values()], dtype=float
        )
        if len(g):
            self.pool_sv = 1.0 - g[:, G_GA].sum() / max(g[:, G_SA].sum(), 1.0)
            self.pool_ga60 = g[:, G_GA].sum() / max(g[:, G_HOURS].sum(), 1e-9)
            self.g_sd = np.array(
                [np.std(self._g_raw(g, c)) or 1.0 for c in goalie_cats], dtype=float
            )
        else:
            self.pool_sv, self.pool_ga60 = 0.9, 3.0
            self.g_sd = np.ones(len(goalie_cats))

    def _g_raw(self, vec: np.ndarray, cat: Category) -> np.ndarray:
        """Per-game contribution of one goalie category; vec may be 1-D or a stack."""
        v = np.asarray(vec, dtype=float)
        if cat.key == "wins":
            return v[..., G_WINS]
        if cat.key == "shutouts":
            return v[..., G_SHO]
        if cat.key == "saves":
            return v[..., G_SA] - v[..., G_GA]
        if cat.key == "save_pct":  # saves above what a pool-average goalie makes
            return (v[..., G_SA] - v[..., G_GA]) - self.pool_sv * v[..., G_SA]
        if cat.key == "gaa":  # goals prevented vs the pool rate (already sign-correct)
            return self.pool_ga60 * v[..., G_HOURS] - v[..., G_GA]
        raise KeyError(f"no goalie accumulator mapping for category {cat.key!r}")

    def skater(self, vec: np.ndarray) -> float:
        return float((vec / self.sk_sd).sum())

    def goalie(self, vec: np.ndarray) -> float:
        return float(
            sum(self._g_raw(vec, c) / sd for c, sd in zip(self.goalie_cats, self.g_sd, strict=True))
        )

    def actual(self, data: ReplayData, pid: int, didx: int) -> float:
        line = data.skater.get(pid, {}).get(didx)
        if line is not None:
            return self.skater(line)
        line = data.goalie.get(pid, {}).get(didx)
        if line is not None:
            return self.goalie(line)
        return 0.0


def projected_pg_values(frame, vm: GameValueModel, skater_keys: list[str]) -> dict[int, float]:
    """Expected per-game value in the model's units, from projected totals."""
    out: dict[int, float] = {}
    for pid, r in frame.iterrows():
        gp = max(float(r.get("proj_gp") or 0), 1.0)
        if r["position"] == "G":
            sa = float(r.get("shots_against") or 0) / gp
            ga = sa * (1.0 - float(r.get("save_pct") or 0))
            vec = np.zeros(G_WIDTH)
            vec[G_WINS] = float(r.get("wins") or 0) / gp
            vec[G_SHO] = float(r.get("shutouts") or 0) / gp
            vec[G_GA], vec[G_SA] = ga, sa
            vec[G_HOURS] = float(r.get("toi_hours") or 0) / gp
            vec[G_STARTS] = 1.0
            out[pid] = vm.goalie(vec)
        else:
            vec = np.array([float(r.get(k) or 0) / gp for k in skater_keys])
            out[pid] = vm.skater(vec)
    return out


def skater_availability(
    conn: sqlite3.Connection, season: str, data: ReplayData, pids: set[int]
) -> dict[int, set[int]]:
    """Morning knowledge: dates where the player's current team has a game.

    Current team = team from their most recent game log at or before the date
    (handles mid-season trades); injuries/scratches are unknowable in the
    morning, so an optimizer can start a player who then doesn't play.
    """
    team_dates: dict[str, set[int]] = {}
    didx = {d: i for i, d in enumerate(data.dates)}
    for h, a, d in conn.execute(
        "SELECT home_team, away_team, game_date FROM nhl_schedule"
        " WHERE season = ? AND game_type = 2",
        (season,),
    ):
        if d in didx:
            team_dates.setdefault(h, set()).add(didx[d])
            team_dates.setdefault(a, set()).add(didx[d])

    logs: dict[int, list[tuple[str, str]]] = {pid: [] for pid in pids}
    for pid, d, team in conn.execute(
        "SELECT player_id, game_date, team_abbrev FROM nhl_game_logs WHERE season = ?"
        " ORDER BY game_date",
        (season,),
    ):
        if pid in logs:
            logs[pid].append((d, team))

    out: dict[int, set[int]] = {}
    for pid, entries in logs.items():
        avail: set[int] = set()
        if entries:
            ei = 0
            team = entries[0][1]  # pre-debut: first known team
            for i, date in enumerate(data.dates):
                while ei < len(entries) and entries[ei][0] <= date:
                    team = entries[ei][1]
                    ei += 1
                if i in team_dates.get(team, set()):
                    avail.add(i)
        out[pid] = avail
    return out


def _daily_optimizer_total(
    roster: list[int],
    positions: dict[int, str],
    pg_value: dict[int, float],
    data: ReplayData,
    shape: LeagueShape,
    avail: dict[int, set[int]],
    goalie_src: GoalieStartSource,
    vm: GameValueModel,
    day_range: range | None = None,
    min_goalie_appearances: int = 0,
) -> float:
    """Optimizer-captured value over the season, or a date-index sub-range so a
    timeline of weekly roster changes can be scored segment by segment."""
    total = 0.0
    for i, assigned in iter_daily_assignments(
        lambda _day: roster,
        positions,
        pg_value,
        data,
        shape,
        avail,
        goalie_src,
        day_range=day_range,
        min_goalie_appearances=min_goalie_appearances,
    ):
        for pid in assigned:
            total += vm.actual(data, pid, i)
    return total


def iter_daily_assignments(
    roster_at: Callable[[int], list[int]],
    positions: dict[int, str],
    pg_value: dict[int, float],
    data: ReplayData,
    shape: LeagueShape,
    avail: dict[int, set[int]],
    goalie_src: GoalieStartSource,
    day_range: range | None = None,
    min_goalie_appearances: int = 0,
) -> Iterator[tuple[int, dict[int, str]]]:
    """Yield (date index, {player_id: slot}) for each day, using morning knowledge.

    `roster_at(day)` supplies the roster for that day so mid-season adds/drops
    replay correctly. When the league sets a weekly goalie floor, a goalie whose
    expected value is otherwise too low to start is forced in once the days left
    in the week can no longer cover the shortfall — missing the floor forfeits
    the goalie categories for the week, which costs far more than a bad start.
    """
    days = list(day_range if day_range is not None else range(len(data.dates)))
    weeks = [int(data.weeks[i]) if len(data.weeks) else 0 for i in days]
    # days remaining in the same week, counting the current one
    days_left_in_week: list[int] = []
    seen: dict[int, int] = {}
    for w in reversed(weeks):
        seen[w] = seen.get(w, 0) + 1
        days_left_in_week.append(seen[w])
    days_left_in_week.reverse()

    starts_this_week = 0
    prev_week: int | None = None
    for k, i in enumerate(days):
        date = data.dates[i]
        week = weeks[k]
        if week != prev_week:
            starts_this_week = 0
            prev_week = week
        goalie_starts = goalie_src.starts(date)
        cands = []
        for pid in roster_at(i):
            if positions.get(pid) == "G":
                p = goalie_starts.get(pid, 0.0)
                if p > 0:
                    cands.append((pid, "G", p * pg_value.get(pid, 0.0)))
            elif i in avail.get(pid, ()):
                cands.append((pid, positions.get(pid, "C"), pg_value.get(pid, 0.0)))

        if min_goalie_appearances:
            short = min_goalie_appearances - starts_this_week
            if short >= days_left_in_week[k]:  # must start every goalie chance from here
                cands = [
                    (pid, pos, max(v, _FORCED_START_VALUE) if pos == "G" else v)
                    for pid, pos, v in cands
                ]

        assigned = optimize_lineup(cands, shape)
        starts_this_week += sum(1 for pid in assigned if positions.get(pid) == "G")
        yield i, assigned


def _hindsight_total(
    roster: list[int],
    positions: dict[int, str],
    data: ReplayData,
    shape: LeagueShape,
    vm: GameValueModel,
) -> float:
    total = 0.0
    for i in range(len(data.dates)):
        vals: dict[int, float] = {}
        cands = []
        for pid in roster:
            v = vm.actual(data, pid, i)
            if v != 0.0:
                vals[pid] = v
                cands.append((pid, positions.get(pid, "C"), v))
        for pid in optimize_lineup(cands, shape):
            total += vals[pid]
    return total


def _set_and_forget_total(
    roster: list[int],
    positions: dict[int, str],
    pg_value: dict[int, float],
    data: ReplayData,
    shape: LeagueShape,
    vm: GameValueModel,
) -> float:
    cands = [(pid, positions.get(pid, "C"), pg_value.get(pid, 0.0)) for pid in roster]
    starters = set(optimize_lineup(cands, shape))
    total = 0.0
    for i in range(len(data.dates)):
        for pid in starters:
            total += vm.actual(data, pid, i)
    return total


@dataclass
class LineupReplayReport:
    n_rosters: int
    season: str
    goalie_accuracy: float
    hindsight: float
    optimizer_perfect: float
    optimizer_noisy: float
    baseline: float
    text: str


def bench_regret_report(
    conn: sqlite3.Connection,
    season: str = "20252026",
    train_seasons: tuple[str, ...] = ("20242025", "20232024", "20222023"),
    n_drafts: int = 2,
    seed: int | None = 123,
    goalie_accuracy: float = 0.9,
    league: LeagueConfig = DEFAULT_LEAGUE,
    progress: Callable[[str], None] | None = None,
) -> LineupReplayReport:
    """Replay drafted rosters over the real season under four lineup policies.

    hindsight-optimal (upper bound) >= optimizer w/ perfect goalie info >=
    optimizer w/ noisy goalie announcements >= set-and-forget baseline is the
    expected ordering; the gap optimizer-vs-baseline is what daily automation
    is worth, and hindsight-vs-optimizer is the bench regret.
    """
    from puckpilot.draft.engine import RosterValuePolicy

    rules = league.draft_rules()
    shape = rules.shape
    rng = np.random.default_rng(seed)

    skater_keys = [c.key for c in league.skater_cats]
    u = build_universe(conn, season, train_seasons, league)
    data = build_replay_data(conn, season, skater_keys)
    vm = GameValueModel(data, set(u.ids.tolist()), league.goalie_cats)
    pg_value = projected_pg_values(u.frame, vm, skater_keys)
    positions = dict(zip(u.ids.tolist(), u.pos.tolist(), strict=True))

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
    hind_g = HindsightGoalieSource(conn, season)
    noisy_g = NoisyGoalieSource(hind_g, accuracy=goalie_accuracy, rng=rng)

    sums = {"hindsight": 0.0, "perfect": 0.0, "noisy": 0.0, "baseline": 0.0}
    for k, roster in enumerate(rosters):
        sums["hindsight"] += _hindsight_total(roster, positions, data, shape, vm)
        sums["perfect"] += _daily_optimizer_total(
            roster,
            positions,
            pg_value,
            data,
            shape,
            avail,
            hind_g,
            vm,
            min_goalie_appearances=league.min_goalie_appearances,
        )
        sums["noisy"] += _daily_optimizer_total(
            roster,
            positions,
            pg_value,
            data,
            shape,
            avail,
            noisy_g,
            vm,
            min_goalie_appearances=league.min_goalie_appearances,
        )
        sums["baseline"] += _set_and_forget_total(roster, positions, pg_value, data, shape, vm)
        if progress and (k + 1) % 6 == 0:
            progress(f"  {k + 1}/{len(rosters)} rosters replayed")

    n = len(rosters)
    h, p, nz, b = (sums[k] / n for k in ("hindsight", "perfect", "noisy", "baseline"))
    lines = [
        f"Bench-regret replay: {n} drafted rosters x {season} season, "
        f"goalie announce accuracy {goalie_accuracy:.0%}",
        "",
        f"{'policy':<28} {'value/roster':>12} {'% of hindsight':>15}",
        f"{'hindsight-optimal':<28} {h:>12.1f} {'100.0%':>15}",
        f"{'optimizer (perfect G info)':<28} {p:>12.1f} {p / h:>14.1%}",
        f"{'optimizer (noisy G info)':<28} {nz:>12.1f} {nz / h:>14.1%}",
        f"{'set-and-forget baseline':<28} {b:>12.1f} {b / h:>14.1%}",
        "",
        f"Daily optimizer vs set-and-forget: +{(nz - b) / b:.1%} value "
        f"({nz - b:+.1f}/roster/season)",
        f"Bench regret vs hindsight (noisy): {h - nz:.1f}/roster/season",
    ]
    return LineupReplayReport(
        n_rosters=n,
        season=season,
        goalie_accuracy=goalie_accuracy,
        hindsight=h,
        optimizer_perfect=p,
        optimizer_noisy=nz,
        baseline=b,
        text="\n".join(lines),
    )
