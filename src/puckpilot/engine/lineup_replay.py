from __future__ import annotations

import sqlite3
from collections.abc import Callable
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
    G_WINS,
    SKATER_KEYS,
    ReplayData,
    build_replay_data,
)
from puckpilot.draft.sim import _default_opponents, build_universe, run_draft
from puckpilot.engine.lineup import optimize_lineup
from puckpilot.engine.valuation import LeagueShape


class GameValueModel:
    """Scalar per-game fantasy value in z-like units, from the pool's real lines.

    Skater line: sum(stat_c / per-game SD of stat_c). Goalie line: wins and
    shutouts over their SDs plus save-impact and GAA-impact (deviation from the
    pool's per-game averages, volume-weighted) over theirs. One number per game
    makes bench regret additive across a season.
    """

    def __init__(self, data: ReplayData, pool_ids: set[int]):
        sk = np.array(
            [v for pid in pool_ids for v in data.skater.get(pid, {}).values()], dtype=float
        )
        self.sk_sd = (
            np.where(sk.std(axis=0) > 0, sk.std(axis=0), 1.0)
            if len(sk)
            else np.ones(len(SKATER_KEYS))
        )
        g = np.array(
            [v for pid in pool_ids for v in data.goalie.get(pid, {}).values()], dtype=float
        )
        if len(g):
            self.pool_sv = 1.0 - g[:, G_GA].sum() / g[:, G_SA].sum()
            self.pool_ga60 = g[:, G_GA].sum() / g[:, G_HOURS].sum()
            sv_imp = (g[:, G_SA] - g[:, G_GA]) - self.pool_sv * g[:, G_SA]
            gaa_imp = self.pool_ga60 * g[:, G_HOURS] - g[:, G_GA]
            self.g_sd = np.array(
                [
                    g[:, G_WINS].std() or 1.0,
                    g[:, G_SHO].std() or 1.0,
                    sv_imp.std() or 1.0,
                    gaa_imp.std() or 1.0,
                ]
            )
        else:
            self.pool_sv, self.pool_ga60 = 0.9, 3.0
            self.g_sd = np.ones(4)

    def skater(self, vec: np.ndarray) -> float:
        return float((vec / self.sk_sd).sum())

    def goalie(self, vec: np.ndarray) -> float:
        sv_imp = (vec[G_SA] - vec[G_GA]) - self.pool_sv * vec[G_SA]
        gaa_imp = self.pool_ga60 * vec[G_HOURS] - vec[G_GA]
        return float(
            vec[G_WINS] / self.g_sd[0]
            + vec[G_SHO] / self.g_sd[1]
            + sv_imp / self.g_sd[2]
            + gaa_imp / self.g_sd[3]
        )

    def actual(self, data: ReplayData, pid: int, didx: int) -> float:
        line = data.skater.get(pid, {}).get(didx)
        if line is not None:
            return self.skater(line)
        line = data.goalie.get(pid, {}).get(didx)
        if line is not None:
            return self.goalie(line)
        return 0.0


def projected_pg_values(frame, vm: GameValueModel) -> dict[int, float]:
    """Expected per-game value in the model's units, from projected totals."""
    out: dict[int, float] = {}
    for pid, r in frame.iterrows():
        gp = max(float(r.get("proj_gp") or 0), 1.0)
        if r["position"] == "G":
            sa = float(r.get("shots_against") or 0) / gp
            ga = sa * (1.0 - float(r.get("save_pct") or 0))
            hours = float(r.get("toi_hours") or 0) / gp
            vec = np.zeros(5)
            vec[G_WINS] = float(r.get("wins") or 0) / gp
            vec[G_SHO] = float(r.get("shutouts") or 0) / gp
            vec[G_GA], vec[G_SA], vec[G_HOURS] = ga, sa, hours
            out[pid] = vm.goalie(vec)
        else:
            vec = np.array([float(r.get(k) or 0) / gp for k in SKATER_KEYS])
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
) -> float:
    """Optimizer-captured value over the season, or a date-index sub-range so a
    timeline of weekly roster changes can be scored segment by segment."""
    total = 0.0
    for i in day_range if day_range is not None else range(len(data.dates)):
        date = data.dates[i]
        goalie_starts = goalie_src.starts(date)
        cands = []
        for pid in roster:
            if positions.get(pid) == "G":
                p = goalie_starts.get(pid, 0.0)
                if p > 0:
                    cands.append((pid, "G", p * pg_value.get(pid, 0.0)))
            elif i in avail.get(pid, ()):
                cands.append((pid, positions.get(pid, "C"), pg_value.get(pid, 0.0)))
        for pid in optimize_lineup(cands, shape):
            total += vm.actual(data, pid, i)
    return total


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
    train_seasons: tuple[str, ...] = ("20242025", "20232024"),
    n_drafts: int = 2,
    seed: int | None = 123,
    goalie_accuracy: float = 0.9,
    progress: Callable[[str], None] | None = None,
) -> LineupReplayReport:
    """Replay drafted rosters over the real season under four lineup policies.

    hindsight-optimal (upper bound) >= optimizer w/ perfect goalie info >=
    optimizer w/ noisy goalie announcements >= set-and-forget baseline is the
    expected ordering; the gap optimizer-vs-baseline is what daily automation
    is worth, and hindsight-vs-optimizer is the bench regret.
    """
    from puckpilot.draft.engine import DraftRules, RosterValuePolicy

    rules = DraftRules()
    shape = rules.shape
    rng = np.random.default_rng(seed)

    u = build_universe(conn, season, train_seasons)
    data = build_replay_data(conn, season)
    vm = GameValueModel(data, set(u.ids.tolist()))
    pg_value = projected_pg_values(u.frame, vm)
    positions = dict(zip(u.ids.tolist(), u.pos.tolist(), strict=True))

    rosters: list[list[int]] = []
    for _ in range(n_drafts):
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
        for ridx in run_draft(u, bots, rules, rng):
            rosters.append([int(u.ids[i]) for i in ridx])

    all_pids = {pid for r in rosters for pid in r}
    avail = skater_availability(conn, season, data, all_pids)
    hind_g = HindsightGoalieSource(conn, season)
    noisy_g = NoisyGoalieSource(hind_g, accuracy=goalie_accuracy, rng=rng)

    sums = {"hindsight": 0.0, "perfect": 0.0, "noisy": 0.0, "baseline": 0.0}
    for k, roster in enumerate(rosters):
        sums["hindsight"] += _hindsight_total(roster, positions, data, shape, vm)
        sums["perfect"] += _daily_optimizer_total(
            roster, positions, pg_value, data, shape, avail, hind_g, vm
        )
        sums["noisy"] += _daily_optimizer_total(
            roster, positions, pg_value, data, shape, avail, noisy_g, vm
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
