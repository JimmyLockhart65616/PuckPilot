from __future__ import annotations

import sqlite3
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

from puckpilot.draft.engine import (
    AdpBot,
    DraftRules,
    GreedyZBot,
    PuntBot,
    RosterValuePolicy,
    Universe,
)
from puckpilot.draft.replay import build_replay_data, replay_roster, roto_standings
from puckpilot.engine import projections
from puckpilot.engine.aggregate import season_aggregates
from puckpilot.engine.valuation import rank_players

UNIVERSE_SIZE = 350
PUNTABLE = ["plus_minus", "pim", "sog"]


def snake_order(n_teams: int, rounds: int) -> list[int]:
    order: list[int] = []
    for r in range(rounds):
        seats = range(n_teams)
        order.extend(seats if r % 2 == 0 else reversed(seats))
    return order


def run_draft(
    u: Universe, bots: list, rules: DraftRules, rng: np.random.Generator
) -> list[list[int]]:
    """Snake draft; returns per-seat lists of universe row indices."""
    n = len(bots)
    avail = np.ones(len(u), dtype=bool)
    rosters: list[list[int]] = [[] for _ in range(n)]
    counts: list[dict[str, int]] = [{} for _ in range(n)]
    remaining = [rules.rounds] * n
    order = snake_order(n, rules.rounds)
    seat_picks: dict[int, list[int]] = {}
    for i, seat in enumerate(order):
        seat_picks.setdefault(seat, []).append(i)
    next_pick = {}  # pick index -> this seat's following pick index (None on last round)
    for picks in seat_picks.values():
        for a, b in zip(picks, picks[1:], strict=False):
            next_pick[a] = b
        next_pick[picks[-1]] = None

    for i, seat in enumerate(order):
        ctx = {"pick_no": i, "next_pick_no": next_pick[i]}
        idx = bots[seat].pick(u, avail, counts[seat], rules, remaining[seat], rng, ctx)
        avail[idx] = False
        rosters[seat].append(idx)
        pos = u.pos[idx]
        counts[seat][pos] = counts[seat].get(pos, 0) + 1
        remaining[seat] -= 1
    return rosters


def build_universe(
    conn: sqlite3.Connection, target_season: str, train_seasons: tuple[str, ...]
) -> Universe:
    """Projection-ranked pool with pseudo-ADP.

    Pseudo-ADP = prior-season ACTUAL value order (what casual drafters chase);
    swaps for real Yahoo ADP when API access lands.
    """
    proj_sk, proj_g = projections.project(conn, target_season, list(train_seasons))
    ranked = rank_players(proj_sk, proj_g)
    act_sk, act_g = season_aggregates(conn, train_seasons[0])
    adp_ranked = rank_players(act_sk, act_g)
    adp = pd.Series(
        np.arange(1, len(adp_ranked) + 1, dtype=float), index=adp_ranked.index, name="adp_rank"
    )
    ranked = ranked.join(adp, how="left")
    ranked["adp_rank"] = ranked["adp_rank"].fillna(999.0)
    return Universe(ranked.head(UNIVERSE_SIZE))


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def two_prop_pvalue(k1: int, n1: int, k2: int, n2: int) -> float:
    """One-sided p-value that proportion 1 exceeds proportion 2 (pooled z-test)."""
    if n1 == 0 or n2 == 0:
        return 1.0
    p1, p2 = k1 / n1, k2 / n2
    pooled = (k1 + k2) / (n1 + n2)
    se = np.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    if se == 0:
        return 1.0
    return float(1 - norm.cdf((p1 - p2) / se))


@dataclass
class SimReport:
    n_sims: int
    top_k: int
    engine_top_rate: float
    engine_ci: tuple[float, float]
    engine_mean_finish: float
    archetypes: dict[str, dict]
    best_bot: str
    p_value: float
    passed: bool
    text: str


def _default_opponents(rng: np.random.Generator) -> list:
    punts = rng.choice(PUNTABLE, size=2, replace=False)
    return [
        *(AdpBot(noise_sd) for noise_sd in (2, 3, 4, 5, 6, 7, 8)),
        GreedyZBot(),
        GreedyZBot(),
        PuntBot((str(punts[0]),)),
        PuntBot((str(punts[1]),)),
    ]


def run_sims(
    conn: sqlite3.Connection,
    n_sims: int,
    seed: int | None = None,
    target_season: str = "20252026",
    train_seasons: tuple[str, ...] = ("20242025", "20232024"),
    rules: DraftRules | None = None,
    top_k: int = 3,
    engine_factory: Callable[[], object] | None = None,
    progress: Callable[[str], None] | None = None,
) -> SimReport:
    """Monte Carlo snake drafts vs bot field, each roster replayed over the
    REAL target season (walk-forward: projections never see target data).

    Success criterion: engine's top-k finish rate beats the best bot archetype's
    per-team rate (one-sided two-proportion z-test).
    """
    rules = rules or DraftRules()
    shape = rules.shape
    engine_factory = engine_factory or RosterValuePolicy
    rng = np.random.default_rng(seed)

    u = build_universe(conn, target_season, train_seasons)
    data = build_replay_data(conn, target_season)
    positions = dict(zip(u.ids.tolist(), u.pos.tolist(), strict=True))
    scalar = dict(zip(u.ids.tolist(), u.z_total.tolist(), strict=True))

    engine_finishes: list[int] = []
    bot_finishes: dict[str, list[int]] = defaultdict(list)
    n_cats = 6
    for s in range(n_sims):
        engine_seat = int(rng.integers(0, shape.n_teams))
        opponents = _default_opponents(rng)
        order = rng.permutation(len(opponents))
        bots = []
        oi = 0
        for seat in range(shape.n_teams):
            if seat == engine_seat:
                bots.append(engine_factory())
            else:
                bots.append(opponents[order[oi]])
                oi += 1

        rosters = run_draft(u, bots, rules, rng)
        sk = np.zeros((shape.n_teams, n_cats))
        g = np.zeros((shape.n_teams, 5))
        for t, ridx in enumerate(rosters):
            ids = [int(u.ids[i]) for i in ridx]
            sk[t], g[t] = replay_roster(ids, positions, scalar, data, shape)
        _points, finish = roto_standings(sk, g)

        engine_finishes.append(int(finish[engine_seat]))
        for t, bot in enumerate(bots):
            if t != engine_seat:
                bot_finishes[bot.name].append(int(finish[t]))
        if progress and (s + 1) % 50 == 0:
            progress(f"  {s + 1}/{n_sims} sims")

    k_e = sum(1 for f in engine_finishes if f <= top_k)
    engine_rate = k_e / n_sims
    archetypes = {}
    for name, fins in bot_finishes.items():
        k = sum(1 for f in fins if f <= top_k)
        archetypes[name] = {
            "n": len(fins),
            "top_rate": k / len(fins),
            "ci": wilson_ci(k, len(fins)),
            "mean_finish": float(np.mean(fins)),
            "k": k,
        }
    best_bot = max(archetypes, key=lambda a: archetypes[a]["top_rate"])
    b = archetypes[best_bot]
    p = two_prop_pvalue(k_e, n_sims, b["k"], b["n"])
    passed = engine_rate > b["top_rate"] and p < 0.05

    ci = wilson_ci(k_e, n_sims)
    baseline = top_k / shape.n_teams
    lines = [
        f"Draft sim: {n_sims} snake drafts, {shape.n_teams} teams, {rules.rounds} rounds, "
        f"target {target_season} (walk-forward), replay on real game logs",
        f"Random-seat baseline top-{top_k} rate: {baseline:.3f}",
        "",
        f"engine     top-{top_k} {engine_rate:.3f}  CI [{ci[0]:.3f}, {ci[1]:.3f}]  "
        f"mean finish {np.mean(engine_finishes):.2f}  (n={n_sims})",
    ]
    for name, a in sorted(archetypes.items(), key=lambda kv: -kv[1]["top_rate"]):
        lines.append(
            f"{name:<10} top-{top_k} {a['top_rate']:.3f}  "
            f"CI [{a['ci'][0]:.3f}, {a['ci'][1]:.3f}]  "
            f"mean finish {a['mean_finish']:.2f}  (n={a['n']})"
        )
    lines += [
        "",
        f"Best bot: {best_bot} ({b['top_rate']:.3f}); one-sided p (engine better): {p:.4f}",
        f"Criterion: engine top-{top_k} > best bot, p < 0.05 -> {'PASS' if passed else 'FAIL'}",
    ]
    return SimReport(
        n_sims=n_sims,
        top_k=top_k,
        engine_top_rate=engine_rate,
        engine_ci=ci,
        engine_mean_finish=float(np.mean(engine_finishes)),
        archetypes=archetypes,
        best_bot=best_bot,
        p_value=p,
        passed=passed,
        text="\n".join(lines),
    )
