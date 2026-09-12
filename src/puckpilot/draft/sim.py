from __future__ import annotations

import sqlite3
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field

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
from puckpilot.draft.h2h import per_category_win_rate, run_h2h_season
from puckpilot.draft.replay import G_WIDTH, build_replay_data, replay_roster, roto_standings
from puckpilot.engine import projections
from puckpilot.engine.aggregate import season_aggregates
from puckpilot.engine.valuation import rank_players
from puckpilot.league import DEFAULT_LEAGUE, LeagueConfig

UNIVERSE_SIZE = 350
# peripheral categories a punt strategy might concede (scoring cats never are)
PUNTABLE_KEYS = {"plus_minus", "pim", "sog", "hits", "blocks", "ppp"}


def snake_order(n_teams: int, rounds: int) -> list[int]:
    order: list[int] = []
    for r in range(rounds):
        seats = range(n_teams)
        order.extend(seats if r % 2 == 0 else reversed(seats))
    return order


def simulate_keepers(
    u: Universe,
    n_teams: int,
    n_keepers: int,
    rng: np.random.Generator,
    pool_factor: float = 1.5,
) -> dict[int, list[int]]:
    """Plausible keeper assignment when the real keeper list is unknown.

    Keepers skew elite but aren't simply the top N — managers also hold mid-round
    value contracts. Modelled as a uniform draw from the top
    `n_teams * n_keepers * pool_factor` by ADP, dealt evenly across seats.
    Replace with the real list (seat -> player ids) whenever it is known.
    """
    if n_keepers <= 0:
        return {}
    need = n_teams * n_keepers
    pool_size = min(len(u), int(need * pool_factor))
    pool = np.argsort(u.adp_rank)[:pool_size]
    chosen = rng.choice(pool, size=need, replace=False)
    return {
        seat: [int(u.ids[i]) for i in chosen[seat * n_keepers : (seat + 1) * n_keepers]]
        for seat in range(n_teams)
    }


def effective_adp(adp_rank: np.ndarray, avail: np.ndarray) -> np.ndarray:
    """ADP re-ranked over available players only: 1 = first off the post-keeper board.

    Unavailable players keep a rank past the end so nothing ever selects them.
    """
    out = np.full(len(adp_rank), float(len(adp_rank) + 1))
    rows = np.flatnonzero(avail)
    out[rows[np.argsort(adp_rank[rows], kind="stable")]] = np.arange(1, len(rows) + 1)
    return out


def keepers_for(
    conn: sqlite3.Connection,
    u: Universe,
    season: str,
    league: LeagueConfig,
    rng: np.random.Generator,
    warn: Callable[[str], None] | None = None,
) -> dict[int, list[int]]:
    """The league's recorded keeper board for `season`, or a simulated draw when
    the config lists none. Keepers outside the universe are reported, never
    dropped quietly — a missing keeper leaves an elite player wrongly available.
    """
    from puckpilot.keepers import keeper_seats, resolve_keeper_ids

    n_teams = league.shape.n_teams
    names = league.keepers_for_season(season)
    if not names:
        return simulate_keepers(u, n_teams, league.n_keepers, rng)
    resolved, unmatched = resolve_keeper_ids(conn, names)
    known = set(u.ids.tolist())
    missing = [n for n, pid in resolved.items() if pid not in known]
    if warn and (unmatched or missing):
        if unmatched:
            warn(f"  keepers not found in nhl_players: {', '.join(unmatched)}")
        if missing:
            warn(f"  keepers outside the ranked universe: {', '.join(missing)}")

    # Where the league records WHO keeps whom, honour it. Availability is the
    # same either way, but which seat holds a keeper decides whose roster the
    # engine reasons about - and on a live board one of those seats is ours.
    owned: dict[int, list[int]] = {}
    for seat, owner_names in league.keeper_owners_for_season(season).items():
        ids = [resolved.get(n) for n in owner_names]
        unknown = [n for n in owner_names if resolved.get(n) not in known]
        if warn and unknown:
            warn(f"  seat {seat} keepers not on the board: {', '.join(unknown)}")
        owned[seat] = [pid for pid in ids if pid is not None and pid in known]

    return keeper_seats(
        [pid for pid in resolved.values() if pid in known], n_teams, rng, owned=owned
    )


def run_draft(
    u: Universe,
    bots: list,
    rules: DraftRules,
    rng: np.random.Generator,
    keepers: dict[int, list[int]] | None = None,
) -> list[list[int]]:
    """Snake draft; returns per-seat lists of universe row indices.

    `keepers` maps seat -> player ids already on that roster: they come off the
    board and pre-fill position counts, but do not consume any of the
    `rules.rounds` picks (rounds already excludes them).
    """
    n = len(bots)
    avail = np.ones(len(u), dtype=bool)
    rosters: list[list[int]] = [[] for _ in range(n)]
    counts: list[dict[str, int]] = [{} for _ in range(n)]
    remaining = [rules.rounds] * n

    if keepers:
        row_of = {int(pid): i for i, pid in enumerate(u.ids)}
        for seat, pids in keepers.items():
            for pid in pids:
                i = row_of.get(int(pid))
                if i is None or not avail[i]:
                    continue
                avail[i] = False
                rosters[seat].append(i)
                counts[seat][u.pos[i]] = counts[seat].get(u.pos[i], 0) + 1
        # Re-base ADP onto the post-keeper board. Keepers are elite, so leaving
        # ADP on the full-board scale would make every remaining player look
        # later-going than they are and wreck any pick-number comparison.
        u = u.with_adp(effective_adp(u.adp_rank, avail))

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
        ctx = {"pick_no": i, "next_pick_no": next_pick[i], "avail": avail}
        idx = bots[seat].pick(u, avail, counts[seat], rules, remaining[seat], rng, ctx)
        avail[idx] = False
        rosters[seat].append(idx)
        pos = u.pos[idx]
        counts[seat][pos] = counts[seat].get(pos, 0) + 1
        remaining[seat] -= 1
    return rosters


def build_universe(
    conn: sqlite3.Connection,
    target_season: str,
    train_seasons: tuple[str, ...],
    league: LeagueConfig = DEFAULT_LEAGUE,
) -> Universe:
    """Projection-ranked pool with pseudo-ADP.

    Pseudo-ADP = prior-season ACTUAL value order (what casual drafters chase);
    swaps for real Yahoo ADP when API access lands.
    """
    kw = {
        "shape": league.shape,
        "skater_cats": league.skater_cats,
        "goalie_cats": league.goalie_cats,
    }
    proj_sk, proj_g = projections.project(conn, target_season, list(train_seasons))
    ranked = rank_players(proj_sk, proj_g, **kw)
    act_sk, act_g = season_aggregates(conn, train_seasons[0])
    adp_ranked = rank_players(act_sk, act_g, **kw)
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
    # Engine's per-category win rate; empty under roto, which has no matchups.
    cat_win_rate: dict[str, float] = field(default_factory=dict)
    finishes: tuple[int, ...] = ()


def puntable_cats(league: LeagueConfig) -> list[str]:
    """Categories a rival plausibly concedes — the peripheral ones, never scoring."""
    return [c.key for c in league.skater_cats if c.key in PUNTABLE_KEYS] or [
        league.skater_cats[-1].key
    ]


def _default_opponents(rng: np.random.Generator, league: LeagueConfig = DEFAULT_LEAGUE) -> list:
    punts = rng.choice(puntable_cats(league), size=2, replace=False)
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
    train_seasons: tuple[str, ...] = ("20242025", "20232024", "20222023"),
    rules: DraftRules | None = None,
    top_k: int = 3,
    engine_factory: Callable[[], object] | None = None,
    league: LeagueConfig = DEFAULT_LEAGUE,
    scoring: str = "h2h",
    progress: Callable[[str], None] | None = None,
) -> SimReport:
    """Monte Carlo snake drafts vs bot field, each roster replayed over the
    REAL target season (walk-forward: projections never see target data).

    scoring='h2h' plays the league's real weekly category matchups plus a
    reseeded playoff bracket; 'roto' keeps the older season-total ranking as a
    regression baseline. Success criterion: engine's top-k finish rate beats the
    best bot archetype's per-team rate (one-sided two-proportion z-test).
    """
    rules = rules or league.draft_rules()
    shape = rules.shape
    engine_factory = engine_factory or RosterValuePolicy
    rng = np.random.default_rng(seed)

    skater_keys = [c.key for c in league.skater_cats]
    u = build_universe(conn, target_season, train_seasons, league)
    data = build_replay_data(conn, target_season, skater_keys)
    positions = dict(zip(u.ids.tolist(), u.pos.tolist(), strict=True))
    scalar = dict(zip(u.ids.tolist(), u.z_total.tolist(), strict=True))

    engine_finishes: list[int] = []
    bot_finishes: dict[str, list[int]] = defaultdict(list)
    cat_wins: dict[str, float] = defaultdict(float)
    cat_n = 0
    n_weeks = max(data.n_weeks, 1)
    for s in range(n_sims):
        engine_seat = int(rng.integers(0, shape.n_teams))
        opponents = _default_opponents(rng, league)
        order = rng.permutation(len(opponents))
        bots = []
        oi = 0
        for seat in range(shape.n_teams):
            if seat == engine_seat:
                bots.append(engine_factory())
            else:
                bots.append(opponents[order[oi]])
                oi += 1

        keepers = keepers_for(conn, u, target_season, league, rng)
        rosters = run_draft(u, bots, rules, rng, keepers)
        sk = np.zeros((shape.n_teams, n_weeks, len(skater_keys)))
        g = np.zeros((shape.n_teams, n_weeks, G_WIDTH))
        for t, ridx in enumerate(rosters):
            ids = [int(u.ids[i]) for i in ridx]
            sk[t], g[t] = replay_roster(ids, positions, scalar, data, shape)

        if scoring == "h2h":
            finish = run_h2h_season(
                sk,
                g,
                league.skater_cats,
                league.goalie_cats,
                skater_keys,
                regular_weeks=league.regular_weeks,
                playoff_teams=league.playoff_teams,
                playoff_weeks=league.playoff_weeks,
            ).finish
        else:
            _points, finish = roto_standings(
                sk, g, league.skater_cats, league.goalie_cats, skater_keys
            )

        engine_finishes.append(int(finish[engine_seat]))
        if scoring == "h2h":
            # Where the edge comes from, not just that there is one. A roster
            # can post a fine z_total and still lose 5-7 categories every week.
            rates = per_category_win_rate(
                sk,
                g,
                league.skater_cats,
                league.goalie_cats,
                skater_keys,
                engine_seat,
                league.regular_weeks,
            )
            for label, v in rates.items():
                cat_wins[label] += v
            cat_n += 1
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
    keeper_note = f", {league.n_keepers} keepers/team" if league.n_keepers else ""
    lines = [
        f"Draft sim: {n_sims} snake drafts, {shape.n_teams} teams, {rules.rounds} rounds"
        f"{keeper_note}, target {target_season} (walk-forward), replay on real game logs",
        f"League: {league.name} ({len(league.all_cats)} cats, {scoring.upper()} scoring)",
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
    cat_rate = {k: v / cat_n for k, v in cat_wins.items()} if cat_n else {}
    if cat_rate:
        # A finish rate says the engine wins; this says where. Sorted worst
        # first, because a category it loses while still spending picks on is
        # the one worth looking at.
        lines += ["", "Engine per-category win rate (worst first):"]
        ordered = sorted(cat_rate.items(), key=lambda kv: kv[1])
        for i in range(0, len(ordered), 6):
            chunk = ordered[i : i + 6]
            lines.append("  " + "   ".join(f"{lbl:<4}{v:.3f}" for lbl, v in chunk))
        won = sum(1 for v in cat_rate.values() if v > 0.5)
        lines.append(f"  wins {won}/{len(cat_rate)} categories on average")

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
        cat_win_rate=cat_rate,
        finishes=tuple(engine_finishes),
        text="\n".join(lines),
    )
