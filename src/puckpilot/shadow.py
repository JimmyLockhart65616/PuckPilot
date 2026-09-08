"""End-to-end shadow season: draft, daily lineups, and waivers over one real season.

Every engine so far has been validated in isolation — projections against
actuals, the draft engine against bots, the lineup optimizer against hindsight,
the waiver engine against standing pat. This composes all of them and scores the
result the way the league actually scores it: weekly head-to-head categories,
then a playoff bracket.

Honest caveat: the 11 opponents are bot archetypes that draft once and then only
set lineups — they never work the waiver wire and never adapt. A real league's
free-agent pool churns, so our waiver advantage here is an upper bound.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from puckpilot.data.goalies import HindsightGoalieSource
from puckpilot.draft.engine import RosterValuePolicy
from puckpilot.draft.h2h import H2HResult, run_h2h_season
from puckpilot.draft.replay import G_WIDTH, build_replay_data
from puckpilot.draft.sim import (
    _default_opponents,
    build_universe,
    keepers_for,
    run_draft,
)
from puckpilot.engine.lineup_replay import (
    GameValueModel,
    iter_daily_assignments,
    projected_pg_values,
    skater_availability,
)
from puckpilot.engine.waivers import Move, best_moves, budget_threshold
from puckpilot.league import DEFAULT_LEAGUE, LeagueConfig


@dataclass
class ShadowSeasonReport:
    season: str
    league_name: str
    seed: int | None
    engine_seat: int
    record: tuple[int, int, int]
    seed_rank: int
    finish: int
    made_playoffs: bool
    champion: bool
    cat_win_rate: dict[str, float]
    acquisitions_used: int
    acquisitions_cap: int | None
    moves: list[Move] = field(repr=False, default_factory=list)
    text: str = ""


def _weekly_category_totals(
    roster_at: Callable[[int], list[int]],
    positions: dict[int, str],
    pg_value: dict[int, float],
    data,
    shape,
    avail: dict[int, set[int]],
    goalie_src,
    min_goalie_appearances: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Accumulate real category production of whoever the optimizer started."""
    n_weeks = max(data.n_weeks, 1)
    sk = np.zeros((n_weeks, len(data.skater_keys)))
    g = np.zeros((n_weeks, G_WIDTH))
    for i, assigned in iter_daily_assignments(
        roster_at,
        positions,
        pg_value,
        data,
        shape,
        avail,
        goalie_src,
        min_goalie_appearances=min_goalie_appearances,
    ):
        w = int(data.weeks[i])
        for pid in assigned:
            line = data.skater.get(pid, {}).get(i)
            if line is not None:
                sk[w] += line
                continue
            line = data.goalie.get(pid, {}).get(i)
            if line is not None:
                g[w] += line
    return sk, g


def run_shadow_season(
    conn: sqlite3.Connection,
    season: str = "20252026",
    train_seasons: tuple[str, ...] = ("20242025", "20232024", "20222023"),
    seed: int | None = 11,
    league: LeagueConfig = DEFAULT_LEAGUE,
    manage_waivers: bool = True,
    progress: Callable[[str], None] | None = None,
) -> ShadowSeasonReport:
    rules = league.draft_rules()
    shape = rules.shape
    rng = np.random.default_rng(seed)
    say = progress or (lambda _m: None)

    skater_keys = [c.key for c in league.skater_cats]
    u = build_universe(conn, season, train_seasons, league)
    data = build_replay_data(conn, season, skater_keys)
    vm = GameValueModel(data, set(u.ids.tolist()), league.goalie_cats)
    proj_pg = projected_pg_values(u.frame, vm, skater_keys)
    positions = dict(zip(u.ids.tolist(), u.pos.tolist(), strict=True))
    avail = skater_availability(conn, season, data, set(u.ids.tolist()))
    goalie_src = HindsightGoalieSource(conn, season)

    didx = {d: i for i, d in enumerate(data.dates)}
    goalie_starts_by_day = {
        didx[d]: set(goalie_src.starts(d)) for d in data.dates if goalie_src.starts(d)
    }

    # --- draft -------------------------------------------------------------
    engine_seat = int(rng.integers(0, shape.n_teams))
    opponents = _default_opponents(rng, league)
    order = rng.permutation(len(opponents))
    bots, oi = [], 0
    for s in range(shape.n_teams):
        if s == engine_seat:
            bots.append(RosterValuePolicy())
        else:
            bots.append(opponents[order[oi]])
            oi += 1
    keepers = keepers_for(conn, u, season, league, rng, warn=say)
    rosters = [[int(u.ids[i]) for i in r] for r in run_draft(u, bots, rules, rng, keepers)]
    drafted = {pid for r in rosters for pid in r}
    say(f"  drafted from seat {engine_seat} ({len(rosters[engine_seat])} players)")

    # --- weekly waivers for our team only ----------------------------------
    n_days = len(data.dates)
    week_starts = list(range(7, n_days, 7))  # first week: settle, no moves
    timeline: list[tuple[int, list[int]]] = [(0, list(rosters[engine_seat]))]
    moves: list[Move] = []
    if manage_waivers:
        roster = list(rosters[engine_seat])
        fa = set(u.ids.tolist()) - drafted
        budget = league.season_acquisitions
        for wi, w0 in enumerate(week_starts):
            if budget is not None and budget <= 0:
                break
            week = range(w0, min(w0 + 7, n_days))
            weekly_cap = league.weekly_acquisitions or 1
            allowed = weekly_cap if budget is None else min(weekly_cap, budget)
            picked = best_moves(
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
                min_gain=budget_threshold(
                    budget, len(week_starts) - wi, league.weekly_acquisitions
                ),
                max_moves=allowed,
            )
            for add, drop, gain in picked:
                roster.remove(drop)
                roster.append(add)
                fa.discard(add)
                fa.add(drop)
                moves.append(Move(w0, add, drop, gain))
                if budget is not None:
                    budget -= 1
            if picked:
                timeline.append((w0, list(roster)))
        say(f"  {len(moves)} waiver moves over {len(week_starts)} weeks")

    def roster_at(day: int) -> list[int]:
        current = timeline[0][1]
        for start, r in timeline:
            if start <= day:
                current = r
            else:
                break
        return current

    # --- play the season ---------------------------------------------------
    sk = np.zeros((max(data.n_weeks, 1), len(skater_keys)))
    g = np.zeros((max(data.n_weeks, 1), G_WIDTH))
    all_sk = np.zeros((shape.n_teams, *sk.shape))
    all_g = np.zeros((shape.n_teams, *g.shape))
    for t in range(shape.n_teams):
        at = roster_at if t == engine_seat else (lambda _d, r=rosters[t]: r)
        all_sk[t], all_g[t] = _weekly_category_totals(
            at,
            positions,
            proj_pg,
            data,
            shape,
            avail,
            goalie_src,
            league.min_goalie_appearances,
        )
        say(f"  team {t + 1}/{shape.n_teams} season replayed")

    res: H2HResult = run_h2h_season(
        all_sk,
        all_g,
        league.skater_cats,
        league.goalie_cats,
        skater_keys,
        regular_weeks=league.regular_weeks,
        playoff_teams=league.playoff_teams,
        playoff_weeks=league.playoff_weeks,
    )

    w, losses, ties = (int(x) for x in res.records[engine_seat])
    cat_rec = res.cat_records[engine_seat]
    seed_rank = int(res.seeds[engine_seat])
    finish = int(res.finish[engine_seat])

    # per-category win rate over the regular season
    per_cat = _per_category_win_rate(all_sk, all_g, league, skater_keys, engine_seat, res)

    report = ShadowSeasonReport(
        season=season,
        league_name=league.name,
        seed=seed,
        engine_seat=engine_seat,
        record=(w, losses, ties),
        seed_rank=seed_rank,
        finish=finish,
        made_playoffs=seed_rank <= league.playoff_teams,
        champion=res.champion == engine_seat,
        cat_win_rate=per_cat,
        acquisitions_used=len(moves),
        acquisitions_cap=league.season_acquisitions,
        moves=moves,
    )

    cap = league.season_acquisitions
    lines = [
        f"Shadow season: {league.name}, {season}, engine at seat {engine_seat + 1}"
        f" of {shape.n_teams} (seed {seed})",
        "  draft -> daily lineups -> weekly waivers -> H2H categories -> playoffs",
        "",
        f"Regular season record: {w}-{losses}-{ties}   "
        f"(categories {int(cat_rec[0])}-{int(cat_rec[1])}-{int(cat_rec[2])})",
        f"Seed: {seed_rank} of {shape.n_teams}   "
        f"Playoffs: {'YES' if report.made_playoffs else 'missed'}   "
        f"Final: {finish}{' (CHAMPION)' if report.champion else ''}",
        f"Acquisitions used: {len(moves)}" + (f" of {cap}" if cap else ""),
        "",
        "Category win rate vs the field (regular season):",
    ]
    for cat in league.all_cats:
        rate = per_cat[cat.label]
        bar = "#" * int(round(rate * 20))
        lines.append(f"  {cat.label:<5} {rate:>6.1%}  {bar}")
    lines += [
        "",
        "Caveat: opponents draft once and set lineups but never work the waiver",
        "wire or adapt, so the waiver edge here is an upper bound.",
    ]
    report.text = "\n".join(lines)
    return report


def _per_category_win_rate(
    all_sk: np.ndarray,
    all_g: np.ndarray,
    league: LeagueConfig,
    skater_keys: list[str],
    seat: int,
    res: H2HResult,
) -> dict[str, float]:
    """Share of regular-season weeks our team beat the league median in each category."""
    from puckpilot.draft.h2h import round_robin_schedule
    from puckpilot.draft.replay import category_totals

    vals = category_totals(all_sk, all_g, league.skater_cats, league.goalie_cats, skater_keys)
    n_weeks = min(league.regular_weeks, vals.shape[1])
    schedule = round_robin_schedule(all_sk.shape[0], n_weeks)
    wins = np.zeros(vals.shape[2])
    played = 0
    for w, pairs in enumerate(schedule):
        for a, b in pairs:
            if seat not in (a, b):
                continue
            opp = b if a == seat else a
            wins += (vals[seat, w] > vals[opp, w]).astype(float)
            played += 1
    return {c.label: float(wins[j] / max(played, 1)) for j, c in enumerate(league.all_cats)}
