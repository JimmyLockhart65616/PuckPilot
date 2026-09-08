"""Head-to-head category scoring: weekly matchups, seeding, and a playoff bracket.

Roto rewards balance across a whole season; H2H rewards winning more categories
than one opponent each week. That difference makes punting a category far more
viable and makes week-to-week consistency matter, so the draft engine has to be
tuned against this objective rather than roto.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from puckpilot.draft.replay import category_totals
from puckpilot.engine.categories import Category


def round_robin_schedule(n_teams: int, n_weeks: int) -> list[list[tuple[int, int]]]:
    """Circle-method round robin, repeated/truncated to n_weeks.

    Returns one list of (team_a, team_b) pairings per week. n_teams must be even.
    """
    teams = list(range(n_teams))
    weeks: list[list[tuple[int, int]]] = []
    fixed, rot = teams[0], teams[1:]
    for w in range(n_weeks):
        r = w % len(rot)
        arrangement = [fixed] + rot[r:] + rot[:r]
        half = n_teams // 2
        weeks.append([(arrangement[i], arrangement[n_teams - 1 - i]) for i in range(half)])
    return weeks


def score_matchup(a_vals: np.ndarray, b_vals: np.ndarray) -> tuple[int, int, int]:
    """(wins, losses, ties) in categories for team A. Inputs already sign-flipped
    so higher is better in every category."""
    wins = int(np.sum(a_vals > b_vals))
    losses = int(np.sum(a_vals < b_vals))
    return wins, losses, len(a_vals) - wins - losses


@dataclass
class H2HResult:
    records: np.ndarray  # (T, 3) matchup W/L/T
    cat_records: np.ndarray  # (T, 3) category W/L/T, the seeding tiebreak
    seeds: np.ndarray  # (T,) 1 = top seed
    finish: np.ndarray  # (T,) 1 = champion; non-playoff teams keep their seed
    champion: int


def _matchup_points(rec: np.ndarray) -> np.ndarray:
    return rec[:, 0] + 0.5 * rec[:, 2]


def run_h2h_season(
    weekly_sk: np.ndarray,
    weekly_g: np.ndarray,
    skater_cats: tuple[Category, ...],
    goalie_cats: tuple[Category, ...],
    skater_keys: list[str],
    *,
    regular_weeks: int,
    playoff_teams: int,
    playoff_weeks: int,
) -> H2HResult:
    """Score a full H2H season from per-week category totals.

    weekly_sk: (T, weeks, n_skater_keys); weekly_g: (T, weeks, G_WIDTH).
    Regular season is a round robin over `regular_weeks`; the top
    `playoff_teams` then play a reseeded single-elimination bracket using the
    following weeks' real production.
    """
    n_teams = weekly_sk.shape[0]
    vals = category_totals(weekly_sk, weekly_g, skater_cats, goalie_cats, skater_keys)
    n_avail = vals.shape[1]
    regular_weeks = min(regular_weeks, n_avail)

    records = np.zeros((n_teams, 3))
    cat_records = np.zeros((n_teams, 3))
    for w, pairs in enumerate(round_robin_schedule(n_teams, regular_weeks)):
        for a, b in pairs:
            cw, cl, ct = score_matchup(vals[a, w], vals[b, w])
            cat_records[a] += (cw, cl, ct)
            cat_records[b] += (cl, cw, ct)
            if cw > cl:
                records[a] += (1, 0, 0)
                records[b] += (0, 1, 0)
            elif cl > cw:
                records[a] += (0, 1, 0)
                records[b] += (1, 0, 0)
            else:
                records[a] += (0, 0, 1)
                records[b] += (0, 0, 1)

    # seed on matchup points, then category win pct as the tiebreak
    cat_pct = _matchup_points(cat_records) / np.maximum(cat_records.sum(axis=1), 1)
    key = _matchup_points(records) + 1e-6 * cat_pct
    seeds = np.empty(n_teams, dtype=int)
    seeds[np.argsort(-key, kind="stable")] = np.arange(1, n_teams + 1)

    finish = seeds.copy()
    alive = list(np.argsort(-key, kind="stable")[:playoff_teams])
    week = regular_weeks
    eliminated_at: dict[int, int] = {}
    for _rnd in range(playoff_weeks):
        if len(alive) <= 1 or week >= n_avail:
            break
        alive.sort(key=lambda t: seeds[t])  # reseed each round
        winners, losers = [], []
        for i in range(len(alive) // 2):
            a, b = alive[i], alive[len(alive) - 1 - i]
            aw, al, _ = score_matchup(vals[a, week], vals[b, week])
            if aw > al or (aw == al and seeds[a] < seeds[b]):  # higher seed wins ties
                winners.append(a)
                losers.append(b)
            else:
                winners.append(b)
                losers.append(a)
        for t in losers:
            eliminated_at[t] = len(alive)
        alive = winners
        week += 1

    # champion 1st, then losers ranked by the round they went out
    champion = int(alive[0]) if alive else int(np.argmin(seeds))
    ordered = [champion] + sorted(eliminated_at, key=lambda t: (eliminated_at[t], seeds[t]))
    for place, t in enumerate(ordered, start=1):
        finish[t] = place
    non_playoff = [t for t in range(n_teams) if t not in ordered]
    for place, t in enumerate(sorted(non_playoff, key=lambda t: seeds[t]), start=len(ordered) + 1):
        finish[t] = place

    return H2HResult(
        records=records,
        cat_records=cat_records,
        seeds=seeds,
        finish=finish,
        champion=champion,
    )
