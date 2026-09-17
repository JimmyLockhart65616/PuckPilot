"""A small draft with everything that goes wrong on a real one.

The integrity tests need more than a clean snake over numbered players: keepers
that make the pick sequence uneven, market-only rows with no projection, NaN
categories, a goalie with no skater stats, and two players who share a name
(there really are two Sebastian Ahos). Kept apart from any one test module so
the page, relay and invariant suites all exercise the same board.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from puckpilot.draft.board import DraftBoard
from puckpilot.draft.engine import AdpBot, DraftRules, GreedyZBot, RosterValuePolicy, Universe
from puckpilot.draft.feed import SimFeed
from puckpilot.engine.categories import CATALOG
from puckpilot.engine.valuation import LeagueShape

SHAPE = LeagueShape(
    n_teams=6,
    slots=(("C", 2), ("L", 1), ("R", 1), ("D", 2), ("G", 1)),
    util_slots=1,
    bench_slots=2,
)
RULES = DraftRules(
    shape=SHAPE,
    rounds=10,
    caps={"C": 4, "L": 3, "R": 3, "D": 4, "G": 2},
    mins={"C": 2, "L": 1, "R": 1, "D": 2, "G": 1},
)
CATS = tuple(CATALOG[k] for k in ("G", "A", "W", "SV%"))
PER_POS = 16
MARKET_NAMES = ("Gavin McKenna", "Ivar Stenberg", "Porter Martone")


def universe() -> Universe:
    rng = np.random.default_rng(7)
    rows, pid = {}, 1
    for pos in ("C", "L", "R", "D", "G"):
        for i in range(PER_POS):
            v = 18.0 - i * 1.3 + float(rng.normal(0, 0.4))
            name = f"{pos} Skater {chr(65 + i)}" if pos != "G" else f"Goalie {chr(65 + i)}"
            if pos in ("C", "D") and i == 2:
                name = "Sebastian Aho"  # one C, one D: a real collision
            goalie = pos == "G"
            rows[pid] = {
                "name": name,
                "position": pos,
                "team": ["EDM", "TOR", "CAR", "NYI"][i % 4],
                "vorp": v,
                "z_total": v,
                "adp_rank": float(pid + int(rng.integers(-6, 7))),
                "goals": np.nan if goalie else 40.0 - i,
                "assists": np.nan if goalie else 50.0 - i,
                "wins": 38.0 - i if goalie else np.nan,
                "save_pct": 0.915 - i * 0.002 if goalie else np.nan,
                "z_goals": 0.0 if goalie else 2.0 - i * 0.2,
                "z_assists": 0.0 if goalie else 1.8 - i * 0.2,
                "age": 21.0 + (i % 14),
                "train_gp": 40.0 + i * 12,
                "source": "projected",
            }
            pid += 1
    for j, name in enumerate(MARKET_NAMES):
        rows[900 + j] = {
            "name": name,
            "position": "L" if j < 2 else "R",
            "team": "TOR",
            "vorp": 4.0 - j,
            "z_total": 4.0 - j,
            "adp_rank": 20.0 + j * 9,
            "goals": np.nan,
            "assists": np.nan,
            "wins": np.nan,
            "save_pct": np.nan,
            "z_goals": np.nan,
            "z_assists": np.nan,
            "age": 19.0,
            "train_gp": np.nan,
            "source": "market",
        }
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "player_id"
    u = Universe(df.sort_values("vorp", ascending=False))
    u.has_market = u.adp_rank < 60
    return u


def board(my_seat: int = 0, keepers: bool = True) -> DraftBoard:
    u = universe()
    kept = {}
    if keepers:
        ids = [int(i) for i in u.ids]
        by_pos = {p: [i for i, pos in zip(ids, u.pos, strict=True) if pos == p] for p in "CLRDG"}
        kept = {0: [by_pos["C"][0]], 3: [by_pos["D"][0], by_pos["G"][0]], 5: [by_pos["L"][0]]}
    return DraftBoard(u, RULES, my_seat=my_seat, keepers=kept, roster_rounds=10)


def sim_feed(seed: int, skip_seats: set[int] | None = None) -> SimFeed:
    """A mixed field: noisy ADP drafters, a greedy one, and the engine itself."""
    bots = [AdpBot(2), AdpBot(4), GreedyZBot(), RosterValuePolicy(), AdpBot(6), AdpBot(3)]
    return SimFeed(bots, np.random.default_rng(seed), skip_seats=skip_seats)
