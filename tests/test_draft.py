import numpy as np
import pandas as pd
import pytest

from puckpilot.draft.engine import (
    AdpBot,
    DraftRules,
    PuntBot,
    Universe,
    VorpPolicy,
    eligible_positions,
)
from puckpilot.draft.replay import ReplayData, replay_roster, roto_standings
from puckpilot.draft.sim import run_draft, snake_order, two_prop_pvalue, wilson_ci
from puckpilot.engine.valuation import LeagueShape


def test_snake_order_reverses_each_round():
    assert snake_order(3, 2) == [0, 1, 2, 2, 1, 0]
    assert snake_order(2, 3) == [0, 1, 1, 0, 0, 1]


def test_eligible_positions_caps_and_forced_minimums():
    rules = DraftRules()
    # G at cap of 3 -> not eligible
    assert "G" not in eligible_positions({"G": 3}, rules, picks_left=10)
    # 17 picks done, only G minimum unmet with 1 pick left -> forced to G
    counts = {"C": 4, "L": 4, "R": 4, "D": 4, "G": 1}
    assert eligible_positions(counts, rules, picks_left=1) == {"G"}


def _universe(n_per_pos=6):
    rows = {}
    pid = 1
    for pos in ("C", "L", "R", "D", "G"):
        for i in range(n_per_pos):
            v = 10.0 - i
            rows[pid] = {
                "name": f"{pos}{i}",
                "position": pos,
                "vorp": v,
                "z_total": v + (1.0 if pos == "G" else 0.0),
                "adp_rank": float(pid),
                "z_plus_minus": 2.0 if pos == "D" else 0.0,
            }
            pid += 1
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "player_id"
    return Universe(df.sort_values("vorp", ascending=False))


SMALL_RULES = DraftRules(
    shape=LeagueShape(n_teams=2, slots=(("C", 1), ("L", 1), ("R", 1), ("D", 1), ("G", 1))),
    rounds=6,
    caps={"C": 2, "L": 2, "R": 2, "D": 2, "G": 2},
    mins={"C": 1, "L": 1, "R": 1, "D": 1, "G": 1},
)


def test_run_draft_unique_picks_and_valid_rosters():
    u = _universe()
    rng = np.random.default_rng(7)
    bots = [VorpPolicy(), AdpBot(noise_sd=2.0)]
    rosters = run_draft(u, bots, SMALL_RULES, rng)
    flat = [i for r in rosters for i in r]
    assert len(flat) == 12
    assert len(set(flat)) == 12  # no player drafted twice
    for r in rosters:
        pos_counts = {}
        for i in r:
            pos_counts[u.pos[i]] = pos_counts.get(u.pos[i], 0) + 1
        for pos, m in SMALL_RULES.mins.items():
            assert pos_counts.get(pos, 0) >= m
        for pos, cap in SMALL_RULES.caps.items():
            assert pos_counts.get(pos, 0) <= cap


def test_survival_discount_defers_market_survivors():
    from puckpilot.draft.engine import RosterValuePolicy

    rows = {
        1: {"name": "taken-soon", "position": "C", "vorp": 5.0, "z_total": 5.0, "adp_rank": 3.0},
        2: {
            "name": "will-survive",
            "position": "C",
            "vorp": 5.2,
            "z_total": 5.2,
            "adp_rank": 200.0,
        },
    }
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "player_id"
    u = Universe(df)
    rng = np.random.default_rng(0)
    avail = np.ones(len(u), dtype=bool)
    ctx = {"pick_no": 5, "next_pick_no": 18}

    # without the discount the slightly higher z wins; with it, the market
    # threat (adp 3, about to be taken) wins because adp 200 survives to next turn
    blind = RosterValuePolicy(survival_discount=0.0)
    aware = RosterValuePolicy(survival_discount=0.3)
    assert u.names[blind.pick(u, avail, {}, SMALL_RULES, 6, rng, ctx)] == "will-survive"
    assert u.names[aware.pick(u, avail, {}, SMALL_RULES, 6, rng, ctx)] == "taken-soon"


def test_punt_bot_devalues_punted_category():
    u = _universe()
    rng = np.random.default_rng(0)
    picky = PuntBot(("plus_minus",))
    avail = np.ones(len(u), dtype=bool)
    idx = picky.pick(u, avail, {}, SMALL_RULES, 6, rng)
    # D players carry z_plus_minus=2; punting +/- should avoid the top D
    assert u.pos[idx] != "D"


def _replay_fixture():
    data = ReplayData()
    data.dates = ["d0", "d1"]
    # three centers, only 1 C slot + 1 util; skater vector is 6 cats, goals first
    data.skater = {
        1: {0: np.array([3.0, 0, 0, 0, 0, 0])},
        2: {0: np.array([2.0, 0, 0, 0, 0, 0])},
        3: {0: np.array([1.0, 0, 0, 0, 0, 0]), 1: np.array([5.0, 0, 0, 0, 0, 0])},
    }
    data.goalie = {10: {0: np.array([1.0, 1.0, 0.0, 30.0, 1.0])}}
    return data


def test_replay_roster_slot_competition_and_util():
    data = _replay_fixture()
    shape = LeagueShape(n_teams=2, slots=(("C", 1), ("G", 1)), util_slots=1)
    positions = {1: "C", 2: "C", 3: "C", 10: "G"}
    scalar = {1: 3.0, 2: 2.0, 3: 1.0, 10: 1.0}
    sk, g = replay_roster([1, 2, 3, 10], positions, scalar, data, shape)
    # d0: player 1 takes C, player 2 takes util, player 3 benched; d1: player 3 plays
    assert sk[0] == pytest.approx(3.0 + 2.0 + 5.0)
    assert g[0] == 1.0  # goalie win counted


def test_replay_goalie_needs_open_slot():
    data = ReplayData()
    data.dates = ["d0"]
    data.goalie = {
        10: {0: np.array([1.0, 0, 2.0, 30.0, 1.0])},
        11: {0: np.array([1.0, 0, 0.0, 30.0, 1.0])},
    }
    shape = LeagueShape(n_teams=2, slots=(("G", 1),), util_slots=1)
    positions = {10: "G", 11: "G"}
    sk, g = replay_roster([10, 11], positions, {10: 2.0, 11: 1.0}, data, shape)
    assert g[0] == 1.0 and g[2] == 2.0  # only the higher-scalar goalie played


def test_roto_standings_directions():
    sk = np.array([[10.0, 0, 0, 0, 0, 0], [5.0, 0, 0, 0, 0, 0]])
    # team 0: better W and GAA (1 GA in 2h = 0.5); team 1: 6 GA in 2h = 3.0
    g = np.array([[2.0, 1.0, 1.0, 60.0, 2.0], [1.0, 0.0, 6.0, 60.0, 2.0]])
    points, finish = roto_standings(sk, g)
    assert finish[0] == 1
    assert points[0] > points[1]


def test_wilson_and_two_prop_sanity():
    lo, hi = wilson_ci(30, 100)
    assert lo < 0.3 < hi
    assert two_prop_pvalue(60, 100, 30, 100) < 0.01  # clearly better
    assert two_prop_pvalue(30, 100, 30, 100) == pytest.approx(0.5, abs=0.01)


def test_run_sims_smoke_on_fixture_db(db):
    from puckpilot.draft.sim import run_sims
    from tests.conftest import add_goalie_game, add_player, add_skater_game

    seasons = ("20232024", "20242025", "20252026")
    pid = 1
    for pos in ("C", "C", "C", "L", "L", "L", "R", "R", "R", "D", "D", "D"):
        add_player(db, pid, f"P{pid}", pos)
        for si, season in enumerate(seasons):
            for gm in range(12):
                add_skater_game(
                    db,
                    pid,
                    season,
                    int(f"9{si}{pid:02d}{gm:02d}"),
                    date=f"2025-01-{gm + 1:02d}",
                    goals=pid % 3,
                    assists=1,
                    shots=2 + pid % 4,
                )
        pid += 1
    for _ in range(2):
        add_player(db, pid, f"G{pid}", "G")
        for si, season in enumerate(seasons):
            for gm in range(12):
                add_goalie_game(
                    db,
                    pid,
                    season,
                    int(f"9{si}{pid:02d}{gm:02d}"),
                    date=f"2025-01-{gm + 1:02d}",
                    goals_against=pid % 3 + 1,
                    decision="W" if gm % 2 else "L",
                )
        pid += 1

    report = run_sims(db, n_sims=3, seed=1, rules=SMALL_RULES, top_k=1)
    assert report.n_sims == 3
    assert 0.0 <= report.engine_top_rate <= 1.0
    assert report.best_bot in report.archetypes
    assert "Draft sim" in report.text
