import numpy as np
import pytest

from puckpilot.data import store
from puckpilot.draft.engine import DraftRules
from puckpilot.draft.replay import ReplayData
from puckpilot.engine.lineup_replay import GameValueModel
from puckpilot.engine.valuation import LeagueShape
from puckpilot.engine.waivers import best_move, blended_pg_value, expected_games

RULES = DraftRules(
    shape=LeagueShape(n_teams=2, slots=(("C", 1), ("D", 1), ("G", 1)), util_slots=0),
    rounds=3,
    caps={"C": 2, "D": 2, "G": 1},
    mins={"C": 1, "D": 1, "G": 1},
)


def _data():
    data = ReplayData()
    data.dates = [f"d{i}" for i in range(20)]
    # pid 1: hot recently (days 10-13); pid 2: cold; pid 3: no games yet
    data.skater = {
        1: {i: np.array([2.0, 1, 0, 0, 1, 4]) for i in range(10, 14)},
        2: {i: np.array([0.0, 0, 0, 0, 0, 1]) for i in range(10, 14)},
    }
    data.goalie = {}
    return data


def test_blended_pg_value_no_lookahead():
    data = _data()
    vm = GameValueModel(data, {1, 2})
    proj = {1: 1.0, 2: 1.0, 3: 1.0}
    # at day 10 no games have happened in [0,10) -> pure projection
    assert blended_pg_value(1, 10, data, vm, proj) == pytest.approx(1.0)
    # at day 14, four hot games pull pid 1 above projection, pid 2 below
    assert blended_pg_value(1, 14, data, vm, proj) > 1.0
    assert blended_pg_value(2, 14, data, vm, proj) < 1.0


def test_blended_pg_value_shrinks_toward_projection():
    data = _data()
    vm = GameValueModel(data, {1, 2})
    hot_game = vm.skater(np.array([2.0, 1, 0, 0, 1, 4]))
    blended = blended_pg_value(1, 14, data, vm, {1: 0.0})
    # 4 games vs shrink k=10 -> weight 4/14 of the hot form
    assert blended == pytest.approx(hot_game * 4 / 14, rel=1e-6)


def test_expected_games_skater_and_goalie():
    week = range(10, 15)  # disjoint from the trailing lookback [0, 10)
    avail = {1: {10, 11, 12, 13, 17}}
    assert expected_games(1, "C", week, 10, avail, {}) == 4.0
    # goalie: 4 team games in lookback, started 2 -> half of 3 week games
    gsd = {i: {10} for i in (5, 7)}
    avail_g = {10: {5, 6, 7, 8, 10, 11, 12}}
    got = expected_games(10, "G", week, 10, avail_g, gsd)
    assert got == pytest.approx(3 * (2 / 4))


def test_best_move_respects_minimums():
    data = _data()
    vm = GameValueModel(data, {1, 2})
    positions = {1: "C", 2: "C", 10: "D", 11: "G", 5: "D"}
    proj = {1: 5.0, 2: 0.1, 10: 0.2, 11: 0.3, 5: 9.0}
    avail = {p: set(range(20)) for p in positions}
    roster = [2, 10, 11]  # 1C 1D 1G, all at minimums
    week = range(0, 7)
    mv = best_move(
        roster, {1, 5}, week, 5, positions, data, vm, proj, avail, {}, RULES, min_gain=0.0
    )
    assert mv is not None
    add, drop, gain = mv
    # only same-position swaps are legal at minimums: D 5 must replace D 10,
    # never the G, and C 1 could only replace C 2
    assert (add, drop) in {(5, 10), (1, 2)}
    assert add == 5  # higher gain: 9.0 vs 0.2 beats 5.0 vs 0.1
    assert gain > 0


def test_best_move_returns_none_below_threshold():
    data = _data()
    vm = GameValueModel(data, {1, 2})
    positions = {1: "C", 2: "C"}
    proj = {1: 1.0, 2: 0.99}
    avail = {1: set(range(20)), 2: set(range(20))}
    mv = best_move(
        [2], {1}, range(0, 7), 5, positions, data, vm, proj, avail, {}, RULES, min_gain=5.0
    )
    assert mv is None


def test_proposals_store_roundtrip(tmp_path):
    conn = store.connect(tmp_path / "t.db")
    store.init_db(conn)
    pid = store.add_proposal(conn, add_pid=8478402, drop_pid=123, reason_json="{}")
    assert store.list_proposals(conn, "pending")[0]["id"] == pid
    store.set_proposal_status(conn, pid, "approved")
    rows = store.list_proposals(conn)
    assert rows[0]["status"] == "approved"
    assert store.list_proposals(conn, "pending") == []
