import numpy as np
import pytest

from puckpilot.data.goalies import HindsightGoalieSource, NoisyGoalieSource
from puckpilot.draft.replay import ReplayData
from puckpilot.engine.lineup import optimize_lineup, slot_instances
from puckpilot.engine.lineup_replay import (
    GameValueModel,
    _daily_optimizer_total,
    _hindsight_total,
    _set_and_forget_total,
)
from puckpilot.engine.valuation import LeagueShape
from tests.conftest import add_goalie_game, add_player

SHAPE = LeagueShape(n_teams=2, slots=(("C", 1), ("D", 1), ("G", 1)), util_slots=1)


def test_slot_instances_expands_shape():
    assert slot_instances(SHAPE) == ["C", "D", "G", "UTIL"]


def test_optimize_lineup_eligibility_and_maximization():
    players = [
        (1, "C", 5.0),
        (2, "C", 3.0),  # second C -> util
        (3, "D", 2.0),
        (4, "G", 1.0),
        (5, "C", 1.0),  # benched: util taken by better C
    ]
    out = optimize_lineup(players, SHAPE)
    assert out[1] == "C"
    assert out[2] == "UTIL"
    assert out[3] == "D"
    assert out[4] == "G"
    assert 5 not in out


def test_optimize_lineup_goalie_never_fills_skater_slot():
    players = [(1, "G", 9.0), (2, "G", 8.0)]  # one G slot only
    out = optimize_lineup(players, SHAPE)
    assert list(out.values()) == ["G"]
    assert out[1] == "G"


def test_optimize_lineup_prefers_empty_slot_over_negative_value():
    out = optimize_lineup([(1, "C", -2.0)], SHAPE)
    assert out == {}


def test_goalie_sources(db):
    add_player(db, 10, "Goalie A", "G")
    add_goalie_game(db, 10, "20252026", 1, date="2026-01-05", started=1, decision="W")
    add_goalie_game(db, 10, "20252026", 2, date="2026-01-07", started=0)
    hind = HindsightGoalieSource(db, "20252026")
    assert hind.starts("2026-01-05") == {10: 1.0}
    assert hind.starts("2026-01-07") == {}  # relief appearance is not a start
    assert hind.starts("2026-02-01") == {}

    all_misses = NoisyGoalieSource(hind, accuracy=0.0, rng=np.random.default_rng(0))
    assert all_misses.starts("2026-01-05") == {}
    perfect = NoisyGoalieSource(hind, accuracy=1.0, rng=np.random.default_rng(0))
    assert perfect.starts("2026-01-05") == {10: 1.0}


def _fixture_data():
    data = ReplayData()
    data.dates = ["d0", "d1"]
    data.skater = {
        1: {0: np.array([2.0, 1, 0, 0, 1, 4]), 1: np.array([1.0, 0, 0, 0, 0, 2])},
        2: {0: np.array([0.0, 1, 0, 0, 0, 1])},
        3: {1: np.array([3.0, 2, 1, 0, 1, 5])},
    }
    data.goalie = {10: {0: np.array([1.0, 0, 2.0, 30.0, 1.0])}}
    return data


def test_policy_ordering_hindsight_beats_all():
    data = _fixture_data()
    vm = GameValueModel(data, {1, 2, 3, 10})
    positions = {1: "C", 2: "C", 3: "D", 10: "G"}
    pg = {1: 1.0, 2: 0.5, 3: 0.8, 10: 0.6}
    avail = {1: {0, 1}, 2: {0}, 3: {1}}

    class Hind:
        def starts(self, date):
            return {10: 1.0} if date == "d0" else {}

    roster = [1, 2, 3, 10]
    h = _hindsight_total(roster, positions, data, SHAPE, vm)
    o = _daily_optimizer_total(roster, positions, pg, data, SHAPE, avail, Hind(), vm)
    b = _set_and_forget_total(roster, positions, pg, data, SHAPE, vm)
    assert h >= o >= b > 0


def test_game_value_model_orders_lines():
    data = _fixture_data()
    vm = GameValueModel(data, {1, 2, 3, 10})
    big = vm.skater(np.array([3.0, 2, 1, 0, 1, 5]))
    small = vm.skater(np.array([0.0, 1, 0, 0, 0, 1]))
    assert big > small
    assert vm.actual(data, 3, 1) == pytest.approx(big)
    assert vm.actual(data, 3, 0) == 0.0
