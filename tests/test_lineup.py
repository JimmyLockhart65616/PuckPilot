import numpy as np
import pytest

from puckpilot.data.goalies import HindsightGoalieSource, NoisyGoalieSource
from puckpilot.draft.replay import G_GA, G_HOURS, G_SA, G_STARTS, G_WIDTH, G_WINS, ReplayData
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
    gv = np.zeros(G_WIDTH)
    gv[G_WINS], gv[G_GA], gv[G_SA], gv[G_HOURS], gv[G_STARTS] = 1.0, 2.0, 30.0, 1.0, 1.0
    data.goalie = {10: {0: gv}}
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
    # on a fixture this small the three policies can tie exactly; allow float slack
    tol = 1e-9
    assert h + tol >= o
    assert o + tol >= b
    assert b > 0


def test_weekly_goalie_minimum_forces_a_start():
    """A goalie too weak to be worth starting must still be played when the
    week's remaining days can no longer satisfy the league minimum."""
    data = ReplayData()
    data.dates = ["2025-10-06", "2025-10-07"]  # same fantasy week
    gv = np.zeros(G_WIDTH)
    gv[G_WINS], gv[G_GA], gv[G_SA], gv[G_HOURS], gv[G_STARTS] = 1.0, 2.0, 30.0, 1.0, 1.0
    data.goalie = {10: {0: gv, 1: gv}}
    data.skater = {1: {0: np.array([2.0, 1, 0, 0, 1, 4])}}

    vm = GameValueModel(data, {1, 10})
    positions = {1: "C", 10: "G"}
    pg = {1: 5.0, 10: -99.0}  # goalie is actively bad by projection
    avail = {1: {0}}

    class AlwaysStarting:
        def starts(self, date):
            return {10: 1.0}

    roster = [1, 10]
    unconstrained = _daily_optimizer_total(
        roster, positions, pg, data, SHAPE, avail, AlwaysStarting(), vm
    )
    forced = _daily_optimizer_total(
        roster,
        positions,
        pg,
        data,
        SHAPE,
        avail,
        AlwaysStarting(),
        vm,
        min_goalie_appearances=2,
    )
    # unconstrained benches the negative-value goalie; the floor plays him twice
    assert forced > unconstrained
    assert forced == pytest.approx(unconstrained + 2 * vm.goalie(gv))


def test_game_value_model_orders_lines():
    data = _fixture_data()
    vm = GameValueModel(data, {1, 2, 3, 10})
    big = vm.skater(np.array([3.0, 2, 1, 0, 1, 5]))
    small = vm.skater(np.array([0.0, 1, 0, 0, 0, 1]))
    assert big > small
    assert vm.actual(data, 3, 1) == pytest.approx(big)
    assert vm.actual(data, 3, 0) == 0.0


# -- multi-position eligibility ---------------------------------------------
#
# `engine/lineup.py` took one position per player until in-season work needed
# Yahoo's real eligibility (172 of 562 mapped players are multi-eligible). The
# LP formulation was always the right shape for it; these pin that widening it
# changes nothing for single-position callers and gets the flex case right.

FLEX_SHAPE = LeagueShape(n_teams=12, slots=(("C", 1), ("R", 1)), util_slots=0)


def test_a_position_set_is_equivalent_to_the_string_for_one_position():
    """Every sim, replay and backtest still passes a bare string."""
    as_str = optimize_lineup([(1, "C", 5.0), (2, "C", 3.0), (3, "D", 2.0)], SHAPE)
    as_set = optimize_lineup(
        [(1, frozenset({"C"}), 5.0), (2, frozenset({"C"}), 3.0), (3, frozenset({"D"}), 2.0)],
        SHAPE,
    )
    assert as_str == as_set


def test_eligibility_widens_and_never_narrows():
    """A C/RW must still be able to take the C slot."""
    out = optimize_lineup([(1, frozenset({"C", "R"}), 5.0)], FLEX_SHAPE)
    assert out[1] in ("C", "R")


def test_the_flex_player_yields_the_slot_only_he_can_be_moved_out_of():
    """The case greedy gets wrong: taking the best slot for the best player
    strands a single-position player who has nowhere else to go."""
    players = [
        (1, frozenset({"C", "R"}), 5.0),  # flex, most valuable
        (2, "C", 4.0),  # C only
        (3, "R", 1.0),  # R only
    ]
    out = optimize_lineup(players, FLEX_SHAPE)
    assert out == {1: "R", 2: "C"}  # total 9.0; greedy would take C then R for 6.0


def test_a_multi_position_skater_can_still_fill_util():
    out = optimize_lineup(
        [(1, "C", 9.0), (2, frozenset({"C", "R"}), 8.0), (3, "D", 1.0), (4, "G", 1.0)], SHAPE
    )
    assert out[1] == "C"
    assert out[2] == "UTIL"


def test_an_unknown_position_is_ineligible_not_a_crash():
    assert optimize_lineup([(1, frozenset({"IR+"}), 9.0)], SHAPE) == {}
    assert optimize_lineup([(1, "Util", 9.0)], SHAPE) == {}
