"""The per-category odds: exact where it matters, and honest about ties."""

from __future__ import annotations

import math

import numpy as np
import pytest

from puckpilot.engine.categories import resolve
from puckpilot.season.odds import (
    PIM_SIZES,
    GoalieGame,
    OddsModel,
    Side,
    compound_pmf,
    count_pmf,
    discrete_odds,
    normal_odds,
    side,
    trials_pmf,
)


def _mean_var(pmf):
    k = np.arange(len(pmf))
    m = float(np.dot(k, pmf))
    return m, float(np.dot((k - m) ** 2, pmf))


def test_a_count_is_poisson_until_widened():
    m, v = _mean_var(count_pmf(6.0))
    assert m == pytest.approx(6.0, abs=1e-6) and v == pytest.approx(6.0, abs=1e-3)
    m, v = _mean_var(count_pmf(6.0, phi=1.5))
    assert m == pytest.approx(6.0, abs=1e-4) and v == pytest.approx(9.0, abs=1e-2)
    assert count_pmf(0.0).tolist() == [1.0]


def test_penalty_minutes_come_in_lumps():
    """Same mean as a Poisson count, far wider - and mostly even numbers."""
    pmf = compound_pmf(16.0)
    m, v = _mean_var(pmf)
    assert m == pytest.approx(16.0, abs=1e-3)
    norm = sum(PIM_SIZES.values())
    second = sum(k * k * p for k, p in PIM_SIZES.items()) / norm
    mean_size = sum(k * p for k, p in PIM_SIZES.items()) / norm
    assert v == pytest.approx(16.0 / mean_size * second, rel=1e-3)
    assert v > 2.5 * m
    assert pmf[16] > pmf[15]  # a 2-minute world favours even totals


def test_chances_are_counted_exactly():
    pmf = trials_pmf([0.5, 0.5])
    assert pmf.tolist() == pytest.approx([0.25, 0.5, 0.25])


def test_even_sides_split_the_win_and_keep_the_tie():
    p = count_pmf(3.0)
    win, tie = discrete_odds(p, p, lead=0.0)
    loss = 1.0 - win - tie
    assert win == pytest.approx(loss)
    assert tie > 0.1  # three a side ties often; a normal curve would say never


def test_a_banked_lead_with_nothing_left_is_certain():
    none = np.array([1.0])
    assert discrete_odds(none, none, lead=2.0) == (1.0, 0.0)
    assert discrete_odds(none, none, lead=0.0) == (0.0, 1.0)
    assert discrete_odds(none, none, lead=-1.0) == (0.0, 0.0)


def test_a_normal_margin_with_integer_ties():
    win, tie = normal_odds(0.0, 4.0, integer=True)
    assert win == pytest.approx((1.0 - tie) / 2.0)
    assert 0.15 < tie < 0.25
    assert normal_odds(3.0, 0.0) == (1.0, 0.0)


def _g(p=1.0, sv=0.905, shots=27.0, win=0.5):
    return GoalieGame(p_start=p, p_win=win, shots=shots, save_pct=sv)


def test_save_percentage_favours_the_better_goalie_and_banked_leads_count():
    m = OddsModel()
    c = resolve("SV%")
    even = m.category(c, Side(goalies=[_g()]), Side(goalies=[_g()]))
    assert even.p_win == pytest.approx(0.5, abs=1e-6)
    # One game of save percentage is mostly noise: .930 against .890 over 27
    # shots is about 65%. Four games each narrows it.
    one = m.category(c, Side(goalies=[_g(sv=0.93)]), Side(goalies=[_g(sv=0.89)]))
    four = m.category(c, Side(goalies=[_g(sv=0.93)] * 4), Side(goalies=[_g(sv=0.89)] * 4))
    assert 0.6 < one.p_win < 0.7
    assert four.p_win > 0.75
    banked = Side(banked={"saves": 184.0, "shots_against": 200.0}, goalies=[_g()])
    behind = Side(banked={"saves": 176.0, "shots_against": 200.0}, goalies=[_g()])
    assert m.category(c, banked, behind).p_win > 0.8


def test_save_percentage_with_no_shots_on_one_side_goes_to_the_other():
    m = OddsModel()
    got = m.category(resolve("SV%"), Side(goalies=[_g()]), Side())
    assert (got.p_win, got.p_tie) == (1.0, 0.0)


def test_a_start_that_may_not_happen_is_most_of_a_goalies_variance():
    """Whether he plays at all dwarfs how many saves he makes if he does."""
    m = OddsModel()
    sure = m.category(resolve("SV"), Side(goalies=[_g(p=1.0)]), Side(goalies=[_g(p=1.0)]))
    coin = m.category(resolve("SV"), Side(goalies=[_g(p=0.5)] * 2), Side(goalies=[_g(p=1.0)]))
    # Same expected saves; the uncertain side is more likely to fall short or tie
    # far less often - it is simply wider.
    assert coin.ours == pytest.approx(sure.ours)
    assert coin.p_tie < sure.p_tie


def test_wins_are_counted_as_chances_per_goalie_game():
    m = OddsModel()
    got = m.category(
        resolve("W"),
        Side(banked={"wins": 2.0}, goalies=[_g(p=1.0, win=0.5)]),
        Side(banked={"wins": 2.0}, goalies=[_g(p=1.0, win=0.5)]),
    )
    # One game each at 50%: win 1/4, tie 1/2, loss 1/4.
    assert (got.p_win, got.p_tie) == (pytest.approx(0.25), pytest.approx(0.5))


def test_a_lower_is_better_category_is_turned_the_right_way():
    m = OddsModel()
    tight = Side(goalies=[_g(sv=0.93)], banked={"goals_against": 4.0, "toi_hours": 3.0})
    leaky = Side(goalies=[_g(sv=0.88)], banked={"goals_against": 12.0, "toi_hours": 3.0})
    got = m.category(resolve("GAA"), tight, leaky)
    assert got.ours < got.theirs and got.p_win > 0.9


def test_expected_categories_is_the_sum_with_ties_at_half():
    m = OddsModel(p_play=1.0)
    cats = (resolve("G"), resolve("SOG"))
    even = Side(skaters={"goals": 5.0, "sog": 60.0})
    w = m.week(cats, even, even)
    assert w.expected == pytest.approx(1.0, abs=1e-6)  # two coin flips
    assert w.of("goals").p_tie > 0


def test_healthy_skaters_miss_some_games():
    full = OddsModel(p_play=1.0).category(
        resolve("G"), Side(skaters={"goals": 10.0}), Side(skaters={"goals": 10.0})
    )
    part = OddsModel(p_play=0.9).category(
        resolve("G"), Side(skaters={"goals": 10.0}), Side(skaters={"goals": 10.0})
    )
    assert full.ours == pytest.approx(10.0) and part.ours == pytest.approx(9.0)


def test_a_side_is_built_from_starts_and_goalie_games():
    s = side(
        banked={"goals": 3.0},
        skater_games={1: 2.0},
        goalie_games={9: [0.6, 0.7]},
        rates={1: {"goals": 0.5}, 9: {"wins": 0.5, "saves": 24.5, "shots_against": 27.0}},
    )
    assert s.skaters["goals"] == pytest.approx(1.0)
    assert [g.p_start for g in s.goalies] == [0.6, 0.7]
    assert s.goalies[0].save_pct == pytest.approx(24.5 / 27.0)
    assert math.isclose(s.banked["goals"], 3.0)


# -- tonight's goalies --------------------------------------------------------


def _goalie_cats():
    return (resolve("W"), resolve("SV"), resolve("SV%"))


def test_with_everything_open_every_goalie_who_plays_starts():
    from puckpilot.season.odds import choose_goalies

    got = choose_goalies(
        OddsModel(),
        _goalie_cats(),
        Side(),
        {1: _g(p=0.9), 2: _g(p=0.8)},
        Side(goalies=[_g()] * 3),
        slots=2,
    )
    assert got.start == frozenset({1, 2})


def test_a_settled_week_protects_a_close_save_percentage():
    """Wins and saves locked up, save percentage a narrow lead: one more start
    can only cost the one category still open, so he sits."""
    from puckpilot.season.odds import choose_goalies

    ours = Side(banked={"wins": 5.0, "saves": 230.0, "shots_against": 250.0})  # .920
    theirs = Side(banked={"wins": 1.0, "saves": 91.0, "shots_against": 100.0})  # .910
    got = choose_goalies(
        OddsModel(), _goalie_cats(), ours, {1: _g(p=0.95, sv=0.905)}, theirs, slots=2
    )
    assert got.start == frozenset()
    assert got.cost_of(frozenset({1})) > 0


def test_the_weekly_minimum_is_a_floor_on_the_choice():
    from puckpilot.season.odds import choose_goalies

    ours = Side(banked={"wins": 5.0, "saves": 230.0, "shots_against": 250.0})
    theirs = Side(banked={"wins": 1.0, "saves": 91.0, "shots_against": 100.0})
    got = choose_goalies(
        OddsModel(),
        _goalie_cats(),
        ours,
        {1: _g(p=0.95, sv=0.905)},
        theirs,
        slots=2,
        at_least=1,
    )
    assert got.start == frozenset({1})
