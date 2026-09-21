"""The weekly category plan.

Most of these pin defects found by running it against the real league, because
each one produced plausible-looking output that was wrong.
"""

from __future__ import annotations

import pytest

from puckpilot.engine.categories import resolve
from puckpilot.season.week import (
    CLOSE_BAND,
    CategoryOutlook,
    _categories_helped,
    _f,
    per_game_rates,
    project_totals,
)


class Frame:
    """The few bits of a projection frame these functions touch."""

    def __init__(self, rows, columns):
        self._rows = rows
        self.columns = columns

    def iterrows(self):
        return iter(self._rows.items())


CATS = (resolve("G"), resolve("PIM"), resolve("SV"), resolve("SA"), resolve("SV%"))
COLS = ["proj_gp", "goals", "pim", "saves", "shots_against", "save_pct"]


# -- NaN, the bug that made every category unreadable -----------------------


def test_a_missing_cell_does_not_poison_the_whole_category():
    """A goalie has no goals column and a skater no saves column; both arrive
    as NaN, and NaN is truthy, so `float(x or 0)` passes it straight through.
    On the real league this made all twelve category totals NaN."""
    assert _f(float("nan")) == 0.0
    assert _f(None) == 0.0
    assert _f("") == 0.0
    assert _f(3) == 3.0


def test_totals_survive_a_roster_of_mixed_positions():
    frame = Frame(
        {
            1: {"proj_gp": 80, "goals": 40, "pim": 20, "saves": float("nan")},
            2: {"proj_gp": 60, "goals": float("nan"), "saves": 1500, "shots_against": 1650},
        },
        COLS,
    )
    rates = per_game_rates(frame, CATS)
    totals = project_totals({1: 4, 2: 3}, rates, CATS)
    assert totals["goals"] == pytest.approx(2.0)
    assert totals["saves"] == pytest.approx(75.0)
    assert all(v == v for v in totals.values())  # no NaN anywhere


def test_a_rate_category_is_the_ratio_of_totals_not_a_mean_of_rates():
    frame = Frame(
        {
            1: {"proj_gp": 60, "saves": 1200, "shots_against": 1300},
            2: {"proj_gp": 60, "saves": 600, "shots_against": 700},
        },
        COLS,
    )
    rates = per_game_rates(frame, CATS)
    totals = project_totals({1: 3, 2: 1}, rates, CATS)
    expected = totals["saves"] / totals["shots_against"]
    assert totals["save_pct"] == pytest.approx(expected)


def test_no_games_means_no_contribution():
    frame = Frame({1: {"proj_gp": 80, "goals": 40}}, COLS)
    rates = per_game_rates(frame, CATS)
    assert project_totals({1: 0}, rates, CATS)["goals"] == 0.0


# -- the outlook ------------------------------------------------------------


def _outlook(ours, theirs, label="G"):
    return CategoryOutlook(category=resolve(label), ours=ours, theirs=theirs)


def test_a_tight_category_is_in_play():
    o = _outlook(10.0, 10.2)
    assert o.verdict == "close"
    assert o.in_play is True


def test_a_lopsided_category_is_not():
    assert _outlook(150.0, 96.0).verdict == "ahead"
    assert _outlook(41.0, 52.0).verdict == "behind"
    assert _outlook(150.0, 96.0).in_play is False


def test_the_band_is_relative_so_it_works_for_big_and_small_categories():
    """SV runs to 150 a week and goals to 11; one absolute threshold cannot
    serve both."""
    small = _outlook(11.0, 11.0 * (1 + CLOSE_BAND / 2))
    big = _outlook(150.0, 150.0 * (1 + CLOSE_BAND / 2))
    assert small.in_play and big.in_play


def test_two_empty_categories_are_not_a_division_by_zero():
    assert _outlook(0.0, 0.0).relative == 0.0


# -- what a candidate actually helps ----------------------------------------


def test_helps_reports_size_not_mere_presence():
    """Nearly every forward has goals, PIM and PPP above zero, so a presence
    test made every candidate read the same and told you nothing."""
    close = {
        "goals": _outlook(11.0, 10.0, "G"),
        "pim": _outlook(21.6, 21.7, "PIM"),
    }
    rate = {"goals": 0.3, "pim": 0.5}
    helps = _categories_helped(rate, games=4, close=close)
    assert helps == ("PIM +2.0", "G +1.2")


def test_a_negligible_contribution_is_not_listed():
    close = {"goals": _outlook(40.0, 30.0, "G")}  # gap of 10
    assert _categories_helped({"goals": 0.05}, games=2, close=close) == ()


def test_categories_that_are_not_close_are_ignored():
    assert _categories_helped({"goals": 1.0}, games=4, close={}) == ()


def test_rate_categories_are_left_out_of_helps():
    """A skater does not move save percentage, and a goalie's effect on it
    depends on the rest of the week's saves."""
    close = {"save_pct": _outlook(0.898, 0.902, "SV%")}
    assert _categories_helped({"saves": 30.0, "shots_against": 33.0}, 4, close) == ()


def test_at_most_three_categories_are_named():
    close = {
        k: _outlook(10.0, 10.0, lbl)
        for k, lbl in (("goals", "G"), ("pim", "PIM"), ("ppp", "PPP"), ("sog", "SOG"))
    }
    rate = dict.fromkeys(["goals", "pim", "ppp", "sog"], 1.0)
    assert len(_categories_helped(rate, games=4, close=close)) == 3
