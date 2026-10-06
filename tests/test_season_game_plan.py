"""The week as a plan: targets, their price, what is given up - in words."""

from __future__ import annotations

from types import SimpleNamespace

from puckpilot.engine.categories import resolve
from puckpilot.season.game_plan import COST, GIVE_UP, SAFE, TARGET, TOSS_UP, GamePlan
from puckpilot.season.odds import CategoryOdds, WeekOdds


def _odds(**chances) -> WeekOdds:
    """label -> expected score; no ties, so p_win is the score."""
    return WeekOdds(tuple(CategoryOdds(resolve(k), p, 0.0, 0.0, 0.0) for k, p in chances.items()))


def _week(now, then=None, moves=0, adds_left=3, **kw):
    base = dict(
        week=2,
        opponent="Rival FC",
        odds=_odds(**now),
        planned=_odds(**then) if then is not None else None,
        targets=tuple(object() for _ in range(moves)),
        adds_left_week=adds_left,
        our_games=46,
        their_games=43,
        banked=False,
        days_left=7,
        ceiling={},
        bench_calls={},
        extra_game={},
    )
    base.update(kw)
    return SimpleNamespace(**base)


NOW = {"SOG": 0.54, "G": 0.46, "SV": 0.69, "HIT": 0.06, "BLK": 0.88, "A": 0.60}


def _roles(gp):
    return {r.category.label: r.role for r in gp.rows}


def test_each_category_gets_one_role():
    then = dict(NOW, SOG=0.68, G=0.55, SV=0.61, A=0.61)
    gp = GamePlan.from_week(_week(NOW, then, moves=2))
    assert _roles(gp) == {
        "SOG": TARGET,
        "G": TARGET,
        "SV": COST,
        "HIT": GIVE_UP,
        "BLK": SAFE,
        "A": TOSS_UP,  # moved a point - not what the moves are for
    }


def test_the_head_says_what_the_moves_are_worth():
    gp = GamePlan.from_week(_week(NOW, dict(NOW, SOG=0.68), moves=2))
    now, then = sum(NOW.values()), sum(NOW.values()) + 0.14
    assert gp.head() == f"Expect {now:.1f} of 6 -> {then:.1f} if the 2 proposed moves are made"


def test_with_no_moves_the_head_says_so_and_points_at_where_a_game_counts():
    gp = GamePlan.from_week(_week(NOW, extra_game={"SOG": 0.04, "G": 0.03, "PIM": 0.001}))
    assert gp.head().endswith("no pickup clears the bar this week")
    title, lines = gp.groups()[0]
    assert title == "An extra skater game counts most in" and lines == ["SOG, G"]


def test_no_acquisitions_left_is_said_plainly():
    gp = GamePlan.from_week(_week(NOW, adds_left=0))
    assert gp.head().endswith("no acquisitions left this week")
    assert "no acquisitions left to change it" in dict(gp.groups())["Giving up"][0]


def test_a_long_shot_is_listed_with_what_every_add_could_do_for_it():
    """HIT at 6% used to be a silent 'hold' because adds could reach it."""
    gp = GamePlan.from_week(_week(NOW, ceiling={"hits": 0.31}))
    assert dict(gp.groups())["Giving up"] == [
        "HIT 6% - all 3 acquisitions left, on it alone, would reach at most 31%"
    ]


def test_the_price_of_the_moves_is_named():
    gp = GamePlan.from_week(_week(NOW, dict(NOW, SOG=0.68, SV=0.61), moves=1))
    groups = dict(gp.groups())
    assert groups["Go after"] == ["SOG 54% -> 68%"]
    assert groups["Paying for it"] == ["SV 69% -> 61%"]


def test_a_rate_reads_as_a_chance_not_a_margin():
    """The old card printed save percentage's margin to one place: "-0.0"."""
    gp = GamePlan.from_week(_week({"SV%": 0.47, "G": 0.5}))
    text = gp.text()
    assert "SV% 47%" in text and "-0.0" not in text


def test_bench_calls_say_whether_the_lineup_has_a_choice_this_week():
    none = GamePlan.from_week(_week(NOW))
    some = GamePlan.from_week(
        _week(NOW, bench_calls={"2026-10-08": 1, "2026-10-06": 2}), lineup_by="this week's odds"
    )
    assert none.notes()[1] == "Lineup: no bench calls - everyone with a game starts"
    assert some.notes()[1] == "Lineup: 3 bench calls (Tue, Thu), decided by this week's odds"


def test_the_levers_line_counts_acquisitions_and_starts():
    gp = GamePlan.from_week(_week(NOW, dict(NOW, SOG=0.68), moves=2, banked=True))
    assert gp.notes()[0] == "3 acquisition(s) left this week, the plan uses 2 · starts left 46 v 43"


def test_no_odds_means_no_plan():
    assert GamePlan.from_week(SimpleNamespace(odds=None)) is None


def test_the_payload_is_all_strings_the_page_only_lays_out():
    gp = GamePlan.from_week(_week(NOW, dict(NOW, SOG=0.68), moves=1))
    p = gp.payload()
    assert p["title"] == "Week 2 vs Rival FC"
    assert all(isinstance(x, str) for g in p["groups"] for x in [g["title"], *g["lines"]])
    assert all(isinstance(x, str) for x in p["notes"])
