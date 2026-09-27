"""The calibration gate's own arithmetic, on cases small enough to check by hand."""

from __future__ import annotations

import pytest

from puckpilot.engine.categories import resolve
from puckpilot.season.calibration import Case, _result, score
from puckpilot.season.odds import OddsModel, Side

G = resolve("G")


def _case(ours_goals, theirs_goals, outcome, day=0):
    return Case(
        season="s",
        week=1,
        day=day,
        days_left=1,
        ours=Side(skaters={"goals": ours_goals}),
        theirs=Side(skaters={"goals": theirs_goals}),
        outcome={"goals": outcome},
    )


def test_a_confident_right_call_scores_better_than_a_confident_wrong_one():
    m = OddsModel(p_play=1.0)
    right = score([_case(12.0, 1.0, 1.0)], m, (G,))
    wrong = score([_case(12.0, 1.0, 0.0)], m, (G,))
    assert right.logloss < 0.05 < wrong.logloss
    assert right.brier < wrong.brier


def test_a_tie_is_scored_by_the_chance_given_to_a_tie():
    """Log-loss on the realised result, three ways: a tie the model said was
    impossible is punished, not ignored."""
    m = OddsModel(p_play=1.0)
    s = score([_case(0.0, 0.0, 0.5)], m, (G,))  # nothing left: a certain tie
    assert s.logloss == pytest.approx(0.0, abs=1e-9)


def test_reliability_bins_compare_what_was_said_with_what_happened():
    m = OddsModel(p_play=1.0)
    cases = [_case(5.0, 5.0, 1.0 if i % 2 else 0.0) for i in range(200)]
    s = score(cases, m, (G,))
    ((lo, hi, n, predicted, realised),) = [b for b in s.reliability if b[2]]
    assert (lo, n) == (pytest.approx(0.5), 200)
    assert predicted == pytest.approx(0.5, abs=1e-6) and realised == pytest.approx(0.5)
    assert s.reliable


def test_the_error_in_categories_won_is_tracked_by_day_of_the_week():
    m = OddsModel(p_play=1.0)
    s = score([_case(5.0, 5.0, 1.0, day=0), _case(0.0, 0.0, 0.5, day=6)], m, (G,))
    assert s.ecats_error[0] == pytest.approx(0.5, abs=1e-6)
    assert s.ecats_error[6] == pytest.approx(0.0, abs=1e-9)


def test_results_read_rates_from_their_parts_and_respect_direction():
    sv = resolve("SV%")
    assert (
        _result(
            sv, {"saves": 91.0, "shots_against": 100.0}, {"saves": 180.0, "shots_against": 200.0}
        )
        == 1.0
    )
    assert (
        _result(sv, {"saves": 0.0, "shots_against": 0.0}, {"saves": 9.0, "shots_against": 10.0})
        == 0.0
    )
    gaa = resolve("GAA")
    assert (
        _result(
            gaa, {"goals_against": 4.0, "toi_hours": 3.0}, {"goals_against": 9.0, "toi_hours": 3.0}
        )
        == 1.0
    )
