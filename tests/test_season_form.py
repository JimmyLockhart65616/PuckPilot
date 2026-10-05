"""Form: per-category rates that have seen the season, usage, and the harness."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from puckpilot.data import store
from puckpilot.season.form import FormRates, Usage
from tests.conftest import add_player, add_skater_game

KEYS = ["goals", "hits"]


def _data(games: dict[int, dict[int, list[float]]], n_dates=10):
    """A ReplayData stand-in: dates and per-player skater lines by date index."""
    return SimpleNamespace(
        dates=[f"2025-11-{d:02d}" for d in range(1, n_dates + 1)],
        skater_keys=KEYS,
        skater={
            pid: {i: np.array(v, dtype=float) for i, v in g.items()} for pid, g in games.items()
        },
    )


def test_a_rate_is_the_season_so_far_shrunk_toward_the_prior():
    data = _data({1: {0: [1, 4], 1: [1, 6], 2: [0, 2]}})
    prior = {1: {"goals": 0.2, "hits": 1.0, "saves": 0.0}, 2: {"goals": 0.5, "hits": 0.0}}
    f = FormRates(data, prior, k={"goals": 8.0, "hits": 2.0})
    got = f.rates("2025-11-03")  # two games before the 3rd
    assert got[1]["goals"] == pytest.approx((2 + 8 * 0.2) / (2 + 8))
    assert got[1]["hits"] == pytest.approx((10 + 2 * 1.0) / (2 + 2))
    assert got[1]["saves"] == 0.0  # a key form does not cover is left as it was
    assert got[2] == prior[2]  # no games this season: the prior alone
    # Before any game, nothing has moved; the third game counts from the 4th on.
    assert f.rates("2025-11-01")[1] == prior[1]
    assert f.rates("2025-11-04")[1]["goals"] == pytest.approx((2 + 8 * 0.2) / (3 + 8))


def test_a_category_never_measured_gets_the_middle_k():
    from puckpilot.season.form import DEFAULT_FORM_K

    data = SimpleNamespace(
        dates=["2025-11-01", "2025-11-02"],
        skater_keys=["shp"],
        skater={1: {0: np.array([1.0])}},
    )
    f = FormRates(data, {1: {"shp": 0.0}})
    assert f.rates("2025-11-02")[1]["shp"] == pytest.approx(1 / (1 + DEFAULT_FORM_K))


def test_a_skater_value_comes_from_his_rates_and_a_goalie_keeps_his_own():
    data = _data({1: {0: [1, 4]}})
    vm = SimpleNamespace(skater=lambda vec: float(vec.sum()))
    f = FormRates(data, {1: {"goals": 0.0, "hits": 0.0}, 9: {"wins": 0.5}}, skaters={1})
    assert f.value(1, "2025-11-02", vm) == pytest.approx(
        sum(f.rates("2025-11-02")[1][k] for k in KEYS)
    )
    assert f.value(9, "2025-11-02", vm) is None  # a goalie
    assert f.value(5, "2025-11-02", vm) is None  # no prior


def test_usage_scales_the_prior_and_power_play_time_scales_ppp():
    data = SimpleNamespace(
        dates=["2025-11-01", "2025-11-02"],
        skater_keys=["goals", "ppp"],
        skater={1: {0: np.array([0.0, 0.0])}},
    )
    usage = SimpleNamespace(ratios=lambda pid, t: (1.44, 4.0))
    f = FormRates(data, {1: {"goals": 1.0, "ppp": 1.0}}, k={"goals": 1.0, "ppp": 1.0}, usage=usage)
    got = f.rates("2025-11-02")[1]
    assert got["goals"] == pytest.approx((0 + 1.0 * 1.2) / 2)  # 1.44 ** 0.5
    assert got["ppp"] == pytest.approx((0 + 1.0 * 1.2 * 2.0) / 2)  # and 4.0 ** 0.5


def _toi(db, pid, season, gid, date, toi, pp=None):
    add_skater_game(db, pid, season, gid, date=date, toi=toi)
    if pp is not None:
        store.upsert_skater_toi(db, [(gid, pid, season, date, None, pp, 0, None)])


def test_usage_compares_recent_ice_time_with_last_season(db):
    add_player(db, 1, "Riser", "C")
    for i in range(10):
        _toi(db, 1, "20242025", 100 + i, f"2025-01-{i + 1:02d}", "15:00", pp=60)
    dates = [f"2025-11-{d:02d}" for d in range(1, 8)]
    for i, d in enumerate(dates[:6]):
        _toi(db, 1, "20252026", 200 + i, d, "21:00", pp=150)
    db.commit()
    u = Usage(db, "20252026", dates)
    toi, pp = u.ratios(1, 6)
    assert toi == pytest.approx(1.4)  # 21 against 15
    assert pp == pytest.approx((150 + 30) / (60 + 30))  # smoothed: 2.0, at the clip
    assert u.ratios(1, 0) == (1.0, 1.0)  # nothing played yet
    assert u.ratios(99, 6) == (1.0, 1.0)  # nobody
    # A huge jump is clipped rather than believed.
    for i, d in enumerate(dates[:6]):
        _toi(db, 1, "20252026", 200 + i, d, "40:00", pp=900)
    db.commit()
    assert Usage(db, "20252026", dates).ratios(1, 6) == (1.5, 2.0)


def test_a_card_shows_ice_time_and_the_power_play(db):
    from puckpilot.season.add_story import usage_line

    add_player(db, 1, "Riser", "C")
    for i in range(10):
        _toi(db, 1, "20242025", 100 + i, f"2025-01-{i + 1:02d}", "15:00", pp=65)
    for i in range(6):
        _toi(db, 1, "20252026", 200 + i, f"2025-11-{i + 1:02d}", "18:30", pp=161)
    db.commit()
    p = SimpleNamespace(nhl_player_id=1, position="C")
    assert usage_line(db, p, "20252026", "2025-11-10") == (
        "ice time 18.5 min a game over his last 5 (15.0 last season); power play 2:41 (1:05)"
    )
    assert usage_line(db, SimpleNamespace(nhl_player_id=1, position="G"), "20252026", "x") == ""
    assert usage_line(db, p, "20252026", "2025-11-01") == ""  # no games before the date


# -- the harness ------------------------------------------------------------------


def _sample(prior, n, total, week_games, week_y):
    from puckpilot.season.form_gate import WINDOWS, Sample

    return Sample(
        prior=np.array(prior, dtype=float),
        windows={w: (n, np.array(total, dtype=float)) for w in WINDOWS},
        week=(week_games, np.array(week_y, dtype=float)),
        rest=(week_games, np.array(week_y, dtype=float)),
        usage={},
        pp_usage={},
    )


def test_the_harness_prefers_the_rate_that_was_true():
    """A player whose season says 1 a game, against a prior of 0.2, who then
    scores 1 a game: form must beat the prior, and more trust in form wins."""
    from puckpilot.season.form_gate import Batch, fit_per_category, score

    data = [_sample([0.2, 2.0], 20, [20.0, 40.0], 5, [5.0, 10.0]) for _ in range(30)]
    batch = Batch.of(data)
    keys = ["goals", "hits"]
    sc = score(keys, batch, {"pre": ("pre",), "form": (None, np.array([5.0, 5.0]))}, "x")
    assert sc.total("form", "week") < sc.total("pre", "week")
    assert sc.vs_pre("form", "week") < 0
    k = fit_per_category(keys, batch, None)
    assert k[0] == min(k)  # goals: the prior was wrong, so the smallest k wins


def test_poisson_deviance_is_zero_only_when_the_forecast_is_right():
    from puckpilot.season.form_gate import _deviance

    assert _deviance(np.array([3.0]), np.array([3.0]))[0] == pytest.approx(0.0)
    assert _deviance(np.array([0.0]), np.array([1.0]))[0] == pytest.approx(2.0)
    assert _deviance(np.array([3.0]), np.array([1.0]))[0] > 0


def test_the_odds_gate_refuses_an_unknown_rate_source(db):
    from puckpilot.league import DEFAULT_LEAGUE
    from puckpilot.season.calibration import build_cases

    with pytest.raises(ValueError, match="rates_mode"):
        build_cases(db, "20252026", ("20242025",), DEFAULT_LEAGUE, rates_mode="hot")


def test_an_arm_names_its_form_options():
    from puckpilot.season.add_gate import parse_arm

    spec = parse_arm("odds-daily-h-f25-x1-r-v-u", 2)
    assert (spec.form_rates, spec.form_value, spec.usage) == (True, True, True)


def test_rate_payloads_are_plain_json():
    """The card stores reasons as JSON; nothing here may leak numpy types."""
    data = _data({1: {0: [1, 4]}})
    f = FormRates(data, {1: {"goals": 0.2, "hits": 1.0}})
    json.dumps({k: float(v) for k, v in f.rates("2025-11-02")[1].items()})
