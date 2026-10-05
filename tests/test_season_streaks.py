"""Hot streaks: detection, role changes, what they are worth, and where they show."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from puckpilot.season import streaks as st

KEYS = ["goals", "hits", "pim", "ppp"]


def _data(lines: list[list[float]], pid=1):
    """One player's season: a game a date, in order."""
    return SimpleNamespace(
        dates=[f"2025-11-{d:02d}" for d in range(1, 31)],
        skater_keys=KEYS,
        skater={pid: {i: np.array(v, dtype=float) for i, v in enumerate(lines)}},
    )


PRIOR = {1: {"goals": 0.2, "hits": 2.0, "pim": 0.5, "ppp": 0.2}}


def _usage(minutes: list[float], pp: list[float] | None = None, pid=1):
    """A `form.Usage` stand-in: (date index, value) per game, in order."""
    return SimpleNamespace(
        _min={pid: list(enumerate(minutes))},
        _pp={pid: list(enumerate(pp or []))},
    )


def _quiet(n):
    return [[0, 2, 0, 0] for _ in range(n)]


def test_a_surge_well_beyond_what_was_expected_is_a_streak():
    hits = _quiet(10) + [[0, 6, 0, 0]] * 5  # 30 hits in 5, about 10 expected
    f = st.StreakFinder(_data(hits), PRIOR)
    [s] = f.streaks(1, "2025-11-16")
    assert s.key == "hits" and s.actual == 30 and s.expected == pytest.approx(10, rel=0.05)
    assert s.p < st.STREAK_ALPHA
    assert not s.role_up  # no ice-time data: never a role change
    assert s.momentum == st.MOMENTUM["plain"]["hits"]
    assert f.multipliers(1, "2025-11-16") == {"hits": 1 + st.MOMENTUM["plain"]["hits"]}


def test_nothing_is_a_streak_before_the_measured_range():
    """Streaks seen before a player's 10th game were measured to mean nothing."""
    early = _quiet(4) + [[0, 6, 0, 0]] * 5
    assert st.StreakFinder(_data(early), PRIOR).streaks(1, "2025-11-10") == ()


def test_an_ordinary_week_is_not_a_streak():
    f = st.StreakFinder(_data(_quiet(15)), PRIOR)
    assert f.streaks(1, "2025-11-16") == ()
    assert f.multipliers(1, "2025-11-16") == {}


def test_more_ice_time_makes_a_scoring_streak_a_role_change():
    goals = _quiet(10) + [[1, 2, 0, 0]] * 5
    plain = st.StreakFinder(_data(goals), PRIOR).streaks(1, "2025-11-16")
    assert [s.key for s in plain] == ["goals"] and plain[0].momentum == 0.0
    assert "mostly luck" in plain[0].describe()
    usage = _usage([15.0] * 10 + [18.0] * 5)
    [role] = st.StreakFinder(_data(goals), PRIOR, usage=usage).streaks(1, "2025-11-16")
    assert role.role_up and role.momentum == st.MOMENTUM["role"]["goals"]
    assert role.usage_now == 18.0 and role.usage_before == 15.0
    assert "a role change" in role.describe() and "18.0 min a game, 15.0 before" in role.describe()


def test_power_play_time_decides_a_ppp_streak():
    ppp = _quiet(10) + [[0, 2, 0, 1]] * 5
    usage = _usage([15.0] * 15, pp=[60.0] * 10 + [150.0] * 5)
    [s] = st.StreakFinder(_data(ppp), PRIOR, usage=usage).streaks(1, "2025-11-16")
    assert s.key == "ppp" and s.role_up  # minutes flat, power-play time up
    assert s.momentum == st.MOMENTUM["role"]["ppp"]
    assert "power-play time (2:30 a game, 1:00 before)" in s.describe()


def test_a_penalty_streak_is_expected_to_fade():
    pim = _quiet(10) + [[0, 2, 6, 0]] * 5
    [s] = st.StreakFinder(_data(pim), PRIOR).streaks(1, "2025-11-16")
    assert s.momentum < 0 and "expect a fade" in s.describe()


def test_a_role_change_in_a_category_with_no_measured_effect_moves_nothing():
    f = st.StreakFinder(
        _data(_quiet(10) + [[0, 2, 6, 0]] * 5),
        PRIOR,
        usage=_usage([15.0] * 10 + [18.0] * 5),
    )
    [s] = f.streaks(1, "2025-11-16")
    assert s.role_up and s.momentum == 0.0  # PIM with more ice time: not measured
    assert "no lasting effect was measured for PIM" in s.describe()


def test_form_rates_carry_a_streak_only_when_asked():
    from puckpilot.season.form import FormRates

    data = _data(_quiet(10) + [[0, 6, 0, 0]] * 5)
    finder = st.StreakFinder(data, PRIOR)
    plain = FormRates(data, PRIOR, streaks=finder).rates("2025-11-16")[1]["hits"]
    moved = FormRates(data, PRIOR, streaks=finder, momentum=True).rates("2025-11-16")[1]["hits"]
    assert moved == pytest.approx(plain * (1 + st.MOMENTUM["plain"]["hits"]))


def test_the_hot_list_puts_the_most_useful_streak_first():
    data = SimpleNamespace(
        dates=[f"2025-11-{d:02d}" for d in range(1, 31)],
        skater_keys=KEYS,
        skater={
            1: {i: np.array(v, dtype=float) for i, v in enumerate(_quiet(10) + [[0, 6, 0, 0]] * 5)},
            2: {i: np.array(v, dtype=float) for i, v in enumerate(_quiet(15))},
            3: {i: np.array(v, dtype=float) for i, v in enumerate(_quiet(10) + [[0, 2, 6, 0]] * 5)},
        },
    )
    prior = {pid: PRIOR[1] for pid in (1, 2, 3)}
    f = st.StreakFinder(data, prior)
    rates = {pid: {"hits": 2.0, "pim": 0.5} for pid in (1, 2, 3)}
    entries = [("Hitter", "TOR", "LW", 1, 5.0), ("Quiet", "TOR", "C", 2, 1.0)]
    entries.append(("Fighter", "MTL", "RW", 3, 2.0))
    hot = st.hot_players(entries, f, "2025-11-16", rates, {"TOR": 3, "MTL": 3}, {"hits": 2.0})
    assert [h.name for h in hot] == ["Hitter", "Fighter"]  # nobody quiet; the fade last
    assert hot[0].extra == {"hits": pytest.approx(st.MOMENTUM["plain"]["hits"] * 2.0 * 3)}
    assert hot[1].score < 0


def test_a_card_says_who_is_hot_and_why():
    from puckpilot.season.add_story import profile_lines

    finder = st.StreakFinder(_data(_quiet(10) + [[0, 6, 0, 0]] * 5), PRIOR)
    runtime = SimpleNamespace(nhl_season="20252026")
    cand = SimpleNamespace(
        nhl_player_id=1,
        name="Hitter",
        team="TOR",
        yahoo_eligible=frozenset({"LW"}),
        position="L",
        player_key="k",
    )
    lines = [x for x in _profile(profile_lines, runtime, cand, finder) if "hot -" in x]
    assert lines and lines[0].startswith("  hot - HIT: 30 in his last 5")


def _profile(profile_lines, runtime, cand, finder):
    import sqlite3

    from puckpilot.data import store

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    store.init_db(conn)
    return profile_lines(conn, runtime, cand, None, "2025-11-16", streaks=finder)


def test_an_arm_can_carry_streaks():
    from puckpilot.season.add_gate import parse_arm

    assert parse_arm("odds-daily-h-f25-x1-r-m", 2).momentum
    assert not parse_arm("odds-daily-h-f25-x1-r", 2).momentum


def test_the_tail_widens_for_penalty_minutes():
    poisson = st.tail(np.array([6.0]), np.array([2.0]), 1.0)[0]
    lumpy = st.tail(np.array([6.0]), np.array([2.0]), 3.84)[0]
    assert poisson < 0.02 < lumpy  # six minutes against two is rare - unless they come in lumps
