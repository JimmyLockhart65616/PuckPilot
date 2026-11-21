"""The season horizon: a swap priced over every week still to come."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from puckpilot.season.outlook import SeasonOutlook, remaining_days, typical_week


def _p(pid, team, pos="C"):
    return SimpleNamespace(player_key=str(pid), nhl_player_id=pid, team=team, position=pos)


def _outlook(**kw):
    base = dict(
        leverage={"goals": 0.10, "hits": 0.0},
        rest_games={"AAA": 60, "BBB": 60},
        goalie_share={9: 0.5},
        rivals=11,
        weeks=17,
        weight=0.5,
    )
    return SeasonOutlook(**{**base, **kw})


RATES = {1: {"goals": 0.40, "hits": 1.0}, 2: {"goals": 0.10, "hits": 3.0}, 9: {"saves": 25.0}}


def test_a_scorer_for_a_hitter_gains_what_the_goals_are_worth_and_nothing_for_the_hits():
    o = _outlook()
    t = SimpleNamespace(player=_p(1, "AAA"), starts=3.0, drop=_p(2, "BBB"), drop_starts=3.0)
    weeks = {"AAA": 3, "BBB": 3}
    # each plays every game: 60 left; goals +0.30 a game x 60 x 0.10; hits are already won
    assert o.season_gain(t, RATES, weeks) == pytest.approx(0.30 * 60 * 0.10)


def test_a_streamers_four_game_week_is_his_share_of_games_not_his_week_repeated():
    o = _outlook()
    week = {"AAA": 4, "BBB": 2}
    streamer = SimpleNamespace(player=_p(1, "AAA"), starts=4.0, drop=None, drop_starts=0.0)
    bench = SimpleNamespace(player=_p(1, "AAA"), starts=2.0, drop=None, drop_starts=0.0)
    full = o.season_gain(streamer, RATES, week)
    half = o.season_gain(bench, RATES, week)
    assert full == pytest.approx(0.40 * 60 * 0.10)  # all four of four: every game left
    assert half == pytest.approx(full / 2)  # two of four: half of them


def test_a_player_with_no_game_this_week_counts_his_share_of_the_season():
    o = _outlook(leverage={"saves": 0.01})
    goalie = SimpleNamespace(player=_p(9, "AAA", "G"), starts=0.0, drop=None, drop_starts=0.0)
    assert o.season_gain(goalie, RATES, {}) == pytest.approx(25.0 * 0.5 * 60 * 0.01)


def _week(n, start, end):
    return SimpleNamespace(
        number=n,
        start=start,
        end=end,
        dates=lambda s=start: [s],
    )


def test_the_season_left_is_the_regular_weeks_after_this_one():
    weeks = [_week(1, "a1", "a7"), _week(2, "b1", "b7"), _week(3, "c1", "c7"), _week(4, "d1", "d7")]
    rt = SimpleNamespace(weeks=weeks, playoff_start_week=4)
    assert remaining_days(rt, weeks[0]) == ["b1", "c1"]
    assert typical_week(rt, weeks[0]).number == 2
    assert typical_week(rt, weeks[2]) is weeks[2]  # none left: this week stands in


def test_with_no_weight_a_swap_is_this_weeks_only():
    from puckpilot.season.week import AddTarget, _with_season

    t = AddTarget(
        player=_p(1, "AAA"), starts=3.0, drop_starts=0.0, deltas={}, helps=(), drop=None,
        score=0.2, timing="", gain=0.2,
    )  # fmt: skip
    assert _with_season(t, None, None, None, [], RATES) is t
    assert _with_season(t, _outlook(weight=0.0), None, None, [], RATES) is t
    o = _outlook()
    o._week_games = {(): {"AAA": 3}}
    got = _with_season(t, o, None, None, [], RATES)
    season = 0.40 * 60 * 0.10
    assert got.week_gain == 0.2 and got.season_gain == pytest.approx(season)
    assert got.gain == pytest.approx(0.2 + 0.5 * season) == pytest.approx(got.score)


def test_the_knob_is_off_by_default_and_never_negative():
    from puckpilot.season.authority import AuthorityError, TransactionAuthority

    assert TransactionAuthority().season_weight == 0.0
    with pytest.raises(AuthorityError):
        TransactionAuthority(season_weight=-0.1)


def test_parse_arm_reads_a_season_weight():
    from puckpilot.season.add_gate import parse_arm

    assert parse_arm("odds-daily-c50").season_weight == 0.5
    assert parse_arm("odds-daily").season_weight == 0.0


def test_a_pinned_stance_weights_its_categorys_season_leverage(monkeypatch):
    from puckpilot.season import odds, outlook
    from puckpilot.season import week as weekmod

    monkeypatch.setattr(odds, "leverage", lambda *a, **k: {"assists": 0.2, "ppp": 0.1})
    monkeypatch.setattr(weekmod, "expected_starts", lambda *a, **k: {})
    monkeypatch.setattr(outlook, "_club_games", lambda *a: {})
    weeks = [_week(1, "a1", "a7"), _week(2, "b1", "b7")]
    rt = SimpleNamespace(weeks=weeks, playoff_start_week=0)
    league = SimpleNamespace(all_cats=())
    ours = SimpleNamespace(team_key="us", players=())
    rival = SimpleNamespace(team_key="r", players=())
    plain = outlook.build_outlook(None, rt, league, weeks[0], ours, [rival], {}, None, None, 1.0)
    tilted = outlook.build_outlook(
        None, rt, league, weeks[0], ours, [rival], {}, None, None, 1.0,
        tilt={"assists": 1.5, "ppp": 0.0},
    )  # fmt: skip
    assert plain.leverage == {"assists": 0.2, "ppp": 0.1}
    assert tilted.leverage == {"assists": pytest.approx(0.3), "ppp": 0.0}


def test_a_card_says_both_parts_of_a_season_price():
    from puckpilot.season.week import AddTarget, _season_noted

    t = AddTarget(
        player=_p(1, "AAA"), starts=3.0, drop_starts=0.0, deltas={}, helps=(), drop=None,
        score=0.5, timing="", gain=0.5, week_gain=0.2, season_gain=0.6,
        detail={"week": ["Starts 3 of 3"]},
    )  # fmt: skip
    got = _season_noted(t, _outlook())
    assert got.detail["week"][0].startswith("Priced +0.50: this week +0.20, and over the rest")
    assert "+0.60 categories" in got.detail["week"][0] and got.detail["week"][1] == "Starts 3 of 3"
    plain = replace_gain(t)
    assert _season_noted(plain, _outlook()) is plain


def replace_gain(t):
    from dataclasses import replace

    return replace(t, week_gain=None, season_gain=None)


def test_a_waiver_claim_waits_for_the_league_waiver_period_within_the_week():
    from puckpilot.season.authority import AuthorityError, TransactionAuthority
    from puckpilot.season.week import _claim_wait

    rt = SimpleNamespace(waiver_days=2)
    fa = SimpleNamespace(on_waivers=False)
    claim = SimpleNamespace(on_waivers=True)
    days = ["d1", "d2", "d3", "d4", "d5"]
    assert _claim_wait(fa, rt, days) == 0
    assert _claim_wait(claim, rt, days) == 2
    assert _claim_wait(claim, SimpleNamespace(waiver_days=0), days) == 1  # never same-day
    assert _claim_wait(claim, SimpleNamespace(waiver_days=9), days) == 4  # within the week
    assert _claim_wait(claim, rt, ["d5"]) == 0  # no later morning to make it on
    assert TransactionAuthority().waiver_pool == 0
    with pytest.raises(AuthorityError):
        TransactionAuthority(waiver_pool=-1)
