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


# -- how far an add can reach --------------------------------------------------


class _Roster:
    def __init__(self, players):
        self.players = players


class _P:
    def __init__(self, pid, team="TOR", undroppable=False, out=False, slot="BN"):
        self.nhl_player_id = pid
        self.team = team
        self.is_undroppable = undroppable
        self.is_out = out
        self.on_ir = False
        self.selected_slot = slot
        self.player_key = f"p.{pid}"
        self.yahoo_eligible = frozenset()
        self.is_editable = True


def _headroom(db, adds_left, pool_rate, drop_rate):
    from puckpilot.data import store
    from puckpilot.season.week import add_headroom
    from tests.test_season_settings import _slots

    for gid, d in enumerate(["2026-10-05", "2026-10-06", "2026-10-07", "2026-10-08"], start=1):
        store.upsert_schedule_game(
            db,
            game_id=gid,
            season="20262027",
            game_type=2,
            game_date=d,
            start_time_utc=None,
            home_team="TOR",
            away_team="MTL",
        )
    db.commit()
    # Two active spots for two players: a full roster, so every add costs a drop.
    rt = _runtime_for_week(roster_positions=_slots(("C", 1, 1), ("BN", 1, 0)))
    rates = {
        1: {"goals": drop_rate},
        2: {"goals": drop_rate},
        9: {"goals": pool_rate},
        10: {"goals": pool_rate},
    }
    return add_headroom(
        db,
        rt,
        rt.week(1),
        _Roster([_P(1), _P(2)]),
        [_P(9), _P(10)],
        rates,
        (resolve("G"),),
        None,
        adds_left=adds_left,
    )


def _runtime_for_week(**over):
    from puckpilot.season.settings import LeagueRuntime, Week
    from tests.test_season_settings import payload

    return LeagueRuntime.from_payload(
        payload(**over), weeks=(Week(1, "2026-10-05", "2026-10-08"),), fetched_at="now"
    )


def test_headroom_counts_every_acquisition_still_available(db):
    """Costing it at one add called a 1.3-goal gap unreachable in week 1,
    which it plainly is not - the league allows three a week."""
    one = _headroom(db, 1, pool_rate=0.5, drop_rate=0.1)["goals"]
    three = _headroom(db, 3, pool_rate=0.5, drop_rate=0.1)["goals"]
    assert three > one


def test_headroom_is_net_of_what_the_drop_takes_with_him(db):
    cheap = _headroom(db, 1, pool_rate=0.5, drop_rate=0.0)["goals"]
    costly = _headroom(db, 1, pool_rate=0.5, drop_rate=0.4)["goals"]
    assert cheap > costly


def test_no_acquisitions_left_means_no_room(db):
    assert _headroom(db, 0, pool_rate=0.9, drop_rate=0.0)["goals"] == 0.0


def test_a_rate_category_stays_unmeasured_however_many_adds_are_left(db):
    from puckpilot.season.week import add_headroom

    rt = _runtime_for_week()
    got = add_headroom(
        db, rt, rt.week(1), _Roster([]), [], {}, (resolve("SV%"),), None, adds_left=3
    )
    assert got["save_pct"] is None


# -- who an add costs -------------------------------------------------------


def _week_games(db):
    """TOR plays four times this week; MTL once now and ten times in November."""
    from puckpilot.data import store

    gid = 1
    for d in ("2026-10-05", "2026-10-06", "2026-10-07", "2026-10-08"):
        store.upsert_schedule_game(
            db,
            game_id=gid,
            season="20262027",
            game_type=2,
            game_date=d,
            start_time_utc=None,
            home_team="TOR",
            away_team="OTT",
        )
        gid += 1
    for d in ["2026-10-08"] + [f"2026-11-{n:02d}" for n in range(1, 11)]:
        store.upsert_schedule_game(
            db,
            game_id=gid,
            season="20262027",
            game_type=2,
            game_date=d,
            start_time_utc=None,
            home_team="MTL",
            away_team="BUF",
        )
        gid += 1
    db.commit()


class _PerGame:
    def __init__(self, pg):
        self.pg = pg

    def per_game(self, pid, day):
        return float(self.pg.get(pid, 0.0))

    def per_game_tilted(self, pid, day, weights):
        return self.per_game(pid, day)


def _rp(key, name, pid, team, slot, status="", eligible=("C", "Util")):
    from puckpilot.season.roster import RosterPlayer

    return RosterPlayer(
        player_key=key,
        yahoo_id=key,
        name=name,
        team=team,
        primary_position="C",
        yahoo_eligible=frozenset(eligible),
        selected_slot=slot,
        nhl_player_id=pid,
        status=status,
    )


def _fa(key, name, pid, team="TOR"):
    from puckpilot.season.pool import PoolPlayer

    return PoolPlayer(
        player_key=key,
        name=name,
        team=team,
        primary_position="C",
        yahoo_eligible=frozenset({"C", "Util"}),
        nhl_player_id=pid,
    )


def _targets_for(db, players, slots, pool=None, rates=None, per_game=None, **kw):
    from puckpilot.league import LeagueConfig
    from puckpilot.season.roster import TeamRoster
    from puckpilot.season.week import _targets
    from tests.test_season_settings import _slots

    _week_games(db)
    rt = _runtime_for_week(roster_positions=_slots(*slots))
    ours = TeamRoster(league_key="999.l.1", team_key="t", date="2026-10-05", players=players)
    goals = resolve("G")
    return _targets(
        db,
        rt,
        LeagueConfig(skater_cats=(goals,), goalie_cats=()),
        rt.week(1),
        ours,
        pool if pool is not None else [_fa("fa.1", "Streamer", 9)],
        rates or {1: {"goals": 0.5}, 2: {"goals": 0.2}, 9: {"goals": 0.3}},
        {},
        None,
        _PerGame(per_game or {1: 1.5, 2: 0.5, 9: 1.0}),
        0.0,
        5,
        **kw,
    )


def test_a_regular_with_a_light_week_is_never_the_cheapest_drop(db):
    """Star plays once this week and eleven more times; Depth plays four times
    and that is his season. Ranked by this week's games, Star was the drop."""
    got = _targets_for(
        db,
        (_rp("p.1", "Star", 1, "MTL", "C"), _rp("p.2", "Depth", 2, "TOR", "C")),
        (("C", 2, 1),),
    )
    assert got and got[0].drop.name == "Depth"


def test_an_open_roster_spot_means_an_add_with_no_drop(db):
    got = _targets_for(
        db,
        (_rp("p.1", "Star", 1, "MTL", "C"), _rp("p.2", "Depth", 2, "TOR", "C")),
        (("C", 2, 1), ("BN", 1, 0)),
    )
    assert got and got[0].drop is None
    assert got[0].extra_starts == got[0].starts


def test_the_spot_an_injured_player_frees_for_ir_is_filled_without_a_drop(db):
    """Sanderson's case end to end: out, IR+-eligible, holding a spot. He is
    not cut for a streamer - he goes to IR and the streamer takes his spot."""
    got = _targets_for(
        db,
        (
            _rp("p.1", "Star", 1, "MTL", "C"),
            _rp("p.2", "Hurt", 2, "TOR", "C", status="O", eligible=("C", "IR+", "Util")),
        ),
        (("C", 2, 1), ("IR+", 1, 0)),
    )
    assert got and got[0].drop is None


def test_a_like_for_like_swap_is_legal_at_the_position_minimum(db):
    """Two centres against a minimum of two: dropping one for another centre
    leaves two. The check used to refuse it for leaving one."""
    from puckpilot.draft.engine import DraftRules
    from puckpilot.season.week import _cheapest_legal_drop

    class Cand:
        position = "C"

    ours = [_rp("p.1", "A", 1, "TOR", "C"), _rp("p.2", "B", 2, "TOR", "C")]
    got = _cheapest_legal_drop(Cand(), ours, {"p.1": 5.0, "p.2": 1.0}, {"C": 2}, DraftRules())
    assert got is not None and got.name == "B"


def test_only_as_many_goalies_count_as_there_are_g_slots(db):
    """Three goalies playing the same night fill two slots; the third's start
    cannot count. Summing every P(start) credited it anyway."""
    from puckpilot.data import store
    from puckpilot.season.goalies import StaticGoalieSource
    from puckpilot.season.roster import RosterPlayer
    from puckpilot.season.week import expected_starts
    from tests.test_season_settings import _slots

    for gid, (home, away) in enumerate((("TOR", "OTT"), ("MTL", "BUF")), start=1):
        store.upsert_schedule_game(
            db,
            game_id=gid,
            season="20262027",
            game_type=2,
            game_date="2026-10-05",
            start_time_utc=None,
            home_team=home,
            away_team=away,
        )
    db.commit()
    rt = _runtime_for_week(roster_positions=_slots(("G", 2, 1), ("BN", 1, 0)))
    goalies = [
        RosterPlayer(
            player_key=f"g.{i}",
            yahoo_id=str(i),
            name=f"G{i}",
            team=team,
            primary_position="G",
            yahoo_eligible=frozenset({"G"}),
            selected_slot="G",
            nhl_player_id=i,
        )
        for i, team in ((1, "TOR"), (2, "MTL"), (3, "OTT"))
    ]
    got = expected_starts(
        db,
        rt,
        rt.week(1),
        goalies,
        StaticGoalieSource({"2026-10-05": {1: 0.8, 2: 0.7, 3: 0.6}}),
        _PerGame({1: 1.0, 2: 1.0, 3: 1.0}),
        days=["2026-10-05"],
    )
    assert got == {1: 0.8, 2: 0.7}


# -- banked plus what is left ------------------------------------------------


def _live_week(db, **kw):
    import pandas as pd

    from puckpilot.data import store
    from puckpilot.league import LeagueConfig
    from puckpilot.season.roster import TeamRoster
    from puckpilot.season.week import build_week_plan
    from tests.test_season_settings import _slots

    for gid, d in enumerate(("2026-10-05", "2026-10-06", "2026-10-07", "2026-10-08"), 1):
        store.upsert_schedule_game(
            db,
            game_id=gid,
            season="20262027",
            game_type=2,
            game_date=d,
            start_time_utc=None,
            home_team="TOR",
            away_team="OTT",
        )
    db.commit()
    rt = _runtime_for_week(roster_positions=_slots(("C", 1, 1), ("BN", 1, 0)))
    ours = TeamRoster(
        league_key="l", team_key="t.5", date="d", players=(_rp("p.1", "Ours", 1, "TOR", "C"),)
    )
    theirs = TeamRoster(
        league_key="l", team_key="t.11", date="d", players=(_rp("p.2", "Theirs", 2, "OTT", "C"),)
    )
    frame = pd.DataFrame({"goals": [41.0, 20.5], "proj_gp": [82.0, 82.0]}, index=[1, 2])
    return build_week_plan(
        db,
        rt,
        LeagueConfig(skater_cats=(resolve("G"),), goalie_cats=()),
        rt.week(1),
        "Them",
        ours,
        theirs,
        [],
        frame,
        None,
        _PerGame({1: 1.0, 2: 1.0}),
        find_targets=kw.pop("find_targets", False),
        **kw,
    )


def test_the_week_is_what_is_banked_plus_what_is_left(db):
    """On Wednesday, Monday and Tuesday are Yahoo's numbers, not projections."""
    p = _live_week(db, banked_ours={"G": 3.0}, banked_theirs={"G": 1.0}, from_day="2026-10-07")
    g = p.outlook[0]
    assert g.banked_ours == 3.0 and g.banked_theirs == 1.0
    assert g.ours == pytest.approx(3.0 + 0.5 * 2)
    assert g.theirs == pytest.approx(1.0 + 0.25 * 2)
    assert (p.our_games, p.their_games, p.days_left) == (2, 2, 2)
    assert p.banked is True


def test_a_game_already_under_way_is_not_counted_twice(db):
    p = _live_week(
        db,
        banked_ours={"G": 3.0},
        banked_theirs={"G": 1.0},
        from_day="2026-10-07",
        started={"TOR", "OTT"},
    )
    assert p.outlook[0].ours == pytest.approx(3.0 + 0.5 * 1)


def test_the_spread_narrows_as_the_week_runs_out(db):
    early = _live_week(db, banked_ours={"G": 0.0}, banked_theirs={"G": 0.0}, from_day="2026-10-05")
    late = _live_week(db, banked_ours={"G": 0.0}, banked_theirs={"G": 0.0}, from_day="2026-10-08")
    assert early.outlook[0].sd > late.outlook[0].sd > 0
    assert late.outlook[0].sd == pytest.approx((0.5 + 0.25) ** 0.5)  # Poisson: var = mean


def test_with_nothing_banked_it_is_the_whole_week_as_before(db):
    p = _live_week(db)
    assert p.banked is False and p.days_left == 4
    assert p.outlook[0].ours == pytest.approx(2.0) and p.outlook[0].banked_ours is None


def test_a_banked_rate_is_rebuilt_from_its_components():
    """A week's SV% is the ratio of the week's totals, so it is carried as
    saves and shots against - SA is display-only but is the denominator."""
    from puckpilot.season.week import banked_components, project_totals

    base = banked_components({"SV": 90.0, "SA": 100.0, "SV%": 0.9, "W": 2.0})
    assert base == {"saves": 90.0, "shots_against": 100.0, "wins": 2.0}
    got = project_totals({}, {}, (resolve("SV%"), resolve("W")), base=base)
    assert got == {"save_pct": pytest.approx(0.9), "wins": 2.0}


def test_gaa_hours_are_backed_out_of_ga_and_gaa():
    from puckpilot.season.week import banked_components

    assert banked_components({"GA": 10.0, "GAA": 2.5}) == {
        "goals_against": 10.0,
        "toi_hours": 4.0,
    }


def test_penalty_minutes_are_lumpier_than_goals_and_wins_are_yes_or_no():
    from puckpilot.season.week import remaining_variance

    cats = (resolve("G"), resolve("PIM"), resolve("W"))
    var = remaining_variance(
        {1: 2.0}, {1: {"goals": 0.5, "pim": 0.5, "wins": 0.5}}, cats, totals={}
    )
    assert var["goals"] == pytest.approx(1.0)
    assert var["pim"] == pytest.approx(3.84)
    assert var["wins"] == pytest.approx(0.5)  # 2 starts x 0.5 x 0.5


def _odds(label, ours, theirs, sd, **kw):
    from puckpilot.season.week import CategoryOutlook

    return CategoryOutlook(category=resolve(label), ours=ours, theirs=theirs, sd=sd, **kw)


def test_the_bands_are_odds_not_shares_of_the_total():
    """Week 1's real case: +0.44 wins on 3.6 is +12% of the total - "safe" under
    the old band - and with two goalie starts either way it is a coin flip."""
    assert _odds("W", 4.0, 3.56, sd=1.5).band == "in play"
    assert _odds("SV", 135.0, 99.3, sd=21.3).band == "likely"
    assert _odds("PPP", 5.0, 8.0, sd=2.6).band == "long shot"


def test_a_lower_is_better_lead_reads_as_a_lead():
    o = _odds("GAA", 2.0, 3.0, sd=0.3)
    assert o.edge == pytest.approx(1.0)
    assert o.band == "likely" and o.verdict == "ahead"


def test_a_banked_certainty_has_no_spread_left():
    assert _odds("G", 9.0, 5.0, sd=0.0).band == "likely"
    assert _odds("G", 5.0, 9.0, sd=0.0).band == "long shot"
    assert _odds("G", 5.0, 5.0, sd=0.0).band == "in play"


def test_reachable_uses_the_odds_when_the_spread_is_known():
    """Behind by 3 with room to move 2.5: not covered outright, but no longer
    a long shot once the levers are pulled, so it is still reachable."""
    near = _odds("SOG", 50.0, 53.0, sd=2.0, lineup_room=0.0, add_room=2.5)
    far = _odds("SOG", 40.0, 53.0, sd=2.0, lineup_room=0.0, add_room=2.5)
    assert near.reachable and not far.reachable


def test_the_plan_carries_calibrated_odds_and_bands_by_them(db):
    from puckpilot.season.odds import OddsModel

    p = _live_week(
        db,
        banked_ours={"G": 6.0},
        banked_theirs={"G": 1.0},
        from_day="2026-10-08",
        odds_model=OddsModel(p_play=1.0),
    )
    g = p.outlook[0]
    assert g.p_win is not None and g.p_win > 0.95  # +5 with a day left
    assert g.band == "likely"
    assert p.expected == pytest.approx(g.expected)


def test_without_an_odds_model_nothing_claims_a_probability(db):
    p = _live_week(db, banked_ours={"G": 6.0}, banked_theirs={"G": 1.0}, from_day="2026-10-08")
    assert p.odds is None and p.expected is None and p.outlook[0].p_win is None


def test_every_run_logs_what_the_odds_said(db):
    import json

    from puckpilot.season.odds import OddsModel, log_week

    p = _live_week(
        db,
        banked_ours={"G": 3.0},
        banked_theirs={"G": 1.0},
        from_day="2026-10-07",
        odds_model=OddsModel(p_play=1.0),
    )
    assert log_week(db, "jimmy", "l", "t.5", p, "2026-10-07")
    row = db.execute("SELECT * FROM week_odds_log").fetchone()
    assert row["week"] == 1 and row["days_left"] == 2
    assert row["expected"] == pytest.approx(p.expected, abs=1e-4)
    assert json.loads(row["cats_json"])["goals"][0] == pytest.approx(p.outlook[0].p_win, abs=1e-4)


# -- a set of adds that can all be made --------------------------------------


def test_the_adds_proposed_never_share_a_drop(db):
    """Priced one at a time, every candidate named the same cheapest drop, and
    approving one made the rest impossible."""
    got = _targets_for(
        db,
        (
            _rp("p.1", "Mid", 1, "TOR", "C"),
            _rp("p.2", "Depth", 2, "TOR", "C"),
            _rp("p.3", "Also Depth", 3, "TOR", "C"),
        ),
        (("C", 3, 1),),
        pool=[_fa("fa.1", "Streamer", 9), _fa("fa.2", "Streamer Two", 10)],
        rates={
            1: {"goals": 0.4},
            2: {"goals": 0.1},
            3: {"goals": 0.1},
            9: {"goals": 0.5},
            10: {"goals": 0.45},
        },
        per_game={1: 1.2, 2: 0.3, 3: 0.35, 9: 1.1, 10: 1.0},
    )
    drops = [t.drop.name for t in got]
    assert len(got) == 2 and len(set(drops)) == 2


def test_a_streamer_never_costs_a_regular(db):
    """Only the bottom stream_spots rotate. With one, the second add would have
    had to cut Mid - a regular - so it is not proposed at all."""
    got = _targets_for(
        db,
        (_rp("p.1", "Mid", 1, "MTL", "C"), _rp("p.2", "Depth", 2, "TOR", "C")),
        (("C", 2, 1),),
        pool=[_fa("fa.1", "Streamer", 9), _fa("fa.2", "Streamer Two", 10)],
        rates={1: {"goals": 0.3}, 2: {"goals": 0.1}, 9: {"goals": 0.5}, 10: {"goals": 0.45}},
        per_game={1: 1.5, 2: 0.3, 9: 0.2, 10: 0.2},
        stream_spots=1,
    )
    assert [t.drop.name for t in got] == ["Depth"]


def test_an_add_that_is_an_upgrade_may_replace_anyone(db):
    """Worth more over the rest of the season than the player it replaces:
    not streaming, just a better player."""
    got = _targets_for(
        db,
        (_rp("p.1", "Mid", 1, "TOR", "C"), _rp("p.2", "Depth", 2, "TOR", "C")),
        (("C", 2, 1),),
        pool=[_fa("fa.1", "Star FA", 9), _fa("fa.2", "Star FA Two", 10)],
        rates={1: {"goals": 0.2}, 2: {"goals": 0.1}, 9: {"goals": 0.9}, 10: {"goals": 0.8}},
        per_game={1: 0.6, 2: 0.3, 9: 3.0, 10: 2.5},
        stream_spots=1,
    )
    assert sorted(t.drop.name for t in got) == ["Depth", "Mid"]


def test_priced_by_the_odds_an_add_says_what_it_moves(db):
    from puckpilot.season.odds import OddsModel, Side
    from puckpilot.season.week import OddsContext

    ctx = OddsContext(
        model=OddsModel(p_play=1.0),
        banked={"goals": 2.0},
        theirs=Side(banked={"goals": 3.0}, skaters={"goals": 1.2}),
        cats=(resolve("G"),),
    )
    got = _targets_for(
        db,
        (_rp("p.1", "Star", 1, "MTL", "C"), _rp("p.2", "Depth", 2, "TOR", "C")),
        (("C", 2, 1), ("BN", 1, 0)),
        odds_ctx=ctx,
    )
    assert got and got[0].gain > 0 and got[0].score == got[0].gain
    assert got[0].helps and got[0].helps[0].startswith("G ")


def test_the_last_acquisitions_are_held_for_the_playoffs(db):
    p = _live_week(db, adds_used_season=60, playoff_reserve=6, find_targets=True)
    assert p.targets == () and any("playoffs" in n for n in p.notes)


def test_a_misspelt_pricing_is_refused():
    from puckpilot.season.authority import AuthorityError, TransactionAuthority

    with pytest.raises(AuthorityError):
        TransactionAuthority(add_scoring="vibes")


def test_the_days_first_run_is_the_one_that_searches(db):
    from puckpilot.season.pool import PoolPlayer, save_pool
    from puckpilot.season.run import pool_read_on

    assert not pool_read_on(db, "l", "2026-10-06")
    save_pool(db, "l", "2026-10-06", [PoolPlayer("k", "A", "TOR", "C", frozenset({"C"}), 1)])
    assert pool_read_on(db, "l", "2026-10-06") and not pool_read_on(db, "l", "2026-10-07")


def test_the_season_cap_limits_a_week_that_still_has_adds(db):
    """One acquisition left for the season and three for the week: one."""
    got = _targets_for(
        db,
        (
            _rp("p.1", "Mid", 1, "TOR", "C"),
            _rp("p.2", "Depth", 2, "TOR", "C"),
            _rp("p.3", "Also Depth", 3, "TOR", "C"),
        ),
        (("C", 3, 1),),
        pool=[_fa("fa.1", "Streamer", 9), _fa("fa.2", "Streamer Two", 10)],
        rates={
            1: {"goals": 0.4},
            2: {"goals": 0.1},
            3: {"goals": 0.1},
            9: {"goals": 0.5},
            10: {"goals": 0.45},
        },
        per_game={1: 1.2, 2: 0.3, 3: 0.35, 9: 1.1, 10: 1.0},
        adds_left=1,
    )
    assert len(got) == 1


def test_the_fewest_cap_wins_and_absent_caps_are_ignored():
    from puckpilot.season.week import _fewest

    assert _fewest(3, 1) == 1 and _fewest(None, 2) == 2 and _fewest(None, None) is None
    assert _fewest(3, -2) == 0
