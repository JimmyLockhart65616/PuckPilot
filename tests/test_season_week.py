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


def _targets_for(db, players, slots):
    from puckpilot.league import LeagueConfig
    from puckpilot.season.pool import PoolPlayer
    from puckpilot.season.roster import TeamRoster
    from puckpilot.season.week import _targets
    from tests.test_season_settings import _slots

    _week_games(db)
    rt = _runtime_for_week(roster_positions=_slots(*slots))
    ours = TeamRoster(league_key="999.l.1", team_key="t", date="2026-10-05", players=players)
    streamer = PoolPlayer(
        player_key="fa.1",
        name="Streamer",
        team="TOR",
        primary_position="C",
        yahoo_eligible=frozenset({"C", "Util"}),
        nhl_player_id=9,
    )
    goals = resolve("G")
    return _targets(
        db,
        rt,
        LeagueConfig(skater_cats=(goals,), goalie_cats=()),
        rt.week(1),
        ours,
        [streamer],
        {1: {"goals": 0.5}, 2: {"goals": 0.2}, 9: {"goals": 0.3}},
        {},
        None,
        _PerGame({1: 1.5, 2: 0.5, 9: 1.0}),
        0.0,
        5,
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
