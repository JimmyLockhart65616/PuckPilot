"""Next season's keepers: the as-of projection, the cost, contracts and protection."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd

from puckpilot.engine.valuation import LeagueShape
from puckpilot.league import LeagueConfig
from puckpilot.season import keeper_value as kv
from tests.conftest import add_player, add_skater_game


def _league(n_keepers=3, placement="last", years=3):
    return LeagueConfig(
        n_keepers=n_keepers,
        keeper_years=years,
        keeper_placement=placement,
        shape=LeagueShape(n_teams=2),
    )


def _p(key, pid, name=None):
    return SimpleNamespace(player_key=key, nhl_player_id=pid, name=name or key)


# -- projecting next season from part of this one -------------------------------


def _schedule(db, season, dates, teams=("TOR", "MTL")):
    from puckpilot.data import store

    for i, d in enumerate(dates):
        store.upsert_schedule_game(
            db,
            game_id=int(season[:4]) * 1000 + i,
            season=season,
            game_type=2,
            game_date=d,
            start_time_utc=None,
            home_team=teams[0],
            away_team=teams[1],
        )


def test_a_season_so_far_is_cut_at_the_date(db):
    from puckpilot.engine.aggregate import season_aggregates, season_games

    add_player(db, 1, "Skater", "C")
    dates = ["2025-10-10", "2025-10-12", "2025-11-20", "2025-12-01"]
    _schedule(db, "20252026", dates)
    for i, d in enumerate(dates):
        add_skater_game(db, 1, "20252026", 9000 + i, date=d, goals=1)
    db.commit()
    sk, _ = season_aggregates(db, "20252026", before="2025-11-01")
    assert int(sk.loc[1, "gp"]) == 2 and int(sk.loc[1, "goals"]) == 2
    assert season_games(db, "20252026", before="2025-11-01") == 2
    assert season_games(db, "20252026") == 4
    full, _ = season_aggregates(db, "20252026")
    assert int(full.loc[1, "gp"]) == 4


def test_without_a_date_the_projection_is_unchanged(db):
    """`as_of=None` must be the old projection to the digit."""
    from puckpilot.engine import projections

    add_player(db, 1, "Skater", "C")
    dates = [f"2025-10-{d:02d}" for d in range(10, 30)]
    _schedule(db, "20252026", dates)
    for i, d in enumerate(dates):
        add_skater_game(db, 1, "20252026", 9000 + i, date=d, goals=i % 2, shots=3)
    db.commit()
    a, _ = projections.project(db, "20262027", ["20252026"])
    b, _ = projections.project(db, "20262027", ["20252026"], as_of=None, target_games=None)
    pd.testing.assert_frame_equal(a, b)


def test_a_partial_season_counts_availability_against_games_played_so_far(db):
    """Ten games into a season a player who has played all ten is fully
    available - not 10/82 of a player."""
    from puckpilot.engine import projections

    add_player(db, 1, "Everyday", "C")
    dates = [f"2025-10-{d:02d}" for d in range(10, 30)]
    _schedule(db, "20252026", dates)
    for i, d in enumerate(dates[:12]):
        add_skater_game(db, 1, "20252026", 9000 + i, date=d, goals=1, shots=3)
    db.commit()
    sk, _ = projections.project(
        db, "20262027", ["20252026"], weights=(1.0,), as_of="2025-10-22", target_games=82
    )
    assert sk.loc[1, "proj_gp"] == 82.0  # 12 of 12, not 12 of 20 or of 82
    assert sk.loc[1, "goals"] > 70  # a goal a game, aged a year


# -- the cost and the contracts -----------------------------------------------


def test_keeping_in_the_last_rounds_costs_what_those_rounds_would_draw():
    vorp = pd.Series({i: float(100 - i) for i in range(100)})
    lg = _league()
    full = lg.shape.n_teams * lg.shape.roster_size  # every roster filled: picks end here
    assert kv.keeper_cost(lg, vorp) == sum(100 - i for i in range(full, full + 2)) / 2
    first = kv.keeper_cost(_league(placement="first"), vorp)
    assert first == sum(100 - i for i in range(6, 12)) / 6  # after everyone's keepers
    assert first > kv.keeper_cost(lg, vorp)
    assert kv.keeper_cost(_league(n_keepers=0), vorp) == 0.0


def test_contracts_count_this_seasons_keep(tmp_path):
    f = tmp_path / "keepers.json"
    f.write_text(
        json.dumps(
            {
                "managers": [
                    {"continuing": [{"nhl_id": 1, "times_kept": 2}, {"nhl_id": 2, "times_kept": 1}]}
                ],
                "declared": [{"nhl_id": 1}, {"nhl_id": 3}, {"nhl_id": None}],
            }
        ),
        encoding="utf-8",
    )
    # 1: kept twice before and again now; 3: a first-year keep; 2 was not kept.
    assert kv.contracts(f) == {1: 3, 3: 1}
    assert kv.contracts(tmp_path / "missing.json") is None


# -- ranking and protection ------------------------------------------------------


def _board(values, times_kept=None, **kw):
    return kv.KeeperBoard(
        league=_league(**kw),
        season="20262027",
        as_of="2026-10-05",
        vorp_next=pd.Series(values, dtype=float),
        cost=0.0,
        times_kept=times_kept,
    )


def test_the_best_eligible_keepers_and_one_more_are_protected():
    board = _board({1: 9.0, 2: 7.0, 3: 5.0, 4: 3.0, 5: 1.0, 6: -2.0}, times_kept={1: 3})
    ranks = board.rank([_p(f"p{i}", i) for i in range(1, 8)], margin=1)
    by = {k.nhl_player_id: k for k in ranks}
    assert not by[1].eligible and by[1].rank is None  # kept three times: expired
    assert [by[i].rank for i in (2, 3, 4, 5, 6)] == [1, 2, 3, 4, 5]
    assert [i for i in by if by[i].protected] == [2, 3, 4, 5]  # 3 keepers + 1
    assert by[7].rank is None and not by[7].protected  # no projection next season
    assert [k.nhl_player_id for k in ranks][:5] == [2, 3, 4, 5, 6]


def test_nobody_below_what_keeping_costs_is_protected():
    board = _board({1: 4.0, 2: -1.0, 3: -3.0})
    ranks = board.rank([_p("a", 1), _p("b", 2), _p("c", 3)], margin=1)
    assert [k.protected for k in ranks] == [True, False, False]


def test_a_league_without_keepers_protects_nobody():
    board = _board({1: 9.0}, n_keepers=0)
    assert not any(k.protected for k in board.rank([_p("a", 1)]))


def test_the_week_s_ranks_are_kept_and_put_on_the_roster(db):
    from puckpilot.season.roster import RosterPlayer, TeamRoster

    board = _board({1: 9.0, 2: 1.0})
    players = tuple(
        RosterPlayer(
            player_key=f"k{i}",
            yahoo_id=str(i),
            name=f"P{i}",
            team="TOR",
            primary_position="C",
            yahoo_eligible=frozenset({"C"}),
            selected_slot="C",
            nhl_player_id=i,
        )
        for i in (1, 2, 3)
    )
    roster = TeamRoster(league_key="l", team_key="t", date="2026-10-05", players=players)
    ranks = board.rank(roster.players, margin=0)
    kv.save(db, "jimmy", "2026-10-05", "2026-10-05", ranks)
    again = kv.load(db, "jimmy", "2026-10-05")
    assert again == ranks
    assert kv.load(db, "jimmy", "2026-10-12") == []
    marked = kv.annotate(roster, again)
    assert [(p.keeper_rank, p.keeper_protected) for p in marked.players] == [
        (1, True),
        (2, True),
        (None, False),
    ]


def test_a_card_says_what_the_swap_does_to_next_season():
    board = _board({1: 9.0, 2: 1.0, 9: 5.0}, times_kept={3: 3})
    roster = [_p("a", 1, "Star"), _p("b", 2, "Depth"), _p("c", 3, "Veteran")]
    drop = SimpleNamespace(player_key="b", nhl_player_id=2, name="Depth", keeper_rank=2)
    lines = kv.card_lines(_p("fa", 9, "Pickup"), drop, roster, board)
    assert lines[0] == (
        "Pickup projects +5.0 over replacement next season - would rank 2 of your keepers "
        "(you keep 3)"
    )
    assert lines[1] == "Depth ranks 2 of your keepers next season"
    old = SimpleNamespace(player_key="c", nhl_player_id=3, name="Veteran", keeper_rank=None)
    assert kv.card_lines(_p("fa", 8, "Unknown"), old, roster, board) == [
        "Unknown: no projection for next season",
        "Veteran has been kept 3 times - not a keeper next season",
    ]


def test_an_arm_can_protect_keepers():
    from puckpilot.season.add_gate import parse_arm

    assert parse_arm("odds-daily-h-f25-x1-k1", 2).keepers == 1
    assert parse_arm("odds-daily", 2).keepers is None


# -- the gate ---------------------------------------------------------------------


def test_the_keeper_gate_scores_a_perfect_ranking_as_no_regret():
    from puckpilot.season.keeper_gate import score_rule

    actual = pd.Series({1: 10.0, 2: 8.0, 3: 6.0, 4: 1.0, 5: -3.0})
    rosters = [[1, 2, 3, 4, 5]]
    perfect = score_rule(actual, actual, {1, 2, 3, 4, 5}, {1, 2, 3, 4, 5}, rosters, k=2, n_keep=3)
    assert perfect.regret == 0.0 and perfect.top_k == 1.0
    assert abs(perfect.spearman - 1.0) < 1e-9
    backwards = score_rule(-actual, actual, {1, 2, 3, 4, 5}, {1, 2, 3, 4, 5}, rosters, 2, 3)
    # keeps 5, 4 and 3: 0 (below replacement counts as replacement) + 1 + 6
    assert backwards.regret == (10 + 8 + 6) - (0 + 1 + 6)
