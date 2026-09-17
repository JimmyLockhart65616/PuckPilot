"""Keeper resolution: the names in a league file becoming players off the board.

A keeper resolved to the wrong player does double damage - a real, draftable
player vanishes and the kept one stays available - so ambiguity is reported and
refused rather than settled by whichever row the table returned first.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from puckpilot.data import store
from puckpilot.draft.engine import Universe
from puckpilot.draft.sim import keepers_for
from puckpilot.engine.categories import Category
from puckpilot.engine.valuation import LeagueShape
from puckpilot.keepers import resolve_keepers, split_qualifier
from puckpilot.league import LeagueConfig


def _players(db):
    store.upsert_player(db, 1, "Sebastian Aho", "C", "CAR")
    store.upsert_player(db, 2, "Sebastian Aho", "D", "NYI")
    store.upsert_player(db, 3, "J.T. Miller", "C", "NYR")
    store.upsert_player(db, 4, "Tim Stützle", "C", "OTT")
    store.upsert_player(db, 5, "Cale Makar", "D", "COL")


def test_a_shared_name_is_reported_not_guessed(db):
    _players(db)
    res = resolve_keepers(db, ("Sebastian Aho",))
    assert res.resolved == {}
    assert list(res.ambiguous) == ["Sebastian Aho"]
    assert len(res.ambiguous["Sebastian Aho"]) == 2


def test_a_qualifier_settles_a_shared_name(db):
    _players(db)
    res = resolve_keepers(db, ("Sebastian Aho (CAR)", "Sebastian Aho (D)", "Sebastian Aho (1)"))
    assert res.resolved == {
        "Sebastian Aho (CAR)": 1,
        "Sebastian Aho (D)": 2,
        "Sebastian Aho (1)": 1,
    }
    assert not res.ambiguous and not res.unmatched


def test_split_qualifier():
    assert split_qualifier("Sebastian Aho (CAR)") == ("Sebastian Aho", "CAR")
    assert split_qualifier("Cale Makar") == ("Cale Makar", "")


def test_unmatched_and_ambiguous_are_kept_apart(db):
    _players(db)
    res = resolve_keepers(db, ("Nobody Real", "Sebastian Aho", "Tim Stutzle"))
    assert res.unmatched == ["Nobody Real"]
    assert list(res.ambiguous) == ["Sebastian Aho"]
    assert res.resolved == {"Tim Stutzle": 4}


def _universe():
    rows = {
        pid: {
            "name": name,
            "position": pos,
            "team": "AAA",
            "vorp": 10.0 - pid,
            "z_total": 10.0 - pid,
            "adp_rank": float(pid),
        }
        for pid, name, pos in [
            (1, "Sebastian Aho", "C"),
            (2, "Sebastian Aho", "D"),
            (3, "J.T. Miller", "C"),
            (4, "Tim Stützle", "C"),
            (5, "Cale Makar", "D"),
        ]
    }
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "player_id"
    return Universe(df.sort_values("vorp", ascending=False))


def _league(owners, by_season=("J.T. Miller", "Tim Stützle", "Cale Makar")):
    return LeagueConfig(
        name="Test",
        shape=LeagueShape(n_teams=4, slots=(("C", 1), ("D", 1)), util_slots=0, bench_slots=1),
        skater_cats=(Category("goals", "G", "skater"),),
        goalie_cats=(Category("wins", "W", "goalie"),),
        n_keepers=1,
        keepers_by_season={"20262027": by_season},
        keeper_owners_by_season={"20262027": owners},
    )


def test_an_owner_spelling_that_differs_from_the_pool_still_lands_on_that_seat(db):
    """'JT Miller' under an owner and 'J.T. Miller' in the pool are one player.
    An exact-string lookup used to deal him to a random seat instead."""
    _players(db)
    for seed in range(5):  # a random deal would eventually miss seat 3
        seats = keepers_for(
            db, _universe(), "20262027", _league({3: ("JT Miller",)}), np.random.default_rng(seed)
        )
        assert seats[3] == [3]


def test_an_owner_cannot_place_a_player_the_pool_does_not_list(db):
    _players(db)
    warnings: list[str] = []
    seats = keepers_for(
        db,
        _universe(),
        "20262027",
        _league({3: ("Sebastian Aho (CAR)",)}),
        np.random.default_rng(0),
        warn=warnings.append,
    )
    assert 1 not in [p for v in seats.values() for p in v]
    assert any("missing from keepers.by_season" in w for w in warnings)


def test_an_ambiguous_pool_name_is_warned_and_not_placed(db):
    _players(db)
    warnings: list[str] = []
    seats = keepers_for(
        db,
        _universe(),
        "20262027",
        _league({}, by_season=("Sebastian Aho", "Cale Makar")),
        np.random.default_rng(0),
        warn=warnings.append,
    )
    placed = sorted(p for v in seats.values() for p in v)
    assert placed == [5]
    assert any("NOT placed" in w for w in warnings)
