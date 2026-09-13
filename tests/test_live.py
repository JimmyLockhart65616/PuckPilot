"""`draft.live`: the pieces with real mechanics, tested without a live database.

`build_live_board` itself needs a real DB and real projections end to end, so
it is exercised through `ppilot draft live` on draft night rather than here.
`attach_market_frame` is the one part of it with anything to get wrong -
reconciling `with_adp`'s array-only update against a frame that never saw it,
and recomputing `has_market` after a sort scrambles row order - so it is pulled
out and tested on its own.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from puckpilot.draft.engine import Universe
from puckpilot.draft.live import attach_market_frame


def _universe(with_source=False):
    rows = {
        1: {"name": "A", "position": "C", "team": "AAA", "vorp": 10.0, "z_total": 10.0},
        2: {"name": "B", "position": "D", "team": "AAA", "vorp": 8.0, "z_total": 8.0},
        3: {"name": "C", "position": "L", "team": "AAA", "vorp": 6.0, "z_total": 6.0},
    }
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "player_id"
    if with_source:
        df["source"] = "projected"
    u = Universe(df)
    u.has_market = np.array([True, True, False])  # 1, 2 market-priced; 3 not
    return u


def _market_frame():
    rows = {
        900: {
            "name": "Rookie",
            "position": "R",
            "team": "BOS",
            "vorp": 9.0,  # deliberately between A and B, so sort order matters
            "z_total": 9.0,
            "adp_rank": 45.0,
            "source": "market",
            "train_gp": 0.0,
        }
    }
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "player_id"
    return df


def test_an_empty_market_frame_returns_the_same_universe():
    u = _universe()
    out = attach_market_frame(u, pd.DataFrame(), adp={1: 5})
    assert out is u


def test_the_market_row_is_added():
    u = _universe()
    out = attach_market_frame(u, _market_frame(), adp={1: 5, 2: 20})
    assert set(out.ids) == {1, 2, 3, 900}
    row = out.frame.loc[900]
    assert row["source"] == "market"
    assert row["position"] == "R"


def test_real_adp_survives_the_merge_not_the_stale_frame_column():
    """`.frame["adp_rank"]` was never touched by `with_adp` - only the numpy
    array was. Merging from the stale frame column would silently revert
    every real player's ADP to the pre-market proxy."""
    u = _universe()
    u = u.with_adp(np.array([5.0, 20.0, 300.0]))  # real Yahoo ADP, post with_adp
    out = attach_market_frame(u, _market_frame(), adp={1: 5, 2: 20})
    assert out.adp_rank[list(out.ids).index(1)] == 5.0
    assert out.adp_rank[list(out.ids).index(2)] == 20.0


def test_has_market_is_recomputed_by_id_after_the_sort_reorders_rows():
    """Positional concatenation of the old has_market array would be wrong:
    sort_values just reordered every row, so membership must be recomputed by
    id, never carried along by position."""
    u = _universe()
    out = attach_market_frame(u, _market_frame(), adp={1: 5, 2: 20})
    ids = list(out.ids)
    got = dict(zip(ids, out.has_market, strict=True))
    assert got[1] and got[2]  # real, market-priced
    assert not got[3]  # real, never market-priced
    assert got[900]  # the new market row


def test_a_market_row_is_added_even_with_no_real_adp_dict():
    """`adp` can be falsy (no Yahoo ADP loaded) while a market frame still
    exists in principle - has_market must not crash on `set(None)`."""
    u = _universe()
    out = attach_market_frame(u, _market_frame(), adp=None)
    assert 900 in set(out.ids)
    assert bool(dict(zip(out.ids, out.has_market, strict=True))[900])


def test_source_defaults_to_projected_for_rows_the_market_frame_did_not_touch():
    u = _universe()
    out = attach_market_frame(u, _market_frame(), adp={1: 5})
    sources = dict(zip(out.ids, out.source, strict=True))
    assert sources[1] == "projected" and sources[900] == "market"


@pytest.mark.parametrize("with_source", [True, False])
def test_merge_works_whether_or_not_the_real_frame_already_has_a_source_column(with_source):
    u = _universe(with_source=with_source)
    out = attach_market_frame(u, _market_frame(), adp={1: 5})
    assert len(out) == 4
