"""Integrity checks against the real, private database.

Everything else in the suite runs on synthetic boards, which is what CI can see.
These run on the data draft night will actually use - the Yahoo player map, the
harvested rooms, the real league file - and answer the questions only that data
can: does every drafted Yahoo player resolve to the right person, can a drafter
find each one by the name Yahoo shows, and do real boards stay sound across a
whole draft.

Opt-in (`PUCKPILOT_LOCAL_DATA=1 pytest tests/test_local_data.py`) and skipped
when the database is absent, because data/ is git-ignored and never in CI.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from puckpilot.config import Settings

DB = Settings().resolved_db_path
ENABLED = os.environ.get("PUCKPILOT_LOCAL_DATA") == "1" and DB.is_file()
pytestmark = pytest.mark.skipif(
    not ENABLED, reason="set PUCKPILOT_LOCAL_DATA=1 with data/puckpilot.db present"
)
SEASON = "20262027"
MOCKS = Path(__file__).resolve().parents[1] / "data" / "mocks"


@pytest.fixture(scope="module")
def conn():
    from puckpilot.data import store

    return store.connect(DB)


@pytest.fixture(scope="module")
def league_key(conn):
    keys = [r[0] for r in conn.execute("SELECT DISTINCT league_key FROM yahoo_player_map")]
    if len(keys) != 1:
        pytest.skip(f"expected one league key in the player map, found {keys}")
    return keys[0]


@pytest.fixture(scope="module")
def live_board(conn, league_key):
    from puckpilot.draft.live import build_live_board
    from puckpilot.league import DEFAULT_LEAGUE
    from puckpilot.yahoo.playermap import load_adp

    adp = load_adp(conn, league_key)
    return build_live_board(
        conn, DEFAULT_LEAGUE, seat=0, season=SEASON, adp=adp, league_key=league_key,
        progress=lambda _m: None,
    )  # fmt: skip


# ---- the Yahoo map ---------------------------------------------------------------


def test_no_yahoo_player_maps_to_the_wrong_person(conn):
    from puckpilot.draft.e2e import same_player

    rows = conn.execute(
        "SELECT y.full_name, n.full_name, y.nhl_player_id FROM yahoo_player_map y"
        " JOIN nhl_players n ON n.player_id = y.nhl_player_id"
    ).fetchall()
    assert rows, "the player map is empty - run `ppilot yahoo playermap`"
    wrong = [(y, n) for y, n, _ in rows if not same_player(y, n)]
    assert wrong == []
    ids = [r[2] for r in rows]
    assert len(ids) == len(set(ids)), "two Yahoo players resolved to one NHL player"


def test_the_draftable_pool_is_mapped(conn, league_key):
    """A 12-team, 16-round draft takes ~190 players; the top 250 by Yahoo ADP
    must all resolve, or the feed goes blind on real picks."""
    rows = conn.execute(
        "SELECT full_name, nhl_player_id FROM yahoo_player_map WHERE league_key = ?"
        " ORDER BY adp_rank LIMIT 250",
        (league_key,),
    ).fetchall()
    unmapped = [name for name, nid in rows if nid is None]
    assert unmapped == []


def test_every_player_drafted_in_a_harvested_room_is_mapped(conn):
    from puckpilot.draft.farm import load_all
    from puckpilot.draft.wsfeed import load_yahoo_id_map

    rooms = load_all(MOCKS)
    if not rooms:
        pytest.skip("no harvested rooms under data/mocks")
    idmap = load_yahoo_id_map(conn)
    drafted = [p["yahoo_id"] for room in rooms for p in room.picks]
    missing = sorted({y for y in drafted if y not in idmap})
    coverage = 1 - sum(1 for y in drafted if y not in idmap) / len(drafted)
    assert coverage >= 0.995, f"{coverage:.2%} of {len(drafted)} room picks map; missing {missing}"


# ---- names a drafter types ------------------------------------------------------


def _board_rows_by_id(board):
    return {int(pid): i for i, pid in enumerate(board.u.ids)}


def _mapped_on_board(conn, board, league_key, limit=250):
    rows = conn.execute(
        "SELECT full_name, nhl_player_id FROM yahoo_player_map WHERE league_key = ?"
        " AND nhl_player_id IS NOT NULL ORDER BY adp_rank LIMIT ?",
        (league_key, limit),
    ).fetchall()
    by_id = _board_rows_by_id(board)
    return [(name, by_id[int(nid)]) for name, nid in rows if int(nid) in by_id]


def test_every_draftable_player_is_findable_by_surname(conn, live_board, league_key):
    """What a drafter types on a clock. MoneyPuck's deleted letters ("Sttzle")
    broke exactly this until `data rosters` repaired the names."""
    board = live_board
    misses = []
    for name, row in _mapped_on_board(conn, board, league_key):
        surname = name.split()[-1]
        if row not in board.find(surname, limit=50, available_only=False):
            misses.append(f"{surname} -> {board.u.names[row]}")
    assert misses == [], "run `ppilot data rosters` to repair lossy names"


def test_nearly_every_player_is_findable_by_his_full_yahoo_name(conn, live_board, league_key):
    """Full names miss only where Yahoo and the NHL disagree on a given name
    ("Mitch" vs "Mitchell Marner") - the surname still finds them."""
    board = live_board
    pairs = _mapped_on_board(conn, board, league_key)
    misses = [
        f"{name} -> {board.u.names[row]}"
        for name, row in pairs
        if row not in board.find(name, limit=50, available_only=False)
    ]
    assert len(misses) <= max(3, len(pairs) // 50), misses


# ---- whole drafts on the real board ---------------------------------------------


def test_a_real_board_stays_sound_through_a_bot_draft(conn, live_board):
    from puckpilot.draft import integrity
    from puckpilot.draft.e2e import sim_polls
    from puckpilot.draft.feed import apply
    from puckpilot.league import DEFAULT_LEAGUE
    from puckpilot.web.server import LiveState

    board = live_board
    state = LiveState(board=board, top=3, board_rows=120, cats=DEFAULT_LEAGUE.all_cats)
    violations = []
    for poll in sim_polls(board, DEFAULT_LEAGUE, np.random.default_rng(7)):
        if board.made % 9 == 0:
            for seat in (0, 7):
                violations += integrity.snapshot_violations(
                    board, state.snapshot(seat), seat, state.policy, top=3, board_rows=120
                )
        apply(board, poll(board))
    assert board.complete
    for seat in (0, 7):
        violations += integrity.snapshot_violations(board, state.snapshot(seat), seat)
    assert violations == [], violations[:10]


def test_a_harvested_room_replays_through_every_hop(conn, league_key):
    from puckpilot.draft import e2e
    from puckpilot.draft.farm import load_all
    from puckpilot.draft.feed import ReplayFeed
    from puckpilot.draft.live import build_live_board
    from puckpilot.draft.wsfeed import load_yahoo_id_map
    from puckpilot.league import DEFAULT_LEAGUE
    from puckpilot.yahoo.playermap import load_adp

    rooms = load_all(MOCKS)
    if not rooms:
        pytest.skip("no harvested rooms under data/mocks")
    room = rooms[-1]
    league = e2e.room_league(DEFAULT_LEAGUE, room.n_teams)
    board = build_live_board(
        conn, league, seat=0, season=SEASON, adp=load_adp(conn, league_key),
        league_key=league_key, progress=lambda _m: None,
    )  # fmt: skip
    feed = ReplayFeed(room.picks, load_yahoo_id_map(conn), interval=0, n_teams=room.n_teams)
    relay = e2e.start_local_relay()
    try:
        result = e2e.Harness(
            board, relay, cats=league.all_cats, seats=(0,), every=16, names=e2e.yahoo_rows(conn)
        ).run(
            e2e.replay_polls(feed, len(room.picks) + 5),
            room.started,
            "harvest",
            room_picks=len(room.picks),
            unmapped=lambda: feed.unmapped,
        )
    finally:
        relay.close()
    assert result.passed, result.summary()
