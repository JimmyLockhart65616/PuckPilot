"""Recommendations and the view stay sound at every pick of a whole draft.

The existing board tests check one moment each. A draft is 150-odd moments, and
the ones that break are the ones nobody wrote a test for: the round a minimum
starts forcing positions, a keeper-shortened seat running out of picks, the
last pick, the snapshot after the last pick. So this drafts the whole thing and
checks every snapshot for several seats against the board it came from.

The checker is tested too, by corrupting snapshots on purpose - a checker that
never fails is indistinguishable from one that checks nothing.
"""

from __future__ import annotations

import copy

import pytest

from puckpilot.draft import integrity
from puckpilot.draft.feed import apply
from puckpilot.web.server import LiveState
from tests import draftkit

SEATS = (0, 3, 5)


def _drive(seed: int, my_seat: int = 0, every: int = 1):
    board = draftkit.board(my_seat=my_seat)
    state = LiveState(board=board, feed=None, top=3, board_rows=40, cats=draftkit.CATS)
    feed = draftkit.sim_feed(seed)
    violations: list[str] = []
    snaps = 0
    guard = 0
    while guard < 500:
        if board.made % every == 0 or board.complete:
            for seat in SEATS:
                snap = state.snapshot(seat)
                snaps += 1
                violations += integrity.snapshot_violations(
                    board, snap, seat, state.policy, top=state.top, board_rows=state.board_rows
                )
        if board.complete:
            break
        apply(board, feed.poll(board))
        guard += 1
    return board, snaps, violations


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_every_snapshot_of_a_whole_draft_is_sound(seed):
    board, snaps, violations = _drive(seed)
    assert board.complete
    assert snaps == (board.made + 1) * len(SEATS)
    assert violations == [], "\n".join(violations[:20])


def test_the_view_after_the_last_pick_is_still_a_valid_view():
    board, _, _ = _drive(seed=4, every=10_000)
    state = LiveState(board=board, feed=None, top=3, cats=draftkit.CATS)
    for seat in SEATS:
        snap = state.snapshot(seat)
        assert snap["shortlist"] == []
        assert snap["on_clock"] is None and snap["round"] is None and snap["picks_away"] is None
        assert integrity.snapshot_violations(board, snap, seat, state.policy) == []


def test_a_seat_whose_keepers_fill_the_last_rounds_stops_being_told_to_pick():
    """In a league that slots keepers into the final rounds, our last live pick
    comes while the room is still drafting. The cards must stop there."""
    from puckpilot.draft.board import DraftBoard

    base = draftkit.board()
    kept = {0: [int(base.u.ids[0])], 3: [int(base.u.ids[1]), int(base.u.ids[2])]}
    board = DraftBoard(
        base.u, draftkit.RULES, my_seat=3, keepers=kept, keeper_rounds={3: [8, 9]}, roster_rounds=10
    )
    state = LiveState(board=board, feed=None, top=3, cats=draftkit.CATS)
    feed = draftkit.sim_feed(11)
    checked = 0
    while not board.complete:
        if board.picks_left(3) == 0:
            snap = state.snapshot(3)
            assert snap["shortlist"] == [] and snap["picks_away"] is None
            assert integrity.snapshot_violations(board, snap, 3, state.policy) == []
            checked += 1
        apply(board, feed.poll(board))
    assert checked >= 10  # seat 3 sat out the last rounds while others drafted


def test_the_market_only_rookies_never_leak_into_the_advice():
    board, _, _ = _drive(seed=5, every=10_000)
    assert board.complete
    fresh = draftkit.board()
    state = LiveState(board=fresh, feed=None, top=3, board_rows=200, cats=draftkit.CATS)
    snap = state.snapshot(0)
    offered = {c["name"] for c in snap["shortlist"]} | {r["name"] for r in snap["board"]}
    assert offered.isdisjoint(draftkit.MARKET_NAMES)
    assert {p["name"] for p in snap["market_watchlist"]} == set(draftkit.MARKET_NAMES)


# ---- the checker is not vacuous -------------------------------------------------


@pytest.fixture
def mid_draft():
    board = draftkit.board()
    feed = draftkit.sim_feed(9)
    for _ in range(14):
        apply(board, feed.poll(board))
    state = LiveState(board=board, feed=None, top=3, board_rows=40, cats=draftkit.CATS)
    snap = state.snapshot(0)
    assert integrity.snapshot_violations(board, snap, 0, top=3, board_rows=40) == []
    return board, snap


def _broken(snap, mutate):
    s = copy.deepcopy(snap)
    mutate(s)
    return s


def _drafted_name(board):
    return next(p.name for p in board.picks if p.row >= 0)


@pytest.mark.parametrize(
    "label, mutate, expect",
    [
        ("off-by-one pick", lambda s, b: s.update(made=s["made"] + 1), "made"),
        (
            "wrong seat on clock",
            lambda s, b: s.update(on_clock=(s["on_clock"] + 1) % 6),
            "on_clock",
        ),
        (
            "drafted player offered",
            lambda s, b: s["shortlist"][0].update(name=_drafted_name(b)),
            "not available",
        ),
        (
            "drafted player on the board",
            lambda s, b: s["board"][5].update(name=_drafted_name(b)),
            "not available",
        ),
        ("NaN on the wire", lambda s, b: s["board"][0].update(vorp=float("nan")), "strict JSON"),
        ("probability > 1", lambda s, b: s["board"][1].update(p_survive=1.4), "outside [0, 1]"),
        (
            "market row in the shortlist",
            lambda s, b: s["shortlist"][1].update(name="Gavin McKenna", position="L"),
            "market-only",
        ),
        (
            "shortlist reordered",
            lambda s, b: s["shortlist"].reverse(),
            "not what the engine ranks first",
        ),
        ("roster lost a player", lambda s, b: s["roster"].pop(), "roster"),
        ("needs drifted", lambda s, b: s.update(needs=[]), "needs"),
        ("n_left stale", lambda s, b: s.update(n_left=s["n_left"] + 1), "n_left"),
        ("wrong position", lambda s, b: s["board"][2].update(position="G"), "at 'G'"),
        (
            "reason with no text",
            lambda s, b: s["shortlist"][0]["reasons"].append({"kind": "pro"}),
            "malformed",
        ),
        ("duplicated board row", lambda s, b: s["board"].append(dict(s["board"][0])), "repeats"),
        (
            "eligibility widened",
            lambda s, b: s["board"][3].update(eligible=s["board"][3]["position"] + "/G"),
            "shows eligibility",
        ),
        (
            "card eligibility without its own position",
            lambda s, b: s["shortlist"][0].update(
                eligible="G" if s["shortlist"][0]["position"] != "G" else "C"
            ),
            "not in its own eligibility",
        ),
        (
            "row id pointing at a drafted player",
            lambda s, b: s["board"][4].update(id=next(p.player_id for p in b.picks if p.row >= 0)),
            "already drafted",
        ),
        (
            "recent picks reordered",
            lambda s, b: s["recent"].reverse() if len(s["recent"]) > 1 else s["recent"].clear(),
            "recent picks",
        ),
    ],
)
def test_the_checker_catches(mid_draft, label, mutate, expect):
    board, snap = mid_draft
    bad = _broken(snap, lambda s: mutate(s, board))
    found = integrity.snapshot_violations(board, bad, 0, top=3, board_rows=None)
    assert any(expect in v for v in found), f"{label}: {found}"


def test_relay_comparison_ignores_only_what_may_legitimately_differ(mid_draft):
    _, snap = mid_draft
    relay_copy = {**snap, "can_undo": False, "relay_age": 3.2, "stale": False}
    relay_copy["seconds_since_pick"] = (snap["seconds_since_pick"] or 0) + 3.2
    local = {**snap, "can_undo": True, "seconds_since_pick": snap["seconds_since_pick"] or 0}
    assert integrity.relay_violations(local, relay_copy) == []

    lagging = copy.deepcopy(relay_copy)
    lagging["made"] -= 1
    assert any("made" in v for v in integrity.relay_violations(local, lagging))

    reordered = copy.deepcopy(relay_copy)
    reordered["board"] = reordered["board"][::-1]
    assert any("board" in v for v in integrity.relay_violations(local, reordered))

    errored = {"error": "no snapshot for seat 3", "seats": ["0"]}
    assert "relay answered an error" in integrity.relay_violations(local, errored)[0]
