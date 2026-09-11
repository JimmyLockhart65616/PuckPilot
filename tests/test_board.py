"""Live draft board: the state machine behind the draft-night console.

These cover the failures that would produce quietly wrong advice rather than an
obvious crash - an uneven keeper board mis-numbering our next pick, a keeper
that never came off the board, or a shortlist that ignores a roster minimum
with picks running out.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from puckpilot.draft.advice import recommend, survivors
from puckpilot.draft.board import DraftBoard, DraftBoardError
from puckpilot.draft.engine import DraftRules, RosterValuePolicy, Universe
from puckpilot.engine.valuation import LeagueShape

SHAPE = LeagueShape(
    n_teams=4,
    slots=(("C", 1), ("L", 1), ("R", 1), ("D", 2), ("G", 1)),
    util_slots=1,
    bench_slots=1,
)
RULES = DraftRules(
    shape=SHAPE,
    rounds=7,
    caps={"C": 3, "L": 3, "R": 3, "D": 4, "G": 2},
    mins={"C": 1, "L": 1, "R": 1, "D": 2, "G": 1},
)


def _universe(n_per_pos=10):
    """Distinct names on purpose: `keepers._norm` strips digits, so numbered
    fixtures would all normalize to the same string and hide real ambiguity."""
    rows = {}
    pid = 1
    for pos in ("C", "L", "R", "D", "G"):
        for i in range(n_per_pos):
            v = 20.0 - i
            rows[pid] = {
                "name": f"Player {pos}{chr(65 + i)}",
                "position": pos,
                "team": "AAA",
                "vorp": v,
                "z_total": v,
                "adp_rank": float(pid),
                "goals": 30.0 - i,
            }
            pid += 1
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "player_id"
    return Universe(df.sort_values("vorp", ascending=False))


def _board(**kw):
    return DraftBoard(_universe(), RULES, my_seat=0, roster_rounds=7, **kw)


# ---- pick sequence --------------------------------------------------------


def test_no_keepers_gives_a_plain_snake():
    b = _board()
    assert len(b.slots) == 4 * 7
    assert [seat for _, seat in b.slots[:8]] == [0, 1, 2, 3, 3, 2, 1, 0]
    assert b.on_the_clock() == 0
    assert b.current_round() == 1


def test_uneven_keepers_give_uneven_pick_counts():
    """The real-league case: 29 eligible keepers across 12 seats cannot be uniform,
    so a seat keeping fewer players must end up with more live picks."""
    b = _board(keepers={0: [1, 2], 1: [11], 2: [], 3: []})
    assert b.picks_left(0) == 5  # 7 rounds - 2 keepers
    assert b.picks_left(1) == 6
    assert b.picks_left(2) == 7
    assert b.picks_left(3) == 7
    assert len(b.slots) == 5 + 6 + 7 + 7


def test_next_pick_no_indexes_the_live_sequence_not_the_full_snake():
    """survival_discount compares ADP rank against this number, so it has to
    count picks that will actually happen, not slots keepers already consumed."""
    b = _board(keepers={0: [1, 2], 1: [11], 2: [], 3: []})
    first = b.next_pick_no(0)
    assert b.slots[first][1] == 0
    while b.on_the_clock() != 0:
        b.record(int(b.u.ids[np.flatnonzero(b.avail)[0]]))
    assert b.made == first
    later = b.next_pick_no(0, after=b.made)
    assert later is None or (later > b.made and b.slots[later][1] == 0)


def test_explicit_keeper_rounds_override_the_default_assumption():
    default = _board(keepers={0: [1, 2]})
    override = _board(keepers={0: [1, 2]}, keeper_rounds={0: [5, 6]})
    assert default.picks_left(0) == override.picks_left(0) == 5
    # the default consumes seat 0's earliest rounds; the override its last two
    assert default.slots[0][1] != 0
    assert override.slots[0][1] == 0


# ---- keepers --------------------------------------------------------------


def test_keepers_leave_the_board_and_prefill_counts_without_using_a_pick():
    b = _board(keepers={2: [1, 2]})
    assert not b.avail[b._row_of[1]] and not b.avail[b._row_of[2]]
    assert b.counts[2]["C"] == 2
    assert b.made == 0
    assert len(b.roster(2)) == 2


def test_unresolvable_keeper_is_reported_never_swallowed():
    """A keeper that fails to place leaves an elite player wrongly draftable."""
    b = _board(keepers={0: [999_999]})
    assert b.unmatched_keepers == [999_999]


# ---- recording and undo ---------------------------------------------------


def test_record_advances_the_clock_and_removes_the_player():
    b = _board()
    pid = int(b.u.ids[0])
    pick = b.record(pid)
    assert pick.seat == 0 and pick.overall == 0
    assert not b.avail[b._row_of[pid]]
    assert b.on_the_clock() == 1


def test_recording_the_same_player_twice_is_refused():
    b = _board()
    pid = int(b.u.ids[0])
    b.record(pid)
    with pytest.raises(DraftBoardError, match="already off the board"):
        b.record(pid)


def test_unknown_player_is_refused():
    with pytest.raises(DraftBoardError, match="not in the ranked universe"):
        _board().record(999_999)


def test_undo_restores_availability_and_the_clock():
    """A mistyped name mid-draft has to be recoverable in one keystroke."""
    b = _board()
    pid = int(b.u.ids[0])
    b.record(pid)
    b.undo()
    assert b.avail[b._row_of[pid]]
    assert b.made == 0 and b.on_the_clock() == 0
    assert b.counts[0].get(str(b.u.pos[0]), 0) == 0
    b.record(pid)  # and it can be re-recorded, e.g. against the right seat


def test_undo_on_an_empty_board_is_a_no_op():
    assert _board().undo() is None


# ---- name lookup ----------------------------------------------------------


def test_find_matches_case_and_punctuation_insensitively():
    b = _board()
    rows = b.find("player ca")
    assert rows and str(b.u.names[rows[0]]) == "Player CA"
    assert b.find("PLAYER  CA!") == rows


def test_find_skips_players_already_drafted():
    b = _board()
    row = b.find("player ca")[0]
    b.record(int(b.u.ids[row]))
    assert row not in b.find("player ca")


# ---- advice ---------------------------------------------------------------


def test_recommend_leads_with_exactly_what_the_policy_would_pick():
    """The console's top row and the simulated engine must never disagree."""
    b = _board()
    policy = RosterValuePolicy()
    top = recommend(b, policy, n=5)
    chosen = policy.pick(
        b.u,
        b.avail,
        b.counts[0],
        b.rules,
        b.picks_left(0),
        np.random.default_rng(0),
        {"pick_no": b.made, "next_pick_no": b.next_pick_no(0)},
    )
    assert top[0].row == chosen


def test_recommend_never_offers_a_drafted_player():
    b = _board()
    taken = {int(b.u.ids[r]) for r in (0, 1, 2)}
    for pid in taken:
        b.record(pid)
    assert not {c.player_id for c in recommend(b, n=20)} & taken


def test_recommend_respects_forced_minimums_when_picks_run_short():
    """With exactly enough picks left to cover the unmet minimums, only needy
    positions may be offered, or the roster finishes illegal."""
    b = _board()
    b.counts[0] = {"C": 1, "L": 1, "R": 1, "D": 2}
    b.slots = [s for s in b.slots if s[1] != 0][:3] + [(6, 0)]
    assert b.picks_left(0) == 1
    assert b.needs(0) == {"G": 1}
    assert {c.position for c in recommend(b, n=10)} == {"G"}


def test_survivors_flags_players_the_room_will_leave():
    b = _board()
    cands = recommend(b, n=10)
    assert all(0.0 <= c.p_survive <= 1.0 for c in cands)
    assert set(survivors(b, cands, threshold=0.0)) == {c.name for c in cands}


def test_last_pick_of_the_draft_discounts_nobody():
    """Nothing survives a draft that is over, so the discount must vanish
    rather than divide by a missing next pick."""
    b = _board()
    b.slots = b.slots[:1]
    assert b.next_pick_no(0) == 0
    cands = recommend(b, n=3)
    b.record(cands[0].player_id)
    assert b.complete
    assert b.next_pick_no(0) is None


def test_recommendation_is_fast_enough_for_a_30_second_clock():
    """The design rule is that nothing model-shaped sits on the draft-time
    path; this is the assertion that keeps it true."""
    b = _board()
    policy = RosterValuePolicy()
    recommend(b, policy, n=15)  # warm numpy
    start = time.perf_counter()
    for _ in range(50):
        recommend(b, policy, n=15)
    per_call_ms = (time.perf_counter() - start) / 50 * 1000
    assert per_call_ms < 50, f"{per_call_ms:.1f}ms per recommendation"


# ---- live console ---------------------------------------------------------


def test_live_console_applies_feed_picks_and_reranks(monkeypatch):
    """Picks arrive from the feed, not the keyboard: manual entry was removed
    once the websocket proved itself at 190/190."""
    import puckpilot.draft.live as live
    from puckpilot.draft.feed import PickEvent

    b = _board()
    first, second = int(b.u.ids[0]), int(b.u.ids[1])
    first_name = str(b.u.names[0])

    class OneShotFeed:
        name = "test"
        last_error = None

        def __init__(self):
            self.sent = False

        def poll(self, board):
            if self.sent:
                return []
            self.sent = True
            return [PickEvent(first, None, "test"), PickEvent(second, None, "test")]

    monkeypatch.setattr(live, "_stdin_thread", lambda q: q.put("q"))
    frames: list[str] = []
    live.run_live(
        b, feed=OneShotFeed(), cfg=live.LiveConfig(top=5, refresh=0.01), out=frames.append
    )

    assert [p.player_id for p in b.picks] == [first, second]
    table = frames[-1].split("-" * 66)[1].split(chr(10) * 2)[0]
    assert first_name not in table  # a drafted player leaves the shortlist


def test_live_console_undo_is_the_only_recovery_hatch(monkeypatch):
    """The feed is the sole pick source, so undo must work - and typing a
    player name must NOT record anything."""
    import puckpilot.draft.live as live
    from puckpilot.draft.feed import PickEvent

    b = _board()
    pid = int(b.u.ids[0])
    name = str(b.u.names[0])

    class OneShotFeed:
        name = "test"
        last_error = None

        def __init__(self):
            self.sent = False

        def poll(self, board):
            if self.sent:
                return []
            self.sent = True
            return [PickEvent(pid, None, "test")]

    def scripted(q):
        for line in (name, "u", "q"):
            q.put(line)

    monkeypatch.setattr(live, "_stdin_thread", scripted)
    frames: list[str] = []
    live.run_live(b, feed=OneShotFeed(), cfg=live.LiveConfig(refresh=0.01), out=frames.append)
    assert b.picks == [], "undo took back the feed's pick; the typed name added nothing"
    assert "Stopped" in frames[-1]


def test_live_console_survives_a_failing_feed(monkeypatch):
    """A broken feed must be reported, not take the console down."""
    import puckpilot.draft.live as live

    class BrokenFeed:
        name = "broken"
        last_error = "ConnectionError: down"

        def poll(self, board):
            return []

    b = _board()
    monkeypatch.setattr(live, "_stdin_thread", lambda q: q.put("q"))
    frames: list[str] = []
    live.run_live(b, feed=BrokenFeed(), cfg=live.LiveConfig(refresh=0.01), out=frames.append)
    assert any("feed error" in f for f in frames)
