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

from puckpilot.draft.advice import can_wait_on, recommend
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


def test_explicit_keeper_rounds_override_the_placement():
    first = _board(keepers={0: [1, 2]}, keeper_placement="first")
    override = _board(keepers={0: [1, 2]}, keeper_rounds={0: [5, 6]}, keeper_placement="first")
    assert first.picks_left(0) == override.picks_left(0) == 5
    # "first" consumes seat 0's earliest rounds; the override its last two
    assert first.slots[0][1] != 0
    assert override.slots[0][1] == 0


def test_last_placement_keeps_the_opening_rounds_a_plain_snake():
    """The real-league bug: keepers in the LAST rounds, modelled in the first,
    put seat 5 on the clock at pick 1 and our seat-3 first pick at #16 instead
    of #4. Every survival probability reads off that number."""
    keepers = {0: [1, 2], 1: [11], 2: [21, 22], 3: [31, 32]}
    b = _board(keepers=keepers, keeper_placement="last")
    assert [seat for _, seat in b.slots[:8]] == [0, 1, 2, 3, 3, 2, 1, 0]
    assert b.slot_numbers[:8] == list(range(1, 9))
    assert b.next_pick_no(3) == 3
    # the uneven keeper counts show up only in the final rounds: seat 1 kept
    # one player, so he alone still picks in round 6
    assert [seat for rnd, seat in b.slots if rnd == 5] == [1]
    assert [seat for rnd, seat in b.slots if rnd == 6] == []
    assert b.picks_left(1) == 6 and b.picks_left(2) == 5


def test_first_placement_with_uniform_keepers_reverses_the_snake():
    """Why "first" is not the default: an odd number of keepers each shifts the
    live draft onto a reverse round, so the LAST seat picks first."""
    b = _board(keepers={s: [1 + 10 * s] for s in range(4)}, keeper_placement="first")
    assert b.on_the_clock() == 3


def test_an_unknown_placement_is_refused():
    with pytest.raises(DraftBoardError, match="keeper_placement"):
        _board(keeper_placement="middle")


# ---- reconciling with the room --------------------------------------------


def test_room_pick_numbers_map_onto_live_slots_around_keeper_rounds():
    b = _board(keepers={0: [1]}, keeper_placement="last")
    # 4 teams x 7 rounds = 28 room picks; round 7 runs forward, so seat 0's
    # round-7 pick is room #25 - and that is his keeper
    assert b.slot_for_room_pick(25) is None
    assert b.slot_for_room_pick(26) == 24
    assert b.slot_for_room_pick(1) == 0
    assert b.live_picks_through(24) == 24
    assert b.live_picks_through(25) == 24
    assert b.live_picks_through(28) == 27


def test_drift_is_how_far_the_board_trails_the_room():
    b = _board()
    assert b.drift(0) == 0
    assert b.drift(3) == 3  # room made three picks, board recorded none
    b.record(int(b.u.ids[0]))
    assert b.drift(3) == 2
    b.record_unknown()
    b.record_unknown()
    b.record_unknown()
    assert b.drift(3) == -1  # entered by hand ahead of a lagging feed


def test_a_late_keeper_comes_off_without_consuming_a_pick():
    """Found on draft night: a keeper nobody declared. Recording him as a pick
    would advance the clock past a pick that has not happened."""
    b = _board(keeper_placement="last")
    b.record(int(b.u.ids[0]))
    pid = int(b.u.ids[5])
    made, before = b.made, len(b.slots)
    pick = b.add_keeper(pid, seat=2)
    assert pick.source == "keeper" and pick.seat == 2
    assert not b.avail[b._row_of[pid]]
    assert b.made == made
    assert len(b.slots) == before - 1
    assert b.picks_left(2) == 6
    assert [p.player_id for p in b.roster(2)] == [pid]


def test_a_late_keeper_can_be_taken_back():
    b = _board()
    pid = int(b.u.ids[5])
    before = list(b.slots)
    b.add_keeper(pid, seat=2)
    assert b.remove_keeper(pid).player_id == pid
    assert b.avail[b._row_of[pid]]
    assert b.slots == before
    assert b.roster(2) == []
    assert b.remove_keeper(pid) is None


def test_a_placeholder_is_named_later_without_counting_the_slot_twice():
    b = _board()
    b.record_unknown(label="?")
    pid = int(b.u.ids[3])
    pick = b.fill_placeholder(0, pid, source="feed")
    assert b.made == 1
    assert pick.player_id == pid and pick.overall == 0
    assert not b.avail[b._row_of[pid]]
    with pytest.raises(DraftBoardError, match="not an unknown placeholder"):
        b.fill_placeholder(0, int(b.u.ids[4]))


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
        b.pick_context(0),
    )
    assert top[0].row == chosen


def test_the_board_owns_the_pick_context():
    """Both the console and a directly-called policy must score the same board.
    They diverged the moment `recommend` started passing `avail` and the test
    built its own dict, so ctx construction lives in one place now."""
    b = _board()
    ctx = b.pick_context(0)
    assert ctx["pick_no"] == b.made
    assert ctx["next_pick_no"] == b.next_pick_no(0)
    assert ctx["avail"] is b.avail


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


def test_can_wait_on_flags_players_the_room_will_leave():
    b = _board()
    cands = recommend(b, n=10)
    assert all(0.0 <= c.p_survive <= 1.0 for c in cands)
    assert set(can_wait_on(b, cands, threshold=0.0)) == {c.name for c in cands}


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


# ---- reason ordering -------------------------------------------------------


def test_reasons_lead_with_timing_not_with_agreement():
    """Explain used to return `pros + cons`, so a card read as every argument
    for followed by every argument against - which buries the fact that decides
    the pick under a list of facts that do not.

    Timing leads because it is the only one that cannot be recovered: a player
    who will not last is a decision now, a category edge keeps until next turn.
    """
    from puckpilot.draft.explain import CATEGORY, NEED, TIMING, Reason, _order_for_test

    reasons = [
        Reason("pro", "carries BLK", CATEGORY),
        Reason("con", "near the cap", NEED),
        Reason("pro", "will not last", TIMING),
        Reason("pro", "starts right away", NEED),
    ]
    ordered = _order_for_test(reasons)
    assert ordered[0].text == "will not last"
    # at equal weight a pro comes before a con, so the card still reads as a case
    assert [r.text for r in ordered[1:3]] == ["starts right away", "near the cap"]
    assert ordered[-1].text == "carries BLK"


def test_a_real_shortlist_puts_survival_first():
    from puckpilot.draft.explain import TIMING, summarize

    b = _board()
    cands = recommend(b, n=10)
    for _cand, reasons in summarize(b, cands, top=3):
        timing = [r for r in reasons if r.weight == TIMING]
        if timing:
            assert reasons[0].weight == TIMING, "a timing fact exists but is not first"


# ---- terminal hand entry and feed health ------------------------------------


def test_console_taken_unknown_and_kept_commands():
    from puckpilot.draft.live import handle_command

    b = _board()
    assert "taken by seat 0" in handle_command(b, "t Player CA")
    assert b.made == 1 and b.picks[0].name == "Player CA"
    assert "taken by seat 2" in handle_command(b, "taken Player DA @2")
    assert b.picks[1].seat == 2
    assert "recorded as unknown" in handle_command(b, "x")
    assert b.made == 3
    assert "no pick used" in handle_command(b, "k Player LA @1")
    assert b.made == 3 and [p.name for p in b.roster(1)] == ["Player LA"]
    assert "back on the board" in handle_command(b, "unk Player LA")
    assert handle_command(b, "Player RA") is None, "a bare name is not a command"


def test_console_commands_refuse_bad_input_with_a_sentence():
    from puckpilot.draft.live import handle_command

    b = _board()
    handle_command(b, "t Player CA")
    assert "already off the board" in handle_command(b, "t Player CA")
    assert "outside" in handle_command(b, "x @9")
    assert "must be a number" in handle_command(b, "t Player DA @two")
    assert "give the seat" in handle_command(b, "k Player DB")
    assert b.made == 1


def test_console_health_says_when_the_board_is_behind_the_room():
    from puckpilot.draft.live import feed_health

    class Quiet:
        name = "websocket"

        def status(self):
            return {"room_picks": 4, "gaps": [], "unmapped_names": ["Ivan Demidov"]}

    b = _board()
    b.warnings = ["no Yahoo ADP"]
    lines = feed_health(b, Quiet())
    assert lines[0] == "no Yahoo ADP"
    assert any("4 PICKS BEHIND THE ROOM" in line for line in lines)
    assert any("Ivan Demidov" in line for line in lines)
    for _ in range(4):
        b.record_unknown()
    assert not any("BEHIND" in line for line in feed_health(b, Quiet()))


# ---- traded picks and the room's clock ------------------------------------------


def test_a_traded_pick_moves_one_slot_and_keepers_still_take_the_last_picks():
    """The real case: seat 3 owns seat 1's round-3 pick; seat 1 owns seat 3's
    round-5 pick. Keepers must still come off each seat's LAST picks."""
    trades = {(2, 1): 3, (4, 3): 1}  # 0-based (round, pick in round) -> seat
    keepers = {s: [1 + 10 * s] for s in range(4)}
    b = _board(keepers=keepers, keeper_placement="last", pick_owners=trades)
    assert b.slots[9][1] == 3 and b.slot_numbers[9] == 10  # round 3, 2nd pick
    assert b.slots[19][1] == 1 and b.slot_numbers[19] == 20  # round 5, 4th pick
    assert b.picks_left(3) == b.picks_left(1) == 6  # 7 picks each, one kept
    # everyone's keeper came off their last pick: the final round is empty
    assert all(rnd < 6 for rnd, _ in b.slots)


def test_no_trades_is_the_old_sequence_exactly():
    for placement in ("first", "last"):
        keepers = {0: [1, 2], 1: [11], 2: [], 3: [31, 32]}
        a = _board(keepers=keepers, keeper_placement=placement)
        b = _board(keepers=keepers, keeper_placement=placement, pick_owners={})
        assert a.slots == b.slots and a.slot_numbers == b.slot_numbers


def test_the_room_clock_disagreeing_with_the_board_is_named():
    b = _board()
    assert b.clock_disagreement(1, 0) is None
    msg = b.clock_disagreement(2, 3)
    assert msg and "seat 3" in msg and "expects seat 1" in msg
