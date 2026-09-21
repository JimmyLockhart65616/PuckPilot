"""Pick feeds: manual entry, bots, and the shared apply path.

Manual entry is the feed that must not fail, because it is the fallback when
every automated source does. These tests are mostly about what happens when the
input is wrong - a misspelling, an ambiguous surname, a duplicate poll - since
that is what actually occurs at speed on draft night.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from puckpilot.draft.board import DraftBoard, DraftBoardError
from puckpilot.draft.engine import AdpBot, DraftRules, Universe
from puckpilot.draft.feed import ManualFeed, PickEvent, SimFeed, apply
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

NAMES = {
    "C": ["Connor McDavid", "Connor Bedard", "Sidney Crosby"],
    "L": ["Tim Stützle", "Brady Tkachuk", "Zach Hyman"],
    "R": ["Nikita Kucherov", "David Pastrnak", "Mikko Rantanen"],
    "D": ["Cale Makar", "Quinn Hughes", "Roman Josi", "Victor Hedman"],
    "G": ["Connor Hellebuyck", "Andrei Vasilevskiy", "Anthony Stolarz"],
}


def _universe():
    """Named stars for the lookup tests, padded so a full draft cannot run the
    board dry (4 seats x 7 rounds = 28 picks)."""
    rows, pid = {}, 1
    for pos, names in NAMES.items():
        filler = [f"Filler {pos}{k}" for k in range(8)]
        for i, name in enumerate([*names, *filler]):
            v = 20.0 - i
            rows[pid] = {
                "name": name,
                "position": pos,
                "team": "AAA",
                "vorp": v,
                "z_total": v,
                "adp_rank": float(pid),
            }
            pid += 1
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "player_id"
    return Universe(df.sort_values("vorp", ascending=False))


def _board(**kw):
    return DraftBoard(_universe(), RULES, my_seat=0, roster_rounds=7, **kw)


# ---- manual entry ---------------------------------------------------------


def test_partial_name_resolves():
    b, feed = _board(), ManualFeed()
    feed.submit(b, "pastrnak")
    (pick,), rejected = apply(b, feed.poll(b))
    assert pick.name == "David Pastrnak" and not rejected


def test_accents_and_case_do_not_matter():
    """'Stutzle' is what a hurried human types; the board stores 'Stützle'."""
    b, feed = _board(), ManualFeed()
    feed.submit(b, "stutzle")
    (pick,), _ = apply(b, feed.poll(b))
    assert pick.name == "Tim Stützle"


def test_ambiguous_input_is_refused_with_the_candidates():
    """Two Connors: guessing here silently drafts the wrong player."""
    b, feed = _board(), ManualFeed()
    with pytest.raises(DraftBoardError, match="ambiguous"):
        feed.submit(b, "connor")


def test_exact_name_wins_even_when_it_prefixes_another():
    b, feed = _board(), ManualFeed()
    feed.submit(b, "Connor McDavid")
    (pick,), _ = apply(b, feed.poll(b))
    assert pick.name == "Connor McDavid"


def test_unknown_name_is_refused_not_guessed():
    b, feed = _board(), ManualFeed()
    with pytest.raises(DraftBoardError, match="no available player"):
        feed.submit(b, "Wayne Gretzky")


def test_player_id_is_accepted_directly():
    b, feed = _board(), ManualFeed()
    pid = int(b.u.ids[0])
    feed.submit(b, str(pid))
    (pick,), _ = apply(b, feed.poll(b))
    assert pick.player_id == pid


def test_poll_drains_the_queue():
    b, feed = _board(), ManualFeed()
    feed.submit(b, "pastrnak")
    assert len(feed.poll(b)) == 1
    assert feed.poll(b) == []


# ---- bulk paste -----------------------------------------------------------


def test_paste_keeps_the_players_and_hands_back_the_noise():
    """A pasted results panel carries headers and team names between the
    picks; a partial reconciliation still beats abandoning the paste."""
    b, feed = _board(), ManualFeed()
    queued, unresolved = feed.submit_many(
        b,
        "Round 1\nDavid Pastrnak\nCale Makar\n\nSome Team Name\nRoman Josi\n",
    )
    assert len(queued) == 3
    assert unresolved == ["Round 1", "Some Team Name"]
    accepted, rejected = apply(b, feed.poll(b))
    assert [p.name for p in accepted] == ["David Pastrnak", "Cale Makar", "Roman Josi"]
    assert not rejected


# ---- apply ----------------------------------------------------------------


def test_duplicate_pick_is_reported_but_does_not_stop_the_draft():
    """A feed re-reporting a pick is a duplicate poll, not a reason to crash."""
    b = _board()
    pid = int(b.u.ids[0])
    accepted, rejected = apply(b, [PickEvent(pid), PickEvent(pid)])
    assert len(accepted) == 1
    assert len(rejected) == 1 and "already off the board" in rejected[0]
    assert b.made == 1


def test_events_can_name_an_explicit_seat():
    b = _board()
    (pick,), _ = apply(b, [PickEvent(int(b.u.ids[0]), seat=2)])
    assert pick.seat == 2


# ---- sim feed -------------------------------------------------------------


def test_sim_feed_picks_for_the_seat_on_the_clock():
    b = _board()
    bots = [AdpBot(noise_sd=0.0) for _ in range(4)]
    feed = SimFeed(bots, np.random.default_rng(0))
    events = feed.poll(b)
    assert len(events) == 1 and events[0].seat == b.on_the_clock()


def test_sim_feed_leaves_our_seat_alone():
    """In an interactive mock the human owns their seat; the bots must not
    pick over the top of them."""
    b = _board()
    bots = [AdpBot(noise_sd=0.0) for _ in range(4)]
    feed = SimFeed(bots, np.random.default_rng(0), skip_seats={0})
    assert feed.poll(b) == []  # seat 0 is on the clock


def test_sim_feed_drives_a_draft_to_completion():
    b = _board()
    bots = [AdpBot(noise_sd=1.0) for _ in range(4)]
    feed = SimFeed(bots, np.random.default_rng(3))
    guard = 0
    while not b.complete and guard < 500:
        apply(b, feed.poll(b))
        guard += 1
    assert b.complete
    assert len({p.player_id for p in b.picks}) == len(b.picks)  # no duplicates


# ---- yahoo feed -----------------------------------------------------------


class FakeSession:
    """Stands in for YahooSession; draft_results returns the whole draft each
    call, exactly as Yahoo does."""

    def __init__(self, pages):
        self.pages = list(pages)
        self.calls = 0

    def draft_results(self, league_key):
        page = self.pages[min(self.calls, len(self.pages) - 1)]
        self.calls += 1
        if isinstance(page, Exception):
            raise page
        return page


def _yahoo_feed(pages, mapping=None):
    from puckpilot.draft.feed import YahooDraftFeed

    b = _board()
    pid = int(b.u.ids[0])
    mapping = mapping if mapping is not None else {"477.p.1": pid}
    return b, YahooDraftFeed(FakeSession(pages), "477.l.1", mapping), pid


def test_yahoo_feed_reports_each_pick_once():
    """Yahoo re-sends the full draft every call; a pick must not double-apply."""
    picks = [{"pick": 1, "team_key": "t.1", "player_key": "477.p.1"}]
    b, feed, pid = _yahoo_feed([picks, picks, picks])
    first = feed.poll(b)
    assert [e.player_id for e in first] == [pid]
    assert feed.poll(b) == []
    assert feed.poll(b) == []


def test_yahoo_feed_attributes_picks_to_the_right_seat():
    from puckpilot.draft.feed import YahooDraftFeed

    b = _board()
    pid = int(b.u.ids[0])
    feed = YahooDraftFeed(
        FakeSession([[{"pick": 1, "team_key": "t.3", "player_key": "477.p.1"}]]),
        "477.l.1",
        {"477.p.1": pid},
    )
    feed.set_seats(["t.1", "t.2", "t.3", "t.4"])
    assert feed.poll(b)[0].seat == 2


def test_yahoo_feed_survives_a_network_failure():
    """A blip mid-draft must not raise: manual entry has to keep working."""
    picks = [{"pick": 1, "team_key": "t.1", "player_key": "477.p.1"}]
    b, feed, pid = _yahoo_feed([ConnectionError("boom"), picks])
    assert feed.poll(b) == []
    assert "ConnectionError" in feed.last_error
    assert [e.player_id for e in feed.poll(b)] == [pid]  # recovers on the next poll
    assert feed.last_error is None


def test_yahoo_feed_records_players_it_cannot_map():
    """A drafted prospect outside our NHL data is noted, not silently skipped -
    otherwise the board quietly disagrees with Yahoo. And the pick still
    consumes its slot, or the clock falls a pick behind the room."""
    from puckpilot.draft.feed import apply

    picks = [{"pick": 1, "team_key": "t.1", "player_key": "477.p.9999"}]
    b, feed, _ = _yahoo_feed([picks], mapping={})
    feed.key_names = {"477.p.9999": "Ivan Demidov"}
    events = feed.poll(b)
    assert [(e.player_id, e.pick_no) for e in events] == [(None, 1)]
    assert "Ivan Demidov" in events[0].label
    assert feed.unmapped == ["477.p.9999"]
    apply(b, events)
    assert b.made == 1 and b.picks[0].row == -1 and "Ivan Demidov" in b.picks[0].name


def test_yahoo_feed_catches_up_after_missed_polls():
    """Reconnecting mid-draft must recover every pick, not just the newest."""
    b = _board()
    ids = [int(x) for x in b.u.ids[:3]]
    mapping = {f"477.p.{i}": pid for i, pid in enumerate(ids, start=1)}
    full = [{"pick": i, "team_key": "t.1", "player_key": f"477.p.{i}"} for i in range(1, 4)]
    from puckpilot.draft.feed import YahooDraftFeed

    feed = YahooDraftFeed(FakeSession([full]), "477.l.1", mapping)
    assert [e.player_id for e in feed.poll(b)] == ids


# ---- replaying a draft that already happened -------------------------------
#
# The console had never been watched end to end, because doing so required a
# live Yahoo room: a lobby, a browser, eleven strangers and forty minutes. That
# is a bad way to find out the interface is wrong.


def _replay_picks(n=6, teams=12):
    return [
        {"pick": i, "yahoo_id": str(i), "seat": (i - 1) % teams + 1, "position": "C"}
        for i in range(1, n + 1)
    ]


def test_replay_delivers_picks_in_draft_order():
    from puckpilot.draft.feed import ReplayFeed

    board = _board()
    ids = [int(board.u.ids[i]) for i in range(6)]
    feed = ReplayFeed(
        [{"pick": 6 - i, "yahoo_id": str(i), "seat": 1} for i in range(6)],
        {str(i): ids[i] for i in range(6)},
        interval=0.0,
    )
    got = []
    while not feed.exhausted:
        got += [e.player_id for e in feed.poll(board)]
    assert got == list(reversed(ids)), "picks must arrive in pick order, not list order"


def test_replay_respects_its_interval():
    """So a replay can be watched at human speed, not just blasted through."""
    from puckpilot.draft.feed import ReplayFeed

    board = _board()
    now = [100.0]
    feed = ReplayFeed(
        _replay_picks(3),
        {"1": int(board.u.ids[0]), "2": int(board.u.ids[1]), "3": int(board.u.ids[2])},
        interval=5.0,
        clock=lambda: now[0],
    )
    assert feed.poll(board), "the first pick should land immediately"
    assert feed.poll(board) == [], "too soon for the second"
    now[0] += 6.0
    assert feed.poll(board), "and it arrives once the interval has passed"


def test_a_room_of_a_different_size_does_not_crash_the_board():
    """The harvested mocks are 14-team rooms and this league is 12, so seat 13
    walked off the end of `counts` with an IndexError - which on draft night,
    with nobody at the keyboard, would have taken the console down."""
    from puckpilot.draft.feed import ReplayFeed, apply

    board = _board()  # 4 teams
    feed = ReplayFeed(
        [{"pick": 1, "yahoo_id": "1", "seat": 13}],
        {"1": int(board.u.ids[0])},
        interval=0.0,
        n_teams=14,
    )
    accepted, rejected = apply(board, feed.poll(board))
    assert accepted and not rejected, "a mismatched room should still drive the board"
    assert accepted[0].seat == board.on_the_clock() or board.made == 1


def test_an_out_of_range_seat_is_refused_not_raised():
    from puckpilot.draft.board import DraftBoardError

    board = _board()
    with pytest.raises(DraftBoardError, match="outside this"):
        board.record(int(board.u.ids[0]), seat=99)


def test_replay_records_players_it_cannot_map():
    """Same honesty as the live feed: the console says how far behind it is."""
    from puckpilot.draft.feed import ReplayFeed, apply

    board = _board()
    feed = ReplayFeed([{"pick": 1, "yahoo_id": "nobody", "seat": 1}], {}, interval=0.0)
    events = feed.poll(board)
    assert [e.player_id for e in events] == [None]
    assert feed.status()["unmapped"] == 1
    apply(board, events)
    assert board.made == 1, "an unmapped pick still happened; the clock must move"


# ---- reconciling a hand-driven board with a recovered feed ------------------
#
# Manual entry and the feed now share one board, so the question these answer
# is whether the two ever count the same pick twice - the error that leaves the
# clock a pick off for the rest of the draft, silently.


def test_an_unidentified_pick_consumes_its_slot():
    b = _board()
    accepted, rejected = apply(b, [PickEvent(None, 1, "feed", label="Prospect X", pick_no=1)])
    assert accepted == [] and b.made == 1
    # pick 1 is seat 0 by the board's order, whatever seat the feed claimed
    assert b.picks[0].name == "Prospect X" and b.picks[0].seat == 0
    assert rejected and "slot consumed" in rejected[0]


def test_a_recovered_feed_names_a_pick_advanced_by_hand():
    """The drill: the socket dies, the drafter presses 'unknown pick', the feed
    comes back and delivers who it was."""
    b = _board()
    b.record_unknown(label="(entered by hand)")
    pid = int(b.u.ids[2])
    accepted, _ = apply(b, [PickEvent(pid, 0, "feed", pick_no=1)])
    assert b.made == 1, "the same slot must not be counted twice"
    assert accepted[0].player_id == pid
    assert not b.avail[b._row_of[pid]]


def test_a_feed_redelivering_a_hand_entered_pick_is_rejected_not_double_counted():
    b = _board()
    pid = int(b.u.ids[0])
    b.record(pid, source="manual")
    accepted, rejected = apply(b, [PickEvent(pid, 0, "feed", pick_no=1)])
    assert accepted == [] and b.made == 1
    assert "already off the board" in rejected[0]


def test_an_unidentified_pick_already_entered_by_hand_is_not_counted_again():
    b = _board()
    b.record_unknown()
    _, rejected = apply(b, [PickEvent(None, 0, "feed", pick_no=1)])
    assert b.made == 1
    assert "already on the board" in rejected[0]


def test_an_unidentified_pick_in_a_keeper_slot_consumes_nothing():
    b = _board(keepers={0: [1]}, keeper_placement="last")
    # room pick 25 is seat 0's round-7 keeper on this board (see test_board)
    _, rejected = apply(b, [PickEvent(None, 0, "feed", pick_no=25)])
    assert b.made == 0
    assert "keeper slot" in rejected[0]


def test_a_replay_skips_mock_picks_of_players_our_league_keeps():
    """A harvested mock has no keepers. Its pick of a player we keep never
    happened in our room, so it must not count toward how far the room got -
    or the drift indicator reads a keeper-sized lie."""
    from puckpilot.draft.feed import ReplayFeed

    board = _board(keepers={1: [1]})
    picks = [
        {"pick": 1, "yahoo_id": "y1", "seat": 1},  # kept on our board
        {"pick": 2, "yahoo_id": "y2", "seat": 2},
    ]
    feed = ReplayFeed(picks, {"y1": 1, "y2": 2}, interval=0.0)
    apply(board, feed.poll(board))
    apply(board, feed.poll(board))
    assert feed.status()["room_picks"] == 1 and feed.skipped_keepers == 1
    assert board.made == 1 and board.drift(feed.status()["room_picks"]) == 0


def test_a_replay_drop_goes_silent_while_the_room_keeps_drafting():
    from puckpilot.draft.feed import ReplayFeed

    board = _board()
    ids = [int(x) for x in board.u.ids[:6]]
    picks = [{"pick": i + 1, "yahoo_id": f"y{i}", "seat": 1} for i in range(6)]
    feed = ReplayFeed(picks, {f"y{i}": pid for i, pid in enumerate(ids)}, drop=(2, 3))
    for _ in range(5):
        apply(board, feed.poll(board))
    status = feed.status()
    assert status["room_picks"] == 5 and status["missed"] == 3
    assert board.made == 2 and board.drift(status["room_picks"]) == 3
    # recovered by hand: the three missed picks entered, drift clears
    for pid in ids[2:5]:
        board.record(pid)
    assert board.drift(feed.status()["room_picks"]) == 0
    apply(board, feed.poll(board))  # the feed is back
    assert board.made == 6 and board.drift(feed.status()["room_picks"]) == 0


# ---- draft night: seat by pick number, draft results, combined sources --------


def test_a_feed_seat_is_overridden_by_the_boards_own_order():
    """Yahoo's websocket sends the TEAM id as the seat; the board's order for
    that pick number is what decides whose roster it lands on."""
    b = _board()
    pid = int(b.u.ids[0])
    apply(b, [PickEvent(pid, 3, "websocket", pick_no=2)])  # pick 2 belongs to seat 1
    assert b.picks[0].seat == 1


class _Results:
    def __init__(self, rows):
        self.rows = rows
        self.calls = 0

    def draft_results(self, key):
        self.calls += 1
        return list(self.rows)


def test_draft_results_catch_up_every_pick_in_order_and_rate_limit():
    from puckpilot.draft.feed import YahooDraftFeed

    b = _board()
    ids = [int(x) for x in b.u.ids[:3]]
    rows = [
        {"pick": 2, "team_key": "t.9", "player_key": "477.p.2"},
        {"pick": 1, "team_key": "t.7", "player_key": "477.p.1"},
        {"pick": 3, "team_key": "t.8", "player_key": "477.p.3"},
    ]
    t = [100.0]
    feed = YahooDraftFeed(
        _Results(rows),
        "477.l.1",
        {f"477.p.{i + 1}": pid for i, pid in enumerate(ids)},
        min_interval=3.0,
        clock=lambda: t[0],
    )
    apply(b, feed.poll(b))
    assert [p.player_id for p in b.picks] == ids
    assert [p.seat for p in b.picks] == [0, 1, 2]  # the board's order, not t.N
    assert feed.status()["room_picks"] == 3
    t[0] += 1.0
    assert feed.poll(b) == [] and feed.session.calls == 1  # too soon
    t[0] += 3.0
    assert feed.poll(b) == [] and feed.session.calls == 2  # nothing new


def test_a_combined_feed_merges_sources_without_counting_a_pick_twice():
    from puckpilot.draft.feed import CombinedFeed

    b = _board()
    pid = int(b.u.ids[0])

    class One:
        name = "x"
        last_error = None

        def __init__(self, room):
            self.room = room

        def poll(self, board):
            return [PickEvent(pid, None, "x", pick_no=1)]

        def status(self):
            return {"chosen": "x", "room_picks": self.room, "unmapped_names": []}

    feed = CombinedFeed([One(1), One(2)])
    accepted, rejected = apply(b, feed.poll(b))
    assert b.made == 1 and len(accepted) == 1 and "already off the board" in rejected[0]
    assert feed.status()["room_picks"] == 2 and feed.status()["chosen"] == "x+x"


def test_the_websocket_clock_reports_no_seat_and_raw_frames_are_logged(tmp_path):
    from puckpilot.draft.wsfeed import replay

    feed = replay([], {})
    feed.frame_log = tmp_path / "frames.log"
    feed.ingest("D|5|7|60")
    assert feed.status()["on_the_clock"] == {"pick": 5}
    assert "D|5|7|60" in feed.frame_log.read_text(encoding="utf-8")


def test_keeper_picks_in_the_results_do_not_count_as_room_progress():
    """Keepers sit at #157-192 in Yahoo's results from the first pick; the
    room's progress is the unbroken run of filled picks from #1."""
    from puckpilot.draft.feed import YahooDraftFeed

    b = _board()
    rows = [{"pick": 1, "team_key": "t.1", "player_key": "477.p.1"}] + [
        {"pick": n, "team_key": "t.2", "player_key": f"477.p.{n}"} for n in (25, 26)
    ]
    feed = YahooDraftFeed(_Results(rows), "477.l.1", {"477.p.1": int(b.u.ids[0])})
    feed.poll(b)
    assert feed.status()["room_picks"] == 1
