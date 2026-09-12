"""The draft-night board build.

`build_live_board` had no test at all, and was missing its keepers entirely -
`DraftBoard` was constructed without them, so on draft night the kept players
would have stayed on the board and topped the shortlist. That is not a cosmetic
bug: with an empty roster the engine mis-states what we still need, ADP is never
re-based, and the slot count is the full roster rather than the picks that will
actually happen, which makes `next_pick_no` - and therefore every survival
probability - wrong.

These use a small synthetic league so they run offline and do not depend on the
real league file, which is private and git-ignored.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from puckpilot.draft.board import DraftBoard
from puckpilot.engine.categories import Category
from puckpilot.engine.valuation import LeagueShape
from puckpilot.keepers import keeper_seats
from puckpilot.league import LeagueConfig

SHAPE = LeagueShape(
    n_teams=4,
    slots=(("C", 1), ("L", 1), ("R", 1), ("D", 2), ("G", 1)),
    util_slots=1,
    bench_slots=1,
)


def _league(**kw):
    return LeagueConfig(
        name="Test League",
        shape=SHAPE,
        skater_cats=(Category("goals", "G", "skater"),),
        goalie_cats=(Category("wins", "W", "goalie"),),
        n_keepers=2,
        **kw,
    )


def _universe():
    from puckpilot.draft.engine import Universe

    rows, pid = {}, 1
    for pos in ("C", "L", "R", "D", "G"):
        for i in range(8):
            v = 20.0 - i
            rows[pid] = {
                "name": f"Player {pos}{chr(65 + i)}",
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


# ---- keeper seat assignment ------------------------------------------------


def test_declared_owners_are_placed_exactly():
    """On a live board one of these seats is ours, and our roster is what the
    engine reasons about. A random deal is fine for a sim and wrong here."""
    seats = keeper_seats(
        [1, 2, 3, 4, 5, 6], n_teams=3, rng=np.random.default_rng(0), owned={2: [5, 6]}
    )
    assert sorted(seats[2]) == [5, 6]
    assert sorted(p for v in seats.values() for p in v) == [1, 2, 3, 4, 5, 6]


def test_a_declared_owner_is_not_dealt_extra_keepers():
    """Striding from seat 0 would pile the leftovers onto whoever already has
    some, which is the opposite of dealing evenly."""
    seats = keeper_seats(
        list(range(1, 7)), n_teams=3, rng=np.random.default_rng(0), owned={0: [1, 2]}
    )
    assert len(seats[0]) == 2
    assert sorted(len(v) for v in seats.values()) == [2, 2, 2]


def test_undeclared_ownership_still_deals_everyone():
    seats = keeper_seats([1, 2, 3, 4], n_teams=2, rng=np.random.default_rng(0))
    assert sorted(p for v in seats.values() for p in v) == [1, 2, 3, 4]


def test_an_out_of_range_seat_is_ignored_not_crashed():
    seats = keeper_seats([1, 2], n_teams=2, rng=np.random.default_rng(0), owned={9: [1]})
    assert sorted(p for v in seats.values() for p in v) == [1, 2]


# ---- what keepers do to the board -----------------------------------------


def test_keepers_leave_the_board_and_shorten_the_draft():
    """The count that matters: live picks, not roster spots. Getting this wrong
    makes next_pick_no wrong, and survival reads straight off it."""
    u = _universe()
    rules = _league().draft_rules()
    kept = {0: [1, 2], 1: [9], 2: [], 3: []}

    bare = DraftBoard(u, rules, my_seat=0)
    with_keepers = DraftBoard(u, rules, my_seat=0, keepers=kept)

    assert len(bare.slots) == 4 * SHAPE.roster_size
    assert len(with_keepers.slots) == len(bare.slots) - 3
    assert not with_keepers.avail[with_keepers._row_of[1]]
    assert with_keepers.unmatched_keepers == []


def test_our_keepers_prefill_the_roster_and_reduce_what_we_need():
    u = _universe()
    board = DraftBoard(u, _league().draft_rules(), my_seat=0, keepers={0: [1, 2]})
    assert len(board.roster(0)) == 2
    # two C keepers means C is no longer a need
    assert "C" not in board.needs(0)
    assert board.made == 0, "keepers must not consume a pick"


def test_adp_is_rebased_once_keepers_are_gone():
    """With keepers still ranked, everyone left looks later-going than they are
    and every comparison against a pick number is shifted."""
    u = _universe()
    bare = DraftBoard(u, _league().draft_rules(), my_seat=0)
    rebased = DraftBoard(u, _league().draft_rules(), my_seat=0, keepers={0: [1, 2], 1: [9]})
    top_bare = bare.u.adp_rank[bare._row_of[3]]
    top_rebased = rebased.u.adp_rank[rebased._row_of[3]]
    assert top_rebased < top_bare


def test_an_unplaceable_keeper_is_reported(caplog):
    board = DraftBoard(_universe(), _league().draft_rules(), my_seat=0, keepers={0: [999_999]})
    assert board.unmatched_keepers == [999_999]


# ---- league config ---------------------------------------------------------


def test_keeper_owners_round_trip_through_the_league_file(tmp_path):
    """The league will publish ownership before the draft; dropping it in must
    be a config edit, not a code change."""
    from puckpilot.league import load_league

    p = tmp_path / "l.toml"
    p.write_text(
        "\n".join(
            [
                'name = "Test League"',
                "[roster]",
                "teams = 4",
                'slots = [{ pos = "C", count = 1 }, { pos = "G", count = 1 }]',
                "util = 0",
                "bench = 1",
                "[keepers]",
                "count = 2",
                "[keepers.by_season]",
                '"20262027" = ["Alpha One", "Beta Two"]',
                "[keepers.owners.20262027]",
                '"3" = ["Alpha One"]',
            ]
        ),
        encoding="utf-8",
    )
    league = load_league(p)
    assert league.keeper_owners_for_season("20262027") == {3: ("Alpha One",)}
    assert league.keeper_owners_for_season("20272028") == {}


def test_a_non_integer_seat_is_refused_with_a_usable_message(tmp_path):
    from puckpilot.league import LeagueConfigError, load_league

    p = tmp_path / "l.toml"
    p.write_text(
        "\n".join(
            [
                'name = "Test League"',
                "[roster]",
                "teams = 4",
                'slots = [{ pos = "C", count = 1 }]',
                "[keepers.owners.20262027]",
                '"mine" = ["Alpha One"]',
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(LeagueConfigError, match="not an integer"):
        load_league(p)


# ---- dynamic positional replacement ----------------------------------------
#
# VORP's replacement level was computed once, pre-draft, and frozen. After a run
# on a position that is simply wrong: with eight defencemen left, the ninth-best
# D is not worth what the pre-draft table said. These pin the re-basing and the
# edges where it could go wrong quietly.


def _depth_board(n_per_pos=12):
    from puckpilot.draft.engine import DraftRules

    rules = DraftRules(shape=SHAPE, rounds=7, caps={"C": 9, "L": 9, "R": 9, "D": 9, "G": 9})
    return DraftBoard(_universe(), rules, my_seat=0), rules


def test_depth_zero_reproduces_the_frozen_vorp_exactly():
    """The disable switch has to be exact, not approximately equal - it is what
    makes every earlier measurement still comparable."""
    import numpy as np

    from puckpilot.draft.engine import RosterValuePolicy

    b, rules = _depth_board()
    p = RosterValuePolicy(replacement_depth=0.0)
    # the precise claim: the BASE score is the frozen array, untouched
    np.testing.assert_array_equal(p._base_score(b.u, rules, b.avail), b.u.vorp)
    # and it stays so once players come off the board
    b.record(int(b.u.ids[0]))
    np.testing.assert_array_equal(p._base_score(b.u, rules, b.avail), b.u.vorp)


def test_a_policy_with_no_avail_falls_back_to_the_frozen_vorp():
    """Bots and any caller that has not been updated must keep working."""
    import numpy as np

    from puckpilot.draft.engine import RosterValuePolicy

    b, rules = _depth_board()
    p = RosterValuePolicy(replacement_depth=0.5)
    np.testing.assert_array_equal(p._base_score(b.u, rules, None), b.u.vorp)
    np.testing.assert_array_equal(p._base_score(b.u), b.u.vorp)


def test_depleting_a_position_raises_the_value_of_what_is_left():
    """The whole point. Take the best defencemen off the board and the ones
    remaining must score HIGHER than they did, because replacement fell."""
    from puckpilot.draft.engine import RosterValuePolicy

    b, rules = _depth_board()
    p = RosterValuePolicy(replacement_depth=0.5, survival_discount=0.0)
    d_rows = [r for r in range(len(b.u.ids)) if str(b.u.pos[r]) == "D"]
    survivor = d_rows[-1]

    before = p.score(b.u, {}, rules, b.pick_context(0))[survivor]
    for r in d_rows[:-2]:
        b.record(int(b.u.ids[r]))
    after = p.score(b.u, {}, rules, b.pick_context(0))[survivor]
    assert after > before, f"scarcity did not raise value: {before:.3f} -> {after:.3f}"


def test_a_position_with_nobody_left_does_not_blow_up():
    from puckpilot.draft.engine import RosterValuePolicy

    b, rules = _depth_board()
    for r in [r for r in range(len(b.u.ids)) if str(b.u.pos[r]) == "G"]:
        b.record(int(b.u.ids[r]))
    out = RosterValuePolicy(replacement_depth=0.5).score(b.u, {}, rules, b.pick_context(0))
    assert np.isfinite(out).all()


def test_a_negative_depth_is_refused():
    from puckpilot.draft.engine import RosterValuePolicy

    with pytest.raises(ValueError, match="replacement_depth"):
        RosterValuePolicy(replacement_depth=-1.0)


def test_replacement_level_clamps_to_what_is_actually_there():
    """The clamp IS the scarcity signal, not an edge case to tolerate."""
    from puckpilot.engine.valuation import replacement_level

    vals = np.array([10.0, 8.0, 6.0])
    assert replacement_level(vals, 2) == 8.0
    assert replacement_level(vals, 99) == 6.0, "must fall back to the worst available"
    assert replacement_level(vals, 0) == 0.0
    assert replacement_level(np.array([]), 5) == 0.0


# ---- positional state shown to the drafter ---------------------------------


def test_supply_counts_who_is_actually_left_per_position():
    b, _ = _depth_board()
    before = b.supply()
    assert before["D"] == 8 and before["G"] == 8
    d_rows = [r for r in range(len(b.u.ids)) if str(b.u.pos[r]) == "D"]
    for r in d_rows[:5]:
        b.record(int(b.u.ids[r]))
    after = b.supply()
    assert after["D"] == 3, "supply must track the board, not the pre-draft pool"
    assert after["G"] == 8, "an untouched position must not move"


def test_depth_after_reads_the_real_pool_not_a_shortlist():
    """The old cliff signal compared against one player on an already-truncated
    shortlist, so it could only say 'better than the next name on this list'."""
    b, _ = _depth_board()
    rows = [r for r in range(len(b.u.ids)) if str(b.u.pos[r]) == "D"]
    top = max(rows, key=lambda r: b.u.vorp[r])
    gap = b.depth_after(top, steps=3)
    assert gap > 0, "the best D must be worth more than the 3rd-next D"
    # and it grows as the position is stripped from underneath him
    for r in sorted(rows, key=lambda r: -b.u.vorp[r])[1:4]:
        b.record(int(b.u.ids[r]))
    assert b.depth_after(top, steps=3) > gap


def test_depth_after_survives_an_empty_position():
    b, _ = _depth_board()
    rows = [r for r in range(len(b.u.ids)) if str(b.u.pos[r]) == "G"]
    keep = rows[0]
    for r in rows[1:]:
        b.record(int(b.u.ids[r]))
    assert b.depth_after(keep) == 0.0


# ---- the displayed probability is not the scoring knob ---------------------


def test_the_shown_probability_uses_the_calibrated_spread():
    """Every consumer of p_survive outside score() is a human. `calibrate` fits
    the spread against real rooms and says 16.0; the sim says 6.0 drafts best.
    Showing the 6.0 curve would put a number in front of the drafter that three
    real drafts say is wrong by nearly 3x in scale."""
    from puckpilot.draft.advice import recommend
    from puckpilot.draft.engine import RosterValuePolicy

    b, _ = _depth_board()
    policy = RosterValuePolicy(survival_spread=6.0, display_spread=16.0)
    cands = recommend(b, policy, n=8)
    ctx = b.pick_context(0)

    shown = {c.row: c.p_survive for c in cands}
    by_display = policy.survival(b.u, ctx, spread=16.0)
    by_scoring = policy.survival(b.u, ctx, spread=6.0)

    for row, value in shown.items():
        assert value == pytest.approx(by_display[row]), "shown value is not the calibrated one"
    # and the two really do differ, or this test proves nothing
    assert any(abs(by_display[r] - by_scoring[r]) > 0.02 for r in shown), (
        "the two spreads produced identical curves; the test cannot discriminate"
    )


def test_survival_defaults_to_the_scoring_spread():
    """score() must keep using the tuned knob, not the display one."""
    import numpy as np

    from puckpilot.draft.engine import RosterValuePolicy

    b, _ = _depth_board()
    policy = RosterValuePolicy(survival_spread=6.0, display_spread=16.0)
    ctx = b.pick_context(0)
    np.testing.assert_array_equal(policy.survival(b.u, ctx), policy.survival(b.u, ctx, spread=6.0))
