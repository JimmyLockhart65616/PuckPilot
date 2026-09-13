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


# ---- the pick clock must keep up with the room -----------------------------
#
# `board.made` is the index into `slots` that produces `on_the_clock`,
# `current_round` and `next_pick_no` - and `survival()` is a logistic in
# `adp_rank - next_pick_no`. So a pick we refuse does not merely lose a name: it
# makes every "lasts N%" on screen wrong for the rest of the draft, and the
# error accumulates. Measured across six real drafts: 63 refusals in 1,145 picks
# before the board was widened, 7 after.


def test_a_player_we_cannot_rank_still_consumes_his_slot():
    from puckpilot.draft.feed import PickEvent, apply

    b, _ = _depth_board()
    before_avail = int(b.avail.sum())
    accepted, rejected = apply(b, [PickEvent(999_999, None, "test")])

    assert b.made == 1, "the pick really happened; the clock must advance"
    assert not accepted and rejected, "and it is still reported, not swallowed silently"
    assert int(b.avail.sum()) == before_avail, "we do not know who he was"
    assert b.roster(0) == [] or all(p.row >= 0 for p in b.roster(0))


def test_a_duplicate_pick_does_NOT_advance_the_clock():
    """The opposite case, and the reason this needed its own exception type: a
    feed re-reporting a pick is a duplicate poll, not a second selection."""
    from puckpilot.draft.feed import PickEvent, apply

    b, _ = _depth_board()
    pid = int(b.u.ids[0])
    apply(b, [PickEvent(pid, None, "test")])
    assert b.made == 1
    accepted, rejected = apply(b, [PickEvent(pid, None, "test")])
    assert b.made == 1, "a duplicate must not consume a slot"
    assert not accepted and rejected


def test_the_clock_stays_with_the_room_across_unknown_picks():
    """The property that actually matters: next_pick_no after N picks is the
    same whether or not some of them were players we could rank."""
    from puckpilot.draft.feed import PickEvent, apply

    known, _ = _depth_board()
    mixed, _ = _depth_board()
    for i in range(6):
        apply(known, [PickEvent(int(known.u.ids[i]), None, "t")])
        apply(mixed, [PickEvent(int(mixed.u.ids[i]) if i % 2 else 999_000 + i, None, "t")])
    assert known.made == mixed.made == 6
    assert known.next_pick_no(0) == mixed.next_pick_no(0)
    assert known.on_the_clock() == mixed.on_the_clock()


def test_undo_of_an_unknown_pick_gives_nothing_back():
    b, _ = _depth_board()
    before = int(b.avail.sum())
    b.record_unknown()
    p = b.undo()
    assert p is not None and p.row == -1
    assert b.made == 0 and int(b.avail.sum()) == before


# ---- universe depth --------------------------------------------------------


def test_widening_the_universe_cannot_change_a_single_vorp():
    """`replacement_adjust` runs over the whole projected frame and the cut is a
    pure truncation applied after, so depth is free. This is the guard on that:
    if it ever stops being true, every tuned constant is invalidated."""
    import pandas as pd

    from puckpilot.engine.valuation import rank_players

    sk = pd.DataFrame(
        {
            "name": [f"S{i}" for i in range(40)],
            "position": ["C"] * 20 + ["D"] * 20,
            "goals": np.linspace(40, 1, 40),
            "assists": np.linspace(50, 2, 40),
        },
        index=pd.Index(range(1, 41), name="player_id"),
    )
    go = pd.DataFrame(
        {"name": ["G1"], "position": ["G"], "wins": [30.0]},
        index=pd.Index([99], name="player_id"),
    )
    from puckpilot.engine.categories import Category

    cats = (Category("goals", "G", "skater"), Category("assists", "A", "skater"))
    ranked = rank_players(sk, go, skater_cats=cats, goalie_cats=(Category("wins", "W", "goalie"),))
    shallow, deep = ranked.head(10), ranked.head(30)
    common = shallow.index.intersection(deep.index)
    assert (shallow.loc[common, "vorp"] == deep.loc[common, "vorp"]).all()


def test_supply_counts_startable_players_not_the_whole_pool():
    """The pool is deliberately deeper than the draft is long, so a raw count
    reads 'D 340 left' and means nothing to a drafter."""
    b, _ = _depth_board()
    supply = b.supply()
    assert all(v <= int((b.avail & (b.u.vorp > 0)).sum()) for v in supply.values())


# ---- where our board and the room disagree ---------------------------------


def _market_board(n_per_pos=10):
    """A board where the naive overall metric and the positional one DISAGREE.

    Replacement level differs hugely by position in a real league - the 28th
    centre sits near z -0.9 while the 48th defenceman is near -6 - so an overall
    rank-vs-ADP comparison measures that structural offset rather than any
    opinion about a player. This fixture reproduces the shape: every D is worth
    less than every C, so a D can be the best D left while ranking far below
    every C overall.
    """
    import pandas as pd

    from puckpilot.draft.engine import Universe

    rows, pid = {}, 1
    for pos, base in (("C", 20.0), ("D", 2.0)):
        for i in range(n_per_pos):
            rows[pid] = {
                "name": f"{pos}{i}",
                "position": pos,
                "team": "AAA",
                "vorp": base - i,
                "z_total": base - i,
                # Centres: the two boards agree exactly, so any "disagreement"
                # found among them is the positional offset, not an opinion.
                # Defencemen: the market's order is the REVERSE of ours, which
                # is a real within-position disagreement the panel must find.
                "adp_rank": float(pid if pos == "C" else n_per_pos - i),
                "train_gp": 200.0,
                "age": 27.0,
            }
            pid += 1
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "player_id"
    u = Universe(df.sort_values("vorp", ascending=False))
    u.has_market = np.ones(len(u), dtype=bool)
    from puckpilot.draft.engine import DraftRules

    rules = DraftRules(shape=SHAPE, rounds=7, caps={"C": 9, "L": 9, "R": 9, "D": 9, "G": 9})
    return DraftBoard(u, rules, my_seat=0), rules


def test_disagreement_is_measured_within_position_not_overall():
    """The whole panel rests on this. Overall, every D ranks below every C, so a
    naive metric reports a huge 'disagreement' for every defenceman and leads
    with players nobody is actually disagreeing about."""
    from puckpilot.draft.advice import market_disagreement

    b, _ = _market_board()
    sleeping, rated = market_disagreement(b, n=20)
    # Every centre is ranked identically by both boards, so none may appear -
    # overall they would all show a large gap purely from the C/D offset.
    assert not any(g.position == "C" for g in sleeping + rated), (
        "centres appear as disagreements only if ranks are computed overall"
    )
    # and the genuine within-position disagreement among D IS found
    assert any(g.position == "D" for g in sleeping + rated)


def test_a_material_gap_is_required():
    """Ours #2 against room #1 is two boards agreeing, not a disagreement."""
    from puckpilot.draft.advice import MIN_RANK_GAP, market_disagreement

    b, _ = _market_board()
    sleeping, rated = market_disagreement(b, n=10)
    for g in sleeping + rated:
        assert abs(g.our_rank - g.market_rank) >= MIN_RANK_GAP


def test_thin_history_outranks_age_as_the_flag():
    """A 26-year-old with 20 NHL games is the same epistemic problem as a
    20-year-old with 20, and age alone would call out only one of them."""
    from puckpilot.draft.advice import FADING_AGE, THIN_EVIDENCE_GP, market_disagreement

    b, _ = _market_board()
    b.u.frame.loc[b.u.frame["name"] == "D9", ["train_gp", "age"]] = [10.0, FADING_AGE + 5]
    sleeping, rated = market_disagreement(b, n=20)
    found = [g for g in sleeping + rated if g.name == "D9"]
    if found:
        assert found[0].flag == "thin history", "evidence must win over age"
    assert THIN_EVIDENCE_GP > 0


def test_our_side_only_offers_startable_players():
    """A bargain at a position where the best left is sub-replacement is not a
    bargain."""
    from puckpilot.draft.advice import market_disagreement

    b, _ = _market_board()
    sleeping, _ = market_disagreement(b, n=20)
    assert all(g.vorp > 0 for g in sleeping)


def test_the_panel_sees_past_the_shortlist():
    """Players the market rates far above us are below `recommend`'s cut by
    construction, so the panel must read the whole board."""
    from puckpilot.draft.advice import market_disagreement, recommend

    b, _ = _market_board()
    _, rated = market_disagreement(b, n=5)
    shortlist = {c.name for c in recommend(b, n=3)}
    assert rated, "expected at least one player the market rates above us"
    assert any(g.name not in shortlist for g in rated)
