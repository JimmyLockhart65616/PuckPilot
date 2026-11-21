"""Multi-position eligibility: slot matching, and the one property that makes
it safe to put behind a flag - with single-position players it is EXACTLY the
old per-position-count logic, in the engine's score, its pick, and the replay.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from puckpilot.draft.eligibility import (
    open_positions,
    parse_yahoo_positions,
    slot_units,
    starts_in_order,
    unfilled_starting_slots,
)
from puckpilot.draft.engine import DraftRules, RosterValuePolicy, Universe
from puckpilot.draft.replay import G_WIDTH, ReplayData, replay_roster
from puckpilot.engine.valuation import LeagueShape

C, L, R, D, G = (frozenset({p}) for p in "CLRDG")
CL = frozenset({"C", "L"})
LR = frozenset({"L", "R"})

SHAPE = LeagueShape(
    n_teams=4, slots=(("C", 2), ("L", 2), ("R", 2), ("D", 4), ("G", 2)), util_slots=1, bench_slots=3
)
RULES = DraftRules(shape=SHAPE, rounds=16)
UNITS = slot_units(SHAPE.slots, SHAPE.util_slots)


def test_parse_yahoo_positions_drops_slot_types():
    assert parse_yahoo_positions("C,LW,Util,IR+") == CL
    assert parse_yahoo_positions("G") == G
    assert parse_yahoo_positions(None) == frozenset()


def test_a_dual_player_moves_aside_for_a_single_position_one():
    """C/L seated at C must slide to L so a C-only player can start - the
    case a greedy fill gets wrong."""
    units = slot_units((("C", 1), ("L", 1)), 0)
    assert starts_in_order((CL, C), units) == (True, True)
    assert starts_in_order((C, C), units) == (True, False)


def test_open_positions_see_through_a_reshuffle():
    units = slot_units((("C", 1), ("L", 1)), 0)
    assert open_positions([CL], units, "CLRDG") == {"C", "L"}
    assert open_positions([CL, C], units, "CLRDG") == set()


def test_util_takes_any_skater_but_never_a_goalie():
    units = slot_units((("C", 1), ("G", 1)), 1)
    assert open_positions([C], units, "CLRDG") == {"C", "L", "R", "D", "G"} - {"C"} | {"C"}
    assert open_positions([C, C, G], units, "CLRDG") == set()


def test_unfilled_starting_slots_counts_what_no_arrangement_can_fill():
    units = slot_units((("C", 1), ("L", 1), ("R", 1)), 0)
    assert unfilled_starting_slots([CL, LR], units) == 1
    assert unfilled_starting_slots([C, C], units) == 2


# ---- equivalence with the single-position path ---------------------------------


def _universe(n=60, seed=0):
    rng = np.random.default_rng(seed)
    pos = rng.choice(list("CLRDG"), size=n, p=[0.22, 0.2, 0.2, 0.28, 0.1])
    df = pd.DataFrame(
        {
            "name": [f"P{i}" for i in range(n)],
            "position": pos,
            "vorp": rng.normal(0, 4, n),
            "z_total": rng.normal(0, 4, n),
            "adp_rank": rng.permutation(n).astype(float) + 1,
        },
        index=pd.Index(range(1000, 1000 + n), name="player_id"),
    ).sort_values("vorp", ascending=False)
    return Universe(df)


def test_single_position_scores_and_picks_match_the_count_path_exactly():
    u = _universe()
    old = RosterValuePolicy()
    new = RosterValuePolicy(multi_position=True)
    rng = np.random.default_rng(7)
    for _ in range(200):
        rows = list(rng.choice(len(u), size=int(rng.integers(0, 16)), replace=False))
        counts: dict[str, int] = {}
        for r in rows:
            counts[str(u.pos[r])] = counts.get(str(u.pos[r]), 0) + 1
        avail = np.ones(len(u), dtype=bool)
        avail[rows] = False
        ctx = {"pick_no": 5, "next_pick_no": 20, "avail": avail, "roster_rows": rows}
        picks_left = int(rng.integers(1, 17))
        np.testing.assert_allclose(
            old.score(u, counts, RULES, ctx), new.score(u, counts, RULES, ctx)
        )
        assert old.pick(u, avail, counts, RULES, picks_left, rng, ctx) == new.pick(
            u, avail, counts, RULES, picks_left, rng, ctx
        )


def test_a_dual_eligible_player_starts_where_a_single_count_benches_him():
    u = _universe()
    c_rows = [r for r in range(len(u)) if u.pos[r] == "C"]
    skaters = [r for r in range(len(u)) if u.pos[r] in "LRD"]
    # both C slots and util full; wings empty
    roster = [*c_rows[:3]]
    counts = {"C": 3}
    candidate = c_rows[3]
    ctx = {"roster_rows": roster}
    plain = RosterValuePolicy(multi_position=True, survival_discount=0)
    base = plain.score(u, counts, RULES, ctx)[candidate]
    dual = u.with_eligibility({int(u.ids[candidate]): CL})
    lifted = plain.score(dual, counts, RULES, ctx)[candidate]
    if u.vorp[candidate] > 0:
        assert lifted == pytest.approx(base / plain.bench_factor)
    assert skaters  # fixture sanity


def test_with_eligibility_never_makes_a_goalie_a_skater_or_trusts_an_empty_set():
    u = _universe()
    g = next(r for r in range(len(u)) if u.pos[r] == "G")
    c = next(r for r in range(len(u)) if u.pos[r] == "C")
    v = u.with_eligibility({int(u.ids[g]): CL, int(u.ids[c]): G})
    assert v.elig_sets[g] == G and v.elig_sets[c] == C


def test_the_replay_with_single_position_eligibility_is_the_greedy_replay():
    data = ReplayData()
    data.dates = ["2025-10-06", "2025-10-07", "2025-10-13"]
    rng = np.random.default_rng(3)
    positions = {}
    for pid in range(1, 21):
        positions[pid] = "G" if pid > 17 else str(rng.choice(list("CLRD")))
        days = {i: rng.random(len(data.skater_keys)) for i in range(3) if rng.random() < 0.8}
        if positions[pid] == "G":
            data.goalie[pid] = {i: rng.random(G_WIDTH) for i in days}
        else:
            data.skater[pid] = days
    scalar = {pid: float(rng.random()) for pid in positions}
    roster = list(positions)
    sk0, g0 = replay_roster(roster, positions, scalar, data, SHAPE)
    single = {pid: frozenset({p}) for pid, p in positions.items()}
    sk1, g1 = replay_roster(roster, positions, scalar, data, SHAPE, single)
    np.testing.assert_allclose(sk0, sk1)
    np.testing.assert_allclose(g0, g1)


def test_the_eligibility_label_leads_with_the_primary_position():
    from puckpilot.draft.advice import eligible_label

    u = _universe()
    c = next(r for r in range(len(u)) if u.pos[r] == "C")
    assert eligible_label(u, c) == "C"
    v = u.with_eligibility({int(u.ids[c]): frozenset({"R", "C", "L"})})
    assert eligible_label(v, c) == "C/L/R"


def test_a_label_never_shows_a_position_yahoo_does_not_allow():
    """Necas: C in the NHL's data, RW-only on Yahoo. "C/R" would offer the
    drafter a centre slot he cannot use."""
    from puckpilot.draft.advice import eligible_label

    u = _universe()
    c = next(r for r in range(len(u)) if u.pos[r] == "C")
    v = u.with_eligibility({int(u.ids[c]): frozenset({"R"})})
    assert eligible_label(v, c) == "R, valued C"
