"""Drop tiers: who a move may let go, set by the manager or by the rule.

The rule before tiers was measured (`stream_spots`): the cheapest few rotate,
anyone else goes only for an upgrade. These pin down that a tier changes
nothing until one is set, what each one allows once it is, and that a pin
reaches every place a drop is chosen - the add search and a make-room move
here, the executor's last check in test_season_transactions.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from puckpilot.season import legality, tiers
from puckpilot.season.tiers import Tier
from tests.test_season_legality import (
    BACK,
    CENTRE,
    DEFENCE,
    GOALIE,
    KEEP,
    RULES,
    SMALL,
    STREAMER,
    p,
    team,
)


def pin(player, tier: str):
    return replace(player, tier_pin=tier)


def rotating(placed) -> set[str]:
    return {k for k, v in placed.items() if v.tier is Tier.ROTATION}


# -- the rule, and the manager's pins ------------------------------------------


def test_with_nothing_pinned_the_rotation_is_the_measured_stream_floor():
    roster = [CENTRE, STREAMER, DEFENCE, GOALIE]
    placed = tiers.board(roster, KEEP, 2)
    assert rotating(placed) == {"streamer", "goalie"}
    assert {v.source for v in placed.values()} == {tiers.AUTO}
    # A tie at the last spot's value rotates too, as the floor always did.
    assert rotating(tiers.board(roster, {**KEEP, "defence": 200.0}, 2)) == {
        "streamer", "goalie", "defence"
    }  # fmt: skip
    assert rotating(tiers.board(roster, KEEP, 0)) == set()


def test_rotation_pins_take_the_spots_first():
    placed = tiers.board([CENTRE, STREAMER, pin(DEFENCE, "rotation"), GOALIE], KEEP, 2)
    assert {k: placed[k].source for k in rotating(placed)} == {
        "defence": tiers.YOU, "streamer": tiers.AUTO
    }  # fmt: skip
    # Pinned past the spots: nobody else rotates by the rule.
    many = [pin(CENTRE, "rotation"), STREAMER, pin(DEFENCE, "rotation"), GOALIE]
    assert rotating(tiers.board(many, KEEP, 2)) == {"centre", "defence"}


def test_a_player_out_tonight_gives_his_spot_to_the_next_cheapest():
    out = p("streamer", "C", "BN", status="O")
    roster = team(CENTRE, out, DEFENCE, GOALIE)
    placed = tiers.board(roster.players, KEEP, 2, droppable=legality.droppable(roster))
    assert placed["streamer"].tier is Tier.UPGRADE
    assert rotating(placed) == {"goalie", "defence"}


def test_yahoo_locks_beat_any_pin_and_a_pin_overrides_keeper_protection():
    locked = pin(p("locked", "C", "C", undroppable=True), "rotation")
    keeper = pin(p("keeper", "C", "C", protected=True), "upgrade")
    assert tiers.is_never(locked) and not tiers.is_never(keeper)
    placed = tiers.board([locked, keeper], {"locked": 1.0, "keeper": 2.0}, 2)
    assert (placed["locked"].tier, placed["locked"].source) == (Tier.NEVER, tiers.YAHOO)
    assert (placed["keeper"].tier, placed["keeper"].source) == (Tier.UPGRADE, tiers.YOU)
    plain = p("plain", "C", "C", protected=True)
    assert tiers.board([plain], {}, 2)["plain"].source == tiers.KEEPER
    assert [q.player_key for q in legality.droppable(team(locked, keeper, plain))] == ["keeper"]


def test_a_clear_upgrade_is_one_by_a_share_of_his_value():
    assert tiers.clears(110.0, 100.0, 0.10) and not tiers.clears(109.0, 100.0, 0.10)
    assert tiers.clears(100.0, 100.0) and not tiers.clears(99.0, 100.0)
    # Of the value's size: a player valued below zero is not made easier to drop.
    assert tiers.clears(-90.0, -100.0, 0.10) and not tiers.clears(-95.0, -100.0, 0.10)


def test_an_unknown_tier_is_refused_and_auto_is_the_rule():
    assert tiers.tier_of("auto") is None and tiers.tier_of("") is None
    assert tiers.tier_of("rotation") is Tier.ROTATION
    with pytest.raises(ValueError, match="unknown tier"):
        tiers.tier_of("sometimes")


def test_pins_are_kept_per_manager_listed_and_taken_off(db):
    tiers.pin(db, "m", "L", "p.1", Tier.ROTATION, name="Kleven", nhl_player_id=7, via="page")
    tiers.pin(db, "m", "L", "p.2", Tier.NEVER, name="Star")
    tiers.pin(db, "rival", "L", "p.1", Tier.NEVER)  # a rival's view is his own
    assert tiers.pins(db, "m", "L") == {"p.1": Tier.ROTATION, "p.2": Tier.NEVER}
    tiers.pin(db, "m", "L", "p.1", None)
    assert tiers.pins(db, "m", "L") == {"p.2": Tier.NEVER}
    assert [(x["name"], x["set_via"]) for x in tiers.listing(db, "m", "L")] == [("Star", "cli")]


def test_annotate_puts_pins_on_and_takes_old_ones_off():
    got = tiers.annotate(team(pin(CENTRE, "never"), STREAMER), {"streamer": Tier.ROTATION})
    assert {q.player_key: q.tier_pin for q in got.players} == {"centre": "", "streamer": "rotation"}


# -- a make-room move ------------------------------------------------------------


def test_a_never_pin_is_never_the_make_room_drop_or_released():
    res = legality.resolve(
        SMALL, team(CENTRE, pin(STREAMER, "never"), DEFENCE, GOALIE, BACK), KEEP, RULES
    )
    assert res.transactions[0].drop.player_key == "defence"  # the streamer is cheaper
    res = legality.resolve(
        SMALL, team(CENTRE, STREAMER, DEFENCE, GOALIE, pin(BACK, "never")), {**KEEP, "back": 10.0},
        RULES,
    )  # fmt: skip
    assert not res.transactions[0].releases


def test_make_room_takes_the_cheapest_its_tiers_let_go():
    """A rotation player, or an upgrade-only one the returning player clearly
    beats; a big enough margin leaves only the rotation one."""
    roster = team(CENTRE, pin(STREAMER, "upgrade"), pin(DEFENCE, "rotation"), GOALIE, BACK)
    (fix,) = legality.resolve(SMALL, roster, KEEP, RULES).transactions
    assert fix.drop.player_key == "streamer" and "Upgrade only" in fix.reason  # 280 beats 40
    (fix,) = legality.resolve(SMALL, roster, KEEP, RULES, upgrade_margin=10.0).transactions
    assert fix.drop.player_key == "defence" and "Rotation" in fix.reason


def test_with_no_drop_a_tier_allows_the_cheapest_legal_one_is_still_named():
    """An illegal roster refuses every add, so the card always says who - and
    says no tier rule chose him."""
    keeper_back = p("back", "D", "IR+", eligible=("D", "Util"), protected=True)
    roster = team(CENTRE, pin(STREAMER, "upgrade"), DEFENCE, GOALIE, keeper_back)
    (fix,) = legality.resolve(
        SMALL, roster, {**KEEP, "back": 10.0}, RULES, stream_spots=0
    ).transactions
    assert fix.drop.player_key == "streamer" and "not a clear upgrade" in fix.reason


def test_nothing_pinned_makes_room_exactly_as_before_tiers():
    for keep in (KEEP, {**KEEP, "back": 10.0}, {**KEEP, "goalie": 1.0}):
        roster = team(CENTRE, STREAMER, DEFENCE, GOALIE, BACK)
        for margin in (0.0, 0.5):
            res = legality.resolve(SMALL, roster, keep, RULES, upgrade_margin=margin)
            cheapest = min(
                (q for q in legality.droppable(roster)
                 if legality.legal_drop("D", q, legality.position_counts(roster), RULES)),
                key=lambda q: keep[q.player_key],
            )  # fmt: skip
            (fix,) = res.transactions
            want = "back" if keep["back"] <= keep[cheapest.player_key] else cheapest.player_key
            assert fix.drop.player_key == want


# -- the add search ----------------------------------------------------------------


def test_a_regular_pinned_to_rotation_may_go_for_a_streamer(db):
    """With one rotation spot and Mid pinned to it, Depth is upgrade only."""
    from tests.test_season_week import _fa, _rp, _targets_for

    got = _targets_for(
        db,
        (pin(_rp("p.1", "Mid", 1, "MTL", "C"), "rotation"), _rp("p.2", "Depth", 2, "TOR", "C")),
        (("C", 2, 1),),
        pool=[_fa("fa.1", "Streamer", 9), _fa("fa.2", "Streamer Two", 10)],
        rates={1: {"goals": 0.3}, 2: {"goals": 0.1}, 9: {"goals": 0.5}, 10: {"goals": 0.45}},
        per_game={1: 1.5, 2: 0.3, 9: 0.2, 10: 0.2},
        stream_spots=1,
    )
    assert [t.drop.name for t in got] == ["Mid"]


def test_a_never_pin_is_not_a_drop_for_the_search(db):
    from tests.test_season_week import _fa, _rp, _targets_for

    got = _targets_for(
        db,
        (_rp("p.1", "Mid", 1, "MTL", "C"), pin(_rp("p.2", "Depth", 2, "TOR", "C"), "never")),
        (("C", 2, 1),),
        pool=[_fa("fa.1", "Star FA", 9)],
        rates={1: {"goals": 0.2}, 2: {"goals": 0.1}, 9: {"goals": 0.9}},
        per_game={1: 0.6, 2: 0.3, 9: 3.0},
        stream_spots=1,
    )
    assert [t.drop.name for t in got] == ["Mid"]


@pytest.mark.parametrize(("margin", "drops"), [(0.0, ["Depth", "Mid"]), (4.5, ["Depth"])])
def test_the_margin_keeps_a_regular_from_a_small_upgrade(db, margin, drops):
    """Star FA Two is worth 4.2 times Mid a game: an upgrade at no margin, not
    at 450% - and Depth, the one rotation spot, goes for anyone."""
    from tests.test_season_week import _fa, _rp, _targets_for

    got = _targets_for(
        db,
        (_rp("p.1", "Mid", 1, "TOR", "C"), _rp("p.2", "Depth", 2, "TOR", "C")),
        (("C", 2, 1),),
        pool=[_fa("fa.1", "Star FA", 9), _fa("fa.2", "Star FA Two", 10)],
        rates={1: {"goals": 0.2}, 2: {"goals": 0.1}, 9: {"goals": 0.9}, 10: {"goals": 0.8}},
        per_game={1: 0.6, 2: 0.3, 9: 3.0, 10: 2.5},
        stream_spots=1,
        upgrade_margin=margin,
    )
    assert sorted(t.drop.name for t in got) == drops
