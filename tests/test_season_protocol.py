"""The week's agreed stance, and what it is allowed to change."""

from __future__ import annotations

import pytest

from puckpilot.engine.categories import resolve
from puckpilot.season import protocol
from puckpilot.season.protocol import (
    APPROVED,
    CHASE,
    CONCEDE,
    HOLD,
    PROPOSED,
    WEIGHTS,
    ProtocolError,
)
from puckpilot.season.week import CategoryOutlook


def outlook(label, ours, theirs, lineup_room=0.0, add_room=0.0):
    return CategoryOutlook(
        category=resolve(label),
        ours=ours,
        theirs=theirs,
        lineup_room=lineup_room,
        add_room=add_room,
    )


def derive(*os, week=2):
    return protocol.derive(list(os), "jimmy", "999.l.1", "999.l.1.t.5", week, "Them")


# -- what gets conceded -----------------------------------------------------


def test_a_gap_bigger_than_every_lever_is_conceded():
    """The case this exists for: behind on hits by more than a re-slot and an
    add together could close."""
    p = derive(outlook("HIT", 41.2, 52.2, lineup_room=0.0, add_room=6.0))
    assert p.stances[0].stance == CONCEDE
    assert "more than a lineup change and an add" in p.stances[0].reason()


def test_a_gap_an_add_could_close_is_not_conceded():
    p = derive(outlook("HIT", 41.2, 52.2, lineup_room=0.0, add_room=20.0))
    assert p.stances[0].stance != CONCEDE


def test_a_category_we_lead_is_never_conceded():
    p = derive(outlook("SV", 135.0, 86.0, lineup_room=0.0, add_room=0.0))
    assert p.stances[0].stance == HOLD


def test_a_rate_category_is_never_conceded_for_being_unmeasurable():
    """A rate has no answer in these units, so neither lever can be measured.
    Treating that as zero room conceded save percentage every week we trailed
    by a thousandth."""
    o = CategoryOutlook(
        category=resolve("SV%"), ours=0.898, theirs=0.902, lineup_room=None, add_room=None
    )
    assert o.measured is False
    assert o.reachable is True
    assert derive(o).stances[0].stance == CHASE


def test_a_close_category_is_chased():
    p = derive(outlook("PPP", 8.5, 7.7, add_room=5.0))
    assert p.stances[0].stance == CHASE


def test_everything_else_is_left_exactly_alone():
    p = derive(outlook("SOG", 94.2, 82.2, add_room=5.0))
    assert p.stances[0].stance == HOLD
    assert p.stances[0].weight == 1.0


# -- what a stance does -----------------------------------------------------


def test_only_an_approved_protocol_changes_a_number():
    p = derive(outlook("HIT", 41.0, 52.0, add_room=1.0), outlook("PPP", 8.5, 7.7, add_room=5.0))
    assert p.status == PROPOSED
    assert p.weights() == {}
    assert p.is_active is False


def test_an_approved_protocol_weights_only_chase_and_concede(db):
    p = protocol.save(
        db,
        derive(
            outlook("HIT", 41.0, 52.0, add_room=1.0),
            outlook("PPP", 8.5, 7.7, add_room=5.0),
            outlook("SOG", 94.0, 82.0, add_room=5.0),
        ),
    )
    live = protocol.decide(db, p.id, True)
    w = live.weights()
    assert w["hits"] == WEIGHTS[CONCEDE]
    assert w["ppp"] == WEIGHTS[CHASE]
    assert "sog" not in w  # hold is inert
    assert live.is_active


def test_a_rejected_protocol_governs_nothing(db):
    p = protocol.save(db, derive(outlook("PPP", 8.5, 7.7, add_room=5.0)))
    protocol.decide(db, p.id, False)
    assert protocol.active(db, "jimmy", "999.l.1", 2) is None


# -- storage ----------------------------------------------------------------


def test_re_deriving_refreshes_the_proposal_rather_than_stacking(db):
    protocol.save(db, derive(outlook("PPP", 8.5, 7.7, add_room=5.0)))
    protocol.save(db, derive(outlook("PPP", 9.0, 7.0, add_room=5.0)))
    assert len(protocol.listing(db, "jimmy", "999.l.1")) == 1


def test_an_approved_protocol_is_not_quietly_replaced(db):
    """The lineup has been acting on it."""
    p = protocol.save(db, derive(outlook("PPP", 8.5, 7.7, add_room=5.0)))
    protocol.decide(db, p.id, True)
    with pytest.raises(ProtocolError, match="reject it first"):
        protocol.save(db, derive(outlook("PPP", 9.0, 7.0, add_room=5.0)))


def test_an_approved_protocol_wins_over_a_later_proposal(db):
    a = protocol.save(db, derive(outlook("PPP", 8.5, 7.7, add_room=5.0), week=3))
    protocol.decide(db, a.id, True)
    got = protocol.load(db, "jimmy", "999.l.1", 3)
    assert got.status == APPROVED


def test_a_decision_cannot_be_made_twice(db):
    p = protocol.save(db, derive(outlook("PPP", 8.5, 7.7, add_room=5.0)))
    protocol.decide(db, p.id, True)
    with pytest.raises(ProtocolError, match="already approved"):
        protocol.decide(db, p.id, False)


def test_stances_survive_a_round_trip(db):
    p = protocol.save(
        db, derive(outlook("HIT", 41.0, 52.0, add_room=1.0), outlook("PPP", 8.5, 7.7, add_room=5.0))
    )
    back = protocol.load_by_id(db, p.id)
    assert [s.category.label for s in back.stances] == ["HIT", "PPP"]
    assert [s.stance for s in back.stances] == [CONCEDE, CHASE]


def test_a_missing_protocol_is_an_error_not_none(db):
    with pytest.raises(ProtocolError, match="no protocol"):
        protocol.load_by_id(db, 999)
