"""The approval queue.

These mostly pin that there is no way round it, because that is the whole
reason it exists: lineup changes run under standing authority, transactions do
not, and the difference has to be structural rather than a setting.
"""

from __future__ import annotations

import pytest

from puckpilot.season import proposals
from puckpilot.season.pool import FREE_AGENT, PoolPlayer
from puckpilot.season.proposals import APPROVED, EXECUTED, PENDING, REJECTED, ProposalError
from puckpilot.season.roster import RosterPlayer
from puckpilot.season.week import AddTarget


def target(name="Add Me", key="p.add", pid=9001, gain=3.0, drop=True, helps=("PPP +1.2",)):
    player = PoolPlayer(
        player_key=key,
        name=name,
        team="TOR",
        primary_position="C",
        yahoo_eligible=frozenset({"C", "Util"}),
        nhl_player_id=pid,
        ownership_type=FREE_AGENT,
    )
    dropped = (
        RosterPlayer(
            player_key="p.drop",
            yahoo_id="drop",
            name="Drop Me",
            team="MTL",
            primary_position="C",
            yahoo_eligible=frozenset({"C"}),
            selected_slot="BN",
            nhl_player_id=8001,
        )
        if drop
        else None
    )
    return AddTarget(
        player=player,
        games=4,
        value=gain + 1.0,
        helps=helps,
        drop=dropped,
        drop_value=1.0,
        timing=player.timing(),
    )


def make(db, *targets, **kw):
    return proposals.propose(db, "jimmy", "999.l.1", "999.l.1.t.5", list(targets), week=2, **kw)


# -- the point of the whole thing -------------------------------------------


def test_an_executor_cannot_take_a_pending_proposal(db):
    [p] = make(db, target())
    assert p.status == PENDING
    with pytest.raises(ProposalError, match="has not been approved"):
        proposals.take_for_execution(db, p.id)


def test_an_executor_cannot_take_a_rejected_one(db):
    [p] = make(db, target())
    proposals.decide(db, p.id, False)
    with pytest.raises(ProposalError, match="not approved"):
        proposals.take_for_execution(db, p.id)


def test_approval_is_what_makes_it_actionable(db):
    [p] = make(db, target())
    proposals.decide(db, p.id, True)
    got = proposals.take_for_execution(db, p.id)
    assert got.status == APPROVED
    assert got.add_pid == 9001


def test_nothing_else_hands_out_a_proposal_for_execution():
    """If another door opens later, this is the test that should fail."""
    public = {n for n in dir(proposals) if not n.startswith("_")}
    assert "take_for_execution" in public
    assert public & {"execute", "run", "apply"} == set()


def test_a_decision_cannot_be_made_twice(db):
    [p] = make(db, target())
    proposals.decide(db, p.id, True)
    with pytest.raises(ProposalError, match="already approved"):
        proposals.decide(db, p.id, False)


# -- ordinary behaviour -----------------------------------------------------


def test_a_proposal_records_why_it_was_made(db):
    [p] = make(db, target(helps=("PPP +1.2", "G +0.8")))
    assert p.reason["helps"] == ["PPP +1.2", "G +0.8"]
    assert p.reason["week"] == 2
    assert "race" in p.reason["timing"]
    assert "PPP +1.2" in p.describe()


def test_re_proposing_the_same_add_does_not_duplicate(db):
    """A job that runs several times a day must not grow the queue each run."""
    make(db, target())
    again = make(db, target())
    assert again == []
    assert len(proposals.pending(db, "jimmy")) == 1


def test_the_pending_queue_is_capped(db):
    made = make(
        db, *[target(name=f"P{i}", key=f"p.{i}", pid=9000 + i) for i in range(8)], max_pending=3
    )
    assert len(made) == 3
    assert len(proposals.pending(db, "jimmy")) == 3


def test_a_decided_proposal_frees_room_in_the_queue(db):
    [p] = make(db, target(), max_pending=1)
    assert make(db, target(name="Other", key="p.other", pid=9002), max_pending=1) == []
    proposals.decide(db, p.id, False)
    assert len(make(db, target(name="Other", key="p.other", pid=9002), max_pending=1)) == 1


def test_an_unmapped_player_is_never_proposed(db):
    """We could not tell an executor who to add."""
    t = target()
    t = AddTarget(
        player=PoolPlayer(
            player_key="p.x",
            name="Unknown",
            team="TOR",
            primary_position="C",
            yahoo_eligible=frozenset({"C"}),
            nhl_player_id=None,
        ),
        games=t.games,
        value=t.value,
        helps=t.helps,
        drop=t.drop,
        drop_value=t.drop_value,
        timing=t.timing,
    )
    assert make(db, t) == []


def test_a_target_without_a_legal_drop_still_proposes(db):
    [p] = make(db, target(drop=False))
    assert p.drop_player_key == ""
    assert "DROP" not in p.describe()


def test_executing_marks_it_and_keeps_the_result(db):
    [p] = make(db, target())
    proposals.decide(db, p.id, True)
    done = proposals.mark_executed(db, p.id, "Yahoo accepted")
    assert done.status == EXECUTED
    assert done.reason["result"] == "Yahoo accepted"
    assert done.executed_at


def test_listing_filters_by_status_and_manager(db):
    [a] = make(db, target(name="A", key="p.a", pid=9101))
    make(db, target(name="B", key="p.b", pid=9102))
    proposals.decide(db, a.id, False)
    assert len(proposals.listing(db, "jimmy", status=REJECTED)) == 1
    assert len(proposals.listing(db, "jimmy", status=PENDING)) == 1
    assert proposals.listing(db, "sam") == []


def test_a_missing_proposal_is_an_error_not_none(db):
    with pytest.raises(ProposalError, match="no proposal"):
        proposals.get(db, 999)


# -- the audit log ----------------------------------------------------------


def test_actions_are_recorded_with_their_reasoning(db):
    proposals.record_action(
        db,
        "jimmy",
        "999.l.1",
        "999.l.1.t.5",
        "2026-10-07",
        "lineup",
        {"moves": ["START X", "BENCH Y"], "gain": 2.1},
        outcome="executed",
    )
    rows = proposals.actions(db, "jimmy")
    assert len(rows) == 1
    assert rows[0]["kind"] == "lineup"
    assert rows[0]["outcome"] == "executed"


def test_actions_can_be_read_back_by_date(db):
    for d in ("2026-10-01", "2026-10-07"):
        proposals.record_action(db, "jimmy", "l", "t", d, "lineup", {}, outcome="dry-run")
    assert len(proposals.actions(db, "jimmy", since="2026-10-05")) == 1


def test_an_approved_add_is_not_proposed_again(db):
    """It is waiting to be executed, not waiting for an opinion."""
    [p] = make(db, target())
    proposals.decide(db, p.id, True)
    assert make(db, target()) == []


def test_a_refusal_is_not_raised_again_the_same_week(db):
    """Asking again the same week about a player you said no to is the
    notification that teaches someone to stop reading them."""
    [p] = make(db, target())
    proposals.decide(db, p.id, False)
    assert make(db, target()) == []


def test_a_new_week_reconsiders_a_refusal(db):
    [p] = make(db, target())
    proposals.decide(db, p.id, False)
    again = proposals.propose(
        db, "jimmy", "999.l.1", "999.l.1.t.5", [target()], week=3
    )
    assert len(again) == 1
