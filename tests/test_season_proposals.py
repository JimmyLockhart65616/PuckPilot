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
    """An `AddTarget` in its post-repricing shape: net starts and net category
    change, rather than an abstract value number."""
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
        starts=4.0,
        drop_starts=2.0,
        deltas={"ppp": 1.2, "sog": 4.4},
        helps=helps,
        drop=dropped,
        score=gain,
        labels={"ppp": "PPP", "sog": "SOG"},
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
    # The net effect in the league's own units, not an abstract score.
    assert p.reason["extra_starts"] == 2.0
    assert "SOG +4.4" in p.reason["moved"]
    assert "+2 starts" in p.describe()


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
        starts=t.starts,
        drop_starts=t.drop_starts,
        deltas=t.deltas,
        helps=t.helps,
        drop=t.drop,
        score=t.score,
        labels=t.labels,
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
    again = proposals.propose(db, "jimmy", "999.l.1", "999.l.1.t.5", [target()], week=3)
    assert len(again) == 1


# -- a newer search replaces the queue ---------------------------------------


def _five_stale(db):
    return make(
        db,
        *(target(name=f"Stale {i}", key=f"p.s{i}", pid=100 + i) for i in range(5)),
    )


def test_stale_proposals_no_longer_block_a_new_search(db):
    """The real case, 2026-09-28: five from the pre-fix engine filled
    max_pending, and the season's first search would have proposed nothing."""
    _five_stale(db)
    blocked = make(db, target(name="New", key="p.new", pid=200))
    assert blocked == []  # the old behaviour: no room
    made = make(db, target(name="New", key="p.new", pid=200), supersede=True)
    assert [p.add_name for p in made] == ["New"]
    assert [p.add_name for p in proposals.pending(db, "jimmy", "999.l.1")] == ["New"]


def test_a_withdrawn_proposal_cannot_be_approved(db):
    [old] = make(db, target(name="Old", key="p.old", pid=300))
    make(db, target(name="New", key="p.new", pid=301), supersede=True)
    with pytest.raises(ProposalError, match="withdrawn by a newer search"):
        proposals.decide(db, old.id, True)


def test_a_proposal_the_new_search_repeats_is_kept_not_duplicated(db):
    [kept] = make(db, target(name="Same", key="p.same", pid=400))
    made = make(db, target(name="Same", key="p.same", pid=400), supersede=True)
    assert made == []
    [still] = proposals.pending(db, "jimmy", "999.l.1")
    assert still.id == kept.id


def test_a_withdrawn_player_can_be_proposed_again_but_a_refused_one_cannot(db):
    [old] = make(db, target(name="Back", key="p.back", pid=500))
    make(db, target(name="Other", key="p.other", pid=501), supersede=True)  # withdraws Back
    again = make(db, target(name="Back", key="p.back", pid=500))
    assert [p.add_name for p in again] == ["Back"]

    [no] = make(db, target(name="No", key="p.no", pid=502))
    proposals.decide(db, no.id, False)
    assert make(db, target(name="No", key="p.no", pid=502), supersede=True) == []


def test_a_proposal_keeps_its_reasons_and_a_rerun_refreshes_them(db):
    """The numbers behind a pending add move every day; the card must not show
    Monday's reasons on Wednesday."""
    from dataclasses import replace

    [p] = make(db, replace(target(), detail={"week": ["Mon 5: Add Me fills an empty C"]}))
    assert p.reason["detail"] == {"week": ["Mon 5: Add Me fills an empty C"]}
    fresh = replace(target(), detail={"week": ["Wed 7: Add Me fills an empty C"]})
    assert make(db, fresh, supersede=True) == []
    [live] = proposals.pending(db, "jimmy")
    assert live.id == p.id
    assert live.reason["detail"] == {"week": ["Wed 7: Add Me fills an empty C"]}


def test_a_recheck_refreshes_what_pays_and_withdraws_the_rest_with_a_reason(db):
    from dataclasses import replace

    [a] = make(db, target(name="A", key="p.a", pid=9101))
    [b] = make(db, target(name="B", key="p.b", pid=9102))
    [c] = make(db, target(name="C", key="p.c", pid=9103))
    proposals.decide(db, c.id, True)  # decided meanwhile: left alone
    fresh = replace(target(name="A", key="p.a", pid=9101), detail={"week": ["Thu 1: A"]})
    proposals.refresh(db, {a.id: fresh}, {b.id: "B is no longer available", c.id: "x"})
    assert proposals.get(db, a.id).reason["detail"] == {"week": ["Thu 1: A"]}
    assert proposals.get(db, a.id).reason["week"] == 2
    assert [p.id for p in proposals.pending(db, "jimmy")] == [a.id]
    [gone] = proposals.withdrawn_since(db, "jimmy", "999.l.1", "2000-01-01")
    assert gone.id == b.id and gone.reason["withdrawn"] == "B is no longer available"
    assert proposals.get(db, c.id).status == APPROVED


def test_a_search_that_drops_a_proposal_says_so(db):
    [p] = make(db, target())
    make(db, target(name="Other", key="p.other", pid=9002), supersede=True)
    assert proposals.get(db, p.id).reason["withdrawn"] == "a newer search no longer proposes it"


def test_an_approved_move_can_be_cancelled_and_then_never_executed(db):
    [p] = make(db, target())
    proposals.decide(db, p.id, True)
    gone = proposals.cancel(db, p.id, "worse for next week (-0.32 categories)")
    assert gone.status == REJECTED
    assert gone.reason["cancelled"] == "worse for next week (-0.32 categories)"
    with pytest.raises(ProposalError, match="not approved"):
        proposals.take_for_execution(db, p.id)
    # Shown with its reason, like any withdrawal.
    [shown] = proposals.withdrawn_since(db, "jimmy", "999.l.1", "2000-01-01")
    assert shown.reason["withdrawn"].startswith("cancelled - worse for next week")


def test_a_made_move_cannot_be_cancelled(db):
    [p] = make(db, target())
    proposals.decide(db, p.id, True)
    proposals.mark_executed(db, p.id, "made")
    with pytest.raises(ProposalError, match="nothing to cancel"):
        proposals.cancel(db, p.id, "too late")
