"""What the phone gets, and what comes back from it."""

from __future__ import annotations

from puckpilot.season import proposals as proposals_mod
from puckpilot.season import protocol as protocol_mod
from puckpilot.season import snapshot
from puckpilot.season.roster import RosterPlayer, TeamRoster
from puckpilot.season.today import LineupPlan, Move
from tests.test_season_proposals import make, target
from tests.test_season_protocol import derive, outlook


def player(name="A Player", slot="BN", team="TOR", status=""):
    return RosterPlayer(
        player_key=f"p.{name}",
        yahoo_id="1",
        name=name,
        team=team,
        primary_position="C",
        yahoo_eligible=frozenset({"C"}),
        selected_slot=slot,
        nhl_player_id=1,
        status=status,
        status_full="Out" if status else "",
    )


def plan(moves=(), out=(), lock="2026-10-07T23:00:00Z"):
    return LineupPlan(
        date="2026-10-07",
        team_key="999.l.1.t.5",
        manager="jimmy",
        moves=tuple(moves),
        out=tuple(out),
        lock_utc=lock,
        lock_team="TOR",
    )


def test_a_snapshot_with_nothing_in_it_still_has_every_key(db):
    s = snapshot.build(db, "jimmy", "999.l.1", "Home Team")
    for key in ("team", "moves", "proposals", "protocol", "week", "roster", "lock_local"):
        assert key in s


def test_moves_carry_the_kind_the_page_colours_by(db):
    p = plan(
        moves=[
            Move(player=player("Started"), to_slot="LW", from_slot="BN"),
            Move(player=player("Benched", slot="LW"), to_slot="BN", from_slot="LW"),
            Move(player=player("Shifted", slot="C"), to_slot="Util", from_slot="C"),
        ]
    )
    s = snapshot.build(db, "jimmy", "999.l.1", "Home Team", plan=p)
    assert [m["kind"] for m in s["moves"]] == ["start", "bench", "move"]
    assert s["moves"][0]["detail"] == "into LW"
    assert s["moves"][2]["detail"] == "C to Util"


def test_the_deadline_is_localised_here_not_in_the_browser(db):
    """The browser knows its own timezone but not the league's."""
    s = snapshot.build(db, "jimmy", "999.l.1", "T", plan=plan())
    assert s["lock_local"] == "7:00 PM"


def test_pending_proposals_appear_with_their_reasoning(db):
    make(db, target(name="Shane Pinto"))
    s = snapshot.build(db, "jimmy", "999.l.1", "T")
    assert len(s["proposals"]) == 1
    assert s["proposals"][0]["add"] == "Shane Pinto"
    # The net effect, in real units, not a score nobody can interpret.
    assert "+2 starts this week" in s["proposals"][0]["why"]
    assert "SOG +4.4" in s["proposals"][0]["why"]
    assert "race" in s["proposals"][0]["timing"]


def test_a_decided_proposal_leaves_the_page(db):
    [p] = make(db, target())
    proposals_mod.decide(db, p.id, True)
    assert snapshot.build(db, "jimmy", "999.l.1", "T")["proposals"] == []


def test_the_roster_carries_a_readable_status(db):
    r = TeamRoster(
        league_key="999.l.1",
        team_key="999.l.1.t.5",
        date="2026-10-07",
        players=(player("Healthy"), player("Hurt", status="O")),
    )
    rows = snapshot.build(db, "jimmy", "999.l.1", "T", roster=r)["roster"]
    assert [x["status"] for x in rows] == ["", "Out"]


# -- decisions coming back --------------------------------------------------


def test_a_decision_from_the_page_is_recorded(db):
    [p] = make(db, target())
    lines = snapshot.apply_decisions(
        db, [{"seq": 1, "kind": "proposal", "id": p.id, "approve": True}]
    )
    assert "approved" in lines[0]
    assert proposals_mod.get(db, p.id).status == "approved"


def test_a_protocol_decision_from_the_page_is_recorded(db):
    p = protocol_mod.save(db, derive(outlook("PPP", 8.5, 7.7, add_room=5.0)))
    snapshot.apply_decisions(db, [{"seq": 1, "kind": "protocol", "id": p.id, "approve": True}])
    assert protocol_mod.load_by_id(db, p.id).status == "approved"


def test_one_stale_tap_does_not_stop_the_others(db):
    """These arrive in batches from a public endpoint."""
    [a] = make(db, target(name="A", key="p.a", pid=9101))
    [b] = make(db, target(name="B", key="p.b", pid=9102))
    proposals_mod.decide(db, a.id, True)  # already decided
    lines = snapshot.apply_decisions(
        db,
        [
            {"seq": 1, "kind": "proposal", "id": a.id, "approve": False},
            {"seq": 2, "kind": "proposal", "id": 9999, "approve": True},
            {"seq": 3, "kind": "nonsense", "id": 1, "approve": True},
            {"seq": 4, "kind": "proposal", "id": b.id, "approve": True},
        ],
    )
    assert len(lines) == 4
    assert any("could not apply" in x for x in lines)
    assert any("unknown kind" in x for x in lines)
    assert proposals_mod.get(db, b.id).status == "approved"


def test_decisions_are_applied_in_the_order_they_were_made(db):
    [a] = make(db, target())
    snapshot.apply_decisions(
        db,
        [
            {"seq": 2, "kind": "proposal", "id": a.id, "approve": False},
            {"seq": 1, "kind": "proposal", "id": a.id, "approve": True},
        ],
    )
    # seq 1 wins and seq 2 finds it already decided
    assert proposals_mod.get(db, a.id).status == "approved"
