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


def test_roster_moves_come_first_with_their_own_kinds(db):
    """Unkinded, an IR move rendered as "START Sanderson in IR+"."""
    p = LineupPlan(
        date="2026-10-07",
        team_key="999.l.1.t.5",
        manager="jimmy",
        moves=(Move(player=player("Started"), to_slot="D", from_slot="BN"),),
        ir_moves=(
            Move(player=player("Hurt", slot="D", status="O"), to_slot="IR+", from_slot="D"),
            Move(player=player("Back", slot="IR+"), to_slot="BN", from_slot="IR+"),
        ),
        ir_alerts=("Someone is stuck in IR",),
    )
    s = snapshot.build(db, "jimmy", "999.l.1", "Home Team", plan=p)
    assert [m["kind"] for m in s["moves"]] == ["ir", "activate", "start"]
    assert "frees a roster spot" in s["moves"][0]["detail"]
    assert s["alerts"] == ["Someone is stuck in IR"]


def test_the_cold_payload_carries_every_key_a_snapshot_does(db):
    """The contract: a key a snapshot has and a cold relay lacks renders as
    "undefined" on a fresh URL."""
    from puckpilot.web.season_relay import SeasonState

    cold = SeasonState().get("jimmy")
    for key in snapshot.build(db, "jimmy", "999.l.1", "Home Team"):
        if key in ("team", "date", "out", "lock_local", "playing", "rostered"):
            continue  # optional on the page by design
        assert key in cold, key


def test_the_week_card_carries_the_score_so_far_and_games_left(db):
    from puckpilot.engine.categories import resolve
    from puckpilot.season.week import CategoryOutlook, WeekPlan

    wp = WeekPlan(
        week=1,
        start="2026-09-29",
        end="2026-10-04",
        opponent="Visitors",
        outlook=(
            CategoryOutlook(resolve("G"), 9.2, 10.4, sd=3.0, banked_ours=4.0, banked_theirs=6.0),
            CategoryOutlook(
                resolve("SV%"), 0.912, 0.905, sd=0.02, banked_ours=0.915, banked_theirs=0.901
            ),
        ),
        our_games=23,
        their_games=27,
        status="midevent",
        days_left=3,
        banked=True,
    )
    s = snapshot.build(db, "jimmy", "999.l.1", "Home Team", week_plan=wp)
    w = s["week"]
    assert w["games_left"] == {"ours": 23, "theirs": 27} and w["days_left"] == 3
    g, sv = w["cats"]
    assert (g["now_ours"], g["now_theirs"], g["state"]) == (4.0, 6.0, "in play")
    assert sv["now_ours"] == 0.915


def test_the_protocol_card_is_on_every_run_not_just_mondays(db):
    """A push replaces the whole page; an Approve button that vanished on the
    next run could not be pressed."""
    p = protocol_mod.save(db, derive(outlook("PPP", 8.5, 7.7, add_room=5.0), week=2))
    s = snapshot.build(db, "jimmy", "999.l.1", "Home Team", week_no=2)
    assert s["protocol"]["id"] == p.id and s["week"] is None


def test_the_week_card_shows_calibrated_odds_when_there_are_some(db):
    from puckpilot.engine.categories import resolve
    from puckpilot.season.odds import CategoryOdds, WeekOdds
    from puckpilot.season.week import CategoryOutlook, WeekPlan

    g = resolve("G")
    wp = WeekPlan(
        week=1,
        start="a",
        end="b",
        opponent="X",
        outlook=(CategoryOutlook(g, 9.0, 8.0, sd=2.0, p_win=0.58, p_tie=0.08),),
        odds=WeekOdds((CategoryOdds(g, 0.58, 0.08, 9.0, 8.0),)),
    )
    w = snapshot.build(db, "jimmy", "999.l.1", "Home Team", week_plan=wp)["week"]
    assert w["expected"] == 0.6 and w["of"] == 1
    assert w["cats"][0]["chance"] == 62


def test_the_snapshot_says_when_the_next_update_is(db):
    from datetime import datetime
    from zoneinfo import ZoneInfo

    nxt = datetime(2026, 9, 29, 11, 0, tzinfo=ZoneInfo("America/Toronto"))
    s = snapshot.build(db, "jimmy", "999.l.1", "Home Team", next_run=nxt)
    assert s["next_run_utc"] == "2026-09-29T15:00:00+00:00"
    assert s["next_local"] == "Tue 11:00 AM"


def test_a_proposals_reasons_reach_the_page_as_titled_sections(db):
    from dataclasses import replace

    detail = {
        "season": ["Add Me would rank 9 of 16 on your roster"],
        "week": ["Mon 5: Add Me fills an empty C"],
    }
    make(db, replace(target(), detail=detail))
    [p] = snapshot.build(db, "jimmy", "999.l.1", "T")["proposals"]
    # Reading order is fixed here, whatever order they were stored in.
    assert [s["title"] for s in p["detail"]] == ["This week, day by day", "Rest of season"]
    assert p["detail"][0]["lines"] == ["Mon 5: Add Me fills an empty C"]


def test_a_proposal_made_before_reasons_were_kept_has_none(db):
    make(db, target())
    [p] = snapshot.build(db, "jimmy", "999.l.1", "T")["proposals"]
    assert p["detail"] == []


def test_the_page_says_whether_tonights_changes_were_made(db):
    from datetime import UTC, datetime

    at = datetime(2026, 10, 10, 15, 2, tzinfo=UTC)
    s = snapshot.build(
        db, "jimmy", "999.l.1", "T", acted={"ok": True, "message": "made 2 change(s)", "at": at}
    )
    assert s["acted"] == {"ok": True, "text": "Made in Yahoo at 11:02 AM - made 2 change(s)"}
    s = snapshot.build(db, "jimmy", "999.l.1", "T", acted={"ok": False, "message": "locked"})
    assert s["acted"] == {"ok": False, "text": "NOT made - locked"}
    assert snapshot.build(db, "jimmy", "999.l.1", "T")["acted"] is None
