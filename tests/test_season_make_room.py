"""The make-room move, end to end: queued, decided, made, and never made for nothing.

A make-room move (`MoveKind.ACTIVATE`) drops one player so another can come off
an IR slot he no longer qualifies for. It is the precondition for every add, so
it is proposed whatever the add bar says, never counted as an acquisition, and
carried out first - but only while the roster still needs it.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from puckpilot.season import notify
from puckpilot.season import proposals as proposals_mod
from puckpilot.season.executors import FunctionExecutor
from puckpilot.season.legality import Fix
from puckpilot.season.moves import MoveKind
from puckpilot.season.roster import RosterPlayer, TeamRoster
from puckpilot.season.run import RunReport, committed_moves
from puckpilot.season.transactions import carry_out

MANAGER = SimpleNamespace(name="m")
TODAY = "2026-10-26"


def rp(key, slot, eligible=("D", "Util"), pos="D", team="OTT"):
    return RosterPlayer(
        player_key=key,
        yahoo_id=key,
        name=key.title(),
        team=team,
        primary_position=pos,
        yahoo_eligible=frozenset(eligible),
        selected_slot=slot,
        nhl_player_id=abs(hash(key)) % 10_000,
    )


def roster(*players):
    return TeamRoster(league_key="L", team_key="T", date=TODAY, players=players)


BACK = rp("back", "IR+")  # cleared to play; IR+ not in his eligibility any more
DROP = rp("streamer", "BN", eligible=("C", "Util"), pos="C", team="MIN")
ILLEGAL = roster(BACK, DROP)
DONE = roster(rp("back", "IR+"))  # the drop made; the lineup brings him back next


def _fix(drop=DROP):
    return Fix(BACK, drop, "IR+", "the least valuable legal drop")


def _approve(db, pid):
    proposals_mod.decide(db, pid, True, to_execute=True)


@pytest.fixture
def sent(monkeypatch):
    out = []
    monkeypatch.setattr(notify, "made", lambda what, url="": out.append(("made", what)) or True)
    monkeypatch.setattr(
        notify, "failed", lambda what, why, url="": out.append(("failed", why)) or True
    )
    return out


def _run(db, reads, make, **kw):
    it = iter(reads)
    report = RunReport(date=TODAY, manager="m")
    made = carry_out(db, MANAGER, "L", "T", report, lambda: next(it), make=make, today=TODAY,
                     **kw)  # fmt: skip
    return made, report


class _Make:
    def __init__(self, *results):
        self.results, self.calls = list(results), []

    def __call__(self, manager, team_key, p):
        self.calls.append((p.id, p.kind))
        return self.results.pop(0)


def _ok():
    return SimpleNamespace(ok=True, submitted=True, message="submitted", lines=["x"], final=False)


# -- the queue --------------------------------------------------------------


def test_a_make_room_move_is_queued_once_and_refreshed(db):
    p, new = proposals_mod.propose_fix(db, "m", "L", "T", _fix(), week=5)
    assert new and p.kind == MoveKind.ACTIVATE.value and p.drop_player_key == "streamer"
    again, new_again = proposals_mod.propose_fix(db, "m", "L", "T", _fix(), week=5)
    assert again.id == p.id and not new_again


def test_a_cheaper_drop_replaces_a_pending_one_but_never_an_approved_one(db):
    p, _ = proposals_mod.propose_fix(db, "m", "L", "T", _fix(), week=5)
    other = rp("other", "BN", eligible=("C", "Util"), pos="C")
    q, new = proposals_mod.propose_fix(db, "m", "L", "T", _fix(other), week=5)
    assert new and q.drop_player_key == "other"
    assert proposals_mod.get(db, p.id).superseded_at  # withdrawn, with the reason
    _approve(db, q.id)
    r, new = proposals_mod.propose_fix(db, "m", "L", "T", _fix(), week=5)
    assert not new and r.id == q.id  # decided: kept


def test_a_make_room_move_no_longer_needed_is_withdrawn(db):
    p, _ = proposals_mod.propose_fix(db, "m", "L", "T", _fix(), week=5)
    assert proposals_mod.retire_fixes(db, "m", "L", still_stuck={"back"}) == []
    gone = proposals_mod.retire_fixes(db, "m", "L", still_stuck=set())
    assert [g.id for g in gone] == [p.id]


def test_the_add_search_neither_withdraws_it_nor_counts_it(db):
    """Supersede and `max_pending` are the add search's own queue."""
    fix, _ = proposals_mod.propose_fix(db, "m", "L", "T", _fix(), week=5)
    target = SimpleNamespace(
        player=SimpleNamespace(player_key="fa1", nhl_player_id=11, name="Fa", team="TOR"),
        drop=None, score=0.4, gain=0.4, starts=3.0, drop_starts=0.0, extra_starts=3.0,
        helps=(), timing="", moved=lambda: "", detail={},
    )  # fmt: skip
    made = proposals_mod.propose(db, "m", "L", "T", [target], week=5, max_pending=1,
                                 supersede=True)  # fmt: skip
    assert len(made) == 1
    assert proposals_mod.get(db, fix.id).is_live


def test_an_unknown_kind_is_refused():
    with pytest.raises(ValueError):
        MoveKind("trade")


# -- carrying it out --------------------------------------------------------


def test_an_approved_make_room_move_is_made_first_and_proven(db, sent):
    swap = db.execute(
        "INSERT INTO waiver_proposals (add_pid, drop_pid, reason_json, status, manager, league_key,"
        " team_key, kind, add_player_key, drop_player_key) VALUES (1, NULL, ?, 'approved', 'm',"
        " 'L', 'T', 'add_drop', 'fa1', '')",
        (json.dumps({"add_name": "Fa", "approved_to_execute": True}),),
    ).lastrowid
    db.commit()
    room, _ = proposals_mod.propose_fix(db, "m", "L", "T", _fix(), week=5)
    _approve(db, room.id)
    make = _Make(_ok())
    made, report = _run(db, [ILLEGAL, DONE], make, read_adds=lambda keys: {})
    assert make.calls == [(room.id, "activate")]  # first; the add waits
    assert made == 1 and proposals_mod.get(db, room.id).status == proposals_mod.EXECUTED
    assert proposals_mod.get(db, swap).status == proposals_mod.APPROVED
    assert any("waits - Back must come off IR+" in s.detail for s in report.steps)
    assert ("made", f"#{room.id} drop Streamer so Back comes back") in sent


def test_a_roster_that_became_legal_loses_nobody(db, sent):
    room, _ = proposals_mod.propose_fix(db, "m", "L", "T", _fix(), week=5)
    _approve(db, room.id)
    back_on_ir = rp("back", "IR+", eligible=("D", "IR+", "Util"))  # injured again
    make = _Make()
    made, _ = _run(db, [roster(back_on_ir, DROP)], make)
    assert made == 0 and make.calls == []
    cancelled = proposals_mod.get(db, room.id)
    assert cancelled.status == proposals_mod.REJECTED
    assert "no longer needs room" in cancelled.reason["cancelled"]


def test_an_executor_that_cannot_make_the_kind_says_so_and_keeps_it(db, sent):
    room, _ = proposals_mod.propose_fix(db, "m", "L", "T", _fix(), week=5)
    _approve(db, room.id)
    adds_only = FunctionExecutor(_Make(), frozenset({MoveKind.ADD_DROP.value}))
    made, report = _run(db, [ILLEGAL], adds_only)
    assert made == 0
    assert proposals_mod.get(db, room.id).status == proposals_mod.APPROVED
    assert any("cannot make a activate move" in why for _, why in sent)


def test_its_drop_is_held_until_it_is_made(db):
    room, _ = proposals_mod.propose_fix(db, "m", "L", "T", _fix(), week=5)
    _approve(db, room.id)
    week = SimpleNamespace(number=5)
    assert [p.id for p in committed_moves(db, "m", "L", ILLEGAL, week)] == [room.id]
    assert committed_moves(db, "m", "L", DONE, week) == []


# -- the card and the push --------------------------------------------------


def test_the_card_says_make_room_and_what_approve_does(db):
    from puckpilot.season.snapshot import MANUAL, build

    proposals_mod.propose_fix(db, "m", "L", "T", _fix(), week=5)
    card = build(db, "m", "L", "Team", executes_moves=True, today=TODAY)["proposals"][0]
    assert card["kind"] == "activate"
    assert card["head"] == "Make room: drop Streamer so Back comes off IR+"
    assert card["executes"] and "drops Streamer" in card["approve_means"]
    assert "refuses every add" in card["why"]
    manual = build(db, "m", "L", "Team", executes_moves=True, today=TODAY,
                   executable_kinds=frozenset({"add_drop"}))["proposals"][0]  # fmt: skip
    assert not manual["executes"] and manual["approve_means"] == MANUAL


def test_the_push_leads_with_it_at_high_priority(monkeypatch):
    got = {}
    monkeypatch.setattr(notify, "send", lambda title, body, **kw: got.update(t=title, b=body, **kw))
    room = SimpleNamespace(kind="activate", add_name="Back", drop_name="Streamer",
                           drop_player_key="streamer", reason={"slot": "IR+"})  # fmt: skip
    pick = SimpleNamespace(kind="add_drop", add_name="Fa", drop_name="", drop_player_key="",
                           reason={"expected_gain": 0.3, "helps": []})  # fmt: skip
    notify.new_pickups([room, pick], "https://page")
    assert got["priority"] == "high" and got["t"].startswith("Roster blocked")
    assert got["b"].splitlines()[0].startswith("Make room: drop Streamer so Back")


# -- the run step -----------------------------------------------------------


def test_an_illegal_roster_fails_the_run_and_proposes_the_fix(db, monkeypatch):
    """The 2026-10-19 case as one run sees it: FAIL, and a card to approve -
    never "ok" with a line underneath it for twelve days."""
    from puckpilot.season import run as runmod
    from puckpilot.season import week as weekmod
    from puckpilot.season.settings import Week
    from tests.test_season_legality import KEEP, RULES, SMALL

    monkeypatch.setattr(weekmod, "keep_values", lambda *a, **k: KEEP)
    players = (
        rp("centre", "C", eligible=("C", "Util"), pos="C"),
        rp("streamer", "BN", eligible=("C", "Util"), pos="C"),
        rp("defence", "D"),
        rp("goalie", "G", eligible=("G",), pos="G"),
        BACK,
    )
    from puckpilot.season.authority import Authority

    manager = SimpleNamespace(
        name="m", league=SimpleNamespace(draft_rules=lambda: RULES), authority=Authority()
    )
    ctx = runmod.WeekContext(week=Week(5, "2026-10-26", "2026-11-01"))
    report = RunReport(date=TODAY, manager="m")
    res = runmod._legality(db, manager, "L", SMALL, TODAY, roster(*players), ctx, (None, None),
                           report, propose=True)  # fmt: skip
    step = report.steps[-1]
    assert step.name == "legality" and not step.ok
    (card,) = proposals_mod.pending(db, "m", "L")
    assert card.kind == "activate" and card.drop_player_key == "streamer"
    assert any(line.startswith(f"proposed: #{card.id} Make room") for line in step.lines)
    # The week is priced on the roster as it will stand: legal, Back on the bench.
    assert res.transactions and {p.player_key for p in ctx.resolved.players} == {
        "centre", "defence", "goalie", "back"
    }  # fmt: skip
    # Once it is legal again, the next run says so and the card goes.
    legal = roster(*[q for q in players if q.player_key != "back"])
    report = RunReport(date=TODAY, manager="m")
    runmod._legality(db, manager, "L", SMALL, TODAY, legal, ctx, (None, None), report,
                     propose=True)  # fmt: skip
    assert proposals_mod.pending(db, "m", "L") == []
    assert report.steps[-1].ok


def test_an_add_into_a_spot_ir_freed_says_the_spot_is_on_loan(db, monkeypatch):
    """How 10-19 began: an add into the spot an injured player freed, with no
    word that his return would cost a player."""
    from puckpilot.season import run as runmod
    from puckpilot.season import week as weekmod
    from tests.test_season_legality import KEEP, SMALL

    monkeypatch.setattr(weekmod, "keep_values", lambda *a, **k: KEEP)
    hurt = rp("back", "IR+", eligible=("D", "IR+", "Util"))  # still injured: legally on IR
    ours = roster(rp("centre", "C", ("C", "Util"), "C"), rp("defence", "D"), hurt,
                  rp("streamer", "BN", ("C", "Util"), "C"))  # fmt: skip
    fill = SimpleNamespace(player=SimpleNamespace(player_key="fa1"), drop=None, detail={})
    swap = SimpleNamespace(player=SimpleNamespace(player_key="fa2"), drop=DROP, detail={})
    runmod._temporary_spots(db, SMALL, ours, None, TODAY, [fill, swap])
    (line,) = fill.detail["room"]
    assert "Back is on IR" in line and "today that would be Streamer" in line
    assert swap.detail == {}


def test_the_card_names_anyone_cheaper_the_roster_minimum_kept(db, monkeypatch):
    """On 10-31 the two goalies were the cheapest players on the roster, and the
    card listed them first without saying why neither was the drop."""
    from puckpilot.season import run as runmod
    from puckpilot.season import week as weekmod
    from tests.test_season_legality import RULES, SMALL

    keep = {"goalie": 5.0, "streamer": 40.0, "centre": 300.0, "defence": 250.0, "back": 280.0}
    monkeypatch.setattr(weekmod, "keep_values", lambda *a, **k: keep)
    goalie = rp("goalie", "G", eligible=("G",), pos="G")
    ours = roster(rp("centre", "C", ("C", "Util"), "C"), DROP, rp("defence", "D"), goalie, BACK)
    fix = Fix(BACK, DROP, "IR+", "the least valuable legal drop")
    lines = runmod._room_lines(fix, ours, None, TODAY, db, SMALL, RULES)
    assert any(line.startswith("Rest-of-season value of those who may go") and "Goalie" not in line
               for line in lines)  # fmt: skip
    assert any("Worth less but kept: Goalie" in line and "minimum at G" in line for line in lines)
