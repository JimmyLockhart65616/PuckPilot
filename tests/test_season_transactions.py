"""Approved moves carried out: only with consent, once, and proven from the roster."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from puckpilot.season import notify
from puckpilot.season import proposals as proposals_mod
from puckpilot.season.run import RunReport
from puckpilot.season.transactions import carry_out

MANAGER = SimpleNamespace(name="m")
TODAY = "2026-10-06"


def _approved(db, pid_add="a1", pid_drop="d1", to_execute=True, **reason):
    reason = {"add_name": "Pickup", "drop_name": "Depth", "approved_to_execute": to_execute,
              **reason}  # fmt: skip
    cur = db.execute(
        "INSERT INTO waiver_proposals (created_at, add_pid, drop_pid, reason_json, status, manager,"
        " league_key, team_key, kind, add_player_key, drop_player_key)"
        " VALUES ('2026-10-06 11:00:00', 1, 2, ?, 'approved', 'm', 'L', 'T', 'add_drop', ?, ?)",
        (json.dumps(reason), pid_add, pid_drop),
    )
    db.commit()
    return cur.lastrowid


def _roster(*players):
    return SimpleNamespace(players=[SimpleNamespace(player_key=k, team=t) for k, t in players])


BEFORE, AFTER = _roster(("d1", "TOR")), _roster(("a1", "MTL"))


class _Make:
    def __init__(self, *results):
        self.results = list(results)
        self.calls = []

    def __call__(self, manager, team_key, p):
        self.calls.append(p.id)
        return self.results.pop(0)


def _result(ok=True, submitted=True, message="submitted", lines=("made",),
            final=False):  # fmt: skip
    return SimpleNamespace(ok=ok, submitted=submitted, message=message, lines=list(lines),
                           final=final)  # fmt: skip


def _run(db, rosters, make, **kw):
    """One run's pass over the queue; `rosters` are the reads, in order."""
    reads = iter(rosters)
    report = RunReport(date=TODAY, manager="m")
    kw.setdefault("today", TODAY)
    made = carry_out(db, MANAGER, "L", "T", report, lambda: next(reads), make=make, **kw)
    return made, report


@pytest.fixture
def sent(monkeypatch):
    out = []
    monkeypatch.setattr(notify, "made", lambda what, url="": out.append(("made", what)) or True)
    monkeypatch.setattr(
        notify, "failed", lambda what, why, url="": out.append(("failed", why)) or True
    )
    return out


def test_a_move_approved_to_be_made_is_made_and_proven(db, sent):
    pid = _approved(db)
    make = _Make(_result())
    made, report = _run(db, [BEFORE, AFTER], make)
    assert made == 1 and make.calls == [pid]
    assert proposals_mod.get(db, pid).status == proposals_mod.EXECUTED
    assert sent == [("made", f"#{pid} add Pickup, drop Depth")]
    row = db.execute("SELECT outcome FROM season_actions WHERE kind = 'transaction'").fetchone()
    assert row[0] == "executed"


def test_an_approval_given_on_make_it_yourself_is_asked_again_not_made(db, sent, monkeypatch):
    """The card said PuckPilot never adds or drops. A yes to that is not a yes
    to PuckPilot doing it - and a manager with an executor does not make
    pickups by hand either (2026-10-06), so it goes back on the page."""
    asked = []
    monkeypatch.setattr(notify, "ask_again", lambda what, url="": asked.append(what) or True)
    pid = _approved(db, to_execute=False, tapped_at="2026-10-06T12:00:00+00:00")
    make = _Make()
    made, report = _run(db, [], make)
    assert made == 0 and make.calls == []
    p = proposals_mod.get(db, pid)
    assert p.status == proposals_mod.PENDING and p.decided_at == "" and not p.superseded_at
    assert "approved_to_execute" not in p.reason and "tapped_at" not in p.reason
    assert "approve again" in report.steps[0].detail
    assert asked == [f"#{pid} add Pickup, drop Depth"]
    row = db.execute(
        "SELECT outcome, message FROM season_actions WHERE kind = 'transaction'"
    ).fetchone()
    assert row[0] == "skipped" and "asked again" in row[1]
    _, again = _run(db, [], make)
    assert again.steps == []  # pending now: nothing for the executor until a fresh tap


def test_without_an_executor_a_make_it_yourself_approval_is_the_person_s(db, sent, monkeypatch):
    """No executor, no one else to make it: as the card said, said once."""
    monkeypatch.setattr("puckpilot.season.transactions.executor", lambda: None)
    pid = _approved(db, to_execute=False)
    report = RunReport(date=TODAY, manager="m")
    assert carry_out(db, MANAGER, "L", "T", report, lambda: BEFORE) == 0
    assert "leaves it to you" in report.steps[0].detail
    again = RunReport(date=TODAY, manager="m")
    carry_out(db, MANAGER, "L", "T", again, lambda: BEFORE)
    assert again.steps == []
    assert proposals_mod.get(db, pid).status == proposals_mod.APPROVED


def test_reopen_refuses_what_was_not_approved_or_was_submitted(db):
    pending = _pending(db, "a9")
    with pytest.raises(proposals_mod.ProposalError, match="not approved"):
        proposals_mod.reopen(db, pending, "why")
    sent_off = _approved(db, pid_add="a2", submitted_at="2026-10-06T12:00:00+00:00")
    with pytest.raises(proposals_mod.ProposalError, match="already submitted"):
        proposals_mod.reopen(db, sent_off, "why")


def test_a_move_already_made_by_hand_is_recorded_not_made_again(db, sent):
    pid = _approved(db)
    make = _Make()
    made, report = _run(db, [AFTER], make)
    assert made == 1 and make.calls == [] and sent == []
    assert proposals_mod.get(db, pid).status == proposals_mod.EXECUTED
    assert "by hand" in report.steps[0].detail


def test_a_drop_who_plays_tonight_is_dropped_before_his_game(db, sent):
    """The move was priced from today's first unplayed game - his game today
    is part of its price, so waiting would make a different move."""
    pid = _approved(db)
    make = _Make(_result())
    made, _ = _run(db, [BEFORE, AFTER], make, playing_today={"TOR"})
    assert made == 1 and make.calls == [pid]


def test_a_drop_whose_game_has_begun_waits(db, sent):
    """Yahoo will not drop him until tomorrow (Error #174)."""
    _approved(db)
    make = _Make()
    _, report = _run(db, [BEFORE], make, playing_today={"TOR"}, started_today={"TOR"})
    assert make.calls == [] and "has begun" in report.steps[0].detail


def test_a_move_queued_for_after_tonight_waits_until_then(db, sent):
    """The late-week check queues a move whose value is next week's alone."""
    pid = _approved(db, after_games_of=TODAY)
    make = _Make(_result())
    _, report = _run(db, [BEFORE], make, playing_today={"TOR"})
    assert make.calls == [] and "after tonight's games" in report.steps[0].detail
    made, _ = _run(db, [BEFORE, AFTER], make, today="2026-10-07")
    assert made == 1 and make.calls == [pid]


def test_a_move_submitted_but_not_seen_is_never_submitted_twice(db, sent):
    pid = _approved(db)
    make = _Make(_result())
    _, report = _run(db, [BEFORE, BEFORE], make)
    assert not report.steps[0].ok and "roster does not show it" in report.steps[0].detail
    assert sent[0][0] == "failed" and "will not be tried again" in sent[0][1]
    _, again = _run(db, [BEFORE], make)
    assert make.calls == [pid]  # not submitted a second time
    assert "will not be tried again" in again.steps[0].detail
    # Yahoo shows it late: done after all, and the phone hears so.
    made, late = _run(db, [AFTER], make)
    assert made == 1 and proposals_mod.get(db, pid).status == proposals_mod.EXECUTED
    assert "seen on the roster now" in late.steps[0].detail and sent[-1][0] == "made"


def test_a_refusal_after_submitting_keeps_yahoos_words(db, sent):
    _approved(db)
    make = _Make(_result(ok=False, message="Yahoo refused: roster locked"))
    _, report = _run(db, [BEFORE, BEFORE], make)
    assert "Yahoo refused: roster locked" in report.steps[0].detail
    assert "will not be tried again" in report.steps[0].detail


def test_a_passing_failure_is_retried_and_told_once(db, sent):
    pid = _approved(db)
    refused = _result(ok=False, submitted=False, message="Yahoo did not answer", lines=())
    make = _Make(refused, refused)
    for _ in range(2):
        _run(db, [BEFORE], make)
    assert make.calls == [pid, pid]  # nothing was submitted, so trying again is safe
    assert [k for k, _ in sent] == ["failed"]  # the same reason is not told twice
    assert proposals_mod.get(db, pid).status == proposals_mod.APPROVED


def test_a_move_that_cannot_be_made_as_approved_is_cancelled(db, sent):
    """The add was taken: retrying on every run would only repeat the failure."""
    pid = _approved(db)
    taken = _result(ok=False, submitted=False, message="no add form - taken?", final=True)
    make = _Make(taken)
    _run(db, [BEFORE], make)
    p = proposals_mod.get(db, pid)
    assert p.status == proposals_mod.REJECTED and "taken" in p.reason["cancelled"]
    assert sent == [("failed", "no add form - taken? - cancelled; nothing more will be tried")]
    _run(db, [BEFORE], make)
    assert make.calls == [pid]


def test_a_drop_gone_from_the_roster_cancels_the_move(db, sent):
    """Adding without that drop would be a move nobody approved."""
    pid = _approved(db)
    make = _Make()
    _, report = _run(db, [_roster(("x9", "TOR"))], make)
    assert make.calls == [] and "no longer on your roster" in report.steps[0].detail
    assert proposals_mod.get(db, pid).status == proposals_mod.REJECTED


def test_an_add_made_another_way_cancels_the_move(db, sent):
    """Added by hand, but the approved drop is still on the roster: making the
    move now would only fail at the add, so say what happened instead."""
    pid = _approved(db)
    make = _Make()
    _, report = _run(db, [_roster(("a1", "MTL"), ("d1", "TOR"))], make)
    assert make.calls == [] and "already on your roster" in report.steps[0].detail
    assert proposals_mod.get(db, pid).status == proposals_mod.REJECTED


def _now(ownership="freeagents", status=""):
    from puckpilot.season.pool import PoolPlayer

    return PoolPlayer(
        player_key="a1", name="Pickup", team="MTL", primary_position="C",
        yahoo_eligible=frozenset({"C"}), status=status, ownership_type=ownership,
    )  # fmt: skip


@pytest.mark.parametrize(
    ("now", "why"),
    [
        (_now(ownership="team"), "no longer available"),
        (_now(ownership="waivers"), "on waivers now"),
        (_now(status="O"), "now listed O"),
    ],
)
def test_an_acquisition_is_not_spent_on_a_different_situation(db, sent, now, why):
    """Approved for a healthy free agent; since then he was taken, put on
    waivers, or listed out. That yes was not given to this."""
    pid = _approved(db)
    make = _Make()
    _, report = _run(db, [BEFORE], make, read_adds=lambda keys: {"a1": now})
    assert make.calls == [] and why in report.steps[0].detail
    assert proposals_mod.get(db, pid).status == proposals_mod.REJECTED


def test_an_add_that_cannot_be_read_is_tried_again_not_cancelled(db, sent):
    pid = _approved(db)
    make = _Make()
    _, report = _run(db, [BEFORE], make, read_adds=lambda keys: {})
    assert make.calls == [] and "could not be read" in report.steps[0].detail
    assert proposals_mod.get(db, pid).status == proposals_mod.APPROVED


def test_a_healthy_free_agent_is_added(db, sent):
    pid = _approved(db)
    make = _Make(_result())
    made, _ = _run(db, [BEFORE, AFTER], make, read_adds=lambda keys: {"a1": _now(status="DTD")})
    assert made == 1 and make.calls == [pid]  # day-to-day is not out


def test_a_drop_who_became_a_protected_keeper_is_kept(db, sent):
    """Dropping him gives up his keeper rights, and that cannot be undone."""
    pid = _approved(db)
    make = _Make()
    _, report = _run(db, [BEFORE], make, protected={"d1"})
    assert make.calls == [] and "keepers protected" in report.steps[0].detail
    assert proposals_mod.get(db, pid).status == proposals_mod.REJECTED


def test_protection_comes_from_the_latest_weekly_ranking(db):
    from puckpilot.season import keeper_value
    from puckpilot.season.run import _protected_keepers

    def rank(key, protected):
        return keeper_value.KeeperRank(key, None, key, 1.0, 1.0, 0, True, 1, protected)

    assert _protected_keepers(db, "m") == set()
    keeper_value.save(db, "m", "2026-09-28", "x", [rank("d1", True), rank("d2", True)])
    keeper_value.save(db, "m", "2026-10-05", "x", [rank("d1", True), rank("d2", False)])
    assert _protected_keepers(db, "m") == {"d1"}
    assert _protected_keepers(db, "other") == set()


def test_a_read_that_fails_is_told_once_and_tried_again(db, sent):
    """2026-10-07 19:10: a dropped read crashed the step, and the approved add
    was simply not there - nobody was told."""
    pid = _approved(db)
    make = _Make(_result())

    def broken():
        raise RuntimeError("Page.evaluate: TypeError: Failed to fetch")

    for _ in range(2):
        report = RunReport(date=TODAY, manager="m")
        assert carry_out(db, MANAGER, "L", "T", report, broken, make=make, today=TODAY) == 0
        assert not report.steps[0].ok and "could not be read" in report.steps[0].detail
    assert make.calls == [] and [k for k, _ in sent] == ["failed"]  # told once
    assert proposals_mod.get(db, pid).status == proposals_mod.APPROVED
    made, _ = _run(db, [BEFORE, AFTER], make)
    assert made == 1 and proposals_mod.get(db, pid).reason["result"].endswith("shows it")


def test_a_roster_lost_after_submitting_is_checked_next_run(db, sent):
    pid = _approved(db)
    make = _Make(_result())
    reads = iter([BEFORE])

    def then_broken():
        try:
            return next(reads)
        except StopIteration:
            raise RuntimeError("Failed to fetch") from None

    report = RunReport(date=TODAY, manager="m")
    carry_out(db, MANAGER, "L", "T", report, then_broken, make=make, today=TODAY)
    assert "could not be read to check it" in report.steps[0].detail
    assert proposals_mod.get(db, pid).reason.get("submitted_at")  # never submitted again
    made, late = _run(db, [AFTER], make)
    assert made == 1 and make.calls == [pid] and sent[-1][0] == "made"


def test_a_move_without_a_drop_reads_as_an_add():
    from puckpilot.season.run import _swap

    assert _swap(SimpleNamespace(add_name="A", drop_name="B", drop_player_key="d")) == "A for B"
    assert _swap(SimpleNamespace(add_name="A", drop_name="", drop_player_key="")) == "A added"


def test_without_an_executor_the_moves_are_left_to_the_person(db, sent, monkeypatch):
    _approved(db)
    monkeypatch.setattr("puckpilot.season.transactions.executor", lambda: None)
    report = RunReport(date=TODAY, manager="m")
    assert carry_out(db, MANAGER, "L", "T", report, lambda: BEFORE) == 0
    assert "no executor installed" in report.steps[0].detail


# -- the consent record, from the card to the queue ---------------------------------


def _pending(db, key="a1", drop=None, **reason):
    cur = db.execute(
        "INSERT INTO waiver_proposals (created_at, add_pid, reason_json, status, manager,"
        " league_key, team_key, kind, add_player_key, drop_player_key) VALUES"
        " ('2026-10-06 11:00:00', 1, ?, 'pending', 'm', 'L', 'T', 'add', ?, ?)",
        (json.dumps({"add_name": "Pickup", **reason}), key, drop),
    )
    db.commit()
    return cur.lastrowid


def test_each_card_says_what_approve_does(db):
    from puckpilot.season.snapshot import MANUAL, build

    _pending(db, "a1", "d1")
    _pending(db, "a2")
    _pending(db, "a3", "d3", timing="on waivers - put a claim in")
    _pending(db, "a4", "d4", after_games_of=TODAY)
    off = build(db, "m", "L", "Team")["proposals"]
    assert {(c["executes"], c["approve_means"]) for c in off} == {(False, MANUAL)}
    on = build(db, "m", "L", "Team", executes_moves=True)["proposals"]
    assert [c["executes"] for c in on] == [True, True, False, True]
    swap, add, claim, queued = (c["approve_means"] for c in on)
    assert "makes this add and drop in Yahoo on its next run" in swap
    assert "makes this add in Yahoo" in add
    assert "never claims" in claim
    assert "after tonight's games" in queued


def test_a_decision_carries_what_the_card_said(db):
    from puckpilot.season.snapshot import apply_decisions

    first_id, second_id = _pending(db, "a1"), _pending(db, "a2")
    lines = apply_decisions(
        db,
        [
            {"seq": 1, "kind": "proposal", "id": first_id, "approve": True, "executes": True,
             "at": 1_780_000_000.0},
            {"seq": 2, "kind": "proposal", "id": second_id, "approve": True},  # an older relay
        ],
    )  # fmt: skip
    first, second = (proposals_mod.get(db, i).reason for i in (first_id, second_id))
    assert first["approved_to_execute"] is True and first["tapped_at"].startswith("2026")
    assert second["approved_to_execute"] is False
    assert "to be made in Yahoo" in lines[0] and "to be made" not in lines[1]


def test_the_relay_keeps_the_flag_and_never_guesses_it():
    from puckpilot.web.season_relay import SeasonState

    state = SeasonState()
    assert state.decide("m", "proposal", 7, True, executes=True)["executes"] is True
    assert state.decide("m", "proposal", 8, True)["executes"] is False


def test_the_authority_says_who_makes_the_move():
    from puckpilot.season.authority import Authority

    on = Authority.from_config({"transactions": {"execute_approved": True}})
    assert on.transactions.execute_approved
    assert any("made in Yahoo on the next run" in x for x in on.transactions.describe())
    off = Authority.from_config({})
    assert any("the manager makes it" in x for x in off.transactions.describe())
