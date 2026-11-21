"""Blockers: what stops the tool working is said loudly - and only while it stands.

The 2026-10-19 block (an illegal roster) sat for twelve days as a line under an
"ok" step. Each kind here has one detector reading what the run already knows.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from puckpilot.season import blockers as b
from puckpilot.season.legality import Resolution, Violation

NOW = datetime(2026, 10, 31, 20, 0, tzinfo=UTC)
ILLEGAL = Resolution(
    (Violation("ir_ineligible", (), "Back is no longer eligible for IR+."),), fixes=()
)


def kinds(found):
    return [x.kind for x in found]


def test_an_illegal_roster_is_a_blocker_that_points_at_its_card():
    found, _ = b.detect(resolution=ILLEGAL, pending_fixes=[SimpleNamespace(id=17)], now=NOW)
    assert kinds(found) == [b.ROSTER_ILLEGAL]
    assert "Approve the make-room move (#17)" in found[0].message
    legal, _ = b.detect(resolution=Resolution((), ()), now=NOW)
    assert legal == []


def test_moves_meant_to_be_made_with_nothing_to_make_them():
    found, _ = b.detect(execute_approved=True, executor=None, now=NOW)
    assert kinds(found) == [b.NO_EXECUTOR]
    adds_only = SimpleNamespace(kinds=frozenset({"add_drop"}))
    found, _ = b.detect(
        execute_approved=True,
        executor=adds_only,
        approved=[SimpleNamespace(id=17, kind="activate"), SimpleNamespace(id=18, kind="add_drop")],
        now=NOW,
    )
    assert kinds(found) == [b.CANNOT_MAKE] and "#17" in found[0].message
    assert "#18" not in found[0].message
    off, _ = b.detect(execute_approved=False, executor=None, now=NOW)
    assert off == []


def test_one_dropped_read_is_retried_two_in_a_row_are_a_blocker():
    found, n = b.detect(read_failed=True, read_error="TimeoutError", previous_failures=0, now=NOW)
    assert found == [] and n == 1
    found, n = b.detect(read_failed=True, read_error="TimeoutError", previous_failures=1, now=NOW)
    assert kinds(found) == [b.READS_FAILING] and n == 2
    found, n = b.detect(read_failed=False, previous_failures=5, now=NOW)
    assert found == [] and n == 0


def test_a_signed_out_profile_is_said_at_once():
    found, _ = b.detect(
        read_failed=True,
        read_error="Yahoo returned 401 for x. The capture profile is not logged in",
        now=NOW,
    )
    assert kinds(found) == [b.LOGGED_OUT]


def test_a_gap_between_runs_is_said_by_the_run_that_ends_it():
    """10-15/16: 35 hours with no run, found by a person days later."""
    found, _ = b.detect(last_run_at=NOW - timedelta(hours=35), now=NOW)
    assert kinds(found) == [b.RUN_GAP] and "35 hours" in found[0].message
    on_time, _ = b.detect(last_run_at=NOW - timedelta(hours=14), now=NOW)
    assert on_time == []


def test_first_sighting_is_kept_and_a_cleared_blocker_forgotten(db):
    first = b.record(db, "m", [b.Blocker(b.ROSTER_ILLEGAL, "x")], now=NOW)
    later = b.record(db, "m", [b.Blocker(b.ROSTER_ILLEGAL, "x")], now=NOW + timedelta(days=2))
    assert first[0].since == later[0].since == NOW.isoformat(timespec="seconds")
    assert b.record(db, "m", [], now=NOW + timedelta(days=3)) == []
    again = b.record(db, "m", [b.Blocker(b.ROSTER_ILLEGAL, "x")], now=NOW + timedelta(days=4))
    assert again[0].since != first[0].since  # a new block, a new start


def test_each_blocker_is_pushed_once_a_day_while_it_stands(db):
    stood = b.record(db, "m", [b.Blocker(b.ROSTER_ILLEGAL, "x")], now=NOW)
    assert [x.kind for x in b.due(db, "m", stood, now=NOW)] == [b.ROSTER_ILLEGAL]
    assert b.due(db, "m", stood, now=NOW + timedelta(hours=1)) == []
    assert len(b.due(db, "m", stood, now=NOW + timedelta(days=1))) == 1


def test_the_last_run_and_failure_count_survive_between_runs(db):
    assert b.last_run(db, "m") is None and b.read_failures(db, "m") == 0
    b.mark_run(db, "m", now=NOW)
    b.set_read_failures(db, "m", 2)
    assert b.last_run(db, "m") == NOW and b.read_failures(db, "m") == 2


def test_a_run_pushes_the_blocker_but_not_twice_with_its_own_card(db, monkeypatch):
    """The make-room card's push already says the roster is illegal."""
    import json

    from puckpilot.season import notify
    from puckpilot.season.run import RunReport, _notify

    told = []
    monkeypatch.setattr(notify, "new_pickups", lambda new, url="": told.append("card") or True)
    monkeypatch.setattr(notify, "blocked", lambda t, m, url="": told.append(t) or True)
    db.execute(
        "INSERT INTO waiver_proposals (created_at, add_pid, reason_json, status, manager,"
        " league_key, team_key, kind, add_player_key, drop_player_key) VALUES"
        " ('2026-10-31 20:00:00', 1, ?, 'pending', 'm', 'L', 'T', 'activate', 'back', 'cheap')",
        (json.dumps({"add_name": "Back", "drop_name": "Cheap"}),),
    )
    db.commit()
    manager = SimpleNamespace(name="m", page=SimpleNamespace(url="", publishes=False))
    stood = b.record(db, "m", [b.Blocker(b.ROSTER_ILLEGAL, "x"), b.Blocker(b.RUN_GAP, "y")])
    _notify(db, manager, "L", RunReport(date="d", manager="m"), "2026-10-31 19:00:00", stood)
    assert told == ["card", b.KINDS[b.RUN_GAP]]
