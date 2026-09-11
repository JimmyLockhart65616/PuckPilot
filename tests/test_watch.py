"""The draft-watch polling budget.

`yahoo watch-draft` polls an undocumented Yahoo endpoint through a live draft to
settle whether `draftresults` fills in during one. The experiment is worth
running; the request volume is what makes it either acceptable or a robot.

An earlier 3-second default over a 90-minute draft came to roughly 3,600
requests - the exact usage pattern `yahoo/session.py` forbids two files away. So
the budget is the part under test, not the verdict logic.
"""

from __future__ import annotations

import pytest

from puckpilot.yahoo.watch import DEFAULT_INTERVAL_S, MAX_REQUESTS, watch


class FakeSession:
    """Counts real calls, so the budget is measured rather than asserted."""

    def __init__(self, status="drafting", picks=3, fail=False):
        self.calls = 0
        self.status = status
        self.picks = picks
        self.fail = fail

    def league_meta(self, key):
        self.calls += 1
        if self.fail:
            raise ConnectionError("down")
        return {"draft_status": self.status}

    def draft_results(self, key):
        self.calls += 1
        if self.fail:
            raise ConnectionError("down")
        return [{"pick": i} for i in range(self.picks)]


def test_the_defaults_are_polite():
    """These two numbers are the whole conduct claim."""
    assert DEFAULT_INTERVAL_S >= 15.0
    assert MAX_REQUESTS <= 500


def test_polling_stops_at_the_budget_and_says_so():
    session = FakeSession()
    report = watch(session, "465.l.12345", interval=0, duration=600, max_requests=10)
    assert session.calls <= 10, f"budget overrun: {session.calls} requests"
    assert "stopped after" in report.note
    assert report.note in report.text  # the user is told, not just the object


def test_the_budget_holds_when_every_request_fails():
    """A blip must not end the experiment - but nor may it turn into a retry
    storm against an endpoint we are trying to be careful with."""
    session = FakeSession(fail=True)
    watch(session, "465.l.12345", interval=0, duration=600, max_requests=8)
    assert session.calls <= 8


def test_a_budget_too_small_for_one_pass_makes_no_request_at_all():
    session = FakeSession()
    watch(session, "465.l.12345", interval=0, duration=600, max_requests=1)
    assert session.calls == 0


def test_a_finished_draft_stops_immediately():
    session = FakeSession(status="postdraft", picks=192)
    watch(session, "465.l.12345", interval=0, duration=600)
    assert session.calls == 2, "polled on past a completed draft"


def test_the_worst_case_run_stays_inside_the_budget():
    """duration/interval passes x 2 requests must not exceed MAX_REQUESTS, or
    the ceiling is decorative."""
    passes = 3600 / DEFAULT_INTERVAL_S
    assert passes * 2 <= MAX_REQUESTS


@pytest.mark.parametrize("status,expect", [("drafting", "LIVE"), ("predraft", "INCONCLUSIVE")])
def test_the_verdict_still_reflects_what_was_observed(status, expect):
    session = FakeSession(status=status, picks=3 if status == "drafting" else 0)
    report = watch(session, "465.l.12345", interval=0, duration=600, max_requests=4)
    assert report.verdict.startswith(expect)
