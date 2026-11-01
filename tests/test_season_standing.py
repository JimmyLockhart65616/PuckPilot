"""Standing approval: a rotation player's drop, made after its window unless stopped.

It is consent given in advance, so what is pinned down here is mostly what it
refuses: a card not shown with its time, one whose time has not come, one the
manager stopped or said otherwise about since, and anything at all while the
grant is off or the kill switch is set.
"""

from __future__ import annotations

import ast
import inspect
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from puckpilot.season import proposals, standing, tiers
from puckpilot.season.authority import Authority, AuthorityError, StandingApproval
from puckpilot.season.proposals import ProposalError
from puckpilot.season.roster import RosterPlayer, TeamRoster
from tests.test_season_proposals import make, target

GRANT = StandingApproval(enabled=True, veto_minutes=120)
T0 = datetime(2026, 11, 2, 12, 0, tzinfo=UTC)
LK = "999.l.1"


def drop(pos="C", key="p.drop"):
    return RosterPlayer(player_key=key, yahoo_id="drop", name="Drop Me", team="MTL",
                        primary_position=pos, yahoo_eligible=frozenset({pos}),
                        selected_slot="BN", nhl_player_id=8001)  # fmt: skip


def roster(*players):
    return TeamRoster(league_key=LK, team_key=f"{LK}.t.5", date="2026-11-02",
                      players=players or (drop(),))  # fmt: skip


ROTATION = {"p.drop": tiers.Placement(tiers.Tier.ROTATION, tiers.AUTO, "one of the 2 worth least")}
UPGRADE = {"p.drop": tiers.Placement(tiers.Tier.UPGRADE, tiers.AUTO, "goes only for more")}


def offer(db, grant=GRANT, placed=ROTATION, ours=None, executes=True):
    return standing.offer(db, "jimmy", LK, grant, placed, ours or roster(), executes, None,
                          week=2, now=T0)  # fmt: skip


def show(db, at=T0):
    standing.stamp(db, "jimmy", LK, at)
    standing.shown(db, "jimmy", LK, at)


def due(db, at, grant=GRANT):
    return standing.approve_due(db, "jimmy", LK, grant, at)


# -- what is covered ---------------------------------------------------------------


def test_a_rotation_drop_is_offered_once_and_an_upgrade_only_one_says_why_not(db):
    [p] = make(db, target())
    assert offer(db, placed=UPGRADE) == []
    assert proposals.get(db, p.id).reason["standing_not"] == "Drop Me is Upgrade only, not Rotation"
    (got,) = offer(db)
    st = got.reason["standing"]
    assert (st["kind"], st["window"]) == ("same_kind", 120)
    assert st["due_at"] is None and st["shown_at"] is None
    assert got.reason["standing_not"] is None
    assert offer(db) == []


def test_what_a_standing_approval_does_not_cover(db, monkeypatch):
    [p] = make(db, target())

    def why(grant=GRANT, ours=None, executes=True, kinds=None, used=0, card=None):
        card = card or proposals.get(db, p.id)
        return standing.refusal(card, grant, ROTATION, ours or roster(), executes, kinds, used)

    assert why() == ""
    assert "no standing approval" in why(grant=StandingApproval())
    assert "does not make" in why(executes=False)
    assert "does not make" in why(kinds=frozenset({"activate"}))
    assert why(grant=StandingApproval(enabled=True, same_kind=False)) == (
        "same-kind swaps are not covered"
    )
    goalie = roster(drop("G"))
    assert why(ours=goalie) == "skater-for-goalie swaps are not covered"
    assert why(ours=goalie, grant=StandingApproval(enabled=True, cross_kind=True)) == ""
    assert "this week's 1" in why(grant=StandingApproval(enabled=True, max_per_week=1), used=1)
    claim = proposals.annotate(db, p.id, timing="on waivers until Thu")
    assert "waiver claim" in why(card=claim)
    [lone] = make(db, target(name="No Drop", key="p.nodrop", pid=9002, drop=False))
    assert "drops nobody" in why(card=lone)
    monkeypatch.setenv(standing.KILL_SWITCH, "1")
    assert standing.KILL_SWITCH in why(card=proposals.get(db, p.id))


def test_a_make_room_move_is_its_own_kind(db):
    from puckpilot.season import legality

    back = RosterPlayer(player_key="p.back", yahoo_id="back", name="Back", team="OTT",
                        primary_position="D", yahoo_eligible=frozenset({"D"}),
                        selected_slot="IR+", nhl_player_id=7001)  # fmt: skip
    fix = legality.Fix(back, drop(), "IR+", "a Rotation player")
    p, _ = proposals.propose_fix(db, "jimmy", LK, f"{LK}.t.5", fix, 2)
    assert standing.refusal(p, GRANT, ROTATION, roster(), True) == ""
    assert standing.move_kind(p, drop(), None) == "make_room"
    no_room = StandingApproval(enabled=True, make_room=False)
    why = standing.refusal(p, no_room, ROTATION, roster(), True)
    assert why == "make-room drops are not covered"


def test_a_weekly_limit_counts_moves_made_and_offered(db):
    make(db, target(), target(name="Second", key="p.add2", pid=9003))
    (first,) = offer(db, grant=StandingApproval(enabled=True, max_per_week=1))
    others = [p for p in proposals.pending(db, "jimmy") if p.id != first.id]
    assert "this week's 1 without a tap are made" in others[0].reason["standing_not"]


# -- shown first, with its time ------------------------------------------------------


def test_approval_in_advance_needs_the_card_shown_with_its_time_and_the_time_come(db):
    [p] = make(db, target())
    with pytest.raises(ProposalError, match="not offered"):
        proposals.approve_standing(db, p.id, "g", T0)
    offer(db)
    with pytest.raises(ProposalError, match="has not been shown"):
        proposals.approve_standing(db, p.id, "g", T0 + timedelta(hours=5))
    show(db)
    with pytest.raises(ProposalError, match="not due until"):
        proposals.approve_standing(db, p.id, "g", T0 + timedelta(minutes=119))
    got = proposals.approve_standing(db, p.id, "g", T0 + timedelta(minutes=120))
    assert got.status == proposals.APPROVED and got.reason["approved_by"] == "standing"
    assert got.reason["approved_to_execute"] is True and got.reason["grant"] == "g"
    assert proposals.take_for_execution(db, p.id).id == p.id
    with pytest.raises(ProposalError, match="already approved"):
        proposals.approve_standing(db, p.id, "g", T0 + timedelta(hours=3))


def test_the_window_starts_when_the_page_shows_the_card(db):
    """A push that failed showed nothing: the next one's time is the card's."""
    [p] = make(db, target())
    offer(db)
    standing.stamp(db, "jimmy", LK, T0)
    standing.stamp(db, "jimmy", LK, T0 + timedelta(hours=1))
    standing.shown(db, "jimmy", LK, T0 + timedelta(hours=1))
    st = proposals.get(db, p.id).reason["standing"]
    assert st["due_at"] == (T0 + timedelta(hours=3)).isoformat(timespec="seconds")
    standing.stamp(db, "jimmy", LK, T0 + timedelta(hours=2))  # shown: its time stays
    assert proposals.get(db, p.id).reason["standing"]["due_at"] == st["due_at"]


def test_a_card_nobody_stopped_is_approved_when_its_window_passes(db):
    [p] = make(db, target())
    offer(db)
    show(db)
    assert due(db, T0 + timedelta(minutes=60)) == ([], [])
    (got,), back = due(db, T0 + timedelta(minutes=121))
    assert back == [] and got.id == p.id and got.reason["grant"] == standing.grant_hash(GRANT)


def test_a_stopped_card_is_never_made(db):
    [p] = make(db, target())
    offer(db)
    show(db)
    proposals.decide(db, p.id, False)
    assert due(db, T0 + timedelta(hours=3)) == ([], [])
    assert proposals.get(db, p.id).status == proposals.REJECTED


def test_a_tier_set_since_puts_the_card_back_to_wait_for_a_tap(db):
    [p] = make(db, target())
    offer(db)
    show(db)
    tiers.pin(db, "jimmy", LK, "p.drop", tiers.Tier.UPGRADE)
    approved, back = due(db, T0 + timedelta(hours=3))
    assert approved == [] and back[0][1] == "you set Drop Me to Upgrade only"
    card = proposals.get(db, p.id)
    assert card.is_live and not card.reason["standing"]
    assert card.reason["standing_not"] == "you set Drop Me to Upgrade only"


def test_taking_off_the_rotation_pin_it_was_offered_on_puts_it_back(db):
    [p] = make(db, target())
    mine = {"p.drop": tiers.Placement(tiers.Tier.ROTATION, tiers.YOU, "set by you")}
    offer(db, placed=mine)
    show(db)
    _, back = due(db, T0 + timedelta(hours=3))
    assert back[0][1] == "you took Rotation off Drop Me"


def test_turning_it_off_or_the_kill_switch_puts_every_card_back(db, monkeypatch):
    make(db, target())
    offer(db)
    show(db)
    _, back = due(db, T0 + timedelta(hours=3), grant=StandingApproval())
    assert back[0][1] == "the standing approval was turned off"
    [p2] = make(db, target(name="Second", key="p.add2", pid=9003))
    offer(db)
    show(db)
    monkeypatch.setenv(standing.KILL_SWITCH, "1")
    approved, back = due(db, T0 + timedelta(hours=3))
    assert approved == [] and standing.KILL_SWITCH in back[0][1]


def test_the_search_refresh_keeps_what_the_card_was_shown_under(db):
    [p] = make(db, target())
    offer(db)
    show(db)
    make(db, target(gain=2.0), supersede=True)  # the same move, today's numbers
    assert proposals.get(db, p.id).reason["standing"]["shown_at"]
    proposals.refresh(db, {p.id: target(gain=1.0)}, {})
    assert proposals.get(db, p.id).reason["standing"]["shown_at"]


def test_only_a_tap_or_a_standing_approval_ever_approves_a_row():
    """If another way to approve appears, this is the test that should fail."""
    tree = ast.parse(inspect.getsource(proposals))
    approvers = set()
    for fn in (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)):
        for call in (n for n in ast.walk(fn) if isinstance(n, ast.Call)):
            if len(call.args) < 2 or not isinstance(call.args[0], ast.Constant):
                continue
            if not str(call.args[0].value).startswith("UPDATE waiver_proposals SET status"):
                continue
            first = call.args[1].elts[0] if isinstance(call.args[1], ast.Tuple) else call.args[1]
            if "APPROVED" in {n.id for n in ast.walk(first) if isinstance(n, ast.Name)}:
                approvers.add(fn.name)
    assert approvers == {"decide", "approve_standing"}


# -- when it runs, and what the manager sees -------------------------------------------


def test_each_card_s_time_is_a_run_for_the_scheduler(db):
    make(db, target())
    offer(db)
    now = datetime.now(UTC).replace(microsecond=0)
    show(db, now)
    when = (now + timedelta(minutes=120)).astimezone(ZoneInfo(standing.DEFAULT_TZ))
    want = (when + timedelta(minutes=1)).replace(second=0) if when.second else when
    assert standing.due_times(db, "jimmy", want.date().isoformat()) == [want.strftime("%H:%M")]
    assert standing.next_due(db, "jimmy") == now + timedelta(minutes=120)


def test_the_run_schedules_the_card_s_time_with_the_locks(monkeypatch):
    from datetime import date

    from puckpilot.season import locks, schedule
    from puckpilot.season import run as runmod

    seen: dict = {}
    monkeypatch.setattr(locks, "roster_teams", lambda *a, **k: {"TOR"})
    monkeypatch.setattr(locks, "locks_for", lambda *a, **k: [])
    monkeypatch.setattr(standing, "due_times", lambda *a, **k: ["21:06"])
    monkeypatch.setattr(schedule, "plan_day", lambda m, repo, times: seen.update(t=times) or [])
    report = runmod.RunReport(date="2026-11-02", manager="jimmy")
    runmod._plan_rest_of_day(None, SimpleNamespace(name="jimmy"),
                             SimpleNamespace(nhl_season="20262027"), date.today().isoformat(),
                             report, "t")  # fmt: skip
    assert seen["t"] == ["21:06"] and "moves due at 21:06 unless stopped" in report.steps[-1].detail


def test_the_run_approves_due_cards_and_says_so(db):
    from puckpilot.season import run as runmod

    [p] = make(db, target())
    offer(db)
    show(db, datetime.now(UTC) - timedelta(hours=3))
    manager = SimpleNamespace(name="jimmy", authority=Authority(standing=GRANT))
    report = runmod.RunReport(date="2026-11-02", manager="jimmy")
    assert runmod._standing_due(db, manager, LK, report) == 1
    assert proposals.get(db, p.id).status == proposals.APPROVED
    assert "window passed" in report.steps[-1].lines[0]


def test_a_card_under_standing_approval_says_when_and_how_to_stop_it(db):
    from puckpilot.season import snapshot

    make(db, target())
    [other] = make(db, target(name="Second", key="p.add2", pid=9003))
    offer(db)
    # The second put back to wait for a tap, as a pin set since would.
    proposals.annotate(db, other.id, standing=None, standing_not="Drop Me is Upgrade only")
    show(db)
    cards = snapshot.build(db, "jimmy", LK, "T", executes_moves=True, standing_on=True)["proposals"]
    soon = next(c for c in cards if c["standing"])
    assert soon["standing"]["due_local"] and "unless you stop it" in soon["standing"]["text"]
    assert soon["executes"] is True and soon["approve_means"].startswith("Nothing to do")
    asks = next(c for c in cards if not c["standing"])
    assert asks["needs_tap"] == "Needs your tap: Drop Me is Upgrade only"


def test_the_push_says_when_each_move_is_made(db, monkeypatch):
    from puckpilot.season import notify

    monkeypatch.setenv(notify.NOTIFY_ENV, "https://ntfy.example/topic")
    [p] = make(db, target())
    sent = []

    def post(*a, **k):
        sent.append((a, k))
        return True

    assert notify.standing_offers([(p, "9:05 PM")], "https://page", post=post)
    assert sent and "9:05 PM unless you stop it" in str(sent[0])


def test_the_ledger_says_a_move_was_approved_in_advance():
    from puckpilot.engine.categories import CATALOG
    from puckpilot.season import ledger

    line = ledger.LedgerLine(19, "add_drop", "2026-11-03", "Pickup", "Drop Me", standing=True)
    assert ledger.summary(line, [CATALOG["G"]])["head"].endswith("(approved in advance)")


# -- the grant itself ------------------------------------------------------------------


def test_the_grant_is_read_from_config_and_wrong_settings_are_refused():
    cfg = {"standing": {"enabled": True, "veto_minutes": 30, "cross_kind": True}}
    a = Authority.from_config(cfg)
    assert a.standing.enabled and a.standing.veto_minutes == 30 and a.standing.covers("cross_kind")
    assert not Authority().standing.enabled
    with pytest.raises(AuthorityError, match="unknown setting"):
        Authority.from_config({"standing": {"enable": True}})
    with pytest.raises(AuthorityError):
        StandingApproval(veto_minutes=-5)
    with pytest.raises(AuthorityError):
        StandingApproval(max_per_week=True)
    words = Authority.from_config({"standing": {"enabled": True}}).describe()
    assert "Moves without a tap: a ROTATION player's drop" in words
    assert "Moves without a tap: NONE" in Authority().describe()
