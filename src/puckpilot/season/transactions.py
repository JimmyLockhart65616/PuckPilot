"""Carry out the adds and drops a person approved - and only those.

The approval queue (`proposals.py`) is the narrow point: `take_for_execution`
refuses anything not approved, and this is its one caller. Nothing here decides
whether to make a move; it makes the moves a person already decided on, the
way they were approved, and proves each one from the roster afterwards.

Three refusals sit in front of every attempt:

- **Consent.** Only an approval given while the page said Approve would make
  the move (`approved_to_execute`, carried from the card that was tapped). An
  approval given on "make it in Yahoo yourself" is reported once and left to
  the person - a yes to one is not a yes to the other.
- **When it was priced for.** A move is priced from the first game not yet
  played, so it is made on the next run - the drop's game today is part of its
  price. Two exceptions wait: a drop whose game today has begun (Yahoo will
  not drop him until tomorrow, Error #174), and a move the late-week check
  queued for after a day's games (`after_games_of`), whose value is next
  week's alone.
- **Once.** A move submitted but never confirmed by the roster is not tried
  again by itself: a second attempt at a transaction that did go through would
  be a second transaction. It is reported, loudly, for a person to look at.

A move that can no longer be made as approved - the drop has left the roster,
the add has been taken - is cancelled with the reason (`proposals.cancel`)
rather than retried on every run. One that failed for a passing reason is
tried again on the next run, and its failure told once.

Making the move in Yahoo is an optional local executor, like the lineup
actuator; without one, every approved move is reported as waiting to be made
by hand.
"""

from __future__ import annotations

from datetime import UTC, datetime

from puckpilot.season import proposals as proposals_mod


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def executor():
    """The installed local executor's `make`, or None."""
    try:  # Optional and unpublished, like the lineup actuator.
        from puckpilot.local.transact import make
    except ImportError:
        return None
    return make


def swapped(roster, p) -> bool:
    """The roster shows the move: the add on it, the drop (if any) gone."""
    keys = {q.player_key for q in roster.players}
    return p.add_player_key in keys and (not p.drop_player_key or p.drop_player_key not in keys)


def carry_out(
    conn,
    manager,
    league_key: str,
    team_key: str,
    report,
    read_roster,
    playing_today=frozenset(),
    started_today=frozenset(),
    today: str = "",
    make=None,
    page_url: str = "",
) -> int:
    """Make every approved move not yet made. Returns how many were made.

    `read_roster()` reads our roster from Yahoo now; `playing_today` and
    `started_today` are the clubs with a game today and those whose game has
    begun; `today` is the date (ISO). `make(manager, team_key, proposal)` is
    the executor (the local module by default). A move made, or one that
    failed for a new reason, is pushed to the phone (season/notify.py).
    """
    from puckpilot.season import notify

    approved = sorted(
        proposals_mod.listing(conn, manager.name, league_key, status=proposals_mod.APPROVED),
        key=lambda p: p.id,
    )
    mine = []
    for p in approved:
        if p.reason.get("approved_to_execute"):
            mine.append(p)
        elif not p.reason.get("manual_noted"):
            report.add(
                "transactions",
                True,
                f"#{p.id} add {p.add_name}{_drop(p)}: approved when the page said to make it "
                f"yourself - PuckPilot leaves it to you",
            )
            proposals_mod.annotate(conn, p.id, manual_noted=_now())
    if not mine:
        return 0
    if make is None:
        make = executor()
        if make is None:
            report.add(
                "transactions",
                True,
                f"{len(mine)} approved - make them in Yahoo (no executor installed)",
                [f"#{p.id} add {p.add_name}{_drop(p)}" for p in mine],
            )
            return 0
    roster = read_roster()
    made = 0
    for p in mine:
        p = proposals_mod.take_for_execution(conn, p.id)
        what = f"#{p.id} add {p.add_name}{_drop(p)}"
        submitted = p.reason.get("submitted_at")
        if swapped(roster, p):
            # Made by hand, or one submitted earlier that the roster shows late.
            how = "made in Yahoo (seen on the roster now)" if submitted else "made in Yahoo by hand"
            proposals_mod.mark_executed(conn, p.id, how)
            _audit(conn, manager, league_key, team_key, p, "executed", how)
            report.add("transactions", True, f"{what}: {how}")
            if submitted:
                notify.made(what, page_url)
            made += 1
            continue
        if submitted:
            report.add(
                "transactions",
                False,
                f"{what}: submitted {submitted} but never seen on the roster - check Yahoo; it "
                f"will not be tried again by itself (`ppilot season proposals --cancel {p.id}` "
                f"once looked at)",
            )
            continue
        drop = next((q for q in roster.players if q.player_key == p.drop_player_key), None)
        if p.drop_player_key and drop is None:
            _failed(conn, manager, league_key, team_key, report, p, what,
                    f"{p.drop_name} is no longer on your roster", page_url, final=True)  # fmt: skip
            continue
        if any(q.player_key == p.add_player_key for q in roster.players):
            # Added, but not as approved: the drop is still here.
            _failed(conn, manager, league_key, team_key, report, p, what,
                    f"{p.add_name} is already on your roster and {p.drop_name} still is too",
                    page_url, final=True)  # fmt: skip
            continue
        if drop is not None and drop.team in started_today:
            report.add(
                "transactions",
                True,
                f"{what}: waits - {p.drop_name}'s game today has begun, and Yahoo will not "
                f"drop him until tomorrow",
            )
            continue
        queued = str(p.reason.get("after_games_of") or "")
        if queued and today and today <= queued:
            night = "tonight's games" if queued == today else f"the games of {queued}"
            report.add("transactions", True, f"{what}: waits - approved to be made after {night}")
            continue
        result = make(manager, team_key, p)
        if not result.submitted:
            _failed(conn, manager, league_key, team_key, report, p, what, result.message,
                    page_url, final=bool(getattr(result, "final", False)),
                    lines=result.lines)  # fmt: skip
            continue
        p = proposals_mod.annotate(conn, p.id, submitted_at=_now())
        roster = read_roster()
        if result.ok and swapped(roster, p):
            proposals_mod.mark_executed(conn, p.id, result.message)
            _audit(conn, manager, league_key, team_key, p, "executed", result.message, result.lines)
            report.add("transactions", True, f"{what}: made in Yahoo", list(result.lines))
            notify.made(what, page_url)
            made += 1
            continue
        # Submitted, and not shown: never submitted again by itself.
        why = result.message if not result.ok else "submitted, but the roster does not show it"
        _failed(conn, manager, league_key, team_key, report, p, what, why, page_url,
                submitted=True, lines=result.lines)  # fmt: skip
    return made


def _failed(
    conn, manager, league_key, team_key, report, p, what, why, page_url,
    final=False, submitted=False, lines=(),
):  # fmt: skip
    """Record and report a move not made, and push it to the phone once per reason.

    `final`: it can no longer be made as approved, so it is cancelled.
    `submitted`: it was sent to Yahoo, so it is left for a person to look at.
    Otherwise nothing was sent, and the next run tries again.
    """
    from puckpilot.season import notify

    if final:
        then = "cancelled; nothing more will be tried"
    elif submitted:
        then = "check Yahoo; it will not be tried again by itself"
    else:
        then = "it will be tried again on the next run"
    _audit(conn, manager, league_key, team_key, p, "failed", why, lines)
    report.add("transactions", False, f"{what}: NOT made - {why} - {then}", list(lines))
    if final or p.reason.get("failed_notice") != why:
        notify.failed(what, f"{why} - {then}", page_url)
    if final:
        proposals_mod.cancel(conn, p.id, f"not made - {why}")
    else:
        proposals_mod.annotate(conn, p.id, failed_notice=why)


def _drop(p) -> str:
    return f", drop {p.drop_name}" if p.drop_player_key else ""


def _audit(conn, manager, league_key, team_key, p, outcome, message, lines=()) -> None:
    proposals_mod.record_action(
        conn,
        manager.name,
        league_key,
        team_key,
        datetime.now(UTC).date().isoformat(),
        "transaction",
        {"proposal": p.id, "add": p.add_name, "drop": p.drop_name, "steps": list(lines)},
        outcome=outcome,
        message=message,
    )
