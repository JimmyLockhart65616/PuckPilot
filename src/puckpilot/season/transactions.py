"""Carry out the adds and drops a person approved - and only those.

The approval queue (`proposals.py`) is the narrow point: `take_for_execution`
refuses anything not approved, and this is its one caller. Nothing here decides
whether to make a move; it makes the moves a person already decided on, the
way they were approved, and proves each one from the roster afterwards.

Three refusals sit in front of every attempt:

- **Consent.** Only an approval given while the page said Approve would make
  the move (`approved_to_execute`, carried from the card that was tapped). A
  yes to "make it in Yahoo yourself" is not a yes to PuckPilot making it - so
  with an executor installed it is not left to the person either (a manager
  who turned this on does not make pickups by hand): the card is reopened, and
  the next push asks again on the words that are true now. Without an
  executor, the move is the person's to make, as the card said.
- **When it was priced for.** A move is priced from the first game not yet
  played, so it is made on the next run - the drop's game today is part of its
  price. Two exceptions wait: a drop whose game today has begun (Yahoo will
  not drop him until tomorrow, Error #174), and a move the late-week check
  queued for after a day's games (`after_games_of`), whose value is next
  week's alone.
- **Once.** A move submitted but never confirmed by the roster is not tried
  again by itself: a second attempt at a transaction that did go through would
  be a second transaction. It is reported, loudly, for a person to look at.

A move that can no longer be made as approved - the drop has left the roster
or become a keeper to protect, the add has been taken, gone on waivers or been
listed out since the approval - is cancelled with the reason
(`proposals.cancel`) rather than retried on every run. An acquisition is not
spent on a yes given to a different situation. One that failed for a passing
reason is tried again on the next run, and its failure told once.

Moves come in kinds (`moves.MoveKind`). An add/drop is checked against the
add as it stands now; a make-room move (`ACTIVATE`: drop one player so another
can come off an IR slot he no longer qualifies for) is checked against the
roster - it is made only while the roster is still illegal and still full, so
a roster that became legal some other way never loses a player for nothing.
Each kind says what the roster must show for the move to count as made.

Making the move in Yahoo is an executor (`executors.py`): the platform's API
once it can write, or a manager's own optional local module. Without one,
every approved move is reported as waiting to be made by hand; with one that
cannot make a kind, that move is reported and pushed, never skipped quietly.
"""

from __future__ import annotations

from datetime import UTC, datetime

from puckpilot.season import proposals as proposals_mod
from puckpilot.season.executors import as_executor, can_make, installed
from puckpilot.season.moves import MoveKind, kind_of

# What a move made and then seen on the roster is recorded as.
SHOWN = "made in Yahoo - the roster shows it"


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def executor():
    """The installed executor (`executors.installed`), or None."""
    return installed()


def swapped(roster, p) -> bool:
    """The roster shows the move made, by its kind's own test (`MoveKind.made_on`)."""
    return kind_of(p.kind).made_on(roster, p.add_player_key, p.drop_player_key)


def still_needed(p, roster, runtime=None) -> str:
    """Why an approved make-room move should not be made now, or "" if it should.

    It exists to make an illegal roster legal. If the returning player is no
    longer stuck - back on the active roster by hand, eligible for his slot
    again - or a spot has opened so he can come back with no drop, the drop
    would cost a player for nothing.
    """
    from puckpilot.season import legality

    back = next((q for q in roster.players if q.player_key == p.add_player_key), None)
    if back is None:
        return f"{p.add_name} is no longer on your roster"
    if not (back.on_ir and back.selected_slot not in back.yahoo_eligible):
        return f"{p.add_name} no longer needs room - the roster is legal without this drop"
    if runtime is not None and p.drop_player_key != p.add_player_key:
        active = sum(1 for q in roster.players if not q.on_ir)
        if active < legality.active_cap(runtime):
            return f"a roster spot is free, so {p.add_name} can come back without a drop"
    return ""


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
    read_adds=None,
    protected=frozenset(),
    runtime=None,
) -> int:
    """Make every approved move not yet made. Returns how many were made.

    `read_roster()` reads our roster from Yahoo now; `playing_today` and
    `started_today` are the clubs with a game today and those whose game has
    begun; `today` is the date (ISO). `make(manager, team_key, proposal)` is
    the executor (`executors.installed()` by default; a bare function is
    taken to make every kind). `read_adds(keys)` reads the adds as they stand
    now ({player_key: pool.PoolPlayer}); `protected` is the player keys kept
    for next season, never dropped; `runtime` the league's slots, for whether
    a make-room move is still needed. A move made, or one that failed for a new
    reason, is pushed to the phone (season/notify.py).
    """
    from puckpilot.season import notify

    approved = sorted(
        proposals_mod.listing(conn, manager.name, league_key, status=proposals_mod.APPROVED),
        key=lambda p: p.id,
    )
    ex = executor() if make is None else as_executor(make)
    mine = []
    for p in approved:
        if p.reason.get("approved_to_execute"):
            mine.append(p)
        elif ex is not None and not p.reason.get("submitted_at"):
            # Approved on "make it yourself", and this manager does not make
            # pickups by hand: ask again, on the card's words now.
            what = _what(p)
            why = "approved when the card said to make it yourself - asked again"
            proposals_mod.reopen(conn, p.id, why)
            # The reason itself is rebuilt by the next re-check; the record is here.
            _audit(conn, manager, league_key, team_key, p, "skipped", why)
            notify.ask_again(what, page_url)
            report.add(
                "transactions",
                True,
                f"{what}: approved when the page said to make it yourself - back on the page "
                f"to approve again; PuckPilot makes it then. Until then it is a card like any "
                f"other, and a later search can withdraw it",
            )
        elif not p.reason.get("manual_noted"):
            report.add(
                "transactions",
                True,
                f"{_what(p)}: approved when the page said to make it yourself - PuckPilot leaves "
                f"it to you",
            )
            proposals_mod.annotate(conn, p.id, manual_noted=_now())
    if not mine:
        return 0
    if ex is None:
        report.add(
            "transactions",
            True,
            f"{len(mine)} approved - make them in Yahoo (no executor installed)",
            [_what(p) for p in mine],
        )
        return 0
    swaps = [p for p in mine if kind_of(p.kind).spends_acquisition]
    try:
        roster = read_roster()
        adds = None
        if read_adds is not None:
            adds = read_adds([p.add_player_key for p in swaps]) if swaps else {}
    except Exception as e:  # noqa: BLE001 - nothing was made; the next run reads again
        # On 2026-10-07 a dropped read crashed the step at 19:10 and nobody was
        # told: the approved add simply was not there. Say so, once per reason.
        for p in mine:
            _failed(conn, manager, league_key, team_key, report, p, _what(p),
                    f"Yahoo could not be read ({type(e).__name__})", page_url)  # fmt: skip
        return 0
    made = 0
    # Make-room moves first: until they are made Yahoo refuses every add.
    mine.sort(key=lambda q: (kind_of(q.kind).spends_acquisition, q.id))
    for p in mine:
        p = proposals_mod.take_for_execution(conn, p.id)
        kind = kind_of(p.kind)
        what = _what(p)
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
        if not can_make(ex, kind.value):
            _failed(conn, manager, league_key, team_key, report, p, what,
                    f"the installed executor cannot make a {kind.value} move - make it in "
                    f"Yahoo yourself", page_url)  # fmt: skip
            continue
        drop = next((q for q in roster.players if q.player_key == p.drop_player_key), None)
        if p.drop_player_key and drop is None:
            _failed(conn, manager, league_key, team_key, report, p, what,
                    f"{p.drop_name} is no longer on your roster", page_url, final=True)  # fmt: skip
            continue
        if kind is MoveKind.ACTIVATE:
            why = still_needed(p, roster, runtime)
            if why:
                _failed(conn, manager, league_key, team_key, report, p, what, why, page_url,
                        final=True)  # fmt: skip
                continue
        elif any(q.player_key == p.add_player_key for q in roster.players):
            # Added, but not as approved: the drop is still here.
            _failed(conn, manager, league_key, team_key, report, p, what,
                    f"{p.add_name} is already on your roster and {p.drop_name} still is too",
                    page_url, final=True)  # fmt: skip
            continue
        if drop is not None and drop.player_key in protected:
            _failed(conn, manager, league_key, team_key, report, p, what,
                    f"{p.drop_name} is now one of the keepers protected for next season",
                    page_url, final=True)  # fmt: skip
            continue
        stuck = _stuck_on_ir(roster) if kind.spends_acquisition else []
        if stuck:
            # Yahoo refuses every add while one stands (Error #845). Not a
            # failure: the make-room move, then the lineup, clear it - this
            # run's second pass or the next run makes the add.
            q = stuck[0]
            report.add(
                "transactions",
                True,
                f"{what}: waits - {q.name} must come off {q.selected_slot} first; Yahoo "
                f"refuses every add until then",
            )
            continue
        if adds is not None and kind.spends_acquisition:
            now = adds.get(p.add_player_key)
            why = _add_refusal(p, now)
            if why:
                # Unreadable is passing; taken, on waivers or out is not.
                _failed(conn, manager, league_key, team_key, report, p, what, why, page_url,
                        final=now is not None)  # fmt: skip
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
        result = ex.make(manager, team_key, p)
        if not result.submitted:
            _failed(conn, manager, league_key, team_key, report, p, what, result.message,
                    page_url, final=bool(getattr(result, "final", False)),
                    lines=result.lines)  # fmt: skip
            continue
        p = proposals_mod.annotate(conn, p.id, submitted_at=_now())
        try:
            roster = read_roster()
        except Exception as e:  # noqa: BLE001 - submitted: the next run checks the roster
            _failed(conn, manager, league_key, team_key, report, p, what,
                    f"submitted, but the roster could not be read to check it "
                    f"({type(e).__name__})", page_url, submitted=True,
                    lines=result.lines)  # fmt: skip
            break  # the rest wait for a run that can see the roster
        if result.ok and swapped(roster, p):
            proposals_mod.mark_executed(conn, p.id, SHOWN)
            _audit(conn, manager, league_key, team_key, p, "executed", SHOWN, result.lines)
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


def _add_refusal(p, now) -> str:
    """Why the add cannot be made as approved now, or "" when it can."""
    if now is None:
        return f"{p.add_name}'s status could not be read in Yahoo"
    if not now.is_available:
        return f"{p.add_name} is no longer available"
    if now.on_waivers:
        return f"{p.add_name} is on waivers now - a claim, which PuckPilot does not make"
    if now.is_out:
        return f"{p.add_name} is now listed {now.status}"
    return ""


def _stuck_on_ir(roster) -> list:
    """Players in an IR slot they no longer qualify for - while one is, every add fails."""
    return [
        q
        for q in getattr(roster, "players", ())
        if getattr(q, "on_ir", False) and q.selected_slot not in q.yahoo_eligible
    ]


def _drop(p) -> str:
    return f", drop {p.drop_name}" if p.drop_player_key else ""


def _what(p) -> str:
    """The move in the run log's words: "#13 add Marco Rossi, drop X", or a
    make-room move's own line."""
    if kind_of(p.kind) is MoveKind.ACTIVATE:
        if p.drop_player_key == p.add_player_key:
            return f"#{p.id} release {p.add_name}"
        return f"#{p.id} drop {p.drop_name} so {p.add_name} comes back"
    return f"#{p.id} add {p.add_name}{_drop(p)}"


def _audit(conn, manager, league_key, team_key, p, outcome, message, lines=()) -> None:
    proposals_mod.record_action(
        conn,
        manager.name,
        league_key,
        team_key,
        datetime.now(UTC).date().isoformat(),
        "transaction",
        {
            "proposal": p.id,
            "kind": p.kind,
            "add": p.add_name,
            "drop": p.drop_name,
            "steps": list(lines),
        },
        outcome=outcome,
        message=message,
    )
