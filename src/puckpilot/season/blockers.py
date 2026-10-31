"""Blockers: anything that stops the tool doing its job, said loudly until it is gone.

On 2026-10-19 a player cleared to play was left in an IR slot with every active
spot full. Yahoo refused every add from then on, and for twelve days the run
printed a line about it under an "ok" lineup step - no push, nothing on the
page above the fold, nothing for the manager to act on. A tool another manager
relies on cannot be quiet about the one thing that stops it working.

A blocker is one such state, found by a detector that reads what the run
already knows - nothing is fetched for it. Each one:

- marks its run step FAIL (the exit code a scheduler sees),
- leads the page as a banner (`snapshot["blockers"]`),
- is pushed to the phone at high priority on the day it is first seen and once
  a day after that while it lasts - often enough not to be missed, rarely
  enough not to be muted.

Its first sighting is kept (`sync_meta` `blocker:{manager}:{kind}`), so the
banner says how long it has stood and a cleared blocker is forgotten.

Adding a kind is one detector in `detect` and one line in `KINDS`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

ROSTER_ILLEGAL = "roster_illegal"
NO_EXECUTOR = "no_executor"
CANNOT_MAKE = "cannot_make"
READS_FAILING = "reads_failing"
LOGGED_OUT = "logged_out"
RUN_GAP = "run_gap"

# What each kind is called on the page and the phone.
KINDS = {
    ROSTER_ILLEGAL: "Roster illegal - no add can be made",
    NO_EXECUTOR: "Approved moves cannot be made here",
    CANNOT_MAKE: "An approved move cannot be made here",
    READS_FAILING: "Yahoo could not be read",
    LOGGED_OUT: "Yahoo is signed out",
    RUN_GAP: "Scheduled runs were missed",
}

# Consecutive runs whose roster read failed before it is a blocker: one
# dropped read is retried by the next run; two in a row is a pattern.
READ_FAILURES = 2

# Longer than this between completed runs and some were missed. The daily
# anchor runs every morning, so a healthy gap is under a day.
RUN_GAP_HOURS = 26.0


@dataclass(frozen=True)
class Blocker:
    kind: str
    message: str
    since: str = ""  # first seen, UTC ISO

    @property
    def title(self) -> str:
        return KINDS.get(self.kind, self.kind)

    def payload(self) -> dict:
        return {"kind": self.kind, "title": self.title, "message": self.message,
                "since": self.since}  # fmt: skip


def _now() -> datetime:
    return datetime.now(UTC)


def detect(
    *,
    resolution=None,
    pending_fixes=(),
    execute_approved: bool = False,
    executor=None,
    approved=(),
    read_failed: bool = False,
    read_error: str = "",
    previous_failures: int = 0,
    last_run_at: datetime | None = None,
    now: datetime | None = None,
) -> tuple[list[Blocker], int]:
    """This run's blockers, and the count of consecutive failed reads to keep.

    Every input is something the run already has: the roster's legality
    (`legality.Resolution`), the make-room proposals open for it, whether the
    manager has moves made for him and with what, the approved moves waiting,
    whether this run's roster read failed and why, and when the last run
    finished.
    """
    now = now or _now()
    out: list[Blocker] = []

    if resolution is not None and not resolution.legal:
        bits = [v.message for v in resolution.violations]
        if pending_fixes:
            ids = ", ".join(f"#{p.id}" for p in pending_fixes)
            bits.append(f"Approve the make-room move ({ids}) to clear it.")
        bits += list(resolution.unresolved)
        out.append(Blocker(ROSTER_ILLEGAL, " ".join(bits)))

    if execute_approved and executor is None:
        out.append(
            Blocker(
                NO_EXECUTOR,
                "PuckPilot is set to make approved moves, but nothing on this machine can "
                "make them - approved moves wait until one is installed or are made by hand.",
            )
        )
    elif execute_approved and executor is not None:
        stuck = [p for p in approved if p.kind not in executor.kinds]
        if stuck:
            out.append(
                Blocker(
                    CANNOT_MAKE,
                    "Approved, and of a kind the installed executor does not make: "
                    + ", ".join(f"#{p.id}" for p in stuck)
                    + " - make them in Yahoo yourself.",
                )
            )

    failures = previous_failures + 1 if read_failed else 0
    if read_failed and "not logged in" in read_error.lower():
        out.append(
            Blocker(
                LOGGED_OUT,
                "The browser profile PuckPilot reads Yahoo through is signed out - nothing "
                "is read, set or made until it is signed in again.",
            )
        )
    elif failures >= READ_FAILURES:
        out.append(
            Blocker(
                READS_FAILING,
                f"The roster read has failed on {failures} runs in a row"
                + (f" (last: {read_error[:160]})" if read_error else "")
                + " - lineups and moves wait on it.",
            )
        )

    if last_run_at is not None:
        gap = now - last_run_at
        if gap > timedelta(hours=RUN_GAP_HOURS):
            hours = gap.total_seconds() / 3600
            out.append(
                Blocker(
                    RUN_GAP,
                    f"No run finished for {hours:.0f} hours, since "
                    f"{last_run_at.astimezone().strftime('%a %b %d %I:%M %p').replace(' 0', ' ')}"
                    f" - lineups were not set in that time. Runs need the computer on and "
                    f"signed in.",
                )
            )
    return out, failures


def _key(manager: str, kind: str) -> str:
    return f"blocker:{manager}:{kind}"


def record(conn, manager: str, found: list[Blocker], now: datetime | None = None) -> list[Blocker]:
    """Keep each blocker's first sighting; forget the cleared ones.

    Returns `found` with `since` filled in. A run gap is found only by the run
    that ends it, so the next run on time forgets it like any cleared blocker.
    """
    from puckpilot.data import store

    now = now or _now()
    stamp = now.isoformat(timespec="seconds")
    kinds = {b.kind for b in found}
    out = []
    for b in found:
        raw = store.get_meta(conn, _key(manager, b.kind))
        state = json.loads(raw) if raw else {}
        since = state.get("since") or stamp
        state = {**state, "since": since}
        store.set_meta(conn, _key(manager, b.kind), json.dumps(state))
        out.append(Blocker(b.kind, b.message, since))
    for kind in KINDS:
        if kind not in kinds:
            conn.execute("DELETE FROM sync_meta WHERE key = ?", (_key(manager, kind),))
    conn.commit()
    return out


def due(conn, manager: str, blockers: list[Blocker], now: datetime | None = None) -> list[Blocker]:
    """Those not yet pushed today - and mark them pushed."""
    from puckpilot.data import store

    now = now or _now()
    today = now.astimezone().date().isoformat()
    out = []
    for b in blockers:
        raw = store.get_meta(conn, _key(manager, b.kind))
        state = json.loads(raw) if raw else {"since": b.since}
        if state.get("pushed_on") == today:
            continue
        state["pushed_on"] = today
        store.set_meta(conn, _key(manager, b.kind), json.dumps(state))
        out.append(b)
    conn.commit()
    return out


def last_run(conn, manager: str) -> datetime | None:
    from puckpilot.data import store

    raw = store.get_meta(conn, f"lastrun:{manager}")
    try:
        return datetime.fromisoformat(raw) if raw else None
    except ValueError:
        return None


def mark_run(conn, manager: str, now: datetime | None = None) -> None:
    from puckpilot.data import store

    store.set_meta(conn, f"lastrun:{manager}", (now or _now()).isoformat(timespec="seconds"))
    conn.commit()


def read_failures(conn, manager: str) -> int:
    from puckpilot.data import store

    raw = store.get_meta(conn, f"readfail:{manager}")
    return int(raw) if raw and raw.isdigit() else 0


def set_read_failures(conn, manager: str, n: int) -> None:
    from puckpilot.data import store

    store.set_meta(conn, f"readfail:{manager}", str(int(n)))
    conn.commit()
