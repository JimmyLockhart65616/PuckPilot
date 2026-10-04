"""Running it without being asked to.

Windows Task Scheduler rather than cron, and several times a day rather than
once, because the deadline moves: a player locks when his own game starts, so a
Saturday matinee locks at one o'clock and a Tuesday at seven. A single evening
run would be too late for every afternoon game of the season.

Three runs, and the reasoning is the schedule's own shape. Across 2026-27 the
earliest puck drop on a game day is usually 19:00 eastern, but weekend matinees
start from about 13:00, so: late morning (catches matinees, and leaves time to
react), mid afternoon (the last chance before an early evening game), and just
before seven (the one that matters most nights).

Every run is idempotent, so three of them cost a little time and change nothing
twice. The job reports rather than raises, and appends to a log, because an
unattended run that leaves no trace is indistinguishable from one that never
happened.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

PREFIX = "PuckPilot"
LOCK_TAG = "lock"

# The one fixed run. Everything else is planned from the schedule, because a
# player locks when his own game starts and 41% of game days have a game before
# 18:45 - so any fixed evening time is too late two days in five. This anchor
# starts that chain, and it is also the morning read: last night's results,
# the week's score and any new pickup to decide on. It was 11:00 until
# 2026-10-04, when the manager said he checks before 9:00 - and 07:00 is
# still after the night's last game and Yahoo's overnight processing, still
# before every game day's first puck drop (one day in 185 starts before
# 11:00, none before 07:00), and first to a free agent everyone else wakes
# up to.
ANCHOR_TIMES = ("07:00",)
DEFAULT_TIMES = ANCHOR_TIMES


@dataclass(frozen=True)
class Task:
    name: str
    time: str
    command: str

    def create_args(self, once: bool = False) -> list[str]:
        # ONCE with no /SD means today, which is the only day a lock-timed run
        # is ever wanted - they are re-planned every run.
        return [
            "schtasks",
            "/Create",
            "/TN",
            self.name,
            "/TR",
            self.command,
            "/SC",
            "ONCE" if once else "DAILY",
            "/ST",
            self.time,
            "/F",
        ]

    def delete_args(self) -> list[str]:
        return ["schtasks", "/Delete", "/TN", self.name, "/F"]


def _python() -> str:
    return str(Path(sys.executable).resolve())


LOG = "data/logs/season.log"


def tasks(
    manager: str,
    repo: Path,
    times: tuple[str, ...] = DEFAULT_TIMES,
    log: str = LOG,
) -> list[Task]:
    """One task per run time, all running the same idempotent command."""
    py = _python()
    out = []
    for t in times:
        cmd = (
            f'cmd /c cd /d "{repo}" && "{py}" -m puckpilot.cli season run '
            f"--manager {manager} --log {log}"
        )
        out.append(Task(name=f"{PREFIX}-{manager}-{t.replace(':', '')}", time=t, command=cmd))
    return out


def lock_tasks(manager: str, repo: Path, times: list[str], log: str = LOG) -> list[Task]:
    """One-shot runs, each timed to land shortly before a lock."""
    py = _python()
    cmd = (
        f'cmd /c cd /d "{repo}" && "{py}" -m puckpilot.cli season run '
        f"--manager {manager} --log {log}"
    )
    return [
        Task(name=f"{PREFIX}-{manager}-{LOCK_TAG}-{t.replace(':', '')}", time=t, command=cmd)
        for t in times
    ]


def plan_day(manager: str, repo: Path, times: list[str], log: str = LOG) -> list[str]:
    """Replace today's one-shot runs with the ones the schedule now implies.

    Cleared and rebuilt rather than reconciled: a game can be rescheduled, and
    a stale task that fires at yesterday's time is a browser launching for no
    reason at an hour nobody is watching.
    """
    out = [f"cleared {n}" for n in _clear_locks(manager)]
    for t in lock_tasks(manager, repo, times, log):
        r = subprocess.run(t.create_args(once=True), capture_output=True, text=True)
        ok = r.returncode == 0
        out.append(
            f"{'scheduled' if ok else 'FAILED'} a run at {t.time}"
            + ("" if ok else f" - {(r.stderr or r.stdout).strip()[:100]}")
        )
    if not times:
        out.append("no further locks today - nothing more to run")
    return out


def _clear_locks(manager: str) -> list[str]:
    tag = f"{PREFIX}-{manager}-{LOCK_TAG}-"
    gone = []
    for name in installed():
        if name.startswith(tag):
            subprocess.run(
                ["schtasks", "/Delete", "/TN", name, "/F"], capture_output=True, text=True
            )
            gone.append(name)
    return gone


def describe(items: list[Task], key_set: bool) -> str:
    lines = [
        f"{len(items)} daily task(s), all running the same command - it is idempotent, "
        f"so extra runs cost time and change nothing twice:",
        "",
    ]
    for t in items:
        lines.append(f"  {t.time}  {t.name}")
    lines += [
        "",
        f"  {items[0].command}" if items else "",
        "",
        "The command syncs last night's games, collects anything you decided on the",
        "page, works out tonight's lineup, and on the first day of a fantasy week",
        "works out the week and proposes adds. No add, drop or claim is ever made -",
        "those wait for your approval. Lineup changes are made in Yahoo only under",
        "standing authority ([authority.lineup] enabled) and only where an actuator",
        "is installed; otherwise they are recommendations on the page.",
        "",
        "Each run also schedules the rest of today from the real game times - a",
        "player locks when his own game starts, so a 1pm matinee needs a run before",
        "lunch and a 10pm west-coast game needs one at 9:40.",
    ]
    if not key_set:
        lines += [
            "",
            "WARNING: PUCKPILOT_MANAGER_KEY is not set for the scheduled task, so it",
            "will compute the lineup but cannot publish it to your phone. Set it as a",
            "USER environment variable (not just in this shell) before installing:",
            '  setx PUCKPILOT_MANAGER_KEY "<your key>"',
        ]
    return "\n".join(lines)


def install(items: list[Task]) -> list[str]:
    out = []
    for t in items:
        r = subprocess.run(t.create_args(), capture_output=True, text=True)
        ok = r.returncode == 0
        out.append(
            f"{'created' if ok else 'FAILED'} {t.name} at {t.time}"
            + ("" if ok else f" - {(r.stderr or r.stdout).strip()[:120]}")
        )
    return out


def remove(manager: str) -> list[str]:
    """Take away every task for this manager, whatever time it runs at.

    Deliberately by prefix rather than by the times currently configured:
    removing only what today's defaults happen to name is how the run times of
    an older version get orphaned, firing on a schedule nobody remembers
    setting. That happened once already.
    """
    tag = f"{PREFIX}-{manager}-"
    out = []
    for name in installed():
        if not name.startswith(tag):
            continue
        r = subprocess.run(
            ["schtasks", "/Delete", "/TN", name, "/F"], capture_output=True, text=True
        )
        out.append(f"{'removed' if r.returncode == 0 else 'could not remove'} {name}")
    return out or ["nothing registered for " + manager]


def installed() -> list[str]:
    """Names of PuckPilot tasks already registered."""
    r = subprocess.run(["schtasks", "/Query", "/FO", "CSV", "/NH"], capture_output=True, text=True)
    if r.returncode != 0:
        return []
    out = []
    for line in r.stdout.splitlines():
        name = line.split(",")[0].strip('"').lstrip("\\")
        if name.startswith(PREFIX):
            out.append(name)
    return sorted(set(out))
