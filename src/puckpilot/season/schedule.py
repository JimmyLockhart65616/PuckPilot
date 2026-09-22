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

# Local times. The last is the one that matters most nights; the first two are
# for matinees and for leaving a person time to disagree.
DEFAULT_TIMES = ("11:00", "15:30", "18:45")


@dataclass(frozen=True)
class Task:
    name: str
    time: str
    command: str

    def create_args(self) -> list[str]:
        return [
            "schtasks",
            "/Create",
            "/TN",
            self.name,
            "/TR",
            self.command,
            "/SC",
            "DAILY",
            "/ST",
            self.time,
            "/F",
        ]

    def delete_args(self) -> list[str]:
        return ["schtasks", "/Delete", "/TN", self.name, "/F"]


def _python() -> str:
    return str(Path(sys.executable).resolve())


def tasks(
    manager: str,
    repo: Path,
    times: tuple[str, ...] = DEFAULT_TIMES,
    log: str = "data/logs/season.log",
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
        "works out the week and proposes adds. Nothing is written to Yahoo.",
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


def remove(items: list[Task]) -> list[str]:
    out = []
    for t in items:
        r = subprocess.run(t.delete_args(), capture_output=True, text=True)
        out.append(f"{'removed' if r.returncode == 0 else 'not found'} {t.name}")
    return out


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
