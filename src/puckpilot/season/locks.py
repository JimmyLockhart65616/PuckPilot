"""When tonight's slots actually close, and therefore when to run.

A daily league locks each player the moment his own game starts, not at one
roster deadline, so "before the lock" is not a single time - it is two to four
of them on a typical day. Measured across 2026-27: the first puck drop is 19:00
local on 80 days but 13:00 on 32 and noon on 10, and **41% of game days have a
game before 18:45**. Any fixed evening schedule is therefore too late on two
days in five.

So the run times are computed from the schedule rather than chosen. Only the
clubs on the roster count: a 13:00 game matters if it holds one of your players
and is irrelevant otherwise, and the difference is the number of times a
browser gets launched on a Saturday.

Nothing here decides anything. It answers "when does something of yours close",
which is the input to both the schedule and the sentence on the page telling a
person how long they have.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

DEFAULT_TZ = "America/Toronto"

# How long before a lock to run. Long enough for a Yahoo read, the model and a
# push; short enough that a late scratch is still caught.
LEAD_MINUTES = 20


@dataclass(frozen=True)
class Lock:
    """One moment at which some of your players stop being editable."""

    utc: datetime
    local: datetime
    teams: tuple[str, ...]
    players: tuple[str, ...]

    @property
    def hhmm(self) -> str:
        return self.local.strftime("%H:%M")

    @property
    def pretty(self) -> str:
        return self.local.strftime("%I:%M %p").lstrip("0")

    def run_at(self, lead: int = LEAD_MINUTES) -> datetime:
        return self.local - timedelta(minutes=lead)

    def describe(self) -> str:
        who = ", ".join(self.players) if self.players else ", ".join(self.teams)
        return f"{self.pretty}  {who}"


def roster_teams(
    conn: sqlite3.Connection, manager: str, team_key: str = "", day: str = ""
) -> dict[str, list[str]]:
    """team -> player names, from the latest roster snapshot on or before `day`.

    Read from what was last stored rather than from Yahoo: planning the day's
    runs must not itself need a browser, or the thing that schedules the work
    becomes the thing most likely to fail.

    Two traps. A roster read *for* a future date is still a snapshot, so the
    newest date is not the newest read - a 10-07 read taken on 09-21 planned
    every lock for days - hence `day`. And each run now also saves the
    opponent's roster under the same manager, so the manager's own team has to
    be named: without `team_key` it is taken to be the team with the most
    snapshots, which the opponents of single weeks never are.
    """
    if not team_key:
        top = conn.execute(
            "SELECT team_key FROM yahoo_roster_snapshots WHERE manager = ? "
            "GROUP BY team_key ORDER BY COUNT(*) DESC LIMIT 1",
            (manager,),
        ).fetchone()
        if not top:
            return {}
        team_key = top[0]
    row = conn.execute(
        "SELECT MAX(date) FROM yahoo_roster_snapshots WHERE manager = ? AND team_key = ?"
        + (" AND date <= ?" if day else ""),
        (manager, team_key, day) if day else (manager, team_key),
    ).fetchone()
    if not row or not row[0]:
        return {}
    rows = conn.execute(
        "SELECT name, team_abbrev, status FROM yahoo_roster_snapshots "
        "WHERE manager = ? AND team_key = ? AND date = ?",
        (manager, team_key, row[0]),
    ).fetchall()
    out: dict[str, list[str]] = {}
    for r in rows:
        if not r["team_abbrev"]:
            continue
        out.setdefault(r["team_abbrev"], []).append(r["name"])
    return out


def locks_for(
    conn: sqlite3.Connection,
    season: str,
    day: str,
    teams: dict[str, list[str]],
    tz: str = DEFAULT_TZ,
) -> list[Lock]:
    """Every distinct moment on `day` when one of these clubs starts a game."""
    zone = ZoneInfo(tz)
    rows = conn.execute(
        "SELECT start_time_utc, home_team, away_team FROM nhl_schedule "
        "WHERE season = ? AND game_type = 2 AND game_date = ? AND start_time_utc IS NOT NULL",
        (season, day),
    ).fetchall()

    grouped: dict[str, set[str]] = {}
    for r in rows:
        mine = {t for t in (r["home_team"], r["away_team"]) if t in teams}
        if mine:
            grouped.setdefault(r["start_time_utc"], set()).update(mine)

    out: list[Lock] = []
    for stamp, clubs in sorted(grouped.items()):
        utc = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
        names = [n for club in sorted(clubs) for n in teams.get(club, [])]
        out.append(
            Lock(
                utc=utc,
                local=utc.astimezone(zone),
                teams=tuple(sorted(clubs)),
                players=tuple(names),
            )
        )
    return out


def upcoming(
    locks: list[Lock], now: datetime | None = None, lead: int = LEAD_MINUTES
) -> list[Lock]:
    """The locks still worth running before."""
    zone = locks[0].local.tzinfo if locks else ZoneInfo(DEFAULT_TZ)
    at = now or datetime.now(zone)
    return [x for x in locks if x.run_at(lead) > at]


def run_times(locks: list[Lock], lead: int = LEAD_MINUTES) -> list[str]:
    """HH:MM local, one per distinct lock, de-duplicated and ordered."""
    seen: list[str] = []
    for x in locks:
        hm = x.run_at(lead).strftime("%H:%M")
        if hm not in seen:
            seen.append(hm)
    return seen


def describe(day: str, locks: list[Lock], tz: str = DEFAULT_TZ) -> str:
    if not locks:
        return f"{day}: none of your players have a game."
    lines = [f"{day}: {len(locks)} lock(s), each closing only the players in that game."]
    for x in locks:
        lines.append(f"  {x.describe()}")
        lines.append(f"      run by {x.run_at().strftime('%I:%M %p').lstrip('0')}")
    return "\n".join(lines)
