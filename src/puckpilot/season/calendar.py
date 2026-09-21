"""The forward date axis: who plays when, and which fantasy week that falls in.

`nhl_schedule` already holds every game of the season, including ones not yet
played, so all of this is offline arithmetic against a table we sync anyway.

Three places in the codebase each rebuilt a fragment of this inline
(`aggregate.season_games`, the team->dates dict inside
`lineup_replay.skater_availability`, a row count in `preflight`). This is the
shared version; it deliberately returns plain dates rather than the integer
indices the replay harnesses use, because a live caller knows what day it is and
an index into a list of past game dates cannot represent tomorrow.

Week boundaries come from the league's own runtime settings, not from a constant
here: Yahoo tells us when week 1 starts and how many there are, so a league with
a different calendar needs no code change.
"""

from __future__ import annotations

import sqlite3
from datetime import date, timedelta

REGULAR_SEASON = 2


def _d(value: str | date) -> date:
    return value if isinstance(value, date) else date.fromisoformat(value)


def teams_playing(conn: sqlite3.Connection, day: str | date, season: str) -> set[str]:
    """Team abbreviations with a regular-season game on `day`."""
    rows = conn.execute(
        "SELECT home_team, away_team FROM nhl_schedule "
        "WHERE season = ? AND game_type = ? AND game_date = ?",
        (season, REGULAR_SEASON, _d(day).isoformat()),
    ).fetchall()
    out: set[str] = set()
    for r in rows:
        out.add(r["home_team"])
        out.add(r["away_team"])
    return out


def opponents_on(conn: sqlite3.Connection, day: str | date, season: str) -> dict[str, str]:
    """team -> the club it faces on `day`. Empty for teams that are idle."""
    rows = conn.execute(
        "SELECT home_team, away_team FROM nhl_schedule "
        "WHERE season = ? AND game_type = ? AND game_date = ?",
        (season, REGULAR_SEASON, _d(day).isoformat()),
    ).fetchall()
    out: dict[str, str] = {}
    for r in rows:
        out[r["home_team"]] = r["away_team"]
        out[r["away_team"]] = r["home_team"]
    return out


def team_game_dates(
    conn: sqlite3.Connection, team: str, start: str | date, end: str | date, season: str
) -> list[str]:
    """Dates `team` plays in [start, end], inclusive, ascending."""
    rows = conn.execute(
        "SELECT game_date FROM nhl_schedule "
        "WHERE season = ? AND game_type = ? AND game_date BETWEEN ? AND ? "
        "  AND (home_team = ? OR away_team = ?) "
        "ORDER BY game_date",
        (season, REGULAR_SEASON, _d(start).isoformat(), _d(end).isoformat(), team, team),
    ).fetchall()
    return [r["game_date"] for r in rows]


def games_by_team(
    conn: sqlite3.Connection, start: str | date, end: str | date, season: str
) -> dict[str, int]:
    """team -> number of games in [start, end]. One query for all 32 clubs."""
    rows = conn.execute(
        "SELECT home_team, away_team FROM nhl_schedule "
        "WHERE season = ? AND game_type = ? AND game_date BETWEEN ? AND ?",
        (season, REGULAR_SEASON, _d(start).isoformat(), _d(end).isoformat()),
    ).fetchall()
    out: dict[str, int] = {}
    for r in rows:
        out[r["home_team"]] = out.get(r["home_team"], 0) + 1
        out[r["away_team"]] = out.get(r["away_team"], 0) + 1
    return out


def back_to_back(
    conn: sqlite3.Connection, team: str, day: str | date, season: str
) -> tuple[bool, bool]:
    """(played yesterday, plays tomorrow) for `team` around `day`.

    Goalie workload's clearest schedule signal: a club on the second half of a
    back-to-back rarely starts the same goalie it started the night before.
    """
    d = _d(day)
    prev, nxt = d - timedelta(days=1), d + timedelta(days=1)
    near = set(team_game_dates(conn, team, prev, nxt, season))
    return (prev.isoformat() in near, nxt.isoformat() in near)


def season_dates(conn: sqlite3.Connection, season: str) -> tuple[str, str]:
    """First and last regular-season game dates. Raises if the season is unsynced."""
    row = conn.execute(
        "SELECT MIN(game_date) AS lo, MAX(game_date) AS hi FROM nhl_schedule "
        "WHERE season = ? AND game_type = ?",
        (season, REGULAR_SEASON),
    ).fetchone()
    if row is None or row["lo"] is None:
        raise CalendarError(
            f"no regular-season schedule for {season}; run `ppilot data sync --schedule-only`"
        )
    return row["lo"], row["hi"]


class CalendarError(RuntimeError):
    """The schedule cannot answer the question asked of it."""
