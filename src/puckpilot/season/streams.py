"""Goalie starts a team will add during the week: the stream.

A week's projection is the roster as it stands, so every goalie a manager adds
after it is volume the projection never saw. Measured live (`season g3`, weeks
1-7 of 2026-27, 84 team-weeks): goalies added in-week made +10.8 saves a
team-week, about a third of the league's whole saves miss. The opponent's
adds are not ours to decide and will happen whatever we do; pricing a week as
if they will not prices every goalie category as closer than it is.

`measure` reads, from weeks already played, how many starts a team-week got
from goalies added during it and what such a start was worth (shots faced,
save %, wins), from `league_transactions` and the game logs - per team, shrunk
toward the league's rate. `GoalieStream.line` turns that into one
pseudo-goalie's expected starts for the days left: adds arrive through the
week and each starts games only after it is made, so the starts still to come
from adds not yet made scale as (days left / days in the week) squared.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass

# The pseudo-players the stream is priced as, one per side; never a real id.
THEIR_STREAM = -2
OUR_STREAM = -1

# Team-weeks of the league's rate a team's own rate is shrunk toward.
SHRINK_WEEKS = 3.0


@dataclass(frozen=True)
class GoalieStream:
    """Starts a team-week gets from goalies added during it, and each start's line."""

    starts: float  # a team-week
    shots: float  # a start
    save_pct: float
    p_win: float
    team_weeks: int  # how many were measured

    def expected(self, days_left: int, days_in_week: int) -> float:
        """Starts still to come from adds not yet made."""
        if days_in_week <= 0 or days_left <= 0:
            return 0.0
        return self.starts * min(days_left / days_in_week, 1.0) ** 2

    def rates(self) -> dict[str, float]:
        """A per-start line in the projection's component keys."""
        saves = self.shots * self.save_pct
        return {
            "saves": saves,
            "shots_against": self.shots,
            "goals_against": self.shots - saves,
            "wins": self.p_win,
            "toi_hours": 1.0,
        }

    def chances(self, days_left: int, days_in_week: int) -> list[float]:
        """The expected starts as yes/no chances of at most 1 each."""
        e = self.expected(days_left, days_in_week)
        if e <= 0:
            return []
        n = int(-(-e // 1))  # ceil
        return [e / n] * n


@dataclass(frozen=True)
class _Added:
    team: str
    starts: int
    shots: float
    saves: float
    wins: int


def _added(conn: sqlite3.Connection, league_key: str, season: str, week) -> list[_Added]:
    """Each goalie added during `week`, with what he did from the add to its end."""
    out: list[_Added] = []
    for team, pid, at in conn.execute(
        "SELECT team_key, nhl_player_id, at FROM league_transactions WHERE league_key = ? "
        "AND action = 'add' AND positions LIKE '%G%' AND nhl_player_id IS NOT NULL "
        "AND substr(at, 1, 10) BETWEEN ? AND ?",
        (league_key, week.start, week.end),
    ):
        starts = wins = 0
        shots = saves = 0.0
        for (stats_json,) in conn.execute(
            "SELECT stats_json FROM nhl_game_logs WHERE player_id = ? AND season = ? "
            "AND game_type = 2 AND game_date BETWEEN ? AND ?",
            (pid, season, at[:10], week.end),
        ):
            s = json.loads(stats_json)
            if int(s.get("gamesStarted", 0) or 0) != 1:
                continue
            starts += 1
            sa = float(s.get("shotsAgainst", 0) or 0)
            shots += sa
            saves += sa - float(s.get("goalsAgainst", 0) or 0)
            wins += s.get("decision") == "W"
        out.append(_Added(team, starts, shots, saves, wins))
    return out


def added_by_week(conn: sqlite3.Connection, league_key: str, season: str, weeks) -> dict:
    """week number -> the goalies added during it, with what each did after."""
    return {w.number: _added(conn, league_key, season, w) for w in weeks}


def measure(
    added: dict,
    weeks: list[int],
    teams: list[str],
    team: str | None = None,
    shrink_weeks: float = SHRINK_WEEKS,
) -> GoalieStream | None:
    """The stream over `weeks` (numbers, all played; from `added_by_week`).

    `team` given: his own starts a week, shrunk toward the league's by
    `shrink_weeks` team-weeks; the line a start is worth stays the league's
    (too few starts per team to say more). None when nothing was measured.
    """
    if not weeks or not teams:
        return None
    rows = [a for w in weeks for a in added.get(w, [])]
    team_weeks = len(weeks) * len(teams)
    starts = sum(a.starts for a in rows)
    rate = starts / team_weeks
    if team is not None:
        own = sum(a.starts for a in rows if a.team == team)
        rate = (own + shrink_weeks * rate) / (len(weeks) + shrink_weeks)
    shots = sum(a.shots for a in rows)
    if starts == 0 or shots <= 0:
        return GoalieStream(0.0, 0.0, 0.0, 0.0, team_weeks)
    return GoalieStream(
        starts=rate,
        shots=shots / starts,
        save_pct=sum(a.saves for a in rows) / shots,
        p_win=sum(a.wins for a in rows) / starts,
        team_weeks=team_weeks,
    )
