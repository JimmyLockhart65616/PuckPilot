"""Who is starting in goal tonight, in descending order of how much we know.

The bench-regret replay measured what goalie information is worth: an optimizer
with *perfect* starting-goalie data captured 92.6% of the hindsight ceiling and
one with announcements 90% accurate captured 92.3%, against 80.3% for
set-and-forget. That 0.3pt gap was read as "goalie precision barely matters".

Measuring the model below says the reading was too comfortable. A schedule-and-
workload model with no announcement at all tops out near **62%**, not 90%, and
nothing tried moved it far:

    trailing workload share only                     59.0%
    strict alternation (never start twice running)   53.0%   <- tandems do not alternate
    share, demoting the previous starter             61.5%   <- adopted
    (same, demoting only on a back-to-back)          60.6%

So the honest position is that 90% is announcement territory and this is the
floor, roughly 30 points below it. It is still the right floor - it needs no
external source, it works on opening night, and workload is the most repeatable
goalie signal there is (games/starts repeat at 0.53 year over year against 0.13
for save percentage). But a real announcement feed is worth chasing, and
`ChainedGoalieSource` exists so one can be dropped in front of this without
touching anything downstream.

All sources implement `data.goalies.GoalieStartSource`: one method returning
P(start) per NHL player id for a date. The optimizer weights value by that
probability, so a source may be confident (1.0/0.0) or hedged (0.62).
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable

# Team games of history to weigh. Swept over 2025-26 and confirmed on 2024-25
# and 2023-24; 10 was best or tied-best in all three, and the plateau across
# 10-20 games is broad enough that the exact value is not load-bearing.
TRAILING_TEAM_GAMES = 10

# Multiplicative demotion for whoever started the team's previous game. 0.5 was
# best on all three seasons; 1.0 (no demotion) costs 2-4 points of accuracy,
# and going below 0.35 costs more than it gains - starters ride, so a model
# that insists on alternating is worse than one that ignores rest entirely.
PREVIOUS_STARTER_DAMPING = 0.5

# Below this, call it "not starting" rather than a long shot: the optimizer
# would otherwise hold a slot open on a 3% chance instead of filling it.
MIN_MEANINGFUL_P = 0.05


class TrailingStartShareSource:
    """P(start) from who has been starting, demoting the previous starter.

    Indexed eagerly for the season, like `HindsightGoalieSource`, because the
    per-date form of this was a 30-day window scan per call.

    `fallback_season` covers opening weeks, when the current season has too few
    games to say anything: last season's share is a better prior than nothing.
    It is consulted only for teams the current season cannot answer for.
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        season: str,
        fallback_season: str | None = None,
        trailing_games: int = TRAILING_TEAM_GAMES,
        previous_damping: float = PREVIOUS_STARTER_DAMPING,
    ):
        self.season = season
        self.trailing_games = trailing_games
        self.previous_damping = previous_damping
        self._starts = _season_starts(conn, season)
        self._team_dates = _team_dates(self._starts)
        self._fallback = _season_starts(conn, fallback_season) if fallback_season else {}
        self._fallback_dates = _team_dates(self._fallback)
        self._schedule = _scheduled_teams(conn, season)
        self._cache: dict[str, dict[int, float]] = {}

    def starts(self, date: str) -> dict[int, float]:
        if date in self._cache:
            return self._cache[date]
        out: dict[int, float] = {}
        for team in self._schedule.get(date, ()):
            out.update(self._team_probabilities(team, date))
        self._cache[date] = out
        return out

    def _team_probabilities(self, team: str, date: str) -> dict[int, float]:
        history = [d for d in self._team_dates.get(team, ()) if d < date]
        starts, previous = self._starts, None
        if not history and self._fallback_dates.get(team):
            history = list(self._fallback_dates[team])
            starts = self._fallback
        if not history:
            return {}
        previous = starts.get((history[-1], team))

        counts: dict[int, int] = {}
        for day in history[-self.trailing_games :]:
            pid = starts.get((day, team))
            if pid:
                counts[pid] = counts.get(pid, 0) + 1
        if not counts:
            return {}

        weights = {
            pid: n * (self.previous_damping if pid == previous else 1.0)
            for pid, n in counts.items()
        }
        total = sum(weights.values())
        if total <= 0:
            return {}
        return {
            pid: round(w / total, 4) for pid, w in weights.items() if w / total >= MIN_MEANINGFUL_P
        }


class ChainedGoalieSource:
    """The first source with an opinion about a goalie wins, per date.

    Lets a confirmed announcement override the workload model for the goalies
    it covers while the model still answers for everyone else. A source that
    knows nothing today returns {} and costs nothing, and one that raises is
    skipped rather than blinding the rest - a dead feed should degrade to the
    floor, not to an empty lineup.
    """

    def __init__(self, *sources):
        self.sources = [s for s in sources if s is not None]

    def starts(self, date: str) -> dict[int, float]:
        out: dict[int, float] = {}
        for src in reversed(self.sources):  # later sources are the weaker prior
            try:
                out.update(src.starts(date))
            except Exception:  # noqa: BLE001
                continue
        return out


class StaticGoalieSource:
    """A fixed answer: for tests, and for a starter confirmed by hand."""

    def __init__(self, by_date: dict[str, dict[int, float]] | None = None):
        self.by_date = by_date or {}

    def starts(self, date: str) -> dict[int, float]:
        return dict(self.by_date.get(date, {}))


# -- indexing ---------------------------------------------------------------


def _season_starts(conn: sqlite3.Connection, season: str | None) -> dict[tuple[str, str], int]:
    """(date, team) -> the goalie who started. Relief appearances are not starts."""
    if not season:
        return {}
    rows = conn.execute(
        "SELECT l.game_date, l.team_abbrev, l.player_id, l.stats_json FROM nhl_game_logs l "
        "JOIN nhl_players p ON p.player_id = l.player_id "
        "WHERE l.season = ? AND p.position = 'G'",
        (season,),
    ).fetchall()
    out: dict[tuple[str, str], int] = {}
    for r in rows:
        if not r["team_abbrev"]:
            continue
        try:
            if int(json.loads(r["stats_json"]).get("gamesStarted", 0) or 0) == 1:
                out[(r["game_date"], r["team_abbrev"])] = r["player_id"]
        except (ValueError, TypeError):
            continue
    return out


def _team_dates(starts: dict[tuple[str, str], int]) -> dict[str, list[str]]:
    out: dict[str, set[str]] = {}
    for date, team in starts:
        out.setdefault(team, set()).add(date)
    return {t: sorted(d) for t, d in out.items()}


def _scheduled_teams(conn: sqlite3.Connection, season: str) -> dict[str, set[str]]:
    """date -> teams with a regular-season game. Includes days not yet played."""
    rows = conn.execute(
        "SELECT game_date, home_team, away_team FROM nhl_schedule "
        "WHERE season = ? AND game_type = 2",
        (season,),
    ).fetchall()
    out: dict[str, set[str]] = {}
    for r in rows:
        out.setdefault(r["game_date"], set()).update({r["home_team"], r["away_team"]})
    return out


def probable_starters(
    source, date: str, among: Iterable[int] | None = None
) -> list[tuple[int, float]]:
    """(goalie, P(start)) for a date, most likely first, optionally filtered."""
    p = source.starts(date)
    if among is not None:
        keep = set(among)
        p = {k: v for k, v in p.items() if k in keep}
    return sorted(p.items(), key=lambda kv: -kv[1])
