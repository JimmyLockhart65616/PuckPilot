"""The schedule of who you play, and when the weeks actually start and end.

One request answers both questions a weekly plan needs, so they are parsed
together: the week boundaries (which must be fetched, because the season's
weeks are not uniformly seven days) and the opponent for each of them.

Structural parsing again. `flatten` recovers the week fields correctly here,
but it merges both teams of a matchup into one dict, so the opponent - the
whole point - is exactly what it loses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from puckpilot.season.roster import _fields
from puckpilot.season.settings import Week


@dataclass(frozen=True)
class Matchup:
    week: int
    start: str
    end: str
    opponent_key: str
    opponent_name: str
    is_playoffs: bool = False
    status: str = ""

    def as_week(self) -> Week:
        return Week(number=self.week, start=self.start, end=self.end)

    def contains(self, day: str) -> bool:
        return self.start <= day <= self.end


def _teams(node: Any) -> list[dict]:
    """The two teams of a matchup, as {team_key, name}."""
    if not isinstance(node, dict):
        return []
    teams = node.get("teams")
    if not isinstance(teams, dict):
        return []
    out = []
    for i in range(int(teams.get("count", 0))):
        entry = teams.get(str(i), {}).get("team")
        if not entry:
            continue
        core = _fields(entry[0]) if isinstance(entry[0], list) else _fields([entry[0]])
        out.append(
            {
                "team_key": str(core.get("team_key", "")),
                "name": str(core.get("name", "")),
            }
        )
    return out


def parse_matchups(payload: dict, our_team_key: str = "") -> list[Matchup]:
    """Every scheduled matchup for a team, ascending by week.

    A matchup we cannot read an opponent from still yields its week - a partial
    calendar beats a computed one, and `LeagueRuntime.week_of` says plainly
    when a date is not covered.
    """
    try:
        team = payload["fantasy_content"]["team"]
    except (KeyError, TypeError):
        return []
    node = next((x["matchups"] for x in team if isinstance(x, dict) and "matchups" in x), None)
    if not isinstance(node, dict):
        return []

    out: list[Matchup] = []
    for i in range(int(node.get("count", 0))):
        m = node.get(str(i), {}).get("matchup")
        if not isinstance(m, dict):
            continue
        try:
            week = int(m.get("week"))
        except (TypeError, ValueError):
            continue
        start, end = str(m.get("week_start", "")), str(m.get("week_end", ""))
        if not start or not end:
            continue

        opp_key = opp_name = ""
        sides = _teams(m.get("0"))
        if our_team_key:
            others = [t for t in sides if t["team_key"] != our_team_key]
            if len(others) == 1:
                opp_key, opp_name = others[0]["team_key"], others[0]["name"]
        elif len(sides) == 2:
            opp_key, opp_name = sides[1]["team_key"], sides[1]["name"]

        out.append(
            Matchup(
                week=week,
                start=start,
                end=end,
                opponent_key=opp_key,
                opponent_name=opp_name,
                is_playoffs=str(m.get("is_playoffs", "0")) == "1",
                status=str(m.get("status", "")),
            )
        )
    out.sort(key=lambda m: m.week)
    return out


def weeks_of(matchups: list[Matchup]) -> tuple[Week, ...]:
    seen: dict[int, Week] = {}
    for m in matchups:
        seen.setdefault(m.week, m.as_week())
    return tuple(seen[n] for n in sorted(seen))


def for_week(matchups: list[Matchup], week: int) -> Matchup | None:
    return next((m for m in matchups if m.week == week), None)


def for_date(matchups: list[Matchup], day: str) -> Matchup | None:
    return next((m for m in matchups if m.contains(day)), None)
