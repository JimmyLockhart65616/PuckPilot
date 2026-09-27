"""The schedule of who you play, and when the weeks actually start and end.

One request answers both questions a weekly plan needs, so they are parsed
together: the week boundaries (which must be fetched, because the season's
weeks are not uniformly seven days) and the opponent for each of them.

Structural parsing again. `flatten` recovers the week fields correctly here,
but it merges both teams of a matchup into one dict, so the opponent - the
whole point - is exactly what it loses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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


# -- the live score ---------------------------------------------------------


@dataclass(frozen=True)
class TeamScore:
    """One side's week so far, as Yahoo reports it."""

    team_key: str
    name: str
    # stat_id -> value; None where Yahoo shows "-" (a save percentage with no
    # shots faced is undefined, not zero).
    stats: dict[int, float | None]
    remaining_games: int | None = None
    live_games: int | None = None
    completed_games: int | None = None


@dataclass(frozen=True)
class LiveMatchup:
    """The score of a week in progress: both sides' totals and who leads what.

    Yahoo sends this inside the same `team/{key}/matchups` response the week
    calendar is read from; for a season it was simply thrown away.
    """

    week: int
    status: str  # preevent / midevent / postevent
    ours: TeamScore
    theirs: TeamScore
    # stat_id -> "ours" / "theirs" / "tie", for the categories Yahoo adjudicates.
    winners: dict[int, str] = field(default_factory=dict)

    @property
    def started(self) -> bool:
        return self.status in ("midevent", "postevent")

    def banked(self, labels: dict[int, str], side: str = "ours") -> dict[str, float]:
        """label -> value for one side, using Yahoo's stat_id -> label map."""
        team = self.ours if side == "ours" else self.theirs
        return {labels[sid]: v for sid, v in team.stats.items() if sid in labels and v is not None}


def _value(v: Any) -> float | None:
    """Yahoo's stat strings: "14", ".915", "-" or "" when undefined."""
    if v in (None, "", "-"):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _score(entry: list) -> TeamScore:
    """One team entry of a matchup: metadata first, then stats blocks."""
    core = _fields(entry[0]) if isinstance(entry[0], list) else _fields([entry[0]])
    tail = _fields(entry[1:])
    stats: dict[int, float | None] = {}
    block = tail.get("team_stats")
    for item in (block or {}).get("stats", []) if isinstance(block, dict) else []:
        st = item.get("stat") if isinstance(item, dict) else None
        if isinstance(st, dict) and "stat_id" in st:
            try:
                stats[int(st["stat_id"])] = _value(st.get("value"))
            except (TypeError, ValueError):
                continue
    games = tail.get("team_remaining_games")
    total = games.get("total") if isinstance(games, dict) else None
    total = total if isinstance(total, dict) else {}

    def count(k: str) -> int | None:
        try:
            return int(total[k])
        except (KeyError, TypeError, ValueError):
            return None

    return TeamScore(
        team_key=str(core.get("team_key", "")),
        name=str(core.get("name", "")),
        stats=stats,
        remaining_games=count("remaining_games"),
        live_games=count("live_games"),
        completed_games=count("completed_games"),
    )


def parse_live(payload: dict, our_team_key: str, week: int | None = None) -> LiveMatchup | None:
    """The live score of one week (default: the first matchup in the response).

    Fields verified against Yahoo's documented matchup shape, not yet against a
    live week of this league - the first in-season fetch is kept raw in
    `matchup_snapshots` so this can be checked and re-parsed.
    """
    try:
        team = payload["fantasy_content"]["team"]
    except (KeyError, TypeError):
        return None
    node = next((x["matchups"] for x in team if isinstance(x, dict) and "matchups" in x), None)
    if not isinstance(node, dict):
        return None
    for i in range(int(node.get("count", 0))):
        m = node.get(str(i), {}).get("matchup")
        if not isinstance(m, dict):
            continue
        try:
            wk = int(m.get("week"))
        except (TypeError, ValueError):
            continue
        if week is not None and wk != week:
            continue
        teams = (m.get("0") or {}).get("teams")
        if not isinstance(teams, dict):
            return None
        sides = []
        for j in range(int(teams.get("count", 0))):
            entry = teams.get(str(j), {}).get("team")
            if isinstance(entry, list) and entry:
                sides.append(_score(entry))
        ours = next((s for s in sides if s.team_key == our_team_key), None)
        theirs = next((s for s in sides if s.team_key != our_team_key), None)
        if ours is None or theirs is None:
            return None
        winners: dict[int, str] = {}
        for w in m.get("stat_winners") or []:
            sw = w.get("stat_winner") if isinstance(w, dict) else None
            if not isinstance(sw, dict) or "stat_id" not in sw:
                continue
            sid = int(sw["stat_id"])
            if str(sw.get("is_tied", "0")) == "1":
                winners[sid] = "tie"
            elif sw.get("winner_team_key") == our_team_key:
                winners[sid] = "ours"
            elif sw.get("winner_team_key"):
                winners[sid] = "theirs"
        return LiveMatchup(
            week=wk,
            status=str(m.get("status", "")),
            ours=ours,
            theirs=theirs,
            winners=winners,
        )
    return None


def weeks_of(matchups: list[Matchup]) -> tuple[Week, ...]:
    seen: dict[int, Week] = {}
    for m in matchups:
        seen.setdefault(m.week, m.as_week())
    return tuple(seen[n] for n in sorted(seen))


def for_week(matchups: list[Matchup], week: int) -> Matchup | None:
    return next((m for m in matchups if m.week == week), None)


def for_date(matchups: list[Matchup], day: str) -> Matchup | None:
    return next((m for m in matchups if m.contains(day)), None)


def current_or_next(matchups: list[Matchup], day: str) -> Matchup | None:
    """The week `day` falls in, or the next one to start.

    Between seasons, and in the days before one opens, there is no current
    week - and "what is the plan for the week coming" is the most reasonable
    question anyone asks then.
    """
    now = for_date(matchups, day)
    if now is not None:
        return now
    return next((m for m in matchups if m.start >= day), None)
