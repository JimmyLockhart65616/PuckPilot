"""The players you could add, and which way the league is moving on them.

Two things make this different from the draft-time player pool.

Ownership is the league's own, and Yahoo reports its trend directly:
`percent_owned` arrives as `{value, delta}` per week, so "who is being picked
up right now" needs no scraped research page and no differencing of our own
snapshots. We keep daily snapshots anyway, because a weekly delta cannot tell
you about a Tuesday, and because our own history is the only record that
survives the week rolling over.

And `ownership_type` separates the two cases that decide whether a target is
worth losing sleep over. A player on waivers is claimed with a request that
processes at the league's waiver time whether you are awake or not; a free
agent is first come, first served. That distinction is the whole answer to
"do I need to stay up for this".

Parsed structurally, not through `flatten`, for the same reason
`season.roster` is: a player entry nests objects that reuse generic key names.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from puckpilot.season.roster import _fields, _positions
from puckpilot.season.settings import OUT_STATUSES, YAHOO_TO_POS
from puckpilot.yahoo.playermap import _team

Progress = Callable[[str], None]

FREE_AGENT = "freeagents"
WAIVERS = "waivers"
PAGE = 25

# Yahoo's status filter -> the ownership it implies.
STATUS_OWNERSHIP = {"FA": FREE_AGENT, "W": WAIVERS}


@dataclass(frozen=True)
class PoolPlayer:
    player_key: str
    name: str
    team: str
    primary_position: str
    yahoo_eligible: frozenset[str]
    nhl_player_id: int | None = None
    status: str = ""
    injury_note: str = ""
    ownership_type: str = ""
    percent_owned: float = 0.0
    percent_owned_delta: float = 0.0

    @property
    def eligible(self) -> frozenset[str]:
        return frozenset(YAHOO_TO_POS[p] for p in self.yahoo_eligible if p in YAHOO_TO_POS)

    @property
    def position(self) -> str:
        return YAHOO_TO_POS.get(self.primary_position, self.primary_position)

    @property
    def is_out(self) -> bool:
        return self.status in OUT_STATUSES

    @property
    def on_waivers(self) -> bool:
        return self.ownership_type == WAIVERS

    @property
    def is_free_agent(self) -> bool:
        return self.ownership_type == FREE_AGENT

    def timing(self, waiver_days: int = 1) -> str:
        """Whether this one has to be raced for, in plain words."""
        if self.on_waivers:
            return (
                f"on waivers - put a claim in and go to bed; it processes in "
                f"{waiver_days} day(s) by priority, not by who is awake"
            )
        return "free agent - first come, first served, so this one is a race"


def parse_player(
    entry: list[Any],
    player_map: Mapping[str, int] | None = None,
    ownership_type: str = "",
) -> PoolPlayer:
    """One `player` entry from a pool response.

    `ownership_type` is stamped by the caller from the status it asked for.
    Yahoo will report it too, but only via a second subresource, and a query
    for `status=FA` cannot come back holding anything else - so asking twice
    would spend a request to learn what we already said.
    """
    core = _fields(entry[0]) if isinstance(entry[0], list) else _fields([entry[0]])
    tail = _fields(entry[1:])

    owned, delta = 0.0, 0.0
    po = tail.get("percent_owned")
    if isinstance(po, list):
        f = _fields(po)
        owned = _num(f.get("value"))
        delta = _num(f.get("delta"))
    elif isinstance(po, dict):
        owned = _num(po.get("value"))
        delta = _num(po.get("delta"))

    own = tail.get("ownership") or core.get("ownership") or {}
    key = str(core.get("player_key", ""))
    name = core.get("name") or {}
    return PoolPlayer(
        player_key=key,
        name=str(name.get("full", "")) if isinstance(name, dict) else str(name),
        team=_team(core.get("editorial_team_abbr")),
        primary_position=str(core.get("primary_position", core.get("display_position", ""))),
        yahoo_eligible=_positions(core.get("eligible_positions")),
        nhl_player_id=(player_map or {}).get(key),
        status=str(core.get("status", "") or ""),
        injury_note=str(core.get("injury_note", "") or ""),
        ownership_type=(
            str(own.get("ownership_type", "")) if isinstance(own, dict) and own else ownership_type
        ),
        percent_owned=owned,
        percent_owned_delta=delta,
    )


def _num(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def fetch_pool(
    session,
    league_key: str,
    status: str = "FA",
    limit: int = 300,
    player_map: Mapping[str, int] | None = None,
    progress: Progress = lambda _m: None,
) -> list[PoolPlayer]:
    """Available players with their ownership trend.

    `status=FA` is unclaimed, `status=W` is on waivers. Both are asked for with
    the ownership and percent-owned subresources, because the two facts that
    decide a pickup - how fast the league is moving and whether you have to
    race - are exactly those.
    """
    out: list[PoolPlayer] = []
    start = 0
    while start < limit:
        path = (
            f"league/{league_key}/players;status={status};sort=AR;"
            f"start={start};count={PAGE}/percent_owned"
        )
        chunk = _page(session, path, player_map, STATUS_OWNERSHIP.get(status, ""))
        if not chunk:
            break
        out.extend(chunk)
        progress(f"  {status}: {len(out)}")
        if len(chunk) < PAGE:
            break
        start += PAGE
    return out[:limit]


def _page(session, path: str, player_map, ownership_type: str = "") -> list[PoolPlayer]:
    lg = session.get(path)["fantasy_content"]["league"]
    node = next((x["players"] for x in lg if isinstance(x, dict) and "players" in x), None)
    if not isinstance(node, dict):
        return []
    out = []
    for i in range(int(node.get("count", 0))):
        entry = node.get(str(i), {}).get("player")
        if entry:
            out.append(parse_player(entry, player_map, ownership_type))
    return out


def save_pool(
    conn: sqlite3.Connection, league_key: str, date: str, players: list[PoolPlayer]
) -> int:
    """Record the pool as read on `date`, replacing any earlier read of that day.

    Replacing rather than merging: rows of players since picked up would
    otherwise linger as free agents. The old code also dated reads by the
    week's start - three future dates were already on file by the 28th - and a
    merge would have mixed those into the real day's pool.
    """
    conn.execute(
        "DELETE FROM yahoo_fa_snapshots WHERE league_key = ? AND date = ?", (league_key, date)
    )
    conn.executemany(
        "INSERT OR REPLACE INTO yahoo_fa_snapshots "
        "(league_key, date, player_key, nhl_player_id, name, team_abbrev, positions, "
        " ownership, percent_owned, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                league_key,
                date,
                p.player_key,
                p.nhl_player_id,
                p.name,
                p.team,
                ",".join(sorted(p.yahoo_eligible)),
                p.ownership_type,
                p.percent_owned,
                p.status,
            )
            for p in players
        ],
    )
    conn.commit()
    return len(players)


def load_pool(conn: sqlite3.Connection, league_key: str, on_or_before: str) -> list[PoolPlayer]:
    """The most recent saved pool, as of a day - for the runs that do not fetch one.

    Stored eligibility has no primary position, so the first startable one
    stands in: good enough for what a saved pool is used for (how far an add
    could move a category), never used to propose one.
    """
    row = conn.execute(
        "SELECT MAX(date) FROM yahoo_fa_snapshots WHERE league_key = ? AND date <= ?",
        (league_key, on_or_before),
    ).fetchone()
    if not row or not row[0]:
        return []
    out: list[PoolPlayer] = []
    for r in conn.execute(
        "SELECT * FROM yahoo_fa_snapshots WHERE league_key = ? AND date = ?",
        (league_key, row[0]),
    ):
        elig = frozenset(x for x in (r["positions"] or "").split(",") if x)
        primary = next((p for p in ("C", "LW", "RW", "D", "G") if p in elig), "")
        out.append(
            PoolPlayer(
                player_key=r["player_key"],
                name=r["name"],
                team=r["team_abbrev"] or "",
                primary_position=primary,
                yahoo_eligible=elig,
                nhl_player_id=r["nhl_player_id"],
                status=r["status"] or "",
                ownership_type=r["ownership"] or "",
                percent_owned=r["percent_owned"] or 0.0,
            )
        )
    return out


def rising(players: list[PoolPlayer], limit: int = 15) -> list[PoolPlayer]:
    """The league is picking these up. Yahoo's own weekly delta, biggest first."""
    moving = [p for p in players if p.percent_owned_delta > 0 and not p.is_out]
    return sorted(moving, key=lambda p: -p.percent_owned_delta)[:limit]


def our_own_trend(
    conn: sqlite3.Connection, league_key: str, date: str, back_days: int = 3
) -> dict[str, float]:
    """player_key -> change in percent owned over our own snapshots.

    Yahoo's `delta` is a week-over-week figure, so it cannot see a Tuesday, and
    it resets when the week rolls. Differencing what we stored does both.
    """
    rows = conn.execute(
        "SELECT player_key, date, percent_owned FROM yahoo_fa_snapshots "
        "WHERE league_key = ? AND date <= ? ORDER BY date",
        (league_key, date),
    ).fetchall()
    first: dict[str, float] = {}
    last: dict[str, float] = {}
    cutoff = _shift(date, -back_days)
    for r in rows:
        if r["date"] < cutoff:
            continue
        key = r["player_key"]
        first.setdefault(key, r["percent_owned"] or 0.0)
        last[key] = r["percent_owned"] or 0.0
    return {k: last[k] - first[k] for k in last if last[k] != first.get(k)}


def _shift(date: str, days: int) -> str:
    from datetime import date as _d
    from datetime import timedelta

    return (_d.fromisoformat(date) + timedelta(days=days)).isoformat()
