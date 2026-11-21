"""Announced starting goalies: the platform's call on who starts tonight.

The start model (`goalies.TrailingStartShareSource`) is history only - the
share of a club's last ten starts - and names tonight's starter about 70% of
the time. Clubs announce, and Yahoo's fantasy API carries the call on each
goalie for the date it covers (probed 2026-11-21, three snapshots, one game
day):

    "starting_status": [{"coverage_type": "date", "date": "D"}, {"is_starting": 1}]

- Absent means no call yet, NOT "not starting". When a club is called, both of
  its listed goalies carry it - 1 for the starter, 0 for the partner - so a 0
  is information. Both at 0 means a third goalie starts.
- One value only: no confirmed/projected split; no flag changed between reads.
- Coverage grows through the day (8, 18, 23 of 28 playing clubs at noon, 14:23
  and 16:47 ET); most evening clubs were called 4.5-10 h before puck drop, some
  only at or after it. So a call supplements the model and never replaces it.
- It is per player, so free agents carry it too; ask by player_keys, because a
  ranked page can leave a club's partner out.

`fetch_calls` reads every mapped goalie in the run's own read session;
`save_calls` keeps every reading (`goalie_start_calls`), flagged or not, so the
accuracy and the lead time are measured from what was seen and when
(`check`); `AnnouncedStartSource` answers from the latest call, for chaining
in front of the model (`goalies.ChainedGoalieSource`).
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from puckpilot.season.pool import PAGE, parse_player
from puckpilot.season.roster import _fields

# P(a goalie called to start does not) and P(one called to sit starts). Neither
# is measured yet: placeholders until `check` has a few hundred calls with
# their games played (`ppilot season starters --check`).
CALL_MISS = 0.03
CALL_SURPRISE = 0.03

# How the run uses the calls (`authority.lineup.announced_starters`): "off"
# reads nothing; "record" reads and stores them and changes no decision - the
# data `check` needs; "use" also chains them in front of the start model.
ANNOUNCED_MODES = ("off", "record", "use")


@dataclass(frozen=True)
class Call:
    """One goalie as one read saw him."""

    player_key: str
    player_id: int | None
    team: str  # NHL abbreviation
    game_date: str  # the date the call covers, or the date read for when there is none
    is_starting: bool | None  # None: no call yet
    status: str  # Yahoo's tag: "", "O", "IR", "IR-LT", "DTD", "NA", ...


def parse_call(entry: list[Any], date: str, player_map: Mapping[str, int] | None = None) -> Call:
    """A `player` entry read with `out=starting_status`."""
    p = parse_player(entry, player_map)
    raw = _fields(entry[1:]).get("starting_status")
    f = _fields(raw) if isinstance(raw, list) else (raw if isinstance(raw, dict) else {})
    flag = f.get("is_starting")
    return Call(
        player_key=p.player_key,
        player_id=p.nhl_player_id,
        team=p.team,
        game_date=str(f.get("date") or date),
        is_starting=None if flag is None else bool(int(flag)),
        status=p.status,
    )


def fetch_calls(
    session,
    league_key: str,
    player_keys: list[str],
    date: str,
    player_map: Mapping[str, int] | None = None,
) -> list[Call]:
    """The named goalies' calls, `PAGE` keys a request."""
    out: list[Call] = []
    for start in range(0, len(player_keys), PAGE):
        keys = ",".join(player_keys[start : start + PAGE])
        body = session.get(f"league/{league_key}/players;player_keys={keys};out=starting_status")
        lg = body["fantasy_content"]["league"]
        node = next((x["players"] for x in lg if isinstance(x, dict) and "players" in x), None)
        if not isinstance(node, dict):
            continue
        for i in range(int(node.get("count", 0))):
            entry = node.get(str(i), {}).get("player")
            if entry:
                out.append(parse_call(entry, date, player_map))
    return out


def goalie_keys(conn: sqlite3.Connection, league_key: str, seasons: tuple[str, ...]) -> list[str]:
    """The league's mapped goalies who have played in any of `seasons`.

    Every goalie a call could matter for - rostered or free - and none of the
    long tail who never dress, which would only cost requests.
    """
    marks = ",".join("?" * len(seasons))
    rows = conn.execute(
        "SELECT m.player_key FROM yahoo_player_map m WHERE m.league_key = ?"
        " AND m.nhl_player_id IS NOT NULL AND ',' || m.positions || ',' LIKE '%,G,%'"
        " AND EXISTS (SELECT 1 FROM nhl_game_logs g WHERE g.player_id = m.nhl_player_id"
        f" AND g.season IN ({marks})) ORDER BY m.player_key",
        (league_key, *seasons),
    )
    return [r[0] for r in rows]


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def save_calls(conn: sqlite3.Connection, calls: list[Call], read_at: str | None = None) -> int:
    """Keep one read; every goalie in it, called or not. Returns rows written."""
    at = read_at or _now()
    conn.executemany(
        "INSERT OR REPLACE INTO goalie_start_calls"
        " (game_date, player_key, player_id, team, is_starting, status, read_at)"
        " VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            (
                c.game_date,
                c.player_key,
                c.player_id,
                c.team,
                None if c.is_starting is None else int(c.is_starting),
                c.status,
                at,
            )
            for c in calls
        ],
    )
    conn.commit()
    return len(calls)


def latest_calls(
    conn: sqlite3.Connection, date: str, as_of: str | None = None
) -> dict[int, tuple[str, bool | None, str]]:
    """player_id -> (team, the latest call, the latest status) for `date`.

    A read that found no call does not erase an earlier one: Yahoo's flag is
    absent before the call, and a call once made was never seen withdrawn.
    `as_of` (a UTC ISO time) leaves out reads after it.
    """
    rows = conn.execute(
        "SELECT player_id, team, is_starting, status, read_at FROM goalie_start_calls"
        " WHERE game_date = ? AND player_id IS NOT NULL"
        + (" AND read_at <= ?" if as_of else "")
        + " ORDER BY read_at",
        (date, as_of) if as_of else (date,),
    )
    out: dict[int, tuple[str, bool | None, str]] = {}
    for pid, team, flag, status, _at in rows:
        prev = out.get(pid)
        call = bool(flag) if flag is not None else (prev[1] if prev else None)
        out[pid] = (team or "", call, status or "")
    return out


class AnnouncedStartSource:
    """P(start) from the latest call, for the goalies of clubs that were called.

    A goalie called to start gets 1 - `miss`; one called to sit gets
    `surprise`; on a called club every other goalie read gets `surprise` too,
    so a third goalie cannot keep the model's share - and when both listed
    goalies are 0, nobody on the club is the starter we know of. A club with no
    call says nothing, so the next source in the chain answers for it.
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        miss: float = CALL_MISS,
        surprise: float = CALL_SURPRISE,
        as_of: str | None = None,
    ):
        self.conn = conn
        self.miss = miss
        self.surprise = surprise
        self.as_of = as_of

    def starts(self, date: str) -> dict[int, float]:
        calls = latest_calls(self.conn, date, self.as_of)
        called = {team for team, call, _ in calls.values() if call is not None and team}
        out: dict[int, float] = {}
        for pid, (team, call, _status) in calls.items():
            if call is True:
                out[pid] = 1.0 - self.miss
            elif call is False or team in called:
                out[pid] = self.surprise
        return out


def statuses(conn: sqlite3.Connection, date: str) -> dict[int, str]:
    """player_id -> Yahoo's latest tag on him, from the calls read for `date`."""
    return {pid: status for pid, (_t, _c, status) in latest_calls(conn, date).items()}


# -- how good the calls are --------------------------------------------------


@dataclass(frozen=True)
class CallCheck:
    """Calls read before puck drop, against who started."""

    called_start: int  # goalie-games called 1 before puck drop
    started_when_called: int
    called_sit: int
    started_when_sat: int
    starts: int  # goalie starts in the games covered
    starts_called: int  # of those, called 1 before puck drop
    lead_hours: list[float]  # first call v puck drop, per called start
    # Brier score on the called goalie-games: the calls at `miss`/`surprise`,
    # and the start model as of that morning - the comparison that decides
    # whether the calls should be used. None without a model.
    brier_calls: float | None = None
    brier_model: float | None = None

    @property
    def miss(self) -> float | None:
        if not self.called_start:
            return None
        return 1.0 - self.started_when_called / self.called_start

    @property
    def surprise(self) -> float | None:
        return self.started_when_sat / self.called_sit if self.called_sit else None

    @property
    def coverage(self) -> float | None:
        return self.starts_called / self.starts if self.starts else None

    def text(self) -> str:
        def pct(x):
            return "-" if x is None else f"{x:.1%}"

        lead = sorted(self.lead_hours)
        mid = f"{lead[len(lead) // 2]:.1f} h" if lead else "-"
        return "\n".join(
            [
                "Announced starters, read before puck drop, v who started:",
                f"  called to start   {self.called_start:4d}  did not start {pct(self.miss)}",
                f"  called to sit     {self.called_sit:4d}  started anyway {pct(self.surprise)}",
                f"  starts covered    {self.starts_called:4d} of {self.starts}"
                f"  ({pct(self.coverage)} called before puck drop)",
                f"  first call before puck drop, median {mid}",
                *(
                    [
                        f"  Brier on the called goalie-games: calls {self.brier_calls:.4f}"
                        f" v start model {self.brier_model:.4f}"
                    ]
                    if self.brier_calls is not None and self.brier_model is not None
                    else []
                ),
            ]
        )


def check(
    conn: sqlite3.Connection,
    since: str = "",
    until: str = "",
    model_on: Callable[[str], dict[int, float]] | None = None,
    miss: float = CALL_MISS,
    surprise: float = CALL_SURPRISE,
) -> CallCheck:
    """Score every stored call made before its game began against the game log.

    Only dates with a goalie log count; a call read after puck drop is not a
    forecast and is left out. `model_on(date)`, when given, is the start
    model's P(start) as of that morning, scored on the same goalie-games.
    """
    rows = conn.execute(
        "SELECT c.game_date, c.player_id, c.team, c.is_starting, c.read_at, s.start_time_utc"
        " FROM goalie_start_calls c JOIN nhl_schedule s ON s.game_date = c.game_date"
        " AND (s.home_team = c.team OR s.away_team = c.team)"
        " WHERE c.player_id IS NOT NULL AND c.is_starting IS NOT NULL"
        " AND c.game_date >= ? AND (? = '' OR c.game_date <= ?)",
        (since, until, until),
    ).fetchall()
    first: dict[tuple[str, int], tuple[int, str, str, str]] = {}
    for date, pid, team, flag, read_at, start_utc in rows:
        if not start_utc or _utc(read_at) >= _utc(start_utc):
            continue
        key = (date, pid)
        if key not in first or read_at < first[key][1]:
            first[key] = (flag, read_at, start_utc, team)
    dates = sorted({d for d, _ in first})
    started = _started(conn, dates)
    cs = sws = csit = ssat = 0
    lead: list[float] = []
    for (date, pid), (flag, read_at, start_utc, _team) in first.items():
        if date not in started:
            continue
        did = pid in started[date]
        if flag:
            cs += 1
            sws += did
            lead.append((_utc(start_utc) - _utc(read_at)).total_seconds() / 3600)
        else:
            csit += 1
            ssat += did
    starts = sum(len(v) for v in started.values())
    called = sum(1 for (d, pid), v in first.items() if v[0] and pid in started.get(d, ()))
    b_calls = b_model = None
    if model_on is not None:
        scored = [(d, pid, v[0]) for (d, pid), v in first.items() if d in started]
        if scored:
            seen: dict[str, dict[int, float]] = {}
            sq_c = sq_m = 0.0
            for d, pid, flag in scored:
                did = float(pid in started[d])
                p_call = 1.0 - miss if flag else surprise
                if d not in seen:
                    seen[d] = model_on(d)
                sq_c += (p_call - did) ** 2
                sq_m += (float(seen[d].get(pid, 0.0)) - did) ** 2
            b_calls, b_model = sq_c / len(scored), sq_m / len(scored)
    return CallCheck(cs, sws, csit, ssat, starts, called, lead, b_calls, b_model)


def _utc(s: str) -> datetime:
    t = datetime.fromisoformat(s.replace("Z", "+00:00"))
    return t if t.tzinfo else t.replace(tzinfo=UTC)


def _started(conn: sqlite3.Connection, dates: list[str]) -> dict[str, set[int]]:
    """date -> goalies who started, for dates with any goalie log."""
    if not dates:
        return {}
    marks = ",".join("?" * len(dates))
    out: dict[str, set[int]] = {}
    for date, pid in conn.execute(
        "SELECT game_date, player_id FROM nhl_game_logs WHERE game_date IN"
        f" ({marks}) AND json_extract(stats_json, '$.gamesStarted') = 1",
        dates,
    ):
        out.setdefault(date, set()).add(pid)
    return out
