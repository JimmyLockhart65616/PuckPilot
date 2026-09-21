"""Getting the live picture out of Yahoo and into the database.

Everything here is a read. The session this drives is read-only by construction
and a test asserts it stays that way; acting on what these reads produce is a
separate, gated path.

The split between this module and `settings`/`roster` is deliberate: those two
parse payloads and have no idea where they came from, so they can be tested
against fixtures with no browser anywhere. This one knows about sessions and
SQLite and is the part that needs a live login.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from datetime import UTC, datetime

from puckpilot.season.roster import TeamRoster, parse_roster
from puckpilot.season.settings import LeagueRuntime, SettingsError, Week
from puckpilot.yahoo.session import flatten

Progress = Callable[[str], None]


def _noop(_: str) -> None:
    pass


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class FetchError(RuntimeError):
    """Yahoo could not tell us something we need."""


# -- discovery --------------------------------------------------------------


def discover_team_key(session, league_key: str) -> str:
    """The team the logged-in user owns in this league.

    Preferred over configuring a team key by hand, because a wrong one reads
    and eventually acts on somebody else's roster.
    """
    for team in session.teams(league_key):
        if str(team.get("is_owned_by_current_login", "0")) == "1":
            key = str(team.get("team_key", ""))
            if key:
                return key
    raise FetchError(
        f"the logged-in Yahoo user does not own a team in {league_key}. "
        f"Set team_key in the manager config, or log in as the right user."
    )


# -- the week calendar ------------------------------------------------------


def fetch_matchups(session, team_key: str) -> list:
    """Every scheduled matchup, parsed structurally.

    `session.matchups` flattens, which merges both teams of a matchup into one
    dict and so loses the opponent - the thing a weekly plan is about.
    """
    from puckpilot.season.matchups import parse_matchups

    return parse_matchups(session.get(f"team/{team_key}/matchups"), our_team_key=team_key)


def fetch_runtime(
    session, league_key: str, team_key: str = "", progress: Progress = _noop
) -> LeagueRuntime:
    """The league's own rules plus its real week calendar."""
    progress(f"settings for {league_key} ...")
    flat = flatten(session.get(f"league/{league_key}/settings"))
    flat.setdefault("league_key", league_key)

    weeks: tuple[Week, ...] = ()
    key = team_key or ""
    if not key:
        try:
            key = discover_team_key(session, league_key)
        except FetchError:
            key = ""
    if key:
        progress(f"week calendar from {key} ...")
        try:
            from puckpilot.season.matchups import weeks_of

            weeks = weeks_of(fetch_matchups(session, key))
        except Exception as e:  # noqa: BLE001 - a missing calendar must not be fatal
            progress(f"  could not read the week calendar: {e}")

    runtime = LeagueRuntime.from_payload(flat, weeks=weeks, fetched_at=_now())
    if not runtime.weeks:
        progress(
            "  WARNING: no week calendar. Weekly numbers are unavailable until "
            "this is fetched; they are never computed."
        )
    return runtime


# -- the roster -------------------------------------------------------------


def fetch_roster(
    session,
    team_key: str,
    date: str | None = None,
    player_map: dict[str, int] | None = None,
) -> TeamRoster:
    """A team's roster for a day, parsed structurally.

    Goes through `session.get` rather than `session.roster` on purpose: the
    latter flattens, and flattening a roster entry reads `is_keeper.status` as
    the player's injury status.
    """
    path = f"team/{team_key}/roster"
    if date:
        path += f";date={date}"
    return parse_roster(session.get(path), team_key=team_key, player_map=player_map)


def fetch_pool(
    session,
    league_key: str,
    status: str = "FA",
    limit: int = 300,
    extra: str = "sort=AR",
    progress: Progress = _noop,
) -> list[dict]:
    """The free-agent (or waiver) pool, paged.

    `status=FA` is unclaimed, `status=W` is on waivers - the distinction that
    decides whether a target is a race or a claim you can place and sleep on.
    """
    out: list[dict] = []
    start = 0
    page = 25
    while start < limit:
        filt = f"status={status};{extra}" if extra else f"status={status}"
        chunk = session.players(league_key, start=start, count=page, extra=filt)
        if not chunk:
            break
        out.extend(chunk)
        progress(f"  {status}: {len(out)}")
        if len(chunk) < page:
            break
        start += page
    return out[:limit]


# -- persistence ------------------------------------------------------------


def save_runtime(conn: sqlite3.Connection, runtime: LeagueRuntime) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO yahoo_league_runtime "
        "(league_key, settings_json, weeks_json, fetched_at) VALUES (?, ?, ?, ?)",
        (
            runtime.league_key,
            json.dumps(_runtime_row(runtime)),
            json.dumps([[w.number, w.start, w.end] for w in runtime.weeks]),
            runtime.fetched_at or _now(),
        ),
    )
    conn.commit()


def load_runtime(conn: sqlite3.Connection, league_key: str) -> LeagueRuntime | None:
    row = conn.execute(
        "SELECT settings_json, weeks_json, fetched_at FROM yahoo_league_runtime "
        "WHERE league_key = ?",
        (league_key,),
    ).fetchone()
    if row is None:
        return None
    data = json.loads(row["settings_json"])
    weeks = tuple(Week(int(n), s, e) for n, s, e in json.loads(row["weeks_json"]))
    data["weeks"] = weeks
    data["fetched_at"] = row["fetched_at"]
    slots = data.pop("slots")
    from puckpilot.season.settings import RosterSlot

    data["slots"] = tuple(RosterSlot(p, int(c), bool(s)) for p, c, s in slots)
    return LeagueRuntime(**data)


def _runtime_row(runtime: LeagueRuntime) -> dict:
    d = {
        f: getattr(runtime, f)
        for f in LeagueRuntime.__dataclass_fields__
        if f not in ("weeks", "fetched_at")
    }
    d["slots"] = [[s.position, s.count, s.starting] for s in runtime.slots]
    return d


def save_roster(conn: sqlite3.Connection, manager: str, roster: TeamRoster) -> int:
    """Record the day's roster. Returns rows written."""
    rows = [
        (
            manager,
            roster.league_key,
            roster.team_key,
            roster.date,
            p.player_key,
            p.nhl_player_id,
            p.name,
            p.team,
            p.selected_slot,
            ",".join(sorted(p.yahoo_eligible)),
            p.status,
            p.injury_note,
            int(p.is_editable),
        )
        for p in roster.players
    ]
    conn.executemany(
        "INSERT OR REPLACE INTO yahoo_roster_snapshots "
        "(manager, league_key, team_key, date, player_key, nhl_player_id, name, "
        " team_abbrev, selected_slot, eligible, status, injury_note, is_editable) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    return len(rows)


def require_runtime(conn: sqlite3.Connection, league_key: str) -> LeagueRuntime:
    rt = load_runtime(conn, league_key)
    if rt is None:
        raise SettingsError(
            f"no cached league settings for {league_key}; run `ppilot season settings --refresh`"
        )
    return rt


def save_pool_if_any(conn: sqlite3.Connection, league_key: str, date: str, players) -> int:
    """Snapshot the pool, tolerating an empty one.

    Yahoo's own `percent_owned.delta` is a week-over-week figure, so it cannot
    see a Tuesday and resets when the week rolls. Our own daily rows can do
    both, and they are the only record that survives that reset.
    """
    if not players:
        return 0
    from puckpilot.season.pool import save_pool

    return save_pool(conn, league_key, date, players)
