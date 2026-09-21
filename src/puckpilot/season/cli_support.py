"""Shared plumbing for the in-season commands.

Kept out of `cli.py` because every one of these steps is the same for the daily
lineup, the weekly plan and the page: resolve the manager, find the league, get
a read session, load the cached rules. Repeating it per command is how the
draft console and the mock loop drifted apart.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from datetime import date as _date
from pathlib import Path

from puckpilot.config import Settings
from puckpilot.season.manager import Manager, ManagerError, available, load_manager
from puckpilot.season.settings import LeagueRuntime

Progress = Callable[[str], None]


class SeasonCliError(RuntimeError):
    """Something the user has to fix before the command can run."""


def resolve_manager(name: str | None, settings: Settings | None = None) -> Manager:
    """The manager for this invocation.

    With one configured, `--manager` is optional; with several it is required,
    because acting on the wrong team is not a recoverable mistake.
    """
    s = settings or Settings()
    names = available(s)
    if name:
        return load_manager(name, s)
    if len(names) == 1:
        return load_manager(names[0], s)
    if not names:
        raise SeasonCliError(
            "no manager configured. Copy managers/example.toml to "
            "managers/<you>.toml and fill in your team."
        )
    raise SeasonCliError(f"--manager is required; configured: {', '.join(names)}")


def resolve_league_key(conn: sqlite3.Connection, manager: Manager) -> str:
    """The Yahoo league key, from the manager config or the player map."""
    if manager.league_key:
        return manager.league_key
    if manager.team_key:
        return manager.team_key.rsplit(".t.", 1)[0]
    from puckpilot.yahoo import playermap

    keys = playermap.mapped_league_keys(conn)
    if len(keys) == 1:
        return keys[0][0]
    if not keys:
        raise SeasonCliError(
            "no league key: set league_key in the manager config, or run "
            "`ppilot yahoo playermap --league-key <key>` first."
        )
    listed = ", ".join(k for k, *_ in keys)
    raise SeasonCliError(f"several leagues are mapped ({listed}); set league_key in the config.")


def open_db(manager: Manager) -> sqlite3.Connection:
    """Connect, and make sure the schema is current.

    `init_db` is idempotent `CREATE TABLE IF NOT EXISTS` plus an additive
    column migration, so this is cheap and self-healing. Without it an existing
    database - which is every database, since the in-season tables are new -
    fails on the first write with "no such table", after the Yahoo read has
    already happened.
    """
    from puckpilot.data import store

    conn = store.connect(manager.resolved_db())
    store.init_db(conn)
    return conn


def open_session(manager: Manager, settings: Settings | None = None):
    """A read-only Yahoo session on this manager's browser profile.

    `check_oauth` stays on. That guard is the thing that retires the browser
    fallback the day the official API is approved, and disabling it here would
    quietly keep a scheduled job on the fallback forever. The cost is that such
    a job stops on that day, so `run_session` turns it into an instruction
    rather than a traceback.
    """
    from puckpilot.yahoo.session import YahooSession

    s = settings or Settings()
    profile = manager.profile_dir or s._resolve(Path("secrets/chrome-profile"))
    if not profile.is_dir():
        raise SeasonCliError(
            f"no logged-in browser profile at {profile}. Set profile_dir in the "
            f"manager config and sign in to Yahoo there once."
        )
    _refuse_if_busy(profile)
    return YahooSession(user_data_dir=profile, headless=True)


def _refuse_if_busy(profile: Path) -> None:
    """Chrome will not open a profile twice, and the failure is otherwise opaque.

    Worth checking here rather than letting Playwright time out: an in-season
    job runs while the person may well have that profile open themselves.
    """
    from puckpilot.draft.capture import ProfileInUse, profile_is_busy

    if profile_is_busy(profile):
        raise ProfileInUse(
            f"{profile} is already open in another Chrome. Close it (or the "
            f"other PuckPilot command using it) and run again."
        )


def run_session(manager: Manager, work, settings: Settings | None = None):
    """Run `work(session)`, turning the fallback's retirement into an instruction.

    `YahooSession` refuses to start once OAuth answers 200, which is correct -
    the documented API should win the moment it is available - but to a
    scheduled job it looks like an unexplained failure on an ordinary morning.
    """
    from puckpilot.yahoo.session import FallbackNoLongerNeeded

    try:
        with open_session(manager, settings) as session:
            return work(session)
    except FallbackNoLongerNeeded as e:
        raise SeasonCliError(
            f"{e}\n"
            f"  Yahoo's official API now answers, so the browser fallback has "
            f"retired itself. Re-consent with `ppilot yahoo probe` and move these "
            f"commands onto the OAuth client."
        ) from e


def load_rules(conn: sqlite3.Connection, league_key: str) -> LeagueRuntime:
    from puckpilot.season.fetch import require_runtime

    return require_runtime(conn, league_key)


def today_str() -> str:
    return _date.today().isoformat()


def report(manager_error: ManagerError | SeasonCliError) -> int:
    print(f"puckpilot: {manager_error}")
    return 2
