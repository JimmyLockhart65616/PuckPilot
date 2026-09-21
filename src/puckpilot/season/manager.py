"""Who the tool is acting for.

The draft board spent a session discovering that `my_seat` belonged in a
parameter rather than baked into the object, because a second manager in the
same league wanted the same view from a different chair. In-season the same
problem is larger: two managers in one league have different rosters, different
opponents, different acquisition budgets and different Yahoo logins, and they
are rivals, so one must not see the other's plan.

So there is no global "my team" anywhere in this package. A `Manager` is passed
in, and everything - the database rows, the browser profile, the page keys, the
standing authority - is keyed by it.

The three layers a manager may use are deliberately separable:

    decide   needs Python and a Yahoo read session
    view     needs a URL and a key, and nothing installed
    act      needs Chrome, Playwright and that manager's own login

A manager on a locked-down machine takes the first two and executes by hand,
which is how the second seat consumed the draft board. `profile_dir` and the
actuator are optional by construction and never assumed.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from puckpilot.config import Settings
from puckpilot.league import DEFAULT_LEAGUE, LeagueConfig, load_league
from puckpilot.season.authority import Authority, AuthorityError


class ManagerError(RuntimeError):
    """The manager config is missing or unusable."""


@dataclass(frozen=True)
class PageConfig:
    """Where this manager's view is published, and the keys that guard it."""

    url: str = ""
    owner_key: str = ""
    guest_key: str = ""

    @property
    def publishes(self) -> bool:
        return bool(self.url)


@dataclass(frozen=True)
class Manager:
    name: str
    league: LeagueConfig = DEFAULT_LEAGUE
    league_key: str = ""
    team_key: str = ""
    profile_dir: Path | None = None
    db_path: Path | None = None
    authority: Authority = field(default_factory=Authority)
    page: PageConfig = field(default_factory=PageConfig)

    @property
    def can_act(self) -> bool:
        """Whether this manager has a browser profile to drive at all."""
        return self.profile_dir is not None

    def resolved_db(self) -> Path:
        return self.db_path or Settings().resolved_db_path

    def describe(self) -> str:
        team = self.team_key or "(discovered at run time)"
        lines = [
            f"manager   {self.name}",
            f"league    {self.league.name} {self.league_key or '(from the player map)'}",
            f"team      {team}",
            f"database  {self.resolved_db()}",
            f"profile   {self.profile_dir or '(none - view and decide only)'}",
            f"page      {self.page.url or '(not published)'}",
            "",
            self.authority.describe(),
        ]
        return "\n".join(lines)


def manager_path(name: str, settings: Settings | None = None) -> Path:
    s = settings or Settings()
    return s._resolve(Path("managers") / f"{name}.toml")


def load_manager(name: str, settings: Settings | None = None) -> Manager:
    """Load `managers/<name>.toml`.

    Raises rather than falling back to a default. A league config may fall back
    - ranking the wrong players is recoverable and loud. Acting on the wrong
    team is neither.
    """
    path = manager_path(name, settings)
    if not path.is_file():
        raise ManagerError(
            f"no manager config at {path}. Copy managers/example.toml to "
            f"managers/{name}.toml and fill in the team."
        )
    try:
        cfg = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as e:
        raise ManagerError(f"{path}: invalid TOML: {e}") from e

    s = settings or Settings()
    league_file = cfg.get("league_file")
    league = load_league(s._resolve(Path(league_file))) if league_file else DEFAULT_LEAGUE

    profile = cfg.get("profile_dir")
    db = cfg.get("db_path")
    page_cfg = cfg.get("page", {}) or {}

    try:
        authority = Authority.from_config(cfg.get("authority", {}) or {})
    except AuthorityError as e:
        raise ManagerError(f"{path}: {e}") from e

    return Manager(
        name=str(cfg.get("name", name)),
        league=league,
        league_key=str(cfg.get("league_key", "")),
        team_key=str(cfg.get("team_key", "")),
        profile_dir=s._resolve(Path(profile)) if profile else None,
        db_path=s._resolve(Path(db)) if db else None,
        authority=authority,
        page=PageConfig(
            url=str(page_cfg.get("url", "")),
            owner_key=str(page_cfg.get("owner_key", "")),
            guest_key=str(page_cfg.get("guest_key", "")),
        ),
    )


def available(settings: Settings | None = None) -> list[str]:
    """Manager names with a config on this machine, for error messages."""
    s = settings or Settings()
    d = s._resolve(Path("managers"))
    if not d.is_dir():
        return []
    return sorted(p.stem for p in d.glob("*.toml") if p.stem != "example")
