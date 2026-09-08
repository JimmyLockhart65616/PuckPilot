"""League settings, loaded from a TOML file rather than hardcoded.

Every engine takes its categories, roster shape, and constraints from a
LeagueConfig, so supporting a different league means writing a config file, not
editing Python. See `leagues/example.toml` for a fully documented template.

Which file loads by default is controlled by the `league_file` setting
(env `LEAGUE_FILE`, or the repo-root .env).
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from puckpilot.draft.engine import DraftRules
from puckpilot.engine.categories import (
    GOALIE_CATS_DEFAULT,
    SKATER_CATS_DEFAULT,
    Category,
    resolve,
)
from puckpilot.engine.valuation import DEFAULT_SHAPE, LeagueShape


class LeagueConfigError(ValueError):
    pass


@dataclass(frozen=True)
class LeagueConfig:
    name: str = "default"
    league_id: str = ""
    shape: LeagueShape = DEFAULT_SHAPE
    skater_cats: tuple[Category, ...] = SKATER_CATS_DEFAULT
    goalie_cats: tuple[Category, ...] = GOALIE_CATS_DEFAULT
    scoring: str = "h2h"  # 'h2h' | 'roto'

    # keepers: n per team, held for at most `keeper_years` seasons after drafting
    n_keepers: int = 0
    keeper_years: int = 3
    # season -> player names under contract going into that season's draft
    keepers_by_season: dict[str, tuple[str, ...]] = field(default_factory=dict)

    # acquisition budget (Yahoo: "max acquisitions" season/week)
    season_acquisitions: int | None = None
    weekly_acquisitions: int | None = None

    # Yahoo enforces a weekly floor on goalie appearances; missing it costs the
    # goalie categories for that week, so it is a hard lineup constraint.
    min_goalie_appearances: int = 0

    # head-to-head schedule
    regular_weeks: int = 19
    playoff_teams: int = 8
    playoff_weeks: int = 3

    @property
    def all_cats(self) -> tuple[Category, ...]:
        return self.skater_cats + self.goalie_cats

    @property
    def draft_rounds(self) -> int:
        """Keepers occupy roster spots but are not drafted."""
        return self.shape.roster_size - self.n_keepers

    def keepers_for_season(self, season: str) -> tuple[str, ...]:
        return tuple(self.keepers_by_season.get(season, ()))

    def draft_rules(self, **overrides) -> DraftRules:
        return DraftRules(shape=self.shape, rounds=self.draft_rounds, **overrides)


def _shape_from(cfg: dict) -> LeagueShape:
    roster = cfg.get("roster", {})
    slots = roster.get("slots")
    if not slots:
        return DEFAULT_SHAPE
    try:
        ordered = tuple((str(s["pos"]), int(s["count"])) for s in slots)
    except (KeyError, TypeError) as e:
        raise LeagueConfigError(
            "roster.slots must be a list of tables with 'pos' and 'count', "
            'e.g. slots = [{ pos = "C", count = 2 }]'
        ) from e
    return LeagueShape(
        n_teams=int(roster.get("teams", 12)),
        slots=ordered,
        util_slots=int(roster.get("util", 0)),
        bench_slots=int(roster.get("bench", 0)),
    )


def _cats_from(labels: list[str] | None, fallback: tuple[Category, ...]) -> tuple[Category, ...]:
    if not labels:
        return fallback
    return tuple(resolve(str(x)) for x in labels)


def load_league(path: str | Path) -> LeagueConfig:
    """Parse a league TOML file. Raises LeagueConfigError with a usable message."""
    p = Path(path)
    if not p.is_file():
        raise LeagueConfigError(f"league config not found: {p}")
    try:
        cfg = tomllib.loads(p.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as e:
        raise LeagueConfigError(f"{p}: invalid TOML: {e}") from e

    scoring = cfg.get("scoring", {})
    schedule = cfg.get("schedule", {})
    keepers = cfg.get("keepers", {})
    tx = cfg.get("transactions", {})
    lineup = cfg.get("lineup", {})

    scoring_type = str(scoring.get("type", "h2h")).lower()
    if scoring_type not in ("h2h", "roto"):
        raise LeagueConfigError(f"{p}: scoring.type must be 'h2h' or 'roto', got {scoring_type!r}")

    by_season = {
        str(k): tuple(str(n) for n in v) for k, v in (keepers.get("by_season") or {}).items()
    }
    return LeagueConfig(
        name=str(cfg.get("name", p.stem)),
        league_id=str(cfg.get("league_id", "")),
        shape=_shape_from(cfg),
        skater_cats=_cats_from(scoring.get("skater"), SKATER_CATS_DEFAULT),
        goalie_cats=_cats_from(scoring.get("goalie"), GOALIE_CATS_DEFAULT),
        scoring=scoring_type,
        n_keepers=int(keepers.get("count", 0)),
        keeper_years=int(keepers.get("years", 3)),
        keepers_by_season=by_season,
        season_acquisitions=tx.get("season_acquisitions"),
        weekly_acquisitions=tx.get("weekly_acquisitions"),
        min_goalie_appearances=int(lineup.get("min_goalie_appearances", 0)),
        regular_weeks=int(schedule.get("regular_weeks", 19)),
        playoff_teams=int(schedule.get("playoff_teams", 8)),
        playoff_weeks=int(schedule.get("playoff_weeks", 3)),
    )


def load_default_league() -> LeagueConfig:
    """The league named by settings, or a generic 12-team default if absent.

    Falling back keeps `ppilot --help`, tests, and a fresh clone working before
    anyone has written a config — but it says so on stderr, because silently
    ranking players for the wrong league is the worst failure this tool has.
    """
    import sys

    from puckpilot.config import Settings

    path = Settings().resolved_league_path
    try:
        return load_league(path)
    except (LeagueConfigError, OSError) as e:
        print(
            f"puckpilot: {e}\n"
            f"  falling back to generic 12-team defaults; copy leagues/example.toml "
            f"and set LEAGUE_FILE to use your league's real settings.",
            file=sys.stderr,
        )
        return LeagueConfig()


DEFAULT_LEAGUE = load_default_league()
