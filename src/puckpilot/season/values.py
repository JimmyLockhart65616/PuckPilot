"""What a player is worth for one game, live.

Assembles the pieces the backtests already use, with two changes that only
matter once the season is actually running.

First, the scale. `GameValueModel` derives its per-category standard deviations
from a season of real game lines, and in October the current season has almost
none - the SDs would be noise, and every value on the board would move week to
week for reasons that have nothing to do with the players. So the model is
built from the last *completed* season. An SD is a unit of measurement, not a
forecast; borrowing last year's ruler is right, and it keeps the numbers on the
page stable from opening night.

Second, "today". The backtests index into a list of dates built from played
games and pass an integer. Here the caller knows the calendar date, so the
index is found by bisection - which also makes the no-lookahead property
structural rather than remembered: everything at or after today's date is
simply not in the window.
"""

from __future__ import annotations

import bisect
import sqlite3
from dataclasses import dataclass

from puckpilot.draft.replay import ReplayData, build_replay_data
from puckpilot.engine.lineup_replay import GameValueModel, projected_pg_values
from puckpilot.engine.waivers import blended_pg_value
from puckpilot.league import DEFAULT_LEAGUE, LeagueConfig


@dataclass
class ValueModel:
    """Per-game value in z-like units, blended toward recent form."""

    vm: GameValueModel
    data: ReplayData
    proj_pg: dict[int, float]
    season: str
    scale_season: str

    def day_index(self, date: str) -> int:
        """Dates strictly before `date`. No lookahead by construction."""
        return bisect.bisect_left(self.data.dates, date)

    def per_game(self, pid: int, date: str) -> float:
        """Expected value of one game for this player, as of `date`.

        The preseason projection shrunk toward a 14-day trailing window, which
        is `waivers.blended_pg_value` - the same function the waiver backtest
        was validated with, and it filters strictly to days before `date`.
        """
        return blended_pg_value(pid, self.day_index(date), self.data, self.vm, self.proj_pg)

    def projected(self, pid: int) -> float:
        """The preseason number alone, for showing what form has moved."""
        return self.proj_pg.get(pid, 0.0)

    def knows(self, pid: int) -> bool:
        return pid in self.proj_pg


def build_value_model(
    conn: sqlite3.Connection,
    season: str,
    train_seasons: tuple[str, ...],
    league: LeagueConfig = DEFAULT_LEAGUE,
    scale_season: str | None = None,
) -> ValueModel:
    """Assemble the live value model.

    `scale_season` is the season whose game lines set the category scales;
    it defaults to the most recent training season, which is the most recent
    completed one.
    """
    from puckpilot.draft.sim import build_universe

    scale = scale_season or train_seasons[0]
    skater_keys = [c.key for c in league.skater_cats]

    universe = build_universe(conn, season, train_seasons, league)
    scale_data = build_replay_data(conn, scale, skater_keys)
    vm = GameValueModel(scale_data, set(universe.ids), league.goalie_cats)

    live = build_replay_data(conn, season, skater_keys)
    proj_pg = projected_pg_values(universe.frame, vm, skater_keys)

    return ValueModel(vm=vm, data=live, proj_pg=proj_pg, season=season, scale_season=scale)
