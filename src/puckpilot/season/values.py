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
from dataclasses import dataclass, field

from puckpilot.draft.replay import ReplayData, build_replay_data
from puckpilot.engine.lineup_replay import GameValueModel, projected_pg_values
from puckpilot.engine.waivers import blended_pg_value
from puckpilot.league import DEFAULT_LEAGUE, LeagueConfig

# Count a goalie's rest-of-season games as his share of his club's
# (`goalies.GoalieWorkload`) rather than all of them. The share itself is far
# better (remaining-share error 0.60 -> 0.11-0.14, `season goalie-check`), but
# the add search does worse with it: add gate `-g-l` against `-g`'s baseline,
# 4 drafts x 2 seasons, pooled -0.11 +/- 0.03 categories a week, three leagues
# worse by more than 2 SE, none better. Valued at his share, a backup goalie
# becomes one of the cheapest players on the roster and is cut for a skater
# streamer, and the goalie categories go with him. Off: the inflated number
# was doing a job the search relies on.
GOALIE_WORKLOAD = False


@dataclass
class ValueModel:
    """Per-game value in z-like units, blended toward recent form."""

    vm: GameValueModel
    data: ReplayData
    proj_pg: dict[int, float]
    season: str
    scale_season: str
    frame: object | None = None
    _tilts: dict = field(default_factory=dict)
    # goalies.GoalieWorkload, when goalie workload is on
    workload: object | None = None
    # keeper_value.KeeperBoard, when the run ranked next season's keepers
    keepers: object | None = None
    # form.FormRates: per-category rates that have seen this season
    form: object | None = None
    # per_game from `form` for skaters rather than the 14-day blend; None
    # follows `form.FORM_VALUE`
    form_value: bool | None = None

    def day_index(self, date: str) -> int:
        """Dates strictly before `date`. No lookahead by construction."""
        return bisect.bisect_left(self.data.dates, date)

    def per_game(self, pid: int, date: str) -> float:
        """Expected value of one game for this player, as of `date`.

        The preseason projection shrunk toward a 14-day trailing window, which
        is `waivers.blended_pg_value` - the same function the waiver backtest
        was validated with, and it filters strictly to days before `date`.
        """
        if self.form is not None:
            from puckpilot.season.form import FORM_VALUE

            if FORM_VALUE if self.form_value is None else self.form_value:
                v = self.form.value(pid, date, self.vm)
                if v is not None:
                    return v
        return blended_pg_value(pid, self.day_index(date), self.data, self.vm, self.proj_pg)

    def rates(self, cats, date: str) -> dict[int, dict[str, float]]:
        """Per-game category rates as of `date`: this season's form when it is
        attached (`season.form`), else the preseason projection's."""
        if self.form is not None:
            return self.form.rates(date)
        from puckpilot.season.week import per_game_rates

        return per_game_rates(self.frame, cats)

    def projected(self, pid: int) -> float:
        """The preseason number alone, for showing what form has moved."""
        return self.proj_pg.get(pid, 0.0)

    def knows(self, pid: int) -> bool:
        return pid in self.proj_pg

    # -- category tilt -----------------------------------------------------

    def tilt(self, pid: int, weights: dict[str, float]) -> float:
        """How much a player is worth under a category stance, as a multiplier.

        The blended value is one scalar, so a stance cannot be applied to it
        directly - the categories have already been summed away. Instead the
        player's projected per-game line is scored twice, once flat and once
        weighted, and the ratio tilts the blended number. That keeps the recent
        form the blend carries, which re-deriving value from projections alone
        would throw away.

        A player whose flat value is near zero has no meaningful ratio, so he
        is left alone rather than multiplied by something enormous.
        """
        if not weights or self.frame is None:
            return 1.0
        key = (pid, tuple(sorted(weights.items())))
        got = self._tilts.get(key)
        if got is None:
            got = self._tilts[key] = self._compute_tilt(pid, weights)
        return got

    def per_game_tilted(self, pid: int, date: str, weights: dict[str, float]) -> float:
        return self.per_game(pid, date) * self.tilt(pid, weights)

    def _compute_tilt(self, pid: int, weights: dict[str, float]) -> float:
        import numpy as np

        try:
            row = self.frame.loc[pid]
        except (KeyError, AttributeError):
            return 1.0
        gp = max(_num(row.get("proj_gp")), 1.0)

        if str(row.get("position")) == "G":
            sa = _num(row.get("shots_against")) / gp
            saves = _num(row.get("saves")) / gp
            raw = {
                "wins": _num(row.get("wins")) / gp,
                "saves": saves,
                "shots_against": sa,
                "save_pct": saves - self.vm.pool_sv * sa,
                "shutouts": _num(row.get("shutouts")) / gp,
            }
            pairs = [
                (c.key, raw.get(c.key, 0.0) / sd)
                for c, sd in zip(self.vm.goalie_cats, self.vm.g_sd, strict=True)
            ]
        else:
            keys = self.data.skater_keys
            sd = np.asarray(self.vm.sk_sd, dtype=float)
            pairs = [(k, _num(row.get(k)) / gp / s) for k, s in zip(keys, sd, strict=False)]

        flat = sum(z for _, z in pairs)
        if abs(flat) < 1e-9:
            return 1.0
        tilted = sum(z * weights.get(k, 1.0) for k, z in pairs)
        return tilted / flat


def _num(v) -> float:
    """A projection cell as a number; NaN and absent both mean zero."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if f != f else f


def build_value_model(
    conn: sqlite3.Connection,
    season: str,
    train_seasons: tuple[str, ...],
    league: LeagueConfig = DEFAULT_LEAGUE,
    scale_season: str | None = None,
    goalie_workload: bool | None = None,
) -> ValueModel:
    """Assemble the live value model.

    `scale_season` is the season whose game lines set the category scales;
    it defaults to the most recent training season, which is the most recent
    completed one. `goalie_workload` defaults to `GOALIE_WORKLOAD`.
    """
    from puckpilot.draft.sim import build_universe

    scale = scale_season or train_seasons[0]
    skater_keys = [c.key for c in league.skater_cats]

    universe = build_universe(conn, season, train_seasons, league)
    scale_data = build_replay_data(conn, scale, skater_keys)
    vm = GameValueModel(scale_data, set(universe.ids), league.goalie_cats)

    live = build_replay_data(conn, season, skater_keys)
    proj_pg = projected_pg_values(universe.frame, vm, skater_keys)
    from puckpilot.season.form import FORM_USAGE, FormRates, Usage
    from puckpilot.season.week import per_game_rates

    skaters = {int(p) for p, pos in universe.frame["position"].items() if pos != "G"}
    form = FormRates(
        live,
        per_game_rates(universe.frame, league.all_cats),
        skaters=skaters,
        usage=Usage(conn, season, live.dates) if FORM_USAGE else None,
    )

    workload = None
    if GOALIE_WORKLOAD if goalie_workload is None else goalie_workload:
        from puckpilot.engine.aggregate import season_games
        from puckpilot.season.goalies import GoalieWorkload, projected_shares

        priors = projected_shares(universe.frame, season_games(conn, season))
        workload = GoalieWorkload(conn, season, priors=priors)

    return ValueModel(
        vm=vm,
        data=live,
        proj_pg=proj_pg,
        season=season,
        scale_season=scale,
        frame=universe.frame,
        workload=workload,
        form=form,
    )
