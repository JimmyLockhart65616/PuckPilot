"""Per-category rates that have seen this season, not just August.

Every forecast of a week - the category odds, what an add would add - used the
preseason projection's per-game rates (`week.per_game_rates`). A player whose
role changed in October was the same player in March. Each rate here is his
season so far, shrunk toward the preseason rate by `k` games:

    rate = (games * season_mean + k * preseason_rate) / (games + k)

with a separate `k` per category, because how fast a player's own numbers
become believable depends on the category: hits settle in a handful of games,
goals take a season. Measured by `ppilot season form-check` (every 7th game
date after the 21st, Poisson deviance against the preseason rates alone; k fit
on 2024-25 by next-week deviance, then held fixed):

                         next week                   rest of season
    season, k per cat    -6.6% / -5.3% / -5.2%       -23.9% / -20.0% / -17.5%
    last 14, k = 10      -1.1% / -0.6% / +0.2%        -1.5% /  +3.1% /  +4.6%

(2023-24 / 2024-25 / 2025-26). The second row is the blend the lineup's value
per game already used: about no better than August for next week, and worse
for the rest of the season. The k's below are that fit. Skaters only: no
goalie rate has been measured here, so goalies keep their preseason rates.
"""

from __future__ import annotations

import bisect

import numpy as np

# Games of the preseason rate a player's season so far is shrunk toward, per
# skater category (fit on 2024-25; see the module docstring).
FORM_RATE_K = {
    "goals": 80.0,
    "assists": 40.0,
    "points": 40.0,
    "pim": 80.0,
    "ppp": 20.0,
    "sog": 20.0,
    "hits": 10.0,
    "blocks": 20.0,
}
# A category never measured gets the middle of the fitted range.
DEFAULT_FORM_K = 40.0

# Whether the weekly plan - category odds, totals, what an add would add -
# prices with these rates (`week.build_week_plan`). On since 2026-10-05:
# gate G1 (`season calibrate --rates form`, fit 2024-25, test 2025-26) test
# log-loss 0.5557 -> 0.5487, Brier 0.1325 -> 0.1302, 10/10 deciles within 5
# points and ahead of the plain normal model - re-run once the harness built
# both seasons' leagues alike (38d1689); before that it read 0.5793 -> 0.5733
# with six of eight skater categories better and no decile worse; gate G2 (`odds-daily-h-f25-x1-r`, 12 teams x 22 weeks) +0.29 +/- 0.08
# and +0.14 +/- 0.06 categories a week, both clear of 2 SE. Refitting the odds
# model's P_PLAY and PHI on these rates (0.80; A 1.20, SOG 1.00, BLK 1.35) gains
# 0.0005 log-loss on the test season, so the constants stay as G2 ran them.
FORM_RATES = True

# Whether a skater's value per game - what the lineup sorts by, and what the
# add search measures a season by - comes from these rates rather than the old
# blend (last 14 game dates, k = 10). The two judges disagree. The live lineup
# replay (24 rosters a season) prefers it in all three: +15.7 +/- 7.4, +7.8
# +/- 6.5, +6.0 +/- 8.1 a roster (+0.1-0.2%). The add gate does not: -v alone
# +0.13 +/- 0.09 and -0.02 +/- 0.07 categories a week, and on top of form rates
# +0.03 and -0.11 - with about 30% fewer adds either way. Not positive in both
# seasons for the decision it most changes, so off.
FORM_VALUE = False

# Usage: before a role change shows in the counting stats it shows in ice
# time. The preseason rate is scaled by (minutes a game over his last
# USAGE_GAMES against last season's) ** USAGE_GAMMA, clipped, and PPP's by
# power-play time the same way (PP_SMOOTH seconds added to both sides, so a
# player with almost none last season cannot come out at ten times it).
# Measured in `season form-check`: best of every variant tried, in all three
# seasons, next week and rest of season, and on players whose ice time moved
# more than 15% - but by little (about 0.25 points of deviance overall, 0.8 on
# PPP next week). A whole exponent was worse for the rest of the season.
USAGE_GAMES = 5
USAGE_GAMMA = 0.5
USAGE_CLIP = (0.7, 1.5)
PP_GAMMA = 0.5
PP_CLIP = (0.5, 2.0)
PP_SMOOTH = 30.0
# Whether form rates use it. The add gate cannot see a gain that small: -r-u
# against -r, 12 teams x 22 weeks, -0.05 +/- 0.07 and +0.04 +/- 0.05
# categories a week. Off; the cards show the ice time either way.
FORM_USAGE = False


class Usage:
    """Each skater's recent ice time against last season's, by date.

    Minutes from the game logs; power-play seconds from `nhl_skater_toi`
    (`data.nhlstats`), which may be empty - then the power-play ratio is 1.
    """

    def __init__(self, conn, season: str, dates: list[str]):
        import json

        from puckpilot.engine.aggregate import toi_seconds

        y = int(season[:4])
        last = f"{y - 1}{y}"
        idx = {d: i for i, d in enumerate(dates)}

        def minutes(s):
            out: dict[int, dict[str, float]] = {}
            for pid, d, stats in conn.execute(
                "SELECT player_id, game_date, stats_json FROM nhl_game_logs WHERE season = ?", (s,)
            ):
                toi = json.loads(stats).get("toi")
                if toi:
                    out.setdefault(pid, {})[d] = toi_seconds(toi) / 60.0
            return out

        def pp(s):
            out: dict[int, dict[str, float]] = {}
            for pid, d, secs in conn.execute(
                "SELECT player_id, game_date, pp_toi_s FROM nhl_skater_toi WHERE season = ?", (s,)
            ):
                if secs is not None:
                    out.setdefault(pid, {})[d] = float(secs)
            return out

        def base(by_player):
            out = {}
            for pid, v in by_player.items():
                if len(v) >= 10:
                    out[pid] = float(np.mean(list(v.values())))
            return out

        self.base_min = base(minutes(last))
        self.base_pp = base(pp(last))
        self.dates = dates
        self._min = {
            pid: sorted((idx[d], m) for d, m in v.items() if d in idx)
            for pid, v in minutes(season).items()
        }
        self._pp = {
            pid: sorted((idx[d], s) for d, s in v.items() if d in idx)
            for pid, v in pp(season).items()
        }

    @staticmethod
    def _recent(series, t: int, n: int) -> list[float]:
        i = bisect.bisect_left(series, (t, float("-inf")))
        return [x for _, x in series[max(0, i - n) : i]]

    def recent_minutes(self, pid: int, t: int, n: int = USAGE_GAMES) -> float | None:
        got = self._recent(self._min.get(pid, []), t, n)
        return float(np.mean(got)) if got else None

    def recent_pp(self, pid: int, t: int, n: int = USAGE_GAMES) -> float | None:
        got = self._recent(self._pp.get(pid, []), t, n)
        return float(np.mean(got)) if got else None

    def ratios(self, pid: int, t: int) -> tuple[float, float]:
        """(ice time, power-play time) against last season, as of date index t."""
        u = pp = 1.0
        now = self.recent_minutes(pid, t)
        if now is not None and pid in self.base_min:
            u = float(np.clip(now / self.base_min[pid], *USAGE_CLIP))
        now_pp = self.recent_pp(pid, t)
        if now_pp is not None and pid in self.base_pp:
            pp = float(np.clip((now_pp + PP_SMOOTH) / (self.base_pp[pid] + PP_SMOOTH), *PP_CLIP))
        return u, pp


class FormRates:
    """Skater per-category rates as of a date, from this season's game lines.

    `data` is the season's `ReplayData` (one vector per player-game over its
    `skater_keys`); `prior` is the preseason per-game rates
    (`week.per_game_rates`). Only games strictly before the date count.
    """

    def __init__(
        self,
        data,
        prior: dict[int, dict[str, float]],
        k: dict[str, float] | None = None,
        skaters: set[int] | None = None,
        usage: Usage | None = None,
        streaks=None,
        momentum: bool = False,
    ):
        self.data = data
        self.prior = prior
        # Scales the prior by recent ice time (`Usage`); None leaves it alone.
        self.usage = usage
        # `streaks.StreakFinder`: with `momentum`, a hot category's rate is
        # raised (or, for a fading one, lowered) by its measured next-week effect.
        self.streaks = streaks
        self.momentum = momentum
        # Who is a skater: the prior's rows carry every category, zero where it
        # does not apply, so it cannot say. None: anyone with a skater line.
        self.skaters = skaters
        self.keys = list(data.skater_keys)
        ks = {**FORM_RATE_K, **(k or {})}
        self.k = np.array([ks.get(key, DEFAULT_FORM_K) for key in self.keys])
        self._players: dict[int, tuple[list[int], np.ndarray]] = {}
        for pid, games in data.skater.items():
            days = sorted(games)
            cum = np.vstack([np.zeros(len(self.keys)), np.cumsum([games[i] for i in days], axis=0)])
            self._players[pid] = (days, cum)
        self._cache: dict[int, dict[int, dict[str, float]]] = {}

    def rates(self, date: str) -> dict[int, dict[str, float]]:
        """player -> rates as of `date`: the prior's, with skater keys moved."""
        t = bisect.bisect_left(self.data.dates, date)
        if t in self._cache:
            return self._cache[t]
        out: dict[int, dict[str, float]] = {}
        for pid, base in self.prior.items():
            got = self._players.get(pid)
            if got is None:
                out[pid] = base
                continue
            days, cum = got
            n = bisect.bisect_left(days, t)
            if n == 0:
                out[pid] = base
                continue
            u = pp = 1.0
            if self.usage is not None:
                u, pp = self.usage.ratios(pid, t)
            rates = dict(base)
            for c, key in enumerate(self.keys):
                if key in base:
                    k = self.k[c]
                    prior = base[key] * u**USAGE_GAMMA * (pp**PP_GAMMA if key == "ppp" else 1.0)
                    rates[key] = (cum[n][c] + k * prior) / (n + k)
            if self.momentum and self.streaks is not None:
                for key, mult in self.streaks.multipliers(pid, date).items():
                    if key in rates:
                        rates[key] *= mult
            out[pid] = rates
        self._cache[t] = out
        return out

    def value(self, pid: int, date: str, vm) -> float | None:
        """A skater's value per game in `vm`'s units, from his rates as of `date`.

        None for a goalie or anyone without a prior: the caller keeps its own
        number for those.
        """
        if pid not in self.prior:
            return None
        if self.skaters is not None and pid not in self.skaters:
            return None
        if self.skaters is None and pid not in self.data.skater:
            return None
        rates = self.rates(date)[pid]
        return vm.skater(np.array([rates.get(k, 0.0) for k in self.keys]))
