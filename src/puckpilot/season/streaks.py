"""Hot streaks: which ones last, and what they are worth over the next week.

A player is *hot* in a category when his last `STREAK_GAMES` games beat what
the model expected of him going in - his season so far shrunk toward the
preseason rate (`season.form`) - by a margin chance gives only `STREAK_ALPHA`
of the time. Measured by `ppilot season streak-check` (2023-24 to 2025-26,
every skater with 10+ games; each figure is hot minus not-hot, so drift that
has nothing to do with streaks cancels):

- **A streak is short.** The last-5 window stays hot for a median of 2-3
  games; 4-9% are still hot five games later.
- **What it says depends on why.** With more ice time - minutes over the
  streak at least `ROLE_UP` above his minutes before it this season, or
  power-play time for PPP - it is a role change, and he runs above the
  model's own (already updated) rate for the next 7 games: goals +14%, points
  +10%, PPP +18%, shots +9%, blocks +8%. Goals, assists and points stay above
  it for 28 games; PPP for about a week; shots and blocks about two.
- **Without more ice time a scoring streak is luck**: goals, assists, points
  and PPP run at the model's rate. Hits are the exception - a hits streak
  runs +8% for about two weeks either way - and penalty minutes fade: a
  week of fights is followed by 10% fewer minutes than the model expects.

`MOMENTUM` holds the next-week effects that cleared 2 standard errors pooled
(player-clustered), with the same sign estimated on every pair of seasons.
Everything else is zero: not "probably small", but not shown to exist.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass

import numpy as np

STREAK_GAMES = 5  # games a streak is judged over
STREAK_ALPHA = 0.10  # how surprising it must be: this tail probability or less
# Games this season before a streak counts. Streaks first seen at games 6-9
# carried no momentum in hits, shots or blocks (-3% to +1% over the next 7
# games) and PPP streaks then ran 26% BELOW the model - this early the model
# itself has already chased them. From the 10th game they hold (above).
MIN_GAMES = 10
ROLE_UP = 0.10  # ice time over the streak this far above his time before it
# Games before the streak this season needed to judge his ice time by them;
# with fewer, his average last season is the baseline.
ROLE_BEFORE_MIN = 5

# Overdispersion (variance / mean): penalty minutes arrive in lumps.
DISPERSION = {"pim": 3.84}

# Excess over the model's rate for the next 7 games, for a player hot in the
# category - with more ice time ("role"), or without ("plain").
MOMENTUM = {
    "role": {
        "goals": 0.145,
        "points": 0.105,
        "ppp": 0.181,
        "sog": 0.086,
        "hits": 0.074,
        "blocks": 0.085,
    },
    "plain": {"hits": 0.085, "blocks": 0.022, "pim": -0.098},
}

# Whether the weekly plan prices hot players at their streak's measured
# effect (`form.FormRates`). The add gate, pricing each season with the table
# the other two seasons measured (`-r-m` against `-r`, 12 teams x 22 weeks):
# +0.03 +/- 0.04 and +0.00 +/- 0.03 categories a week. Real per player, too
# few decisions moved to show; off. The streaks are shown instead - on the
# add cards and in `ppilot season hot` - for a person choosing a streamer.
STREAK_MOMENTUM = False

LABELS = {
    "goals": "G",
    "assists": "A",
    "points": "P",
    "pim": "PIM",
    "ppp": "PPP",
    "sog": "SOG",
    "hits": "HIT",
    "blocks": "BLK",
}


def tail(s, e, disp) -> np.ndarray:
    """P(X >= s) for X with mean e: Poisson, or negative binomial where the
    category's dispersion (variance over mean, broadcast by column) exceeds 1."""
    from scipy import stats

    e = np.maximum(np.asarray(e, dtype=float), 1e-9)
    s = np.asarray(s, dtype=float)
    disp = np.broadcast_to(np.asarray(disp, dtype=float), e.shape)
    out = np.asarray(stats.poisson.sf(s - 1, e), dtype=float)
    over = disp > 1.0
    if over.any():
        r = e[over] / (disp[over] - 1.0)
        out[over] = stats.nbinom.sf(s[over] - 1, r, 1.0 / disp[over])
    return out


@dataclass(frozen=True)
class Streak:
    """One player hot in one category, as of a date."""

    key: str
    actual: float  # over the last STREAK_GAMES games
    expected: float  # what the model expected of those games going in
    p: float  # how likely that was by chance
    role_up: bool  # with more ice time (power-play time, for PPP)
    usage_now: float | None  # minutes a game over the streak (PP seconds, for PPP)
    usage_before: float | None  # the same, before it this season
    momentum: float  # measured excess over the model for the next 7 games

    @property
    def label(self) -> str:
        return LABELS.get(self.key, self.key)

    def describe(self) -> str:
        """'SOG: 22 in his last 5 (14.3 expected), with more ice time (19.8 min a
        game, 16.9 before) - a role change: shots have run +9% over the model
        the next week'."""
        head = (
            f"{self.label}: {self.actual:g} in his last {STREAK_GAMES} "
            f"({self.expected:.1f} expected)"
        )
        if self.role_up and self.usage_now is not None and self.usage_before is not None:
            if self.key == "ppp":
                head += (
                    f", with more power-play time ({_clock(self.usage_now)} a game, "
                    f"{_clock(self.usage_before)} before)"
                )
            else:
                head += (
                    f", with more ice time ({self.usage_now:.1f} min a game, "
                    f"{self.usage_before:.1f} before)"
                )
        if self.momentum > 0:
            why = "a role change" if self.role_up and self.key != "hits" else "it tends to last"
            return f"{head} - {why}: streaks like it ran {self.momentum:+.0%} over the model"
        if self.momentum < 0:
            return f"{head} - expect a fade: streaks like it ran {self.momentum:+.0%} under"
        if self.role_up:
            return f"{head} - no lasting effect was measured for {self.label}"
        return f"{head} - without more ice time this is mostly luck"


def _clock(seconds: float) -> str:
    m, s = divmod(int(round(seconds)), 60)
    return f"{m}:{s:02d}"


class StreakFinder:
    """Which skaters are hot in which categories, as of any date of a season.

    `data` is the season's `ReplayData`; `prior` the preseason per-game rates
    (`week.per_game_rates`); `usage` a `form.Usage` for ice time (without it,
    no streak is ever a role change). Only games strictly before the date count.
    """

    def __init__(self, data, prior, usage=None, k=None, momentum=None):
        from puckpilot.season.form import DEFAULT_FORM_K, FORM_RATE_K

        self.data = data
        self.prior = prior
        self.usage = usage
        self.keys = list(data.skater_keys)
        ks = {**FORM_RATE_K, **(k or {})}
        self.k = np.array([ks.get(key, DEFAULT_FORM_K) for key in self.keys])
        self.disp = np.array([DISPERSION.get(key, 1.0) for key in self.keys])
        self.momentum = momentum if momentum is not None else MOMENTUM
        self._players: dict[int, tuple[list[int], np.ndarray]] = {}
        for pid, games in data.skater.items():
            days = sorted(games)
            cum = np.vstack([np.zeros(len(self.keys)), np.cumsum([games[i] for i in days], axis=0)])
            self._players[pid] = (days, cum)
        self._cache: dict[tuple[int, int], tuple[Streak, ...]] = {}

    def streaks(self, pid: int | None, date: str) -> tuple[Streak, ...]:
        """His hot categories as of `date`, strongest effect first."""
        if pid is None or pid not in self.prior or pid not in self._players:
            return ()
        t = bisect.bisect_left(self.data.dates, date)
        key = (pid, t)
        if key in self._cache:
            return self._cache[key]
        days, cum = self._players[pid]
        n = bisect.bisect_left(days, t)
        out: list[Streak] = []
        if n >= MIN_GAMES:
            pre = n - STREAK_GAMES
            base = self.prior[pid]
            prior = np.array([base.get(k, 0.0) for k in self.keys])
            r_pre = (cum[pre] + self.k * prior) / (pre + self.k)
            s = cum[n] - cum[pre]
            e = STREAK_GAMES * r_pre
            p = tail(s, e, self.disp)
            start = days[pre]
            minutes = self._usage("_min", pid, start, t)
            pp = self._usage("_pp", pid, start, t)
            for ci, cat in enumerate(self.keys):
                if p[ci] > STREAK_ALPHA:
                    continue
                now, before = pp if cat == "ppp" else minutes
                role_up = (
                    now is not None
                    and before is not None
                    and before > 0
                    and now >= (1.0 + ROLE_UP) * before
                )
                m = self.momentum["role" if role_up else "plain"].get(cat, 0.0)
                out.append(
                    Streak(
                        key=cat,
                        actual=float(s[ci]),
                        expected=float(e[ci]),
                        p=float(p[ci]),
                        role_up=bool(role_up),
                        usage_now=now,
                        usage_before=before,
                        momentum=float(m),
                    )
                )
        out.sort(key=lambda x: -abs(x.momentum))
        got = tuple(out)
        self._cache[key] = got
        return got

    def multipliers(self, pid: int | None, date: str) -> dict[str, float]:
        """category -> 1 + the streak's measured effect, for hot categories."""
        return {s.key: 1.0 + s.momentum for s in self.streaks(pid, date) if s.momentum}

    def _usage(self, attr: str, pid: int, start: int, t: int):
        """(mean over the streak games, mean before them this season)."""
        if self.usage is None:
            return None, None
        series = getattr(self.usage, attr).get(pid, [])
        idx = [i for i, _ in series]
        lo = bisect.bisect_left(idx, start)
        hi = bisect.bisect_left(idx, t)
        during = [v for _, v in series[lo:hi]]
        before = [v for _, v in series[:lo]]
        return (
            float(np.mean(during)) if during else None,
            float(np.mean(before)) if before else None,
        )


@dataclass(frozen=True)
class HotPlayer:
    """A player with at least one streak, and what it should add this week."""

    name: str
    team: str
    positions: str
    nhl_player_id: int
    streaks: tuple[Streak, ...]
    extra: dict[str, float]  # category -> expected over his rate for his games left this week
    score: float  # extra summed in each category's game-to-game spread
    owned: float | None = None


def hot_players(entries, finder: StreakFinder, date: str, rates, games_left, scale) -> list:
    """Players with streaks, the most useful this week first.

    `entries`: (name, team, positions, nhl id, percent owned or None).
    `rates`: player -> category -> rate (the model's, as of `date`);
    `games_left`: club -> games this week from `date`; `scale`: category ->
    its game-to-game spread, so a goal and a hit can be added up.
    """
    out = []
    for name, team, pos, pid, owned in entries:
        found = finder.streaks(pid, date)
        if not found:
            continue
        r = rates.get(pid, {})
        g = games_left.get(team, 0)
        extra = {s.key: s.momentum * r.get(s.key, 0.0) * g for s in found if s.momentum}
        score = sum(v / scale.get(k, 1.0) for k, v in extra.items())
        out.append(
            HotPlayer(
                name=name,
                team=team or "",
                positions=pos or "",
                nhl_player_id=pid,
                streaks=found,
                extra=extra,
                score=score,
                owned=owned,
            )
        )
    out.sort(key=lambda h: -h.score)
    return out
