"""Gate for form: does pulling a player's rates toward how he has been playing
forecast his next week - and the rest of his season - better than August did?

The weekly plan, the category odds and the add search all use preseason rates
(`week.per_game_rates`), so a breakout moves none of them. The lineup's value
per game does blend toward form (`waivers.blended_pg_value`: the last 14 game
dates, shrunk with 10 games of prior), but that blend was never tuned.

Each variant predicts a category rate as

    rate = (n * form + k * prior) / (n + k)

from the n games the player played in its window before the date (`form` is
their mean), and the preseason projection's rate (`prior`). Scored on what he
did over the next `HORIZON` game dates, and over the rest of the season, as
Poisson deviance of rate x the games he actually played - availability is a
separate question, so it is held fixed. Lower is better; shown against the
preseason rates alone, so 0.0% is August's forecast.

Variants: `pre` (prior only), `w<window>-k<k>` (window in game dates, `all` for
the season so far), and `w<window>-pc` (a separate k for every category,
chosen on `FIT_SEASON` and then held fixed for the other seasons).
"""

from __future__ import annotations

import bisect
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from puckpilot.league import LeagueConfig

SEASONS = ("20232024", "20242025", "20252026")
FIT_SEASON = "20242025"

# Game dates skipped at the start (no form yet), sampling stride, and the
# forecast horizon: about a fantasy week of game dates.
SKIP_DATES = 21
STRIDE = 7
HORIZON = 7

WINDOWS = (14, 28, None)  # None: the season so far
KS = (5.0, 10.0, 20.0, 40.0)
PC_GRID = (2.0, 5.0, 10.0, 20.0, 40.0, 80.0, 160.0)

# A rate of zero with a count above zero has infinite deviance; floor it.
MU_FLOOR = 1e-3

# Usage: the prior scaled by recent ice time over last season's, u ** gamma,
# with u clipped so one long night cannot double a forecast. Recent is the
# last n games played ("all": the season so far).
USAGE_CLIP = (0.7, 1.5)
USAGE_RECENT = (5, 10, "all")
USAGE_GAMMAS = (0.5, 1.0)
# A player whose ice time moved this much is a role change, scored apart.
USAGE_CHANGED = 0.15

# Power-play time, for PPP alone: the same ratio from `nhl_skater_toi`, with
# PP_SMOOTH seconds added to both sides so a player with almost none last
# season cannot come out at ten times it, and a wider clip.
PP_CLIP = (0.5, 2.0)
PP_SMOOTH = 30.0
PP_RECENT = (5, 10)
PP_GAMMAS = (0.5, 1.0)


def _deviance(y: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """Poisson deviance per element (y, mu same shape)."""
    mu = np.maximum(mu, MU_FLOOR)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(y > 0, y * np.log(y / mu), 0.0)
    return 2.0 * (term - (y - mu))


@dataclass
class _Player:
    days: np.ndarray  # sorted date indices he played
    cum: np.ndarray  # cumulative stat vectors, cum[j] = sum of the first j games
    toi: np.ndarray | None = None  # minutes in each of those games
    pp: np.ndarray | None = None  # power-play seconds in each of those games


@dataclass
class Sample:
    """What every variant needs at one (player, date): his windows and futures."""

    prior: np.ndarray
    windows: dict  # window -> (games, summed stats)
    week: tuple[int, np.ndarray]  # (games, summed stats) over the horizon
    rest: tuple[int, np.ndarray]  # (games, summed stats) to the season's end
    # recent ice time over last season's, per USAGE_RECENT; 1.0 when unknown
    usage: dict = field(default_factory=dict)
    # the same for power-play time, per PP_RECENT
    pp_usage: dict = field(default_factory=dict)


@dataclass
class FormScores:
    season: str
    keys: list[str]
    # variant -> horizon -> per-category summed deviance
    deviance: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    n: dict[str, int] = field(default_factory=dict)

    def total(self, variant: str, horizon: str) -> float:
        return float(self.deviance[variant][horizon].sum())

    def vs_pre(self, variant: str, horizon: str) -> float:
        base = self.total("pre", horizon)
        return (self.total(variant, horizon) - base) / base if base else float("nan")


def _priors(conn, season: str, keys: list[str]) -> dict[int, np.ndarray]:
    """Preseason per-game rates for skaters, aligned with `keys`."""
    from puckpilot.engine import projections

    y = int(season[:4])
    train = [f"{y - i}{y - i + 1}" for i in range(1, 4)]
    sk, _ = projections.project(conn, season, train)
    out = {}
    for pid, row in sk.iterrows():
        gp = float(row.get("proj_gp") or 0.0)
        if gp <= 0:
            continue
        out[int(pid)] = np.array([float(row.get(k) or 0.0) / gp for k in keys])
    return out


def _toi(conn, season: str) -> dict[int, dict[str, float]]:
    """player -> date -> minutes played, from the game logs."""
    import json

    from puckpilot.engine.aggregate import toi_seconds

    out: dict[int, dict[str, float]] = {}
    for pid, date, stats in conn.execute(
        "SELECT player_id, game_date, stats_json FROM nhl_game_logs WHERE season = ?", (season,)
    ):
        toi = json.loads(stats).get("toi")
        if toi:
            out.setdefault(pid, {})[date] = toi_seconds(toi) / 60.0
    return out


def _pp(conn, season: str) -> dict[int, dict[str, float]]:
    """player -> date -> power-play seconds, from `nhl_skater_toi` (may be empty)."""
    out: dict[int, dict[str, float]] = {}
    for pid, date, pp in conn.execute(
        "SELECT player_id, game_date, pp_toi_s FROM nhl_skater_toi WHERE season = ?", (season,)
    ):
        if pp is not None:
            out.setdefault(pid, {})[date] = float(pp)
    return out


def samples(conn: sqlite3.Connection, season: str, league: LeagueConfig) -> tuple[list, list]:
    """(keys, [Sample]) for every sampled (player, date) with a prior and a future."""
    from puckpilot.draft.replay import build_replay_data

    keys = [c.key for c in league.skater_cats]
    data = build_replay_data(conn, season, keys)
    priors = _priors(conn, season, keys)
    y = int(season[:4])
    last = _toi(conn, f"{y - 1}{y}")
    base = {pid: float(np.mean(list(v.values()))) for pid, v in last.items() if len(v) >= 10}
    now = _toi(conn, season)
    pp_last = _pp(conn, f"{y - 1}{y}")
    pp_base = {pid: float(np.mean(list(v.values()))) for pid, v in pp_last.items() if len(v) >= 10}
    pp_now = _pp(conn, season)
    players: dict[int, _Player] = {}
    for pid, games in data.skater.items():
        if pid not in priors:
            continue
        days = np.array(sorted(games), dtype=int)
        stack = np.array([games[i] for i in days])
        cum = np.vstack([np.zeros(len(keys)), np.cumsum(stack, axis=0)])
        players[pid] = _Player(days=days, cum=cum)
        mins = now.get(pid, {})
        players[pid].toi = np.array([mins.get(data.dates[i], np.nan) for i in days])
        secs = pp_now.get(pid, {})
        players[pid].pp = np.array([secs.get(data.dates[i], np.nan) for i in days])

    n_dates = len(data.dates)
    out = []
    for t in range(SKIP_DATES, n_dates, STRIDE):
        for pid, p in players.items():
            j = bisect.bisect_left(p.days, t)
            jw = bisect.bisect_left(p.days, t + HORIZON)
            if jw == j:
                continue  # did not play in the horizon: nothing to score
            windows = {}
            for w in WINDOWS:
                lo = 0 if w is None else bisect.bisect_left(p.days, t - w)
                windows[w] = (j - lo, p.cum[j] - p.cum[lo])
            end = len(p.days)
            usage = {}
            for n in USAGE_RECENT:
                recent = p.toi[:j] if n == "all" else p.toi[max(0, j - n) : j]
                recent = recent[~np.isnan(recent)] if recent is not None else recent
                if pid in base and recent is not None and len(recent):
                    usage[n] = float(np.clip(recent.mean() / base[pid], *USAGE_CLIP))
                else:
                    usage[n] = 1.0
            pp_usage = {}
            for n in PP_RECENT:
                recent = p.pp[max(0, j - n) : j]
                recent = recent[~np.isnan(recent)]
                if pid in pp_base and len(recent):
                    ratio = (recent.mean() + PP_SMOOTH) / (pp_base[pid] + PP_SMOOTH)
                    pp_usage[n] = float(np.clip(ratio, *PP_CLIP))
                else:
                    pp_usage[n] = 1.0
            out.append(
                Sample(
                    prior=priors[pid],
                    windows=windows,
                    week=(jw - j, p.cum[jw] - p.cum[j]),
                    rest=(end - j, p.cum[end] - p.cum[j]),
                    usage=usage,
                    pp_usage=pp_usage,
                )
            )
    return keys, out


@dataclass
class Batch:
    """Every sample as arrays, so a variant is one vectorised pass."""

    prior: np.ndarray  # (samples, categories)
    windows: dict  # window -> (games (samples,), totals (samples, categories))
    week: tuple[np.ndarray, np.ndarray]
    rest: tuple[np.ndarray, np.ndarray]
    usage: dict = field(default_factory=dict)  # recent -> (samples,) ice-time ratio
    pp_usage: dict = field(default_factory=dict)  # recent -> (samples,) power-play ratio
    ppp: int | None = None  # the PPP column, if the league scores it

    @classmethod
    def of(cls, data: list[Sample]) -> Batch:
        def pair(get):
            return (
                np.array([get(s)[0] for s in data], dtype=float),
                np.array([get(s)[1] for s in data], dtype=float),
            )

        return cls(
            prior=np.array([s.prior for s in data], dtype=float),
            windows={w: pair(lambda s, w=w: s.windows[w]) for w in WINDOWS},
            week=pair(lambda s: s.week),
            rest=pair(lambda s: s.rest),
            usage={n: np.array([s.usage.get(n, 1.0) for s in data]) for n in USAGE_RECENT},
            pp_usage={n: np.array([s.pp_usage.get(n, 1.0) for s in data]) for n in PP_RECENT},
        )

    def rate(
        self, window, k: np.ndarray, usage=None, gamma: float = 0.0, pp=None, pp_gamma: float = 0.0
    ) -> np.ndarray:
        n, total = self.windows[window]
        prior = self.prior
        if usage is not None and gamma:
            prior = prior * (self.usage[usage] ** gamma)[:, None]
        if pp is not None and pp_gamma and self.ppp is not None:
            prior = prior.copy()
            prior[:, self.ppp] *= self.pp_usage[pp] ** pp_gamma
        return (total + k * prior) / (n[:, None] + k)

    def changed(self, usage) -> np.ndarray:
        """Samples whose ice time moved by more than USAGE_CHANGED."""
        return np.abs(self.usage[usage] - 1.0) > USAGE_CHANGED

    def subset(self, mask: np.ndarray) -> Batch:
        return Batch(
            prior=self.prior[mask],
            windows={w: (n[mask], t[mask]) for w, (n, t) in self.windows.items()},
            week=(self.week[0][mask], self.week[1][mask]),
            rest=(self.rest[0][mask], self.rest[1][mask]),
            usage={k: v[mask] for k, v in self.usage.items()},
            pp_usage={k: v[mask] for k, v in self.pp_usage.items()},
            ppp=self.ppp,
        )

    def deviance(self, rate: np.ndarray, horizon: str) -> np.ndarray:
        """Per-category deviance summed over samples, for rate x games played."""
        games, y = getattr(self, horizon)
        return _deviance(y, rate * games[:, None]).sum(axis=0)


def score(
    keys: list[str],
    batch: Batch,
    variants: dict[str, tuple],
    season: str,
) -> FormScores:
    """variants: name -> (window, k vector) or ("pre",)."""
    out = FormScores(season=season, keys=keys)
    for name, spec in variants.items():
        rate = batch.prior if spec[0] == "pre" else batch.rate(spec[0], spec[1], *spec[2:])
        out.deviance[name] = {h: batch.deviance(rate, h) for h in ("week", "rest")}
        out.n[name] = len(batch.prior)
    return out


def fit_per_category(keys: list[str], batch: Batch, window) -> np.ndarray:
    """The k for each category that forecasts the next week best on `batch`."""
    losses = np.array(
        [batch.deviance(batch.rate(window, np.full(len(keys), k)), "week") for k in PC_GRID]
    )
    return np.array([PC_GRID[i] for i in np.argmin(losses, axis=0)])


def _wname(w) -> str:
    return "all" if w is None else str(w)


@dataclass(frozen=True)
class FormGateReport:
    text: str
    scores: dict[str, FormScores]
    per_category_k: dict[str, np.ndarray]


def form_gate_report(
    conn: sqlite3.Connection,
    league: LeagueConfig,
    seasons: tuple[str, ...] = SEASONS,
    fit_season: str = FIT_SEASON,
    progress: Callable[[str], None] | None = None,
) -> FormGateReport:
    say = progress or (lambda _m: None)
    loaded = {}
    for season in dict.fromkeys((fit_season, *seasons)):
        say(f"{season}: samples")
        keys_s, data = samples(conn, season, league)
        batch = Batch.of(data)
        batch.ppp = keys_s.index("ppp") if "ppp" in keys_s else None
        loaded[season] = (keys_s, batch)
    keys = loaded[fit_season][0]
    say(f"{fit_season}: fitting a k per category")
    pc = {w: fit_per_category(keys, loaded[fit_season][1], w) for w in WINDOWS}

    variants: dict[str, tuple] = {"pre": ("pre",)}
    for w in WINDOWS:
        for k in KS:
            variants[f"w{_wname(w)}-k{k:g}"] = (w, np.full(len(keys), k))
        variants[f"w{_wname(w)}-pc"] = (w, pc[w])
    for n in USAGE_RECENT:
        for g in USAGE_GAMMAS:
            variants[f"wall-pc-u{n}-g{g:g}"] = (None, pc[None], n, g)
    for n in PP_RECENT:
        for g in PP_GAMMAS:
            variants[f"wall-pc-pp{n}-g{g:g}"] = (None, pc[None], None, 0.0, n, g)
            variants[f"wall-pc-u5-pp{n}-g{g:g}"] = (None, pc[None], 5, 0.5, n, g)

    scores = {}
    for season in seasons:
        say(f"{season}: scoring")
        k_, data = loaded[season]
        scores[season] = score(k_, data, variants, season)

    lines = [
        "Form: per-category rates pulled toward recent games, scored on what came next",
        f"(Poisson deviance against preseason rates alone; negative is better; "
        f"every {STRIDE}th game date after the first {SKIP_DATES})",
        "",
        f"  {'variant':12}"
        + "".join(f"{s[2:4] + '-' + s[6:] + ' wk':>11}" for s in seasons)
        + "".join(f"{s[2:4] + '-' + s[6:] + ' rest':>12}" for s in seasons),
    ]
    for name in variants:
        if name == "pre":
            continue
        lines.append(
            f"  {name:12}"
            + "".join(f"{scores[s].vs_pre(name, 'week'):>11.2%}" for s in seasons)
            + "".join(f"{scores[s].vs_pre(name, 'rest'):>12.2%}" for s in seasons)
        )
    lines += [
        "",
        f"Players whose ice time moved by more than {USAGE_CHANGED:.0%} (last 10 games against "
        f"last season), same measure:",
    ]
    picked = ["wall-pc"] + [v for v in variants if v.startswith("wall-pc-u")]
    for season in seasons:
        k_, batch = loaded[season]
        sub_ = batch.subset(batch.changed(10))
        sc = score(k_, sub_, {v: variants[v] for v in ("pre", *picked)}, season)
        cells = [f"{v} {sc.vs_pre(v, 'week'):+.1%}/{sc.vs_pre(v, 'rest'):+.1%}" for v in picked]
        lines.append(f"  {season} (n={sc.n['pre']}): " + "  ".join(cells))
    if "ppp" in keys:
        c = keys.index("ppp")
        lines += ["", "PPP alone (next week / rest of season, against preseason PPP):"]
        for name in ["wall-pc"] + [v for v in variants if "-pp" in v or v.startswith("wall-pc-u5")]:
            cells = []
            for season in seasons:
                sc = scores[season]
                pre = sc.deviance["pre"]
                cells.append(
                    f"{(sc.deviance[name]['week'][c] - pre['week'][c]) / pre['week'][c]:+.1%}/"
                    f"{(sc.deviance[name]['rest'][c] - pre['rest'][c]) / pre['rest'][c]:+.1%}"
                )
            lines.append(f"  {name:22}" + "  ".join(f"{x:>15}" for x in cells))
    lines += ["", f"per-category k (fit on {fit_season}, next-week deviance):"]
    for w in WINDOWS:
        pairs = zip(keys, pc[w], strict=True)
        lines.append(f"  window {_wname(w):>3}: " + "  ".join(f"{k} {v:g}" for k, v in pairs))
    lines += ["", "samples: " + ", ".join(f"{s} {scores[s].n['pre']}" for s in seasons)]
    return FormGateReport(
        text="\n".join(lines), scores=scores, per_category_k={_wname(w): pc[w] for w in WINDOWS}
    )
