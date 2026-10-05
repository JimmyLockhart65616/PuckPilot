"""Hot streaks: how often a surge lasts, per category, and for whom.

A player is *hot* in a category when his last `WINDOW` games beat what he was
expected to produce going in by a margin that would happen by chance only
`ALPHA` of the time (a Poisson tail, overdispersed for penalty minutes). The
expectation going in is the model's own: the season so far shrunk toward the
preseason rate (`season.form`), as of the game before the streak began.

For every such onset - the game it first becomes true - his production over
the games that follow is compared with that same pre-streak expectation, in
windows: the next 1-3 games, 4-7, 8-14 and 15-28. Everyone who was not hot is
the control: the same comparison, so any drift that has nothing to do with
streaks (projections that run low for young players, say) cancels out. What is
left - hot minus control - is how much of a streak is still there, and for
how long.

Then the same against the current model's forecast at the moment of onset,
which already gives the streak games their weight: anything left there is what
the tool would gain by reacting more than it does.

Splits that matter for a roster's last spots: players whose preseason value
sits around replacement level, and streaks that came with more ice time (or
power-play time, for PPP) - a role change - against streaks that did not.

Standard errors are clustered by player: one player's games are not
independent draws.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from puckpilot.league import LeagueConfig
from puckpilot.season.streaks import (
    DISPERSION,
    MIN_GAMES,
    ROLE_BEFORE_MIN,
    ROLE_UP,
    STREAK_ALPHA,
    STREAK_GAMES,
    tail,
)

SEASONS = ("20232024", "20242025", "20252026")

# The definitions are the live ones (`season.streaks`), so what is measured
# here is exactly what the tool flags.
WINDOW = STREAK_GAMES
ALPHA = STREAK_ALPHA
TOI_UP = ROLE_UP
AHEAD = ((0, 3), (3, 7), (7, 14), (14, 28))  # future windows, games after onset

# Bands by preseason value per game among skaters. A 12-team league starts
# about 156 skaters; the next 150 are the waiver wire and the bottom of rosters.
BANDS = (("top", 0, 156), ("replacement", 156, 306), ("deep", 306, 10_000))


def _tail(s, e, disp):
    return tail(s, e, disp)


def _role_up(series: np.ndarray, js: np.ndarray, last: float | None = None) -> np.ndarray:
    """For each evaluated game: was his ice time over the streak at least
    TOI_UP above his ice time before it this season? With fewer than
    ROLE_BEFORE_MIN games before it, his average last season stands in."""
    flags = np.zeros(len(js), dtype=bool)
    for i, j in enumerate(js):
        during = series[j - WINDOW : j]
        before = series[: j - WINDOW]
        during, before = during[~np.isnan(during)], before[~np.isnan(before)]
        base = before.mean() if len(before) >= ROLE_BEFORE_MIN else last
        if len(during) and base:
            flags[i] = during.mean() >= (1.0 + TOI_UP) * base
    return flags


@dataclass
class Cases:
    """One row per (player, game) evaluated, per category column."""

    keys: list[str]
    pid: list[np.ndarray] = field(default_factory=list)
    band: list[np.ndarray] = field(default_factory=list)
    hot: list[np.ndarray] = field(default_factory=list)  # (rows, cats)
    onset: list[np.ndarray] = field(default_factory=list)
    # games the trailing window stays hot from this onset (0 where no onset)
    run: list[np.ndarray] = field(default_factory=list)
    streak: list[np.ndarray] = field(default_factory=list)  # (rows, cats, 2): actual, expected
    toi_up: list[np.ndarray] = field(default_factory=list)  # (rows,)
    pp_up: list[np.ndarray] = field(default_factory=list)  # (rows,)
    # (rows, cats, windows, 3): actual, expected before the streak, expected now
    ahead: list[np.ndarray] = field(default_factory=list)

    def stack(self) -> dict[str, np.ndarray]:
        return {name: np.concatenate(getattr(self, name)) for name in FIELDS}


FIELDS = ("pid", "band", "hot", "onset", "run", "streak", "toi_up", "pp_up", "ahead")


def _season_means(by_player: dict[int, dict[str, float]]) -> dict[int, float]:
    """player -> his average over a season of 10+ games."""
    return {pid: float(np.mean(list(v.values()))) for pid, v in by_player.items() if len(v) >= 10}


def _priors(conn, season: str, keys: list[str]) -> dict[int, np.ndarray]:
    from puckpilot.season.form_gate import _priors as priors

    return priors(conn, season, keys)


def _minutes(conn, season: str) -> dict[int, dict[str, float]]:
    from puckpilot.season.form_gate import _toi

    return _toi(conn, season)


def _pp_seconds(conn, season: str) -> dict[int, dict[str, float]]:
    from puckpilot.season.form_gate import _pp

    return _pp(conn, season)


def cases(
    conn: sqlite3.Connection,
    season: str,
    league: LeagueConfig,
    min_games: int = MIN_GAMES,
    max_games: int | None = None,
) -> Cases:
    """Every (player, game) from his `min_games`-th (up to `max_games`), with
    what came next."""
    from puckpilot.draft.replay import build_replay_data
    from puckpilot.season.form import DEFAULT_FORM_K, FORM_RATE_K

    keys = [c.key for c in league.skater_cats]
    data = build_replay_data(conn, season, keys)
    priors = _priors(conn, season, keys)
    k = np.array([FORM_RATE_K.get(key, DEFAULT_FORM_K) for key in keys])
    disp = np.array([DISPERSION.get(key, 1.0) for key in keys])

    # Preseason value per game: each category in units of its game-to-game
    # spread, as GameValueModel scores a line. Ranked into BANDS.
    lines = np.array([v for g in data.skater.values() for v in g.values()], dtype=float)
    sd = np.where(lines.std(axis=0) > 0, lines.std(axis=0), 1.0)
    value = {pid: float((p / sd).sum()) for pid, p in priors.items()}
    order = sorted(value, key=lambda pid: -value[pid])
    band_of = {}
    for b, (_, lo, hi) in enumerate(BANDS):
        for pid in order[lo:hi]:
            band_of[pid] = b

    minutes, pp = _minutes(conn, season), _pp_seconds(conn, season)
    y = int(season[:4])
    last_min = _season_means(_minutes(conn, f"{y - 1}{y}"))
    last_pp = _season_means(_pp_seconds(conn, f"{y - 1}{y}"))
    out = Cases(keys=keys)
    for pid, games in data.skater.items():
        if pid not in priors or len(games) < MIN_GAMES + 1:
            continue
        days = sorted(games)
        x = np.array([games[i] for i in days], dtype=float)
        n = len(days)
        cum = np.vstack([np.zeros(len(keys)), np.cumsum(x, axis=0)])
        prior = priors[pid]
        mins = np.array([minutes.get(pid, {}).get(data.dates[i], np.nan) for i in days])
        secs = np.array([pp.get(pid, {}).get(data.dates[i], np.nan) for i in days])

        js = np.arange(max(min_games, WINDOW + 1), n if max_games is None else min(n, max_games))
        if not len(js):
            continue
        pre_n = (js - WINDOW)[:, None]
        r_pre = (cum[js - WINDOW] + k * prior) / (pre_n + k)
        s = cum[js] - cum[js - WINDOW]
        e = WINDOW * r_pre
        p = _tail(s, e, disp)
        hot = p <= ALPHA
        prev = np.vstack([np.zeros((1, len(keys)), dtype=bool), hot[:-1]])
        onset = hot & ~prev
        r_now = (cum[js] + k * prior) / (js[:, None] + k)

        ahead = np.full((len(js), len(keys), len(AHEAD), 3), np.nan)
        for w, (a, b) in enumerate(AHEAD):
            ok = js + b <= n
            jj = js[ok]
            ahead[ok, :, w, 0] = cum[jj + b] - cum[jj + a]
            ahead[ok, :, w, 1] = (b - a) * r_pre[ok]
            ahead[ok, :, w, 2] = (b - a) * r_now[ok]

        out.pid.append(np.full(len(js), pid))
        out.band.append(np.full(len(js), band_of.get(pid, len(BANDS) - 1)))
        out.hot.append(hot)
        out.onset.append(onset)
        run = np.zeros(hot.shape, dtype=int)
        for ci in range(len(keys)):
            count = 0
            for i in range(len(js) - 1, -1, -1):
                count = count + 1 if hot[i, ci] else 0
                run[i, ci] = count if onset[i, ci] else 0
        out.run.append(run)
        out.streak.append(np.stack([s, e], axis=-1))
        out.toi_up.append(_role_up(mins, js, last_min.get(pid)))
        out.pp_up.append(_role_up(secs, js, last_pp.get(pid)))
        out.ahead.append(ahead)
    return out


@dataclass(frozen=True)
class Excess:
    value: float  # actual / expected - 1, pooled
    se: float  # clustered by player
    n: int  # cases


def excess(actual: np.ndarray, expected: np.ndarray, pid: np.ndarray) -> Excess:
    """Ratio-of-sums excess with a player-clustered standard error."""
    ok = ~np.isnan(actual) & ~np.isnan(expected)
    actual, expected, pid = actual[ok], expected[ok], pid[ok]
    if not len(actual) or expected.sum() <= 0:
        return Excess(float("nan"), float("nan"), 0)
    r = actual.sum() / expected.sum()
    resid = actual - r * expected
    _, inv = np.unique(pid, return_inverse=True)
    by_player = np.bincount(inv, weights=resid)
    se = float(np.sqrt((by_player**2).sum()) / expected.sum())
    return Excess(float(r - 1.0), se, int(len(actual)))


def _diff(a: Excess, b: Excess) -> tuple[float, float]:
    return a.value - b.value, float(np.hypot(a.se, b.se))


@dataclass(frozen=True)
class StreakReport:
    text: str
    rows: dict  # (category, subset) -> per-window (hot - control, se), and onset counts


def _section(title: str, keys, c, mask_hot, mask_ctrl, ref: int, labels) -> tuple[list, dict]:
    """Rows of hot-minus-control excess per window for one subset."""
    lines = [title]
    head = f"  {'cat':5}{'onsets':>7}{'streak':>8}" + "".join(f"{lab:>14}" for lab in labels)
    lines.append(head)
    rows = {}
    for ci, key in enumerate(keys):
        h = mask_hot[:, ci]
        ctl = mask_ctrl[:, ci]
        if h.sum() < 20:
            lines.append(f"  {key[:5]:5}{int(h.sum()):>7}   (too few)")
            continue
        streak = excess(c["streak"][h, ci, 0], c["streak"][h, ci, 1], c["pid"][h])
        cells, per = [], []
        for w in range(len(AHEAD)):
            hot_x = excess(c["ahead"][h, ci, w, 0], c["ahead"][h, ci, w, ref], c["pid"][h])
            ctl_x = excess(c["ahead"][ctl, ci, w, 0], c["ahead"][ctl, ci, w, ref], c["pid"][ctl])
            d, se = _diff(hot_x, ctl_x)
            per.append((d, se))
            cells.append(f"{d:+7.1%}+/-{se:4.1%}")
        rows[key] = {"onsets": int(h.sum()), "streak": streak.value, "ahead": per}
        lines.append(
            f"  {key[:5]:5}{int(h.sum()):>7}{streak.value:>+8.0%}"
            + "".join(f"{x:>14}" for x in cells)
        )
    return lines, rows


def streak_report(
    conn: sqlite3.Connection,
    league: LeagueConfig,
    seasons: tuple[str, ...] = SEASONS,
    progress: Callable[[str], None] | None = None,
) -> StreakReport:
    say = progress or (lambda _m: None)
    parts = []
    for season in seasons:
        say(f"{season}: cases")
        parts.append(cases(conn, season, league))
    keys = parts[0].keys
    merged = Cases(keys=keys)
    for p in parts:
        for name in FIELDS:
            getattr(merged, name).extend(getattr(p, name))
    c = merged.stack()
    labels = [f"g{a + 1}-{b}" for a, b in AHEAD]
    onset, hot = c["onset"], c["hot"]
    ctrl = ~hot
    rep = (c["band"] == 1)[:, None]
    toi_up = c["toi_up"][:, None]
    ppp = keys.index("ppp") if "ppp" in keys else None
    role_up = np.repeat(toi_up, len(keys), axis=1)
    if ppp is not None:
        role_up[:, ppp] = c["pp_up"]

    lines = [
        f"Hot streaks: last {WINDOW} games beat the expectation going in at p <= {ALPHA:.2f}",
        f"({', '.join(s[2:4] + '-' + s[6:] for s in seasons)} pooled; skaters with "
        f"{MIN_GAMES}+ games; each cell: hot minus not-hot, actual over expected - 1, "
        f"player-clustered SE)",
        "",
        f"How often: a skater is hot in a category on {hot.mean():.1%} of games on average "
        f"({ALPHA:.0%} would be pure chance).",
        "",
    ]
    rows: dict = {}
    for title, mh, mc, ref, key in (
        ("All skaters - against his expectation before the streak", onset, ctrl, 1, "all"),
        (
            "Replacement level (preseason ranks 157-306) - before the streak",
            onset & rep,
            ctrl & rep,
            1,
            "replacement",
        ),
        (
            "Streak WITH more ice time (PPP: power-play time) - before the streak",
            onset & role_up,
            ctrl,
            1,
            "role_up",
        ),
        (
            "Streak WITHOUT more ice time - before the streak",
            onset & ~role_up,
            ctrl,
            1,
            "role_flat",
        ),
        (
            "All skaters - against the current model at onset (form rates)",
            onset,
            ctrl,
            2,
            "vs_model",
        ),
        (
            "Replacement level - against the current model at onset",
            onset & rep,
            ctrl & rep,
            2,
            "replacement_vs_model",
        ),
        (
            "Streak WITH more ice time - against the current model at onset",
            onset & role_up,
            ctrl,
            2,
            "role_up_vs_model",
        ),
        (
            "Streak WITHOUT more ice time - against the current model at onset",
            onset & ~role_up,
            ctrl,
            2,
            "role_flat_vs_model",
        ),
    ):
        sec, r = _section(title, keys, c, mh, mc, ref, labels)
        lines += sec + [""]
        rows[key] = r

    # How long a streak stays a streak, and the next week against the model.
    lines.append(
        "Run length: games the last-5 window stays hot from onset "
        "(median / mean / still hot after 5 games)"
    )
    for ci, key in enumerate(keys):
        runs = c["run"][:, ci][onset[:, ci]]
        if len(runs):
            lines.append(
                f"  {key[:5]:5} {np.median(runs):>4.0f} / {runs.mean():>4.1f} / "
                f"{(runs > 5).mean():>4.0%}"
            )
    lines += [
        "",
        "Next 7 games beyond the current model (games 1-7 pooled; hot minus not-hot):",
        f"  {'cat':5}{'all hot':>17}{'+ice time':>17}{'no ice time':>17}{'replacement':>17}"
        f"{'repl.+ice':>17}",
    ]
    summary = {}
    for ci, key in enumerate(keys):
        cells = []
        for mask in (onset, onset & role_up, onset & ~role_up, onset & rep, onset & rep & role_up):
            got = []
            for m in (mask[:, ci], ctrl[:, ci]):
                week = c["ahead"][m, ci, :2]  # games 1-3 and 4-7
                ok = ~np.isnan(week[:, 1, 0])
                got.append(excess(week[ok, :, 0].sum(1), week[ok, :, 2].sum(1), c["pid"][m][ok]))
            cells.append(_diff(*got))
        summary[key] = cells
        lines.append(f"  {key[:5]:5}" + "".join(f"{d:>+10.1%}+/-{se:4.1%}" for d, se in cells))
    rows["next_week_vs_model"] = summary
    return StreakReport(text="\n".join(lines), rows=rows)


def momentum_table(
    conn: sqlite3.Connection, league: LeagueConfig, seasons: tuple[str, ...]
) -> dict[str, dict[str, float]]:
    """`streaks.MOMENTUM` as these seasons alone measure it.

    The excess over the model for the next 7 games of every hot game (not just
    onsets - the tool flags a player on any day of a streak), with more ice
    time and without, kept only where it clears 2 standard errors. The add
    gate prices a season with the table its other seasons give, so nothing it
    is judged on helped set the numbers.
    """
    parts = [cases(conn, s, league) for s in seasons]
    keys = parts[0].keys
    merged = Cases(keys=keys)
    for p in parts:
        for name in FIELDS:
            getattr(merged, name).extend(getattr(p, name))
    c = merged.stack()
    hot = c["hot"]
    role = np.repeat(c["toi_up"][:, None], len(keys), axis=1)
    if "ppp" in keys:
        role[:, keys.index("ppp")] = c["pp_up"]
    out: dict[str, dict[str, float]] = {"role": {}, "plain": {}}
    for ci, key in enumerate(keys):
        for name, mask in (("role", hot & role), ("plain", hot & ~role)):
            got = []
            for m in (mask[:, ci], ~hot[:, ci]):
                week = c["ahead"][m, ci, :2]
                ok = ~np.isnan(week[:, 1, 0])
                got.append(excess(week[ok, :, 0].sum(1), week[ok, :, 2].sum(1), c["pid"][m][ok]))
            d, se = _diff(*got)
            if abs(d) > 2 * se:
                out[name][key] = round(float(d), 3)
    return out
