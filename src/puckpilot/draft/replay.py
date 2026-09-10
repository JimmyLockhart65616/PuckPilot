from __future__ import annotations

import json
import sqlite3
from datetime import date as _date
from datetime import timedelta

import numpy as np
from scipy.stats import rankdata

from puckpilot.engine.aggregate import toi_seconds
from puckpilot.engine.categories import SKATER_CATS_DEFAULT, Category
from puckpilot.engine.valuation import LeagueShape

# skater category -> game-log JSON key. hits/blocks are absent from the game log
# and come from nhl_boxscore_stats instead (see BOX_KEYS).
LOG_KEYS = {
    "goals": "goals",
    "assists": "assists",
    "points": "points",
    "plus_minus": "plusMinus",
    "pim": "pim",
    "ppp": "powerPlayPoints",
    "sog": "shots",
    "shp": "shorthandedPoints",
    "gwg": "gameWinningGoals",
}
BOX_KEYS = {
    "hits": "hits",
    "blocks": "blockedShots",
}

SKATER_KEYS = [c.key for c in SKATER_CATS_DEFAULT]

# goalie accumulator layout — a superset from which any goalie category derives
G_WINS, G_SHO, G_GA, G_SA, G_HOURS, G_STARTS = range(6)
G_WIDTH = 6


def goalie_values(acc: np.ndarray, cats: tuple[Category, ...]) -> np.ndarray:
    """Category values from goalie accumulators. Works on a single vector or a
    stack (last axis = accumulator), returning the same leading shape."""
    a = np.asarray(acc, dtype=float)
    sa, ga, hours = a[..., G_SA], a[..., G_GA], a[..., G_HOURS]
    with np.errstate(invalid="ignore", divide="ignore"):
        out = []
        for c in cats:
            if c.key == "wins":
                v = a[..., G_WINS]
            elif c.key == "shutouts":
                v = a[..., G_SHO]
            elif c.key == "saves":
                v = sa - ga
            elif c.key == "shots_against":
                # A scoring category in the real league (Yahoo stat_id 24,
                # sort_order=1 -> higher is better). It reads as a bad thing but
                # as a counting stat it is a pure workload proxy, which is the
                # most repeatable goalie signal there is.
                v = sa
            elif c.key == "save_pct":
                v = np.where(sa > 0, 1.0 - ga / np.where(sa > 0, sa, 1.0), 0.0)
            elif c.key == "gaa":
                v = np.where(hours > 0, ga / np.where(hours > 0, hours, 1.0), 0.0)
            else:
                raise KeyError(f"no goalie accumulator mapping for category {c.key!r}")
            out.append(v)
    return np.stack(out, axis=-1)


def week_indices(dates: list[str]) -> np.ndarray:
    """Date index -> fantasy week index. Yahoo weeks run Monday-Sunday.

    Real game dates are ISO; synthetic test labels fall back to fixed 7-entry
    blocks, which preserves the only property callers rely on (a monotonic
    grouping of consecutive dates).
    """
    if not dates:
        return np.zeros(0, dtype=int)
    try:
        d0 = _date.fromisoformat(dates[0])
    except ValueError:
        return np.arange(len(dates), dtype=int) // 7
    start = d0 - timedelta(days=d0.weekday())
    return np.array([(_date.fromisoformat(d) - start).days // 7 for d in dates], dtype=int)


class ReplayData:
    """Per-player per-date stat vectors for one real season (hindsight ground truth)."""

    def __init__(self, skater_keys: list[str] | None = None) -> None:
        self._dates: list[str] = []
        self.weeks: np.ndarray = np.zeros(0, dtype=int)
        self.skater_keys: list[str] = skater_keys or list(SKATER_KEYS)
        self.skater: dict[int, dict[int, np.ndarray]] = {}
        self.goalie: dict[int, dict[int, np.ndarray]] = {}

    @property
    def dates(self) -> list[str]:
        return self._dates

    @dates.setter
    def dates(self, value: list[str]) -> None:
        """Week indices are derived from the calendar, so they stay in step."""
        self._dates = list(value)
        self.weeks = week_indices(self._dates)

    def player_dates(self, pid: int) -> list[int]:
        d = self.skater.get(pid) or self.goalie.get(pid) or {}
        return sorted(d)

    @property
    def n_weeks(self) -> int:
        return int(self.weeks.max()) + 1 if len(self.weeks) else 0


def build_replay_data(
    conn: sqlite3.Connection, season: str, skater_keys: list[str] | None = None
) -> ReplayData:
    keys = skater_keys or list(SKATER_KEYS)
    goalie_ids = {
        r[0] for r in conn.execute("SELECT player_id FROM nhl_players WHERE position = 'G'")
    }
    # LEFT JOIN so games without a boxscore still count, with hits/blocks 0
    rows = conn.execute(
        "SELECT l.player_id, l.game_date, l.stats_json, b.stats_json FROM nhl_game_logs l"
        " LEFT JOIN nhl_boxscore_stats b"
        "   ON b.game_id = l.game_id AND b.player_id = l.player_id"
        " WHERE l.season = ?",
        (season,),
    ).fetchall()

    data = ReplayData(keys)
    data.dates = sorted({r[1] for r in rows})
    didx = {d: i for i, d in enumerate(data.dates)}

    for pid, date, stats_json, box_json in rows:
        s = json.loads(stats_json)
        i = didx[date]
        if pid in goalie_ids:
            vec = np.zeros(G_WIDTH)
            vec[G_WINS] = 1.0 if s.get("decision") == "W" else 0.0
            vec[G_SHO] = float(s.get("shutouts") or 0)
            vec[G_GA] = float(s.get("goalsAgainst") or 0)
            vec[G_SA] = float(s.get("shotsAgainst") or 0)
            vec[G_HOURS] = toi_seconds(s["toi"]) / 3600 if s.get("toi") else 0.0
            vec[G_STARTS] = float(s.get("gamesStarted") or 0)
            data.goalie.setdefault(pid, {})[i] = vec
        else:
            box = json.loads(box_json) if box_json else {}
            vec = np.array(
                [
                    float((box.get(BOX_KEYS[k]) if k in BOX_KEYS else s.get(LOG_KEYS[k])) or 0)
                    for k in keys
                ]
            )
            data.skater.setdefault(pid, {})[i] = vec
    return data


def replay_roster(
    roster: list[int],
    positions: dict[int, str],
    scalar: dict[int, float],
    data: ReplayData,
    shape: LeagueShape,
) -> tuple[np.ndarray, np.ndarray]:
    """Replay one roster over the season with daily greedy lineup fill.

    Players are prioritized by `scalar` (draft-time projected value — no
    hindsight leaks into who gets the slot). Single-position eligibility makes
    greedy fill optimal: position slots first, then util for skaters.

    Returns per-week (weeks, n_skater_cats) and (weeks, G_WIDTH) accumulators;
    sum over axis 0 for season totals.
    """
    slot_base = dict(shape.slots)
    order = sorted(roster, key=lambda p: scalar.get(p, 0.0), reverse=True)
    by_date: dict[int, list[int]] = {}
    for pid in order:
        for i in data.player_dates(pid):
            by_date.setdefault(i, []).append(pid)

    n_weeks = max(data.n_weeks, 1)
    sk_total = np.zeros((n_weeks, len(data.skater_keys)))
    g_total = np.zeros((n_weeks, G_WIDTH))
    for i, pids in by_date.items():
        w = int(data.weeks[i])
        slots = slot_base.copy()
        util = shape.util_slots
        for pid in pids:
            pos = positions.get(pid)
            if pos == "G":
                if slots.get("G", 0) > 0:
                    slots["G"] -= 1
                    g_total[w] += data.goalie[pid][i]
            elif slots.get(pos, 0) > 0:
                slots[pos] -= 1
                sk_total[w] += data.skater[pid][i]
            elif util > 0:
                util -= 1
                sk_total[w] += data.skater[pid][i]
    return sk_total, g_total


def category_totals(
    sk: np.ndarray,
    g: np.ndarray,
    skater_cats: tuple[Category, ...],
    goalie_cats: tuple[Category, ...],
    skater_keys: list[str],
) -> np.ndarray:
    """Stack skater + goalie category values, sign-flipped so higher is always better.

    sk: (..., n_skater_keys) totals; g: (..., G_WIDTH) accumulators.
    """
    col = {k: i for i, k in enumerate(skater_keys)}
    sk_vals = np.stack([sk[..., col[c.key]] for c in skater_cats], axis=-1)
    g_vals = goalie_values(g, goalie_cats)
    vals = np.concatenate([sk_vals, g_vals], axis=-1)
    signs = np.array([1.0 if c.higher_is_better else -1.0 for c in skater_cats + goalie_cats])
    return vals * signs


def roto_standings(
    sk: np.ndarray,
    g: np.ndarray,
    skater_cats: tuple[Category, ...] = SKATER_CATS_DEFAULT,
    goalie_cats: tuple[Category, ...] | None = None,
    skater_keys: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Roto points across teams. sk: (T, weeks, cats) or (T, cats); g likewise.

    Returns (points per team, finish position per team, 1 = champion).
    """
    from puckpilot.engine.categories import GOALIE_CATS_DEFAULT

    goalie_cats = GOALIE_CATS_DEFAULT if goalie_cats is None else goalie_cats
    skater_keys = skater_keys or [c.key for c in skater_cats]
    if sk.ndim == 3:  # sum weekly totals into season totals
        sk, g = sk.sum(axis=1), g.sum(axis=1)

    vals = category_totals(sk, g, skater_cats, goalie_cats, skater_keys)
    points = np.zeros(sk.shape[0])
    for j in range(vals.shape[1]):
        points += rankdata(vals[:, j])  # best value gets T points, average ties
    finish = rankdata(-points, method="ordinal")
    return points, finish
