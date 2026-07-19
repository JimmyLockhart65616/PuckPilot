from __future__ import annotations

import json
import sqlite3

import numpy as np
from scipy.stats import rankdata

from puckpilot.engine.aggregate import toi_seconds
from puckpilot.engine.categories import SKATER_CATS_DEFAULT
from puckpilot.engine.valuation import LeagueShape

SKATER_KEYS = [c.key for c in SKATER_CATS_DEFAULT]
# game-log JSON key per skater category column
LOG_KEYS = {
    "goals": "goals",
    "assists": "assists",
    "plus_minus": "plusMinus",
    "pim": "pim",
    "ppp": "powerPlayPoints",
    "sog": "shots",
}
# goalie accumulator layout: wins, shutouts, goals_against, shots_against, toi_hours
G_WINS, G_SHO, G_GA, G_SA, G_HOURS = range(5)


class ReplayData:
    """Per-player per-date stat vectors for one real season (hindsight ground truth)."""

    def __init__(self) -> None:
        self.dates: list[str] = []
        self.skater: dict[int, dict[int, np.ndarray]] = {}
        self.goalie: dict[int, dict[int, np.ndarray]] = {}

    def player_dates(self, pid: int) -> list[int]:
        d = self.skater.get(pid) or self.goalie.get(pid) or {}
        return sorted(d)


def build_replay_data(conn: sqlite3.Connection, season: str) -> ReplayData:
    goalie_ids = {
        r[0] for r in conn.execute("SELECT player_id FROM nhl_players WHERE position = 'G'")
    }
    rows = conn.execute(
        "SELECT player_id, game_date, stats_json FROM nhl_game_logs WHERE season = ?", (season,)
    ).fetchall()

    data = ReplayData()
    data.dates = sorted({r[1] for r in rows})
    didx = {d: i for i, d in enumerate(data.dates)}

    for pid, date, stats_json in rows:
        s = json.loads(stats_json)
        i = didx[date]
        if pid in goalie_ids:
            vec = np.array(
                [
                    1.0 if s.get("decision") == "W" else 0.0,
                    float(s.get("shutouts") or 0),
                    float(s.get("goalsAgainst") or 0),
                    float(s.get("shotsAgainst") or 0),
                    toi_seconds(s["toi"]) / 3600 if s.get("toi") else 0.0,
                ]
            )
            data.goalie.setdefault(pid, {})[i] = vec
        else:
            vec = np.array([float(s.get(LOG_KEYS[k]) or 0) for k in SKATER_KEYS])
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
    Returns (skater cat totals, goalie accumulator).
    """
    slot_base = dict(shape.slots)
    order = sorted(roster, key=lambda p: scalar.get(p, 0.0), reverse=True)
    by_date: dict[int, list[int]] = {}
    for pid in order:
        for i in data.player_dates(pid):
            by_date.setdefault(i, []).append(pid)

    sk_total = np.zeros(len(SKATER_KEYS))
    g_total = np.zeros(5)
    for i, pids in by_date.items():
        slots = slot_base.copy()
        util = shape.util_slots
        for pid in pids:
            pos = positions.get(pid)
            if pos == "G":
                if slots.get("G", 0) > 0:
                    slots["G"] -= 1
                    g_total += data.goalie[pid][i]
            elif slots.get(pos, 0) > 0:
                slots[pos] -= 1
                sk_total += data.skater[pid][i]
            elif util > 0:
                util -= 1
                sk_total += data.skater[pid][i]
    return sk_total, g_total


def roto_standings(sk: np.ndarray, g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Roto points across teams. sk: (T, 6) skater cat totals; g: (T, 5) goalie acc.

    Categories: 6 skater cats + W + SHO + SV% (higher better) and GAA (lower).
    Returns (points per team, finish position per team, 1 = champion).
    """
    cols = [sk[:, j] for j in range(sk.shape[1])]
    cols.append(g[:, G_WINS])
    cols.append(g[:, G_SHO])
    with np.errstate(invalid="ignore", divide="ignore"):
        sv = np.where(g[:, G_SA] > 0, 1.0 - g[:, G_GA] / g[:, G_SA], 0.0)
        gaa = np.where(g[:, G_HOURS] > 0, g[:, G_GA] / g[:, G_HOURS], np.inf)
    cols.append(sv)
    cols.append(-gaa)  # lower GAA is better

    points = np.zeros(sk.shape[0])
    for vals in cols:
        points += rankdata(vals)  # best value gets T points, average ties
    finish = rankdata(-points, method="ordinal")
    return points, finish
