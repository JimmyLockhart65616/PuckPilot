from __future__ import annotations

import json
import sqlite3
from typing import Protocol

import numpy as np


class GoalieStartSource(Protocol):
    """Who starts in net tonight. In-season this reads projected starting
    goalies from a public source (wired up when the season nears); in backtests
    it's hindsight truth, optionally degraded by a noise model."""

    def starts(self, date: str) -> dict[int, float]:
        """P(start) per goalie player_id for the given date (YYYY-MM-DD)."""
        ...


class HindsightGoalieSource:
    """Ground truth from game logs: gamesStarted == 1 on that date."""

    def __init__(self, conn: sqlite3.Connection, season: str):
        goalie_ids = {
            r[0] for r in conn.execute("SELECT player_id FROM nhl_players WHERE position = 'G'")
        }
        self._by_date: dict[str, dict[int, float]] = {}
        rows = conn.execute(
            "SELECT player_id, game_date, stats_json FROM nhl_game_logs WHERE season = ?",
            (season,),
        )
        for pid, date, stats_json in rows:
            if pid not in goalie_ids:
                continue
            if json.loads(stats_json).get("gamesStarted") == 1:
                self._by_date.setdefault(date, {})[pid] = 1.0

    def starts(self, date: str) -> dict[int, float]:
        return self._by_date.get(date, {})


class NoisyGoalieSource:
    """Hindsight starts degraded to morning-announcement uncertainty.

    Each true start is reported with probability `accuracy`; misses are simply
    absent (we'd bench our starter), which is the conservative failure mode.
    False positives (announced starter who then doesn't play) are not modeled.
    """

    def __init__(
        self,
        hindsight: HindsightGoalieSource,
        accuracy: float = 0.9,
        rng: np.random.Generator | None = None,
    ):
        self._hindsight = hindsight
        self.accuracy = accuracy
        self._rng = rng or np.random.default_rng()

    def starts(self, date: str) -> dict[int, float]:
        truth = self._hindsight.starts(date)
        return {pid: p for pid, p in truth.items() if self._rng.random() < self.accuracy}
