"""Save % from shot quality: does it forecast better than the projection's?

A goalie's save % repeats weakly season to season (r ~ .13 measured): most of
it is the shots he faced. MoneyPuck's goalie rows carry the expected goals of
those shots (`mp_season_stats`), so a goalie's goals saved above expected per
shot on goal is his part of it, and the league's rate the rest.

`validate` asks it walk-forward, season by season: the shot-quality forecast
of a goalie's save % is the previous season's league save % plus `weight`
times his goals saved above expected per shot over the previous seasons
(shot-weighted); `weight` is fitted on the first target season only and
judged on the later ones, against the tool's own preseason projection
(`projections.project`, the `save_pct` the in-season odds use). Error is
shot-weighted mean squared error on the goalies with enough shots in both.
Nothing here is used by any engine unless it wins every judged season.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass

MIN_SHOTS = 300  # in the target season, to be scored
HISTORY = 2  # previous seasons of shot quality
WEIGHT_GRID = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5)


def _prev(season: str, k: int = 1) -> str:
    y = int(season[:4]) - k
    return f"{y}{y + 1}"


def actual(conn, season: str) -> dict[int, tuple[float, float]]:
    """goalie -> (save %, shots) over the season's game logs."""
    acc: dict[int, list[float]] = {}
    for pid, stats_json in conn.execute(
        "SELECT player_id, stats_json FROM nhl_game_logs WHERE season = ? AND game_type = 2 "
        "AND json_extract(stats_json, '$.shotsAgainst') IS NOT NULL",
        (season,),
    ):
        s = json.loads(stats_json)
        sa = float(s.get("shotsAgainst") or 0)
        ga = float(s.get("goalsAgainst") or 0)
        a = acc.setdefault(pid, [0.0, 0.0])
        a[0] += sa - ga
        a[1] += sa
    return {pid: (sv / sh, sh) for pid, (sv, sh) in acc.items() if sh > 0}


def gsax(conn, seasons: list[str]) -> dict[int, tuple[float, float]]:
    """goalie -> (goals saved above expected per shot on goal, shots) over `seasons`."""
    acc: dict[int, list[float]] = {}
    marks = ",".join("?" * len(seasons))
    for pid, stats_json in conn.execute(
        f"SELECT player_id, stats_json FROM mp_season_stats WHERE season IN ({marks}) "
        "AND kind = 'goalie' AND situation = 'all'",
        seasons,
    ):
        s = json.loads(stats_json)
        on = float(s.get("ongoal") or 0)
        if on <= 0:
            continue
        a = acc.setdefault(pid, [0.0, 0.0])
        a[0] += float(s.get("xGoals") or 0) - float(s.get("goals") or 0)
        a[1] += on
    return {pid: (g / n, n) for pid, (g, n) in acc.items() if n > 0}


def league_sv(conn, season: str) -> float:
    rows = actual(conn, season)
    shots = sum(sh for _, sh in rows.values())
    return sum(sv * sh for sv, sh in rows.values()) / shots if shots else 0.9


@dataclass(frozen=True)
class SeasonCheck:
    season: str
    goalies: int
    mse_projection: float
    mse_quality: float
    weight: float

    @property
    def better(self) -> bool:
        return self.mse_quality < self.mse_projection


def _mse(pairs: list[tuple[float, float, float]]) -> float:
    w = sum(n for _, _, n in pairs)
    return sum(n * (a - b) ** 2 for a, b, n in pairs) / w if w else float("nan")


def season_inputs(conn, season: str):
    """(actual, projection's save %, shot quality, league save % last season)."""
    from puckpilot.engine.projections import project

    train = [_prev(season, k) for k in (1, 2, 3)]
    _sk, goalies = project(conn, season, train)
    proj = {int(p): float(v) for p, v in goalies["save_pct"].items() if v and v == v}
    quality = gsax(conn, [_prev(season, k) for k in range(1, HISTORY + 1)])
    return actual(conn, season), proj, quality, league_sv(conn, _prev(season))


def check_season(inputs, weight: float, season: str) -> SeasonCheck:
    real, proj, quality, base = inputs
    pairs_p, pairs_q = [], []
    for pid, (sv, shots) in real.items():
        if shots < MIN_SHOTS or pid not in proj or pid not in quality:
            continue
        pairs_p.append((proj[pid], sv, shots))
        pairs_q.append((base + weight * quality[pid][0], sv, shots))
    return SeasonCheck(season, len(pairs_p), _mse(pairs_p), _mse(pairs_q), weight)


def validate(conn: sqlite3.Connection, seasons: list[str]) -> list[SeasonCheck]:
    """Fit the weight on the first season, judge it on every later one."""
    if len(seasons) < 2:
        raise ValueError("needs a season to fit on and at least one to judge")
    inputs = {s: season_inputs(conn, s) for s in seasons}
    first = seasons[0]
    weight = min(WEIGHT_GRID, key=lambda w: check_season(inputs[first], w, first).mse_quality)
    return [check_season(inputs[s], weight, s) for s in seasons]


def text(checks: list[SeasonCheck]) -> str:
    out = [
        "Save % from shot quality v the preseason projection (shot-weighted MSE, x1e5):",
        f"  weight on goals saved above expected per shot: {checks[0].weight:g}"
        f" (fitted on {checks[0].season})",
    ]
    for i, c in enumerate(checks):
        role = "fit" if i == 0 else "judged"
        out.append(
            f"  {c.season} ({role}, {c.goalies} goalies): projection {c.mse_projection * 1e5:.2f}"
            f"  shot quality {c.mse_quality * 1e5:.2f}  {'better' if c.better else 'worse'}"
        )
    judged = checks[1:]
    verdict = all(c.better for c in judged)
    out.append(f"  verdict: {'PASS' if verdict else 'FAIL'} (better in every judged season)")
    return "\n".join(out)
