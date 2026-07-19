from __future__ import annotations

import json
import sqlite3

import pandas as pd

# game-log JSON key -> our column name (skaters)
SKATER_LOG_KEYS = {
    "goals": "goals",
    "assists": "assists",
    "points": "points",
    "plusMinus": "plus_minus",
    "pim": "pim",
    "powerPlayPoints": "ppp",
    "shots": "sog",
    "shorthandedPoints": "shp",
    "gameWinningGoals": "gwg",
}

SKATER_COLS = list(SKATER_LOG_KEYS.values())


def toi_seconds(toi: str) -> int:
    """'62:13' -> 3733. Minutes exceed 59 for goalies in OT games."""
    m, s = toi.split(":")
    return int(m) * 60 + int(s)


def season_aggregates(conn: sqlite3.Connection, season: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(skaters, goalies) per-player season totals from nhl_game_logs.

    Skaters: gp + summed counting stats (SKATER_COLS).
    Goalies: gp, starts, wins, shutouts, shots_against, goals_against, toi_hours,
    plus derived save_pct and gaa (per 60). Split by nhl_players.position == 'G'.
    Both frames are indexed by player_id and carry name/team/position.
    """
    players = pd.read_sql_query(
        "SELECT player_id, full_name AS name, team_abbrev AS team, position FROM nhl_players",
        conn,
        index_col="player_id",
    )
    goalie_ids = set(players.index[players["position"] == "G"])

    skater_rows: list[dict] = []
    goalie_rows: list[dict] = []
    cur = conn.execute(
        "SELECT player_id, stats_json FROM nhl_game_logs WHERE season = ?", (season,)
    )
    for pid, stats_json in cur:
        s = json.loads(stats_json)
        if pid in goalie_ids:
            sa = s.get("shotsAgainst") or 0
            ga = s.get("goalsAgainst") or 0
            goalie_rows.append(
                {
                    "player_id": pid,
                    "starts": s.get("gamesStarted", 0),
                    "wins": 1 if s.get("decision") == "W" else 0,
                    "shutouts": s.get("shutouts", 0),
                    "shots_against": sa,
                    "goals_against": ga,
                    "toi_hours": toi_seconds(s["toi"]) / 3600 if s.get("toi") else 0.0,
                }
            )
        else:
            row = {"player_id": pid}
            for key, col in SKATER_LOG_KEYS.items():
                row[col] = s.get(key) or 0
            skater_rows.append(row)

    def _finish(rows: list[dict]) -> pd.DataFrame:
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        agg = df.groupby("player_id").sum()
        agg["gp"] = df.groupby("player_id").size()
        return agg.join(players, how="left")

    skaters = _finish(skater_rows)
    goalies = _finish(goalie_rows)
    if not goalies.empty:
        goalies["save_pct"] = 1.0 - goalies["goals_against"] / goalies["shots_against"].where(
            goalies["shots_against"] > 0
        )
        goalies["gaa"] = goalies["goals_against"] / goalies["toi_hours"].where(
            goalies["toi_hours"] > 0
        )
    return skaters, goalies


def season_games(conn: sqlite3.Connection, season: str, default: int = 82) -> int:
    """Games each team plays that season (82, or 84 from 2026-27) from the schedule."""
    row = conn.execute(
        """
        SELECT MAX(n) FROM (
            SELECT team, COUNT(*) n FROM (
                SELECT home_team team FROM nhl_schedule WHERE season = ? AND game_type = 2
                UNION ALL
                SELECT away_team FROM nhl_schedule WHERE season = ? AND game_type = 2
            ) GROUP BY team
        )
        """,
        (season, season),
    ).fetchone()
    return row[0] or default
