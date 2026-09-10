from __future__ import annotations

import sqlite3
from pathlib import Path

# Game-log stat lines stay as JSON until Phase 2 pins which categories matter;
# then hot columns get promoted and indexed.
SCHEMA = """
CREATE TABLE IF NOT EXISTS sync_meta (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS nhl_players (
    player_id   INTEGER PRIMARY KEY,
    full_name   TEXT NOT NULL,
    position    TEXT,
    team_abbrev TEXT,
    updated_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS nhl_schedule (
    game_id        INTEGER PRIMARY KEY,
    season         TEXT NOT NULL,
    game_type      INTEGER NOT NULL,
    game_date      TEXT NOT NULL,
    start_time_utc TEXT,
    home_team      TEXT NOT NULL,
    away_team      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_nhl_schedule_date ON nhl_schedule (game_date);

CREATE TABLE IF NOT EXISTS mp_season_stats (
    player_id  INTEGER NOT NULL,
    season     TEXT NOT NULL,
    kind       TEXT NOT NULL CHECK (kind IN ('skater', 'goalie')),
    situation  TEXT NOT NULL,
    name       TEXT NOT NULL,
    team       TEXT,
    position   TEXT,
    stats_json TEXT NOT NULL,
    PRIMARY KEY (player_id, season, situation)
);
CREATE INDEX IF NOT EXISTS idx_mp_season_stats_season ON mp_season_stats (season, kind);

CREATE TABLE IF NOT EXISTS waiver_proposals (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    add_pid     INTEGER NOT NULL,
    drop_pid    INTEGER,
    reason_json TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'approved', 'rejected', 'executed'))
);

CREATE TABLE IF NOT EXISTS nhl_game_logs (
    player_id       INTEGER NOT NULL,
    game_id         INTEGER NOT NULL,
    season          TEXT NOT NULL,
    game_type       INTEGER NOT NULL,
    game_date       TEXT NOT NULL,
    team_abbrev     TEXT,
    opponent_abbrev TEXT,
    is_home         INTEGER,
    stats_json      TEXT NOT NULL,
    PRIMARY KEY (player_id, game_id)
);
CREATE INDEX IF NOT EXISTS idx_nhl_game_logs_season ON nhl_game_logs (season, game_type);

-- Per-game hits/blocks live only in the boxscore endpoint, not the player game
-- log, so HIT/BLK categories need this table joined onto nhl_game_logs.
CREATE TABLE IF NOT EXISTS nhl_boxscore_stats (
    game_id     INTEGER NOT NULL,
    player_id   INTEGER NOT NULL,
    season      TEXT NOT NULL,
    team_abbrev TEXT,
    stats_json  TEXT NOT NULL,
    PRIMARY KEY (game_id, player_id)
);
CREATE INDEX IF NOT EXISTS idx_nhl_boxscore_player ON nhl_boxscore_stats (player_id, season);

-- Biographical data (birth date drives age curves). Kept in its own table so
-- adding it needs no migration of nhl_players.
CREATE TABLE IF NOT EXISTS nhl_player_bio (
    player_id   INTEGER PRIMARY KEY,
    birth_date  TEXT,
    height_in   INTEGER,
    weight_lb   INTEGER,
    shoots      TEXT,
    updated_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Yahoo identifies drafted players only by player_key, so every pick arriving
-- from a live draft has to cross this bridge to reach an NHL player_id. Built
-- once before the draft; nhl_player_id is NULL for players we cannot match
-- (minor leaguers, late call-ups) rather than dropped, so the gap is visible.
CREATE TABLE IF NOT EXISTS yahoo_player_map (
    player_key      TEXT PRIMARY KEY,
    league_key      TEXT NOT NULL,
    full_name       TEXT NOT NULL,
    team_abbrev     TEXT,
    positions       TEXT,          -- comma-separated Yahoo eligibility
    nhl_player_id   INTEGER,
    adp_rank        INTEGER,       -- 1-based order from Yahoo's own ADP sort
    updated_at      TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS ix_yahoo_map_nhl ON yahoo_player_map(nhl_player_id);
"""


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


def table_names(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    return {r["name"] for r in rows}


def get_meta(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM sync_meta WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else None


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO sync_meta (key, value, updated_at) VALUES (?, ?, datetime('now'))"
        " ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
        (key, value),
    )


def upsert_player(
    conn: sqlite3.Connection,
    player_id: int,
    full_name: str,
    position: str | None,
    team_abbrev: str | None,
) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO nhl_players (player_id, full_name, position, team_abbrev)"
        " VALUES (?, ?, ?, ?)",
        (player_id, full_name, position, team_abbrev),
    )


def upsert_schedule_game(
    conn: sqlite3.Connection,
    *,
    game_id: int,
    season: str,
    game_type: int,
    game_date: str,
    start_time_utc: str | None,
    home_team: str,
    away_team: str,
) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO nhl_schedule"
        " (game_id, season, game_type, game_date, start_time_utc, home_team, away_team)"
        " VALUES (?, ?, ?, ?, ?, ?, ?)",
        (game_id, season, game_type, game_date, start_time_utc, home_team, away_team),
    )


def upsert_mp_season_stat(
    conn: sqlite3.Connection,
    *,
    player_id: int,
    season: str,
    kind: str,
    situation: str,
    name: str,
    team: str | None,
    position: str | None,
    stats_json: str,
) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO mp_season_stats"
        " (player_id, season, kind, situation, name, team, position, stats_json)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (player_id, season, kind, situation, name, team, position, stats_json),
    )


def add_proposal(
    conn: sqlite3.Connection, add_pid: int, drop_pid: int | None, reason_json: str
) -> int:
    cur = conn.execute(
        "INSERT INTO waiver_proposals (add_pid, drop_pid, reason_json) VALUES (?, ?, ?)",
        (add_pid, drop_pid, reason_json),
    )
    return int(cur.lastrowid)


def set_proposal_status(conn: sqlite3.Connection, proposal_id: int, status: str) -> None:
    conn.execute(
        "UPDATE waiver_proposals SET status = ? WHERE id = ?",
        (status, proposal_id),
    )


def list_proposals(conn: sqlite3.Connection, status: str | None = None) -> list[sqlite3.Row]:
    if status is None:
        return conn.execute("SELECT * FROM waiver_proposals ORDER BY id").fetchall()
    return conn.execute(
        "SELECT * FROM waiver_proposals WHERE status = ? ORDER BY id", (status,)
    ).fetchall()


def upsert_player_bio(
    conn: sqlite3.Connection,
    *,
    player_id: int,
    birth_date: str | None,
    height_in: int | None,
    weight_lb: int | None,
    shoots: str | None,
) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO nhl_player_bio"
        " (player_id, birth_date, height_in, weight_lb, shoots)"
        " VALUES (?, ?, ?, ?, ?)",
        (player_id, birth_date, height_in, weight_lb, shoots),
    )


def upsert_boxscore_rows(
    conn: sqlite3.Connection,
    rows: list[tuple[int, int, str, str | None, str]],
) -> None:
    """Bulk upsert of (game_id, player_id, season, team_abbrev, stats_json)."""
    conn.executemany(
        "INSERT OR REPLACE INTO nhl_boxscore_stats"
        " (game_id, player_id, season, team_abbrev, stats_json)"
        " VALUES (?, ?, ?, ?, ?)",
        rows,
    )


def upsert_game_log(
    conn: sqlite3.Connection,
    *,
    player_id: int,
    game_id: int,
    season: str,
    game_type: int,
    game_date: str,
    team_abbrev: str | None,
    opponent_abbrev: str | None,
    is_home: int | None,
    stats_json: str,
) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO nhl_game_logs"
        " (player_id, game_id, season, game_type, game_date,"
        "  team_abbrev, opponent_abbrev, is_home, stats_json)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            player_id,
            game_id,
            season,
            game_type,
            game_date,
            team_abbrev,
            opponent_abbrev,
            is_home,
            stats_json,
        ),
    )
