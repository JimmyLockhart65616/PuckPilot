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

-- A transaction can only be executed from an approved row here. That is the
-- whole mechanism behind "lineups are pre-authorised, transactions are not":
-- the executor takes a proposal id and has no path that creates one itself.
CREATE TABLE IF NOT EXISTS waiver_proposals (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    add_pid     INTEGER NOT NULL,
    drop_pid    INTEGER,
    reason_json TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'approved', 'rejected', 'executed')),
    manager         TEXT NOT NULL DEFAULT '',
    league_key      TEXT NOT NULL DEFAULT '',
    team_key        TEXT NOT NULL DEFAULT '',
    kind            TEXT NOT NULL DEFAULT 'add_drop',
    add_player_key  TEXT,
    drop_player_key TEXT,
    decided_at      TEXT,
    executed_at     TEXT
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

-- In-season. Everything below is keyed by `manager` as well as league, because
-- two people in the same league may run this against the same database and the
-- draft board already taught us what happens when "my team" is implicit.

-- Yahoo's own description of the league: the week calendar, lock mode, waiver
-- rules and caps. Cached whole rather than shredded into columns so a setting
-- we have not thought about yet is still there when we want it.
CREATE TABLE IF NOT EXISTS yahoo_league_runtime (
    league_key   TEXT PRIMARY KEY,
    settings_json TEXT NOT NULL,
    weeks_json   TEXT NOT NULL DEFAULT '[]',
    fetched_at   TEXT NOT NULL DEFAULT (datetime('now'))
);

-- One row per player per day: what Yahoo had slotted, and what it knew about
-- his health. Kept as history so "what did we actually start" is answerable
-- after the fact, which is the only way an autonomous lineup change can be
-- audited.
CREATE TABLE IF NOT EXISTS yahoo_roster_snapshots (
    manager       TEXT NOT NULL,
    league_key    TEXT NOT NULL,
    team_key      TEXT NOT NULL,
    date          TEXT NOT NULL,
    player_key    TEXT NOT NULL,
    nhl_player_id INTEGER,
    name          TEXT NOT NULL,
    team_abbrev   TEXT,
    selected_slot TEXT,
    eligible      TEXT,
    status        TEXT,
    injury_note   TEXT,
    is_editable   INTEGER NOT NULL DEFAULT 1,
    fetched_at    TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (manager, team_key, date, player_key)
);
CREATE INDEX IF NOT EXISTS ix_roster_snap_date ON yahoo_roster_snapshots (date);

-- The free-agent and waiver pool, sampled daily. Differencing percent_owned
-- between two days is the trending-adds signal, and it is this league's own
-- ownership rather than a site-wide average.
CREATE TABLE IF NOT EXISTS yahoo_fa_snapshots (
    league_key     TEXT NOT NULL,
    date           TEXT NOT NULL,
    player_key     TEXT NOT NULL,
    nhl_player_id  INTEGER,
    name           TEXT NOT NULL,
    team_abbrev    TEXT,
    positions      TEXT,
    ownership      TEXT,
    percent_owned  REAL,
    status         TEXT,
    PRIMARY KEY (league_key, date, player_key)
);
CREATE INDEX IF NOT EXISTS ix_fa_snap_date ON yahoo_fa_snapshots (league_key, date);

-- The audit log. Every lineup change made under standing authority and every
-- transaction executed after approval lands here, with the reasoning that
-- produced it, so the criteria can be argued with from evidence.
CREATE TABLE IF NOT EXISTS season_actions (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    manager      TEXT NOT NULL,
    league_key   TEXT NOT NULL,
    team_key     TEXT NOT NULL,
    date         TEXT NOT NULL,
    kind         TEXT NOT NULL,
    detail_json  TEXT NOT NULL,
    outcome      TEXT NOT NULL DEFAULT 'planned'
        CHECK (outcome IN ('planned', 'dry-run', 'executed', 'failed', 'skipped')),
    message      TEXT,
    created_at   TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS ix_season_actions_date ON season_actions (manager, date);

-- A stance for the week, agreed before it starts: which categories are out of
-- reach and which are live. Approving one is what licenses the daily lineup to
-- weigh players by it, so the status here is load-bearing, not a label.
CREATE TABLE IF NOT EXISTS week_protocols (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    manager      TEXT NOT NULL,
    league_key   TEXT NOT NULL,
    team_key     TEXT NOT NULL,
    week         INTEGER NOT NULL,
    opponent     TEXT,
    stances_json TEXT NOT NULL,
    status       TEXT NOT NULL DEFAULT 'proposed'
        CHECK (status IN ('proposed', 'approved', 'rejected')),
    decided_at   TEXT,
    created_at   TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS ix_week_protocols ON week_protocols (manager, league_key, week);

-- The live score of a week, as Yahoo reported it at each run. Intra-week state
-- cannot be fetched after the fact - Yahoo only ever says "now" - so this log is
-- the only record of how a week unfolded, and the only live data the win
-- probabilities can ever be checked against. The raw payload is kept because
-- the parser was written before a live week existed to test it on.
CREATE TABLE IF NOT EXISTS matchup_snapshots (
    manager      TEXT NOT NULL,
    league_key   TEXT NOT NULL,
    team_key     TEXT NOT NULL,
    week         INTEGER NOT NULL,
    fetched_at   TEXT NOT NULL,
    status       TEXT,
    opponent_key TEXT,
    ours_json    TEXT NOT NULL,
    theirs_json  TEXT NOT NULL,
    winners_json TEXT,
    raw_json     TEXT,
    PRIMARY KEY (manager, team_key, week, fetched_at)
);

-- What the odds said at each run, so live weeks can be scored against how they
-- ended (gate G3) - the same test the replay passed, on the real league.
CREATE TABLE IF NOT EXISTS week_odds_log (
    manager    TEXT NOT NULL,
    league_key TEXT NOT NULL,
    team_key   TEXT NOT NULL,
    week       INTEGER NOT NULL,
    logged_at  TEXT NOT NULL,
    day        TEXT NOT NULL,
    days_left  INTEGER,
    expected   REAL,
    cats_json  TEXT NOT NULL,
    PRIMARY KEY (manager, team_key, week, logged_at)
);
"""

# `waiver_proposals` predates the in-season work and shipped without a manager,
# a league or Yahoo's own player keys. There is no migration framework here and
# `CREATE TABLE IF NOT EXISTS` will not alter an existing table, so this is the
# narrow version of one: additive columns only, idempotent, no rewrites.
ADDED_COLUMNS: dict[str, dict[str, str]] = {
    "waiver_proposals": {
        "manager": "TEXT NOT NULL DEFAULT ''",
        "league_key": "TEXT NOT NULL DEFAULT ''",
        "team_key": "TEXT NOT NULL DEFAULT ''",
        "kind": "TEXT NOT NULL DEFAULT 'add_drop'",
        "add_player_key": "TEXT",
        "drop_player_key": "TEXT",
        "decided_at": "TEXT",
        "executed_at": "TEXT",
        # Withdrawn by a newer search. A column rather than a status, because
        # the status CHECK cannot change without rewriting the table.
        "superseded_at": "TEXT",
    },
}


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    _add_missing_columns(conn)
    conn.commit()


def _add_missing_columns(conn: sqlite3.Connection) -> list[str]:
    """Bring pre-existing tables up to the current column set.

    Only ever adds, and only columns with a default, so it cannot lose data and
    cannot fail halfway into an inconsistent shape. Returns what it added, so a
    caller can say so rather than changing a database silently.
    """
    added = []
    for table, columns in ADDED_COLUMNS.items():
        have = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        if not have:  # table does not exist yet; the schema above just made it
            continue
        for name, decl in columns.items():
            if name not in have:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")
                added.append(f"{table}.{name}")
    return added


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


def _ascii_only(name: str) -> str:
    return "".join(ch for ch in name if ord(ch) < 128)


def is_lossy_copy(lossy: str, proper: str) -> bool:
    """`lossy` is `proper` with its non-ASCII letters deleted - not folded.

    MoneyPuck's CSVs spell Tim Stützle "Tim Sttzle" and Alexis Lafrenière
    "Alexis Lafrenire". A folded "Stutzle" still matches what a drafter types;
    a deleted letter does not, so `DraftBoard.find("stutzle")` found nobody.
    The test is deliberately that narrow: a name that differs any other way
    ("Mitch" vs "Mitchell") is a spelling choice, not damage, and is left alone.
    """
    return lossy != proper and _ascii_only(proper) == lossy


def upsert_player(
    conn: sqlite3.Connection,
    player_id: int,
    full_name: str,
    position: str | None,
    team_abbrev: str | None,
) -> None:
    # Never overwrite a repaired name with MoneyPuck's lossy copy of it, or every
    # `data sync` would undo `repair_player_name`.
    row = conn.execute(
        "SELECT full_name FROM nhl_players WHERE player_id = ?", (player_id,)
    ).fetchone()
    if row and is_lossy_copy(full_name, row[0]):
        full_name = row[0]
    conn.execute(
        "INSERT OR REPLACE INTO nhl_players (player_id, full_name, position, team_abbrev)"
        " VALUES (?, ?, ?, ?)",
        (player_id, full_name, position, team_abbrev),
    )


def repair_player_name(conn: sqlite3.Connection, player_id: int, proper: str) -> int:
    """Replace a stored name only when it is a lossy copy of `proper`."""
    row = conn.execute(
        "SELECT full_name FROM nhl_players WHERE player_id = ?", (player_id,)
    ).fetchone()
    if not row or not is_lossy_copy(row[0], proper):
        return 0
    return conn.execute(
        "UPDATE nhl_players SET full_name = ? WHERE player_id = ?", (proper, player_id)
    ).rowcount


def update_player_team(conn: sqlite3.Connection, player_id: int, team_abbrev: str) -> int:
    """Set a player's team WITHOUT touching name or position.

    `upsert_player` is INSERT OR REPLACE, so calling it per season means the
    last season processed wins - which is how `team_abbrev` ended up holding
    2022-23 teams for 436 players who played in 2025-26. A team change needs a
    targeted write, not a whole-row replace.
    """
    cur = conn.execute(
        "UPDATE nhl_players SET team_abbrev = ? WHERE player_id = ? AND"
        " (team_abbrev IS NULL OR team_abbrev != ?)",
        (team_abbrev, player_id, team_abbrev),
    )
    return cur.rowcount


def insert_player_if_missing(
    conn: sqlite3.Connection,
    player_id: int,
    full_name: str,
    position: str | None,
    team_abbrev: str | None,
) -> bool:
    """Add a player `nhl_players` has never heard of, without touching a row
    that already exists there - `INSERT OR IGNORE`, not `upsert_player`'s
    whole-row replace, for the same reason `update_player_team` exists.

    This is what makes a pre-debut roster player nameable: MoneyPuck is the
    normal discovery mechanism but only lists players with >=1 NHL game, so a
    rookie who has not played yet has no row until something else creates one.
    Returns whether a row was actually inserted.
    """
    cur = conn.execute(
        "INSERT OR IGNORE INTO nhl_players (player_id, full_name, position, team_abbrev)"
        " VALUES (?, ?, ?, ?)",
        (player_id, full_name, position, team_abbrev),
    )
    return cur.rowcount > 0


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
