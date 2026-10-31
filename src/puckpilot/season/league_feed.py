"""The whole league, not just our matchup: transactions, standings, every week's
scores, and every team's roster as each week began.

Until 2026-10-31 the tool read two rosters a week (ours and the opponent's) and
one score. Asked "how are the other teams doing, and what are they picking up?",
it could not say - and its own evidence had the same blind spot: the replay
gate freezes the other eleven teams, while the live league had made 116 adds in
six weeks, a quarter of them goalies. This feed records what the league
actually does, so that

- a manager can see it (`ppilot season league`),
- live calibration covers every matchup, not one in six (`season/g3.py`), and
- the replay's rivals can be made to behave like real ones (`add_gate`).

A `LeagueSource` is the platform boundary: the parsing of one platform's
payloads lives in its implementation, and the tables below hold nothing
platform-shaped beyond keys (`data/store.py` holds them, with the rest of the
schema). `YahooLeagueSource` reads through the read-only session; the official
client can implement the same four methods.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol


@dataclass(frozen=True)
class TxnMove:
    action: str  # add | drop
    team_key: str
    player_key: str
    name: str
    nhl_team: str
    positions: str
    source_type: str


@dataclass(frozen=True)
class Txn:
    txn_key: str
    at: str
    type: str
    status: str
    moves: tuple[TxnMove, ...]


@dataclass(frozen=True)
class Standing:
    team_key: str
    name: str
    rank: int | None
    wins: int | None
    losses: int | None
    ties: int | None
    moves: int | None
    trades: int | None


@dataclass(frozen=True)
class TeamWeek:
    week: int
    team_key: str
    name: str
    opponent_key: str
    status: str
    cats_won: float | None
    completed_games: int | None
    stats: dict[str, float | None]


class LeagueSource(Protocol):
    def transactions(self, league_key: str) -> list[Txn]: ...

    def standings(self, league_key: str) -> list[Standing]: ...

    def scoreboard(self, league_key: str, week: int) -> list[TeamWeek]: ...

    def roster(self, team_key: str, date: str, player_map=None): ...


# -- Yahoo ------------------------------------------------------------------


def _flat(obj, out=None) -> dict:
    """Yahoo's list-of-single-key-dicts objects, merged (first value wins)."""
    out = {} if out is None else out
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (str, int, float)) or v is None:
                out.setdefault(k, v)
            else:
                _flat(v, out)
    elif isinstance(obj, list):
        for item in obj:
            _flat(item, out)
    return out


def _int(v) -> int | None:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _num(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def parse_transactions(payload: dict) -> list[Txn]:
    league = payload["fantasy_content"]["league"]
    node = next((x["transactions"] for x in league if isinstance(x, dict) and "transactions" in x),
                {})  # fmt: skip
    out = []
    for i in range(int(node.get("count", 0) or 0)):
        t = node[str(i)]["transaction"]
        meta = t[0]
        players = t[1].get("players", {}) if len(t) > 1 and isinstance(t[1], dict) else {}
        moves = []
        for j in range(int(players.get("count", 0) or 0)):
            p = players[str(j)]["player"]
            info = _flat(p[0])
            data = _flat(p[1].get("transaction_data", {}))
            action = str(data.get("type", ""))
            team = (
                data.get("destination_team_key") if action == "add" else data.get("source_team_key")
            )
            moves.append(
                TxnMove(
                    action=action,
                    team_key=str(team or ""),
                    player_key=str(info.get("player_key", "")),
                    name=str(info.get("full", "")),
                    nhl_team=str(info.get("editorial_team_abbr", "")),
                    positions=str(info.get("display_position", "")),
                    source_type=str(data.get("source_type", "")),
                )
            )
        at = datetime.fromtimestamp(int(meta.get("timestamp", 0) or 0), UTC)
        out.append(
            Txn(
                txn_key=str(meta.get("transaction_key", "")),
                at=at.isoformat(timespec="seconds"),
                type=str(meta.get("type", "")),
                status=str(meta.get("status", "")),
                moves=tuple(moves),
            )
        )
    return out


def parse_standings(payload: dict) -> list[Standing]:
    league = payload["fantasy_content"]["league"]
    node = next(x["standings"] for x in league if isinstance(x, dict) and "standings" in x)
    teams = node[0]["teams"]
    out = []
    for i in range(int(teams.get("count", 0) or 0)):
        t = teams[str(i)]["team"]
        meta = _flat(t[0])
        st = next((x["team_standings"] for x in t[1:] if isinstance(x, dict)
                   and "team_standings" in x), {})  # fmt: skip
        totals = st.get("outcome_totals", {}) if isinstance(st, dict) else {}
        out.append(
            Standing(
                team_key=str(meta.get("team_key", "")),
                name=str(meta.get("name", "")),
                rank=_int(st.get("rank")) if isinstance(st, dict) else None,
                wins=_int(totals.get("wins")),
                losses=_int(totals.get("losses")),
                ties=_int(totals.get("ties")),
                moves=_int(meta.get("number_of_moves")),
                trades=_int(meta.get("number_of_trades")),
            )
        )
    return out


def parse_scoreboard(payload: dict, week: int) -> list[TeamWeek]:
    league = payload["fantasy_content"]["league"]
    board = next(x["scoreboard"] for x in league if isinstance(x, dict) and "scoreboard" in x)
    matchups = board["0"]["matchups"]
    out = []
    for i in range(int(matchups.get("count", 0) or 0)):
        m = matchups[str(i)]["matchup"]
        status = str(m.get("status", ""))
        pair = []
        for k in ("0", "1"):
            t = m["0"]["teams"][k]["team"]
            meta = _flat(t[0])
            body = t[1]
            stats = {
                str(x["stat"]["stat_id"]): _num(x["stat"].get("value"))
                for x in body.get("team_stats", {}).get("stats", [])
            }
            games = _flat(body.get("team_remaining_games", {}))
            pair.append(
                (
                    str(meta.get("team_key", "")),
                    str(meta.get("name", "")),
                    _num(body.get("team_points", {}).get("total")),
                    _int(games.get("completed_games")),
                    stats,
                )
            )
        for (key, name, won, done, stats), (other, *_rest) in (
            (pair[0], pair[1]),
            (pair[1], pair[0]),
        ):
            out.append(TeamWeek(week, key, name, other, status, won, done, stats))
    return out


class YahooLeagueSource:
    """The four reads, through a read-only Yahoo session (`yahoo.session`)."""

    def __init__(self, session):
        self.session = session

    def transactions(self, league_key: str) -> list[Txn]:
        return parse_transactions(self.session.get(f"league/{league_key}/transactions;count=500"))

    def standings(self, league_key: str) -> list[Standing]:
        return parse_standings(self.session.get(f"league/{league_key}/standings"))

    def scoreboard(self, league_key: str, week: int) -> list[TeamWeek]:
        return parse_scoreboard(
            self.session.get(f"league/{league_key}/scoreboard;week={week}"), week
        )

    def roster(self, team_key: str, date: str, player_map=None):
        from puckpilot.season.fetch import fetch_roster

        return fetch_roster(self.session, team_key, date, player_map=player_map)


# -- storage ----------------------------------------------------------------


def save_transactions(conn, league_key: str, txns: list[Txn], player_map=None) -> int:
    rows = []
    for t in txns:
        for m in t.moves:
            pid = (player_map or {}).get(m.player_key)
            rows.append(
                (league_key, t.txn_key, t.at, t.type, t.status, m.team_key, m.action,
                 m.player_key, pid, m.name, m.nhl_team, m.positions, m.source_type)
            )  # fmt: skip
    # A read without the player map must not erase an id an earlier one found.
    conn.executemany(
        "INSERT INTO league_transactions (league_key, txn_key, at, type, status, "
        "team_key, action, player_key, nhl_player_id, name, nhl_team, positions, source_type) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
        "ON CONFLICT (league_key, txn_key, player_key, action) DO UPDATE SET "
        "status = excluded.status, "
        "nhl_player_id = COALESCE(excluded.nhl_player_id, league_transactions.nhl_player_id)",
        rows,
    )
    conn.commit()
    return len(rows)


def save_standings(conn, league_key: str, date: str, rows: list[Standing]) -> int:
    conn.executemany(
        "INSERT OR REPLACE INTO league_standings (league_key, date, team_key, name, rank, wins, "
        "losses, ties, moves, trades) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [(league_key, date, s.team_key, s.name, s.rank, s.wins, s.losses, s.ties, s.moves,
          s.trades) for s in rows],
    )  # fmt: skip
    conn.commit()
    return len(rows)


def save_week(conn, league_key: str, rows: list[TeamWeek]) -> int:
    now = datetime.now(UTC).isoformat(timespec="seconds")
    conn.executemany(
        "INSERT OR REPLACE INTO league_week_stats (league_key, week, team_key, name, "
        "opponent_key, status, cats_won, completed_games, stats_json, fetched_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [(league_key, r.week, r.team_key, r.name, r.opponent_key, r.status, r.cats_won,
          r.completed_games, json.dumps(r.stats), now) for r in rows],
    )  # fmt: skip
    conn.commit()
    return len(rows)


def week_final(conn, league_key: str, week: int) -> bool:
    row = conn.execute(
        "SELECT COUNT(*) FROM league_week_stats WHERE league_key = ? AND week = ? "
        "AND status = 'postevent'",
        (league_key, week),
    ).fetchone()
    return bool(row and row[0])


def has_roster(conn, team_key: str, date: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM yahoo_roster_snapshots WHERE team_key = ? AND date = ? LIMIT 1",
        (team_key, date),
    ).fetchone()
    return row is not None


# -- the sync ---------------------------------------------------------------


def sync(
    conn: sqlite3.Connection,
    source: LeagueSource,
    league_key: str,
    runtime,
    manager: str,
    today: str,
    player_map=None,
    rosters: bool = True,
) -> dict:
    """Bring the league tables up to `today`. Idempotent: a final week's
    scores and a week-start roster already saved are not read again."""
    from puckpilot.season.fetch import save_roster

    out = {"transactions": 0, "standings": 0, "weeks": 0, "rosters": 0}
    out["transactions"] = save_transactions(
        conn, league_key, source.transactions(league_key), player_map
    )
    standing = source.standings(league_key)
    out["standings"] = save_standings(conn, league_key, today, standing)
    teams = [s.team_key for s in standing]
    for w in runtime.weeks:
        if w.start > today:
            break
        if not week_final(conn, league_key, w.number):
            board = source.scoreboard(league_key, w.number)
            out["weeks"] += bool(save_week(conn, league_key, board))
        if not rosters:
            continue
        for team in teams:
            if has_roster(conn, team, w.start):
                continue
            save_roster(conn, manager, source.roster(team, w.start, player_map))
            out["rosters"] += 1
    return out


# -- what it says -------------------------------------------------------------


def summary(conn, league_key: str) -> dict:
    """Per team: standing, adds, goalie adds, and the latest moves - the view a
    manager asks for when asking what the other teams are doing."""
    date = conn.execute(
        "SELECT MAX(date) FROM league_standings WHERE league_key = ?", (league_key,)
    ).fetchone()[0]
    standing = conn.execute(
        "SELECT * FROM league_standings WHERE league_key = ? AND date = ? ORDER BY rank",
        (league_key, date),
    ).fetchall()
    adds = {
        r[0]: (r[1], r[2])
        for r in conn.execute(
            "SELECT team_key, COUNT(*), SUM(CASE WHEN positions LIKE '%G%' THEN 1 ELSE 0 END) "
            "FROM league_transactions WHERE league_key = ? AND action = 'add' GROUP BY team_key",
            (league_key,),
        )
    }
    return {
        "date": date,
        "teams": [
            {
                "team_key": s[2],
                "name": s[3],
                "rank": s[4],
                "record": f"{s[5]}-{s[6]}-{s[7]}",
                "adds": adds.get(s[2], (0, 0))[0],
                "goalie_adds": adds.get(s[2], (0, 0))[1] or 0,
            }
            for s in standing
        ],
    }
