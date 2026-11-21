"""Every injury and availability tag the runs see, and when they first saw it.

Two questions no replay can answer, because no historical data carries Yahoo's
tags:

- How often does a day-to-day player actually play his next game? That is
  what `start_questionable` and the odds' P(play) for a tagged player should be
  set from, and today both are judgement.
- How often does a tag change after the last run but before the game? That is
  whether the run times are right.

Each run's roster read overwrites the day's snapshot, so the answer was being
thrown away. This keeps it, for every roster a run reads (ours and the
opponent's): the tag each player was last seen with, and a row whenever it
changes - bracketed between the last run that saw the old tag and the first
that saw the new one, which is as closely as a change can be timed from runs.

Started 2026-09-30, the day Jake Sanderson went day-to-day between runs.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime

from puckpilot.season.settings import is_out_status

# Tags whose next game is worth following up: did he play it?
WATCHED = ("DTD", "GTD", "O", "NA", "SUSP")


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _when(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


@dataclass(frozen=True)
class Change:
    player_key: str
    name: str
    team: str
    old: str | None  # None: the first time this player was seen at all
    new: str
    note: str
    old_seen_at: str | None
    new_seen_at: str

    def describe(self) -> str:
        was = "first seen" if self.old is None else (self.old or "healthy")
        return f"{self.name} ({self.team}): {was} -> {self.new or 'healthy'}" + (
            f" ({self.note})" if self.note else ""
        )


def record(conn: sqlite3.Connection, roster, seen_at: str | None = None) -> list[Change]:
    """Note each player's tag as this run saw it; return what changed.

    A player seen for the first time is a change only when he arrives tagged -
    a healthy newcomer is the baseline, not news.
    """
    seen_at = seen_at or _now()
    changes: list[Change] = []
    for p in roster.players:
        status = p.status or ""
        row = conn.execute(
            "SELECT status, last_seen_at FROM player_status "
            "WHERE league_key = ? AND player_key = ?",
            (roster.league_key, p.player_key),
        ).fetchone()
        if row is None:
            conn.execute(
                "INSERT INTO player_status "
                "(league_key, player_key, status, since_at, last_seen_at) VALUES (?, ?, ?, ?, ?)",
                (roster.league_key, p.player_key, status, seen_at, seen_at),
            )
            if status:
                changes.append(_log(conn, roster, p, None, status, None, seen_at))
            continue
        if row["status"] == status:
            conn.execute(
                "UPDATE player_status SET last_seen_at = ? WHERE league_key = ? AND player_key = ?",
                (seen_at, roster.league_key, p.player_key),
            )
            continue
        changes.append(_log(conn, roster, p, row["status"], status, row["last_seen_at"], seen_at))
        conn.execute(
            "UPDATE player_status SET status = ?, since_at = ?, last_seen_at = ? "
            "WHERE league_key = ? AND player_key = ?",
            (status, seen_at, seen_at, roster.league_key, p.player_key),
        )
    conn.commit()
    return changes


def _log(conn, roster, p, old, new, old_seen_at, new_seen_at) -> Change:
    conn.execute(
        "INSERT INTO player_status_changes (league_key, team_key, player_key, nhl_player_id, "
        " name, team_abbrev, old_status, new_status, injury_note, old_seen_at, new_seen_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            roster.league_key,
            roster.team_key,
            p.player_key,
            p.nhl_player_id,
            p.name,
            p.team,
            old,
            new,
            p.injury_note or "",
            old_seen_at,
            new_seen_at,
        ),
    )
    return Change(
        p.player_key, p.name, p.team, old, new, p.injury_note or "", old_seen_at, new_seen_at
    )


# -- what happened next ---------------------------------------------------------


@dataclass(frozen=True)
class Outcome:
    change: Change
    game_date: str | None  # his club's next game after the tag was first seen
    result: str  # "played", "did not play", "upcoming", "no game found"
    # A game that started between the last run without the tag and the first
    # with it: the tag arrived too late for any run to act on.
    missed_game: str | None = None


def outcomes(conn: sqlite3.Connection, league_key: str, season: str, today: str) -> list[Outcome]:
    """Every watched tag seen so far, with whether the player played next."""
    out: list[Outcome] = []
    rows = conn.execute(
        "SELECT * FROM player_status_changes WHERE league_key = ? ORDER BY new_seen_at",
        (league_key,),
    ).fetchall()
    for r in rows:
        if r["new_status"] not in WATCHED and not is_out_status(r["new_status"]):
            continue
        ch = Change(
            r["player_key"],
            r["name"],
            r["team_abbrev"],
            r["old_status"],
            r["new_status"],
            r["injury_note"] or "",
            r["old_seen_at"],
            r["new_seen_at"],
        )
        games = conn.execute(
            "SELECT game_id, game_date, start_time_utc FROM nhl_schedule WHERE season = ? "
            "AND game_type = 2 AND (home_team = ? OR away_team = ?) AND start_time_utc IS NOT NULL "
            "ORDER BY start_time_utc",
            (season, r["team_abbrev"], r["team_abbrev"]),
        ).fetchall()
        seen = _when(r["new_seen_at"])
        before = _when(r["old_seen_at"]) if r["old_seen_at"] else None
        missed = next(
            (
                g["game_date"]
                for g in games
                if before is not None and before < _when(g["start_time_utc"]) <= seen
            ),
            None,
        )
        nxt = next((g for g in games if _when(g["start_time_utc"]) > seen), None)
        if nxt is None:
            out.append(Outcome(ch, None, "no game found", missed))
            continue
        if nxt["game_date"] >= today:
            out.append(Outcome(ch, nxt["game_date"], "upcoming", missed))
            continue
        played = conn.execute(
            "SELECT 1 FROM nhl_game_logs WHERE player_id = ? AND game_id = ?",
            (r["nhl_player_id"], nxt["game_id"]),
        ).fetchone()
        out.append(Outcome(ch, nxt["game_date"], "played" if played else "did not play", missed))
    return out


def summary(results: list[Outcome]) -> list[str]:
    """Play rate by tag, over the tags whose next game has happened."""
    by: dict[str, list[bool]] = {}
    for o in results:
        if o.result in ("played", "did not play"):
            by.setdefault(o.change.new, []).append(o.result == "played")
    lines = [
        f"{tag}: {sum(v)} of {len(v)} played their next game ({sum(v) / len(v):.0%})"
        for tag, v in sorted(by.items())
    ]
    late = [o for o in results if o.missed_game]
    if late:
        lines.append(
            f"{len(late)} tag(s) arrived after a game had started and before any run saw them"
        )
    return lines or ["No tagged player has played (or missed) a game since logging began."]
