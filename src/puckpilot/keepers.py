"""Keeper-list helpers.

Which players are on contract is league configuration, not code - it lives in
the league's TOML file (`[keepers.by_season]`). This module only resolves those
names to NHL player ids and decides which seat holds each keeper.
"""

from __future__ import annotations

import re
import sqlite3
import unicodedata

import numpy as np


def _norm(name: str) -> str:
    """Fold accents, punctuation and case so 'Tim Stuetzle' matches 'Tim Stutzle'."""
    s = unicodedata.normalize("NFKD", name)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z]", "", s.lower())


def resolve_keeper_ids(
    conn: sqlite3.Connection, names: tuple[str, ...]
) -> tuple[dict[str, int], list[str]]:
    """Map keeper names to NHL player ids. Returns (resolved, unmatched).

    Unmatched names are returned rather than silently dropped: a missed keeper
    would leave an elite player wrongly available on the simulated draft board.
    """
    rows = conn.execute("SELECT player_id, full_name FROM nhl_players").fetchall()
    by_norm: dict[str, int] = {}
    for r in rows:
        pid, full = (r["player_id"], r["full_name"]) if hasattr(r, "keys") else (r[0], r[1])
        by_norm.setdefault(_norm(full), int(pid))

    resolved: dict[str, int] = {}
    unmatched: list[str] = []
    for n in names:
        pid = by_norm.get(_norm(n))
        if pid is None:
            unmatched.append(n)
        else:
            resolved[n] = pid
    return resolved, unmatched


def keeper_seats(
    player_ids: list[int],
    n_teams: int,
    rng: np.random.Generator,
    owned: dict[int, list[int]] | None = None,
) -> dict[int, list[int]]:
    """Deal kept players across seats as evenly as possible.

    Ownership is not recorded on most keeper sheets, so by default this
    randomizes which rival holds which keeper while keeping the board (the set
    of unavailable players) exactly right. That is fine for a simulation, where
    only availability matters.

    It is NOT fine for a live board, where our own roster drives what the engine
    thinks we still need. Pass `owned` - seat -> player ids known to be held by
    that seat - and those are placed exactly; everything left over is dealt
    round-robin across the seats with the most room, so declaring only your own
    keepers still produces a sane board.
    """
    owned = {int(s): [int(p) for p in ids] for s, ids in (owned or {}).items()}
    out: dict[int, list[int]] = {s: [] for s in range(n_teams)}
    placed: set[int] = set()
    for seat, ids in owned.items():
        if not 0 <= seat < n_teams:
            continue
        for pid in ids:
            if pid not in placed:
                out[seat].append(pid)
                placed.add(pid)

    rest = [int(p) for p in player_ids if int(p) not in placed]
    rng.shuffle(rest)
    # Fill the emptiest seats first rather than striding from seat 0, or a
    # declared owner would be dealt extra keepers on top of the ones they hold.
    for pid in rest:
        seat = min(range(n_teams), key=lambda s: (len(out[s]), s))
        out[seat].append(pid)
    return out
