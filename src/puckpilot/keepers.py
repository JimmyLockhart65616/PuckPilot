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
    player_ids: list[int], n_teams: int, rng: np.random.Generator
) -> dict[int, list[int]]:
    """Deal kept players across seats as evenly as possible.

    Ownership is not recorded on most keeper sheets, so this randomizes which
    rival holds which keeper while keeping the board (the set of unavailable
    players) exactly right.
    """
    ids = list(player_ids)
    rng.shuffle(ids)
    out: dict[int, list[int]] = {s: [] for s in range(n_teams)}
    for i, pid in enumerate(ids):
        out[i % n_teams].append(int(pid))
    return out
