"""Keeper-list helpers.

Which players are on contract is league configuration, not code - it lives in
the league's TOML file (`[keepers.by_season]`). This module only resolves those
names to NHL player ids and decides which seat holds each keeper.
"""

from __future__ import annotations

import re
import sqlite3
import unicodedata
from dataclasses import dataclass, field

import numpy as np


def _norm(name: str) -> str:
    """Fold accents, punctuation and case so 'Tim Stuetzle' matches 'Tim Stutzle'."""
    s = unicodedata.normalize("NFKD", name)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z]", "", s.lower())


# "Sebastian Aho (CAR)", "Sebastian Aho (D)" or "Sebastian Aho (8478427)": the
# qualifier a league file uses when two NHL players share a name.
_QUALIFIED = re.compile(r"^(?P<name>.*?)\s*\((?P<q>[^()]+)\)\s*$")


def split_qualifier(entry: str) -> tuple[str, str]:
    """('Sebastian Aho', 'CAR') from 'Sebastian Aho (CAR)'; ('Name', '') otherwise."""
    m = _QUALIFIED.match(entry)
    if not m:
        return entry.strip(), ""
    return m.group("name").strip(), m.group("q").strip()


@dataclass
class KeeperResolution:
    """What a keeper list resolved to. Nothing is ever dropped silently.

    `ambiguous` is separate from `unmatched` because the fix differs: an
    unmatched name is a typo or a player we have never synced, while an
    ambiguous one is a real player sharing a name with another real player
    (there are two Sebastian Ahos and two Elias Petterssons) and needs a
    qualifier in the league file - `"Sebastian Aho (CAR)"`.
    """

    resolved: dict[str, int] = field(default_factory=dict)
    unmatched: list[str] = field(default_factory=list)
    # entry -> the candidates it could mean, as "Name (TEAM, POS, id)"
    ambiguous: dict[str, list[str]] = field(default_factory=dict)


def resolve_keepers(conn: sqlite3.Connection, names: tuple[str, ...]) -> KeeperResolution:
    """Map keeper names to NHL player ids.

    A normalized-name collision is never settled by taking whichever row the
    table returned first: that removes a real, draftable player from the board
    and leaves the kept one on it, which is the worst board error there is.
    Such a name resolves only when its qualifier - team, position, or player id
    - narrows the candidates to exactly one.
    """
    rows = conn.execute("SELECT player_id, full_name, position, team_abbrev FROM nhl_players")
    by_norm: dict[str, list[tuple[int, str, str, str]]] = {}
    for r in rows.fetchall():
        pid, full, pos, team = (int(r[0]), str(r[1]), str(r[2] or ""), str(r[3] or ""))
        by_norm.setdefault(_norm(full), []).append((pid, full, pos, team))

    out = KeeperResolution()
    for entry in names:
        name, qualifier = split_qualifier(entry)
        cands = by_norm.get(_norm(name), [])
        if qualifier:
            q = qualifier.upper()
            cands = [
                c
                for c in cands
                if (q.isdigit() and c[0] == int(q)) or q in (c[3].upper(), c[2].upper())
            ]
        if not cands:
            out.unmatched.append(entry)
        elif len(cands) > 1:
            out.ambiguous[entry] = [f"{c[1]} ({c[3]}, {c[2]}, {c[0]})" for c in cands]
        else:
            out.resolved[entry] = cands[0][0]
    return out


def resolve_keeper_ids(
    conn: sqlite3.Connection, names: tuple[str, ...]
) -> tuple[dict[str, int], list[str]]:
    """Map keeper names to NHL player ids. Returns (resolved, unmatched).

    Ambiguous names are returned with the unmatched ones: neither may be placed.
    Use `resolve_keepers` to tell the two apart.
    """
    res = resolve_keepers(conn, names)
    return res.resolved, res.unmatched + list(res.ambiguous)


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
