"""Bridge Yahoo player keys to NHL player ids.

`draftresults` identifies a pick only by `player_key` ("477.p.6743"). Every
engine downstream is keyed on the NHL player id that MoneyPuck and the NHL API
share. Without this map a live pick cannot be marked off the board, so it is
built once before the draft rather than looked up on the clock.

Matching is by normalized name, reusing `keepers._norm` so accents and
punctuation do not matter. Two guards on top of that, because a wrong match is
far worse than a missing one - it would remove the wrong player from the board:

- an NHL id is claimed by at most one Yahoo key, best ADP first;
- when a normalized name is ambiguous across several NHL players, team abbrev
  breaks the tie, and if it cannot, the player is left unmatched.

Unmatched players are recorded with a NULL id rather than dropped, so the size
and shape of the gap is inspectable instead of silent.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, field

from puckpilot.keepers import _norm
from puckpilot.yahoo.session import YahooSession

# Yahoo pages the player list; 25 per request is its documented maximum.
PAGE = 25
# Yahoo's own abbreviations differ from the NHL API's in a handful of cases.
TEAM_ALIASES = {
    "LA": "LAK",
    "SJ": "SJS",
    "TB": "TBL",
    "NJ": "NJD",
    "WSH": "WSH",
    "MON": "MTL",
    "CLS": "CBJ",
    "ANH": "ANA",
    "PHX": "ARI",
}

Progress = Callable[[str], None]


def _noop(_: str) -> None:
    pass


def _team(abbrev: str | None) -> str:
    a = (abbrev or "").upper()
    return TEAM_ALIASES.get(a, a)


@dataclass
class MapReport:
    total: int = 0
    matched: int = 0
    unmatched: list[str] = field(default_factory=list)
    ambiguous: list[str] = field(default_factory=list)
    fallbacks: list[str] = field(default_factory=list)  # matched, but not exactly

    @property
    def match_rate(self) -> float:
        return self.matched / self.total if self.total else 0.0

    @property
    def text(self) -> str:
        lines = [
            f"Yahoo player map: {self.matched}/{self.total} matched ({self.match_rate:.1%})",
        ]
        if self.fallbacks:
            # Surfaced, never silent: a fuzzy match that went to the wrong
            # player would take the wrong name off the board on draft night.
            lines.append(f"  matched via fallback ({len(self.fallbacks)}), verify these:")
            lines += [f"    {f}" for f in self.fallbacks[:15]]
        if self.ambiguous:
            lines.append(
                f"  ambiguous (same name, team did not disambiguate): "
                f"{', '.join(self.ambiguous[:8])}"
            )
        if self.unmatched:
            shown = ", ".join(self.unmatched[:12])
            more = f" (+{len(self.unmatched) - 12} more)" if len(self.unmatched) > 12 else ""
            lines.append(f"  unmatched: {shown}{more}")
            lines.append(
                "  Unmatched players are usually prospects with no NHL game logs. "
                "They stay draftable in Yahoo but carry no projection."
            )
        return "\n".join(lines)


def fetch_players(
    session: YahooSession,
    league_key: str,
    limit: int = 600,
    progress: Progress = _noop,
) -> list[dict]:
    """Every player in the league pool, in Yahoo's own ADP order."""
    out: list[dict] = []
    start = 0
    while start < limit:
        page = session.players(league_key, start=start, count=PAGE, extra="sort=AR")
        if not page:
            break
        out.extend(page)
        start += PAGE
        progress(f"  fetched {len(out)} players")
        if len(page) < PAGE:
            break
    return out


@dataclass
class NhlIndex:
    """Lookup structures over known NHL players, for layered matching."""

    by_name: dict[str, list[int]] = field(default_factory=dict)
    by_name_team: dict[tuple[str, str], int] = field(default_factory=dict)
    by_last: dict[str, list[int]] = field(default_factory=dict)
    names: dict[int, str] = field(default_factory=dict)
    teams: dict[int, str] = field(default_factory=dict)


def _last(name: str) -> str:
    return _norm(name.split()[-1]) if name.split() else ""


def _nhl_index(conn: sqlite3.Connection) -> NhlIndex:
    idx = NhlIndex()
    for row in conn.execute("SELECT player_id, full_name, team_abbrev FROM nhl_players"):
        pid, name, team = int(row[0]), row[1], row[2]
        key = _norm(name)
        idx.by_name.setdefault(key, []).append(pid)
        idx.by_name_team.setdefault((key, _team(team)), pid)
        idx.by_last.setdefault(_last(name), []).append(pid)
        idx.names[pid] = name
        idx.teams[pid] = _team(team)
    return idx


def _resolve(idx: NhlIndex, name: str, team: str) -> tuple[int | None, str]:
    """Best NHL id for a Yahoo name, plus how it was found.

    Three layers, each narrower than the last. The fallbacks exist because two
    real mismatch classes turn up among *stars*, where a miss is expensive:

    - nickname vs given name ("Mitch Marner" / "Mitchell Marner") and
      transliteration variants ("Egor" / "Yegor Chinakhov");
    - upstream names where an accented letter was DROPPED rather than folded -
      MoneyPuck supplies "Tim Sttzle" and "Alexis Lafrenire". Yahoo spells them
      correctly, so exact and even last-name matching both fail.

    Every fallback requires the team to agree, which is what keeps a fuzzy match
    from ever silently taking the wrong player off the board.
    """
    norm = _norm(name)

    exact = idx.by_name.get(norm, [])
    if len(exact) == 1:
        return exact[0], "exact"
    if len(exact) > 1:
        hit = idx.by_name_team.get((norm, team))
        return (hit, "exact+team") if hit else (None, "ambiguous")

    from difflib import SequenceMatcher

    # Layer 2: surname plus team. Catches nicknames and transliterations.
    all_last = idx.by_last.get(_last(name), [])
    same_last = [p for p in all_last if idx.teams.get(p) == team]
    if len(same_last) == 1:
        return same_last[0], "surname+team"

    # Layer 3: a surname unique across the whole league, with the given names
    # close enough to be the same person. This layer has to exist because
    # `nhl_players.team_abbrev` is the LAST-PLAYED team, not the current one, so
    # every player traded in the offseason fails any team-constrained check -
    # Marner reads as TOR here and VGK on Yahoo. Requiring a unique surname plus
    # name similarity keeps it safe without trusting the stale team.
    if len(all_last) == 1:
        score = SequenceMatcher(None, norm, _norm(idx.names[all_last[0]])).ratio()
        if score >= 0.70:
            return all_last[0], f"surname-unique{score:.2f}"

    # Layer 4: closest full name on the same team, only if clearly close.

    best, best_score = None, 0.0
    for pid, other in idx.names.items():
        if idx.teams.get(pid) != team:
            continue
        score = SequenceMatcher(None, norm, _norm(other)).ratio()
        if score > best_score:
            best, best_score = pid, score
    if best is not None and best_score >= 0.85:
        return best, f"fuzzy{best_score:.2f}"
    return None, "unmatched"


def build_map(
    conn: sqlite3.Connection,
    session: YahooSession,
    league_key: str,
    limit: int = 600,
    progress: Progress = _noop,
) -> MapReport:
    """Fetch the Yahoo pool, match it to NHL ids, and persist the mapping."""
    players = fetch_players(session, league_key, limit=limit, progress=progress)
    idx = _nhl_index(conn)

    report = MapReport(total=len(players))
    claimed: set[int] = set()
    rows = []
    for rank, p in enumerate(players, start=1):
        name = p.get("full") or ""
        key = p.get("player_key")
        if not key or not name:
            continue
        team = _team(p.get("editorial_team_abbr"))
        nhl_id, how = _resolve(idx, name, team)
        if how == "ambiguous":
            report.ambiguous.append(name)
        elif how not in ("exact", "unmatched") and nhl_id is not None:
            report.fallbacks.append(f"{name} -> {idx.names[nhl_id]} ({how})")

        if nhl_id is not None and nhl_id in claimed:
            # Already taken by a better-ADP Yahoo entry; never double-assign.
            nhl_id = None
        if nhl_id is None:
            report.unmatched.append(name)
        else:
            claimed.add(nhl_id)
            report.matched += 1

        rows.append(
            (
                key,
                league_key,
                name,
                team,
                ",".join(p.get("eligible_positions") or []),
                nhl_id,
                rank,
            )
        )

    conn.executemany(
        "INSERT OR REPLACE INTO yahoo_player_map"
        " (player_key, league_key, full_name, team_abbrev, positions, nhl_player_id, adp_rank)"
        " VALUES (?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    return report


def load_map(conn: sqlite3.Connection, league_key: str) -> dict[str, int]:
    """player_key -> nhl_player_id, for the keys that resolved."""
    return {
        r[0]: int(r[1])
        for r in conn.execute(
            "SELECT player_key, nhl_player_id FROM yahoo_player_map"
            " WHERE league_key = ? AND nhl_player_id IS NOT NULL",
            (league_key,),
        )
    }


def load_adp(conn: sqlite3.Connection, league_key: str) -> dict[int, int]:
    """nhl_player_id -> Yahoo ADP rank. Replaces the pseudo-ADP in build_universe."""
    return {
        int(r[0]): int(r[1])
        for r in conn.execute(
            "SELECT nhl_player_id, adp_rank FROM yahoo_player_map"
            " WHERE league_key = ? AND nhl_player_id IS NOT NULL",
            (league_key,),
        )
    }
