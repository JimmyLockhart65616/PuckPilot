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

import re
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
    positions: dict[int, str] = field(default_factory=dict)


def _last(name: str) -> str:
    return _norm(name.split()[-1]) if name.split() else ""


# Short forms that share no letters a spelling comparison could find.
NICKNAMES = {
    "bob": "robert",
    "rob": "robert",
    "bobby": "robert",
    "bill": "william",
    "billy": "william",
    "will": "william",
    "dick": "richard",
    "ted": "edward",
    "ned": "edward",
    "chuck": "charles",
    "jack": "john",
}


def _given(name: str) -> list[str]:
    """'Anthony (AJ) Spellacy' -> ['anthony', 'aj']: the given names, surname dropped."""
    parts = name.split()[:-1]
    return [_norm(t) for t in re.split(r"[\s\-()]+", " ".join(parts)) if _norm(t)]


def same_given(a: str, b: str) -> bool:
    """Whether two given names could be one person's: the same initial, a close
    spelling (Egor / Yegor), a shared token (AJ / Anthony (AJ)) or a common
    nickname (Bob / Robert). Tarin / Konnor and William / John are not."""
    ga, gb = _given(a), _given(b)
    if not ga or not gb:
        return True
    if ga[0][0] == gb[0][0] or set(ga) & set(gb):
        return True
    x, y = "".join(ga), "".join(gb)
    if NICKNAMES.get(x) == y or NICKNAMES.get(y) == x:
        return True
    from difflib import SequenceMatcher

    return SequenceMatcher(None, x, y).ratio() >= 0.6


def position_class(positions: str | None) -> str:
    """F, D or G from a position or a Yahoo eligibility list; '' when unknown."""
    got = {x.strip().upper() for x in re.split(r"[,/ ]+", positions or "") if x.strip()}
    if "G" in got:
        return "G"
    if got & {"C", "LW", "RW", "L", "R", "F", "W"}:
        return "F"
    if "D" in got:
        return "D"
    return ""


def _fits(idx: NhlIndex, pid: int, name: str, ycls: str) -> bool:
    """A fuzzy candidate may be this Yahoo player: a given name that could be
    his, and no forward / defence / goalie disagreement."""
    if not same_given(name, idx.names[pid]):
        return False
    ncls = position_class(idx.positions.get(pid))
    return not (ycls and ncls and ycls != ncls)


def _nhl_index(conn: sqlite3.Connection) -> NhlIndex:
    idx = NhlIndex()
    for row in conn.execute("SELECT player_id, full_name, team_abbrev, position FROM nhl_players"):
        pid, name, team, pos = int(row[0]), row[1], row[2], row[3]
        key = _norm(name)
        idx.by_name.setdefault(key, []).append(pid)
        idx.by_name_team.setdefault((key, _team(team)), pid)
        idx.by_last.setdefault(_last(name), []).append(pid)
        idx.names[pid] = name
        idx.teams[pid] = _team(team)
        idx.positions[pid] = pos or ""
    return idx


def _resolve(idx: NhlIndex, name: str, team: str, positions: str = "") -> tuple[int | None, str]:
    """Best NHL id for a Yahoo name, plus how it was found.

    Three layers, each narrower than the last. The fallbacks exist because two
    real mismatch classes turn up among *stars*, where a miss is expensive:

    - nickname vs given name ("Mitch Marner" / "Mitchell Marner") and
      transliteration variants ("Egor" / "Yegor Chinakhov");
    - upstream names where an accented letter was DROPPED rather than folded -
      MoneyPuck supplies "Tim Sttzle" and "Alexis Lafrenire". Yahoo spells them
      correctly, so exact and even last-name matching both fail.

    Every fallback requires the team to agree, which is what keeps a fuzzy match
    from ever silently taking the wrong player off the board - and, given
    Yahoo's `positions`, a given name that could be his and no forward /
    defence / goalie disagreement. Without those, a surname and a club were
    enough: Tarin Smith came back as Konnor Smith, and a centre named William
    Moore as a defenceman named John. Two players with one name on one club
    (Vancouver has two Elias Petterssons) are told apart by position.
    """
    norm = _norm(name)
    ycls = position_class(positions)

    exact = idx.by_name.get(norm, [])
    if len(exact) == 1:
        return exact[0], "exact"
    if len(exact) > 1:
        on_team = [p for p in exact if idx.teams.get(p) == team]
        if len(on_team) == 1:
            return on_team[0], "exact+team"
        if ycls:
            fit = [p for p in (on_team or exact) if position_class(idx.positions.get(p)) == ycls]
            if len(fit) == 1:
                return fit[0], "exact+position"
        return None, "ambiguous"

    from difflib import SequenceMatcher

    # Layer 2: surname plus team. Catches nicknames and transliterations.
    all_last = idx.by_last.get(_last(name), [])
    same_last = [p for p in all_last if idx.teams.get(p) == team and _fits(idx, p, name, ycls)]
    if len(same_last) == 1:
        return same_last[0], "surname+team"

    # Layer 3: a surname unique across the whole league, with the given names
    # close enough to be the same person. This layer has to exist because
    # `nhl_players.team_abbrev` is the LAST-PLAYED team, not the current one, so
    # every player traded in the offseason fails any team-constrained check -
    # Marner reads as TOR here and VGK on Yahoo. Requiring a unique surname plus
    # name similarity keeps it safe without trusting the stale team.
    if len(all_last) == 1 and _fits(idx, all_last[0], name, ycls):
        score = SequenceMatcher(None, norm, _norm(idx.names[all_last[0]])).ratio()
        if score >= 0.70:
            return all_last[0], f"surname-unique{score:.2f}"

    # Layer 4: closest full name on the same team, only if clearly close.

    best, best_score = None, 0.0
    for pid, other in idx.names.items():
        if idx.teams.get(pid) != team or not _fits(idx, pid, name, ycls):
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
        positions = ",".join(p.get("eligible_positions") or [])
        nhl_id, how = _resolve(idx, name, team, positions)
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


def reresolve_unmatched(conn: sqlite3.Connection, progress: Progress = _noop) -> MapReport:
    """Re-try every `yahoo_player_map` row with no NHL id, against `nhl_players`
    as it stands right now - without asking Yahoo for anything.

    `build_map` resolves once, against whatever `nhl_players` contained at that
    moment. A name that failed then is not stuck failing forever: `data sync`
    (roster sync in particular, which inserts pre-debut players `nhl_players`
    has never held) can make a previously-unmatched name resolvable later, and
    re-running the *whole* Yahoo fetch just to pick that up means another
    browser session for zero new information from Yahoo's side. This only
    touches `nhl_player_id`, never `adp_rank` or `positions` - the fetched
    fields are not being re-fetched, so nothing here may overwrite them.

    Across every league key at once. A resolved NHL id already claimed by
    another mapped row (any league) is skipped, same non-double-assignment
    guard as `build_map`.
    """
    idx = _nhl_index(conn)
    claimed = {
        int(r[0])
        for r in conn.execute(
            "SELECT DISTINCT nhl_player_id FROM yahoo_player_map WHERE nhl_player_id IS NOT NULL"
        )
    }
    rows = conn.execute(
        "SELECT player_key, full_name, team_abbrev, positions FROM yahoo_player_map"
        " WHERE nhl_player_id IS NULL"
    ).fetchall()

    report = MapReport(total=len(rows))
    updates = []
    for key, name, team, positions in rows:
        nhl_id, how = _resolve(idx, name, team or "", positions or "")
        if how == "ambiguous":
            report.ambiguous.append(name)
        if nhl_id is not None and nhl_id in claimed:
            nhl_id = None  # already spoken for by a different Yahoo entry
        if nhl_id is None:
            report.unmatched.append(name)
            continue
        if how not in ("exact",):
            report.fallbacks.append(f"{name} -> {idx.names[nhl_id]} ({how})")
        claimed.add(nhl_id)
        report.matched += 1
        updates.append((nhl_id, key))

    conn.executemany("UPDATE yahoo_player_map SET nhl_player_id = ? WHERE player_key = ?", updates)
    conn.commit()
    progress(f"  re-resolved {report.matched}/{report.total} previously-unmatched players")
    return report


def recheck_map(conn: sqlite3.Connection, progress: Progress = _noop) -> list[str]:
    """Resolve every stored row again, as `build_map` would today - no Yahoo read.

    The rows keep what was fetched (name, team, eligibility, ADP); only the NHL
    id is decided again, best ADP first within each league, so a match an older
    resolver got wrong is corrected now rather than at the next full rebuild.
    Returns one line per changed row.
    """
    idx = _nhl_index(conn)
    rows = conn.execute(
        "SELECT player_key, league_key, full_name, team_abbrev, positions, nhl_player_id"
        " FROM yahoo_player_map ORDER BY league_key, adp_rank"
    ).fetchall()
    claimed: dict[str, set[int]] = {}
    changes, updates = [], []
    for key, league, name, team, positions, old in rows:
        taken = claimed.setdefault(league, set())
        new, _how = _resolve(idx, name, team or "", positions or "")
        if new is not None and new in taken:
            new = None
        if new is not None:
            taken.add(new)
        if new != old:
            updates.append((new, key, league))
            was = idx.names.get(old, old) if old is not None else "nobody"
            now = idx.names.get(new, new) if new is not None else "nobody"
            changes.append(f"{name} ({team} {positions}): {was} -> {now}")
    conn.executemany(
        "UPDATE yahoo_player_map SET nhl_player_id = ? WHERE player_key = ? AND league_key = ?",
        updates,
    )
    conn.commit()
    progress(f"  rechecked {len(rows)} rows, {len(changes)} changed")
    return changes


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


def load_eligibility(
    conn: sqlite3.Connection, league_key: str | None = None
) -> dict[int, frozenset[str]]:
    """nhl_player_id -> the positions Yahoo lets him fill ("C,LW,Util" -> {C, L}).

    Stored by `build_map` since the map was first built and never read until
    now. Newest row wins when several leagues map the same player.
    """
    from puckpilot.draft.eligibility import parse_yahoo_positions

    sql = (
        "SELECT nhl_player_id, positions FROM yahoo_player_map"
        " WHERE nhl_player_id IS NOT NULL AND positions IS NOT NULL"
    )
    params: tuple = ()
    if league_key:
        sql += " AND league_key = ?"
        params = (league_key,)
    out: dict[int, frozenset[str]] = {}
    for pid, raw in conn.execute(sql + " ORDER BY updated_at", params):
        got = parse_yahoo_positions(raw)
        if got:
            out[int(pid)] = got
    return out


def position_corrections(conn: sqlite3.Connection, league_key: str | None = None) -> dict[int, str]:
    """nhl_player_id -> the position to VALUE him at, for skaters whose NHL
    position is one Yahoo does not let him play at all.

    Martin Necas is a C in the NHL's data and RW-only on Yahoo. VORP subtracts
    a per-position replacement level, and a centre's sits about two VORP above
    a winger's, so he is valued against the wrong pool - and can never fill the
    C slot the roster accounting gives him. The correction is Yahoo's first
    listed position. Anyone Yahoo allows at his NHL position is left alone.

    Measured 2026-09-16 as `build_universe(position_overrides=...)` and NOT
    wired into the board: n=1000 over two seeds, top-3 fell 0.739 -> 0.718 and
    0.740 -> 0.708 on target 2025-26 while rising 0.310 -> 0.393 and 0.323 ->
    0.409 on 2024-25. Seasons opposite in sign, so it failed the pre-registered
    gate. `draft preflight` names the affected players instead.
    """
    from puckpilot.draft.eligibility import YAHOO_TO_POS

    nhl = {
        int(r[0]): str(r[1]) for r in conn.execute("SELECT player_id, position FROM nhl_players")
    }
    sql = (
        "SELECT nhl_player_id, positions FROM yahoo_player_map"
        " WHERE nhl_player_id IS NOT NULL AND positions IS NOT NULL"
    )
    params: tuple = ()
    if league_key:
        sql += " AND league_key = ?"
        params = (league_key,)
    out: dict[int, str] = {}
    for pid, raw in conn.execute(sql + " ORDER BY updated_at", params):
        listed = [YAHOO_TO_POS[p] for p in str(raw).split(",") if p in YAHOO_TO_POS]
        current = nhl.get(int(pid))
        if not listed or current in (None, "G") or "G" in listed:
            continue
        if current not in listed:
            out[int(pid)] = listed[0]
        else:
            out.pop(int(pid), None)
    return out


def mapped_league_keys(conn: sqlite3.Connection) -> list[tuple[str, int, str]]:
    """(league_key, rows with an ADP, last updated) for every league in the map."""
    return [
        (str(r[0]), int(r[1]), str(r[2]))
        for r in conn.execute(
            "SELECT league_key, SUM(adp_rank IS NOT NULL AND nhl_player_id IS NOT NULL),"
            " MAX(updated_at) FROM yahoo_player_map GROUP BY league_key ORDER BY league_key"
        )
    ]


def resolve_adp_key(
    conn: sqlite3.Connection, explicit: str | None, yahoo_arg: str | None
) -> tuple[str | None, list[str]]:
    """Which league's Yahoo ADP the board should use, and what to say about it.

    Every survival probability on screen and the survival discount in the
    score are computed off ADP. Without it the board falls back to a proxy
    (last season's actual value order) - and it used to do that silently:
    `--yahoo` given bare, or a key with a typo, simply produced no ADP and no
    message. So every way of not getting ADP is named here, and the keys that
    DO exist are listed, because a typo is the likeliest cause.

    An explicit key wins; then a key given to `--yahoo`; then, if the map holds
    exactly one league, that one.
    """
    notes: list[str] = []
    keys = mapped_league_keys(conn)
    listing = ", ".join(f"{k} ({n} priced, updated {u})" for k, n, u in keys) or "none"
    key = explicit or (yahoo_arg if yahoo_arg and "." in str(yahoo_arg) else None)
    if key is None:
        if len(keys) == 1:
            key = keys[0][0]
            notes.append(f"using Yahoo ADP from the only mapped league, {key}")
        else:
            notes.append(
                "WARNING: no ADP league key given and the player map does not name exactly "
                f"one league (mapped: {listing}). Pass --adp-league-key."
            )
            return None, notes
    if not load_adp(conn, key):
        notes.append(
            f"WARNING: no Yahoo ADP for league key {key!r} - mapped keys: {listing}. "
            "Run `ppilot yahoo playermap --league-key <key>`, or fix the key."
        )
        return None, notes
    return key, notes


def pool_adp(conn: sqlite3.Connection, league_key: str | None = None) -> dict[str, float]:
    """Bare Yahoo player id -> ADP rank, over the whole fetched pool.

    Keyed to match the draft-room websocket, which sends "6743" where the map
    table stores "477.p.6743" - the same convention as
    `wsfeed.load_yahoo_id_map`.

    This is the ADP the survival calibration should fit against. It is Yahoo's
    own pre-draft market view of ~400 players, independent of any one draft, so
    it is neither sparse (the in-draft advice channel covered 26 of 192 picks in
    the 2026-09-08 mock) nor circular (draft order cannot stand in for the rank
    the draft is being measured against).
    """
    sql = "SELECT player_key, adp_rank FROM yahoo_player_map WHERE adp_rank IS NOT NULL"
    params: tuple = ()
    if league_key:
        sql += " AND league_key = ?"
        params = (league_key,)
    return {str(k).rsplit(".", 1)[-1]: float(rank) for k, rank in conn.execute(sql, params)}
