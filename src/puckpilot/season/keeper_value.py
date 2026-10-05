"""Next season's keepers, so that this season's moves do not give one away.

A drop in a keeper league is permanent in a way it is not in a redraft one: the
player goes back to the pool, and whoever holds him when the season ends holds
his keeper rights (`yahoo.keeperhistory` - eligibility is the season-end
holder). Everything the add engine measured was this season - a player's value
a game times the games left - so a young forward having a slow October, worth
more next year than anyone on the wire, could look like the cheapest player on
the roster.

So each player gets a keeper value: his projected value over replacement next
season, made the way the preseason projection is but with this season so far
as its most recent year (`projections.project(as_of=...)`), less what keeping
him costs - the draft pick he uses. The best few eligible players are protected:
never proposed as a drop. The two seasons are never mixed into one number. A
keeper and a streamer are not measured in the same unit, and how much of this
year to give up for next is the manager's call, not a weight's.

Measured before it was trusted: `season.keeper_gate` replays past seasons and
asks which way of ranking a roster on a given date would have kept the players
who turned out best the next year.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from puckpilot.league import LeagueConfig

# Protect one more than the league lets a team keep: the line between the last
# keeper and the next one moves from week to week, and a drop is forever.
KEEPER_MARGIN = 1


def next_season(season: str) -> str:
    y = int(season[:4]) + 1
    return f"{y}{y + 1}"


def project_next(
    conn: sqlite3.Connection, league: LeagueConfig, season: str, as_of: str | None
) -> pd.Series:
    """player id -> projected value over replacement next season, as of `as_of`.

    This season (so far, if `as_of` falls inside it) and the two before it,
    projected for a season one year older and as long as this one - next
    season's schedule does not exist yet. Indexed by NHL player id.
    """
    from puckpilot.engine import projections
    from puckpilot.engine.aggregate import season_games
    from puckpilot.engine.valuation import rank_players

    y = int(season[:4])
    train = [f"{y - i}{y - i + 1}" for i in range(3)]
    skaters, goalies = projections.project(
        conn,
        next_season(season),
        train,
        as_of=as_of,
        target_games=season_games(conn, season),
    )
    ranked = rank_players(
        skaters,
        goalies,
        shape=league.shape,
        skater_cats=league.skater_cats,
        goalie_cats=league.goalie_cats,
    )
    if ranked.empty:
        return pd.Series(dtype=float)
    out = ranked["vorp"].astype(float)
    out.index = out.index.astype(int)
    return out


def keeper_cost(league: LeagueConfig, vorp_next: pd.Series) -> float:
    """What keeping a player costs, in the same units: the pick he uses.

    "last": keepers fill each seat's final rounds, so a keeper stands in for
    the player that seat would have taken there - about the best one left once
    every roster in the league is full, which is close to replacement, so
    keeping is nearly free. "first": a keeper uses one of the seat's earliest
    picks - about the best player left once every team's keepers are off the
    board, which is not cheap at all. A league without keepers costs nothing
    because it keeps nothing.
    """
    n, k, size = league.shape.n_teams, league.n_keepers, league.shape.roster_size
    if k <= 0 or vorp_next.empty:
        return 0.0
    ordered = vorp_next.sort_values(ascending=False).to_numpy()
    lo, hi = (n * k, 2 * n * k) if league.keeper_placement == "first" else (n * size, n * size + n)
    window = ordered[lo:hi]
    return float(window.mean()) if len(window) else float(ordered[-1])


def contracts(path: Path) -> dict[int, int] | None:
    """NHL id -> times kept going into next season, or None without the file.

    `ppilot yahoo keepers` saves each season's contracts going into its draft
    (`continuing`, with how many times each had been kept) and the keepers then
    declared. A declared keeper's count is his count going in plus this season;
    anyone not declared starts again at nothing - whoever holds him at the end
    of the season may keep him for the first time.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    before: dict[int, int] = {}
    for m in data.get("managers", []):
        for c in m.get("continuing", []):
            if c.get("nhl_id") is not None:
                before[int(c["nhl_id"])] = int(c.get("times_kept", 0) or 0)
    out: dict[int, int] = {}
    for d in data.get("declared", []):
        if d.get("nhl_id") is not None:
            pid = int(d["nhl_id"])
            out[pid] = before.get(pid, 0) + 1
    return out


@dataclass(frozen=True)
class KeeperRank:
    """One rostered player as a keeper for next season."""

    player_key: str
    nhl_player_id: int | None
    name: str
    vorp_next: float | None
    value: float | None  # vorp_next less the keeper cost
    times_kept: int  # going into next season, counting this one
    eligible: bool
    rank: int | None  # among eligible players with a value; 1 is the best
    protected: bool


@dataclass
class KeeperBoard:
    """Next season's values league-wide, the keeper cost and the contracts."""

    league: LeagueConfig
    season: str
    as_of: str
    vorp_next: pd.Series
    cost: float
    times_kept: dict[int, int] | None  # None: contracts unknown

    def value(self, pid: int | None) -> float | None:
        if pid is None or pid not in self.vorp_next.index:
            return None
        return float(self.vorp_next[pid]) - self.cost

    def kept(self, pid: int | None) -> int:
        return (self.times_kept or {}).get(pid, 0) if pid is not None else 0

    def eligible(self, pid: int | None) -> bool:
        return self.kept(pid) < self.league.keeper_years

    def rank(self, players, margin: int = KEEPER_MARGIN) -> list[KeeperRank]:
        """The roster as next season's keepers, best first; unranked at the end.

        The top `n_keepers + margin` eligible players worth more than they cost
        are protected. A league that keeps nobody protects nobody.
        """
        n_protect = self.league.n_keepers + margin if self.league.n_keepers > 0 else 0
        rows = []
        for p in players:
            pid = p.nhl_player_id
            v = self.value(pid)
            rows.append((p, v, self.kept(pid), self.eligible(pid)))
        ranked = sorted(
            (r for r in rows if r[3] and r[1] is not None), key=lambda r: (-r[1], r[0].name)
        )
        rank_of = {r[0].player_key: i for i, r in enumerate(ranked, start=1)}
        out = []
        for p, v, kept, eligible in rows:
            r = rank_of.get(p.player_key)
            vorp = None if v is None else v + self.cost
            out.append(
                KeeperRank(
                    player_key=p.player_key,
                    nhl_player_id=p.nhl_player_id,
                    name=p.name,
                    vorp_next=vorp,
                    value=v,
                    times_kept=kept,
                    eligible=eligible,
                    rank=r,
                    protected=r is not None and r <= n_protect and (v or 0.0) > 0.0,
                )
            )
        out.sort(key=lambda k: (k.rank is None, k.rank or 0, k.name))
        return out


def board(
    conn: sqlite3.Connection,
    league: LeagueConfig,
    season: str,
    as_of: str,
    contracts_path: Path | None = None,
) -> KeeperBoard:
    vorp = project_next(conn, league, season, as_of)
    return KeeperBoard(
        league=league,
        season=season,
        as_of=as_of,
        vorp_next=vorp,
        cost=keeper_cost(league, vorp),
        times_kept=contracts(contracts_path) if contracts_path is not None else None,
    )


# -- the week's decision -------------------------------------------------------


@dataclass(frozen=True)
class SavedPlayer:
    player_key: str
    nhl_player_id: int | None
    name: str


def saved_roster(conn: sqlite3.Connection, manager: str, team_key: str = "") -> list[SavedPlayer]:
    """The manager's own roster as the last run saved it - no browser needed.

    Each run also saves the opponent's roster under the same manager, so the
    team is the one named, or else the one with the most snapshots.
    """
    if not team_key:
        top = conn.execute(
            "SELECT team_key FROM yahoo_roster_snapshots WHERE manager = ? "
            "GROUP BY team_key ORDER BY COUNT(*) DESC LIMIT 1",
            (manager,),
        ).fetchone()
        if not top:
            return []
        team_key = top[0]
    rows = conn.execute(
        "SELECT player_key, nhl_player_id, name FROM yahoo_roster_snapshots "
        "WHERE manager = ? AND team_key = ? AND date = (SELECT MAX(date) FROM "
        "yahoo_roster_snapshots WHERE manager = ? AND team_key = ?)",
        (manager, team_key, manager, team_key),
    ).fetchall()
    return [SavedPlayer(r[0], r[1], r[2] or "") for r in rows]


def load(conn: sqlite3.Connection, manager: str, week_start: str) -> list[KeeperRank]:
    """This week's keeper ranks for a manager, as decided on its first run."""
    rows = conn.execute(
        "SELECT player_key, nhl_player_id, name, vorp_next, value, times_kept, eligible, "
        "rank, protected FROM keeper_ranks WHERE manager = ? AND week_start = ?",
        (manager, week_start),
    ).fetchall()
    out = [
        KeeperRank(
            player_key=r[0],
            nhl_player_id=r[1],
            name=r[2] or "",
            vorp_next=r[3],
            value=r[4],
            times_kept=int(r[5]),
            eligible=bool(r[6]),
            rank=r[7],
            protected=bool(r[8]),
        )
        for r in rows
    ]
    out.sort(key=lambda k: (k.rank is None, k.rank or 0, k.name))
    return out


def save(
    conn: sqlite3.Connection, manager: str, week_start: str, as_of: str, ranks: list[KeeperRank]
) -> None:
    conn.executemany(
        "INSERT OR REPLACE INTO keeper_ranks (manager, week_start, player_key, nhl_player_id, "
        "name, vorp_next, value, times_kept, eligible, rank, protected, as_of) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                manager,
                week_start,
                k.player_key,
                k.nhl_player_id,
                k.name,
                k.vorp_next,
                k.value,
                k.times_kept,
                int(k.eligible),
                k.rank,
                int(k.protected),
                as_of,
            )
            for k in ranks
        ],
    )
    conn.commit()


def annotate(roster, ranks: list[KeeperRank]):
    """The roster with each player's keeper rank and protection set.

    A player added since the week's ranks were decided has none: he is
    ranked with everyone else next week.
    """
    from dataclasses import replace

    by_key = {k.player_key: k for k in ranks}
    players = tuple(
        replace(p, keeper_rank=k.rank, keeper_protected=k.protected)
        if (k := by_key.get(p.player_key)) is not None
        else p
        for p in roster.players
    )
    return replace(roster, players=players)


def card_lines(cand, drop, roster, keepers: KeeperBoard) -> list[str]:
    """What the swap does to next season's keepers, for a proposal's card."""
    n = keepers.league.n_keepers
    if n <= 0:
        return []
    mates = [p for p in roster if drop is None or p.player_key != drop.player_key]
    ranked = keepers.rank(mates)
    out = []
    v = keepers.value(cand.nhl_player_id)
    if v is None:
        out.append(f"{cand.name}: no projection for next season")
    else:
        above = sum(1 for k in ranked if k.rank is not None and (k.value or 0.0) > v)
        out.append(
            f"{cand.name} projects {v + keepers.cost:+.1f} over replacement next season - "
            f"would rank {above + 1} of your keepers (you keep {n})"
        )
    if drop is not None:
        if not keepers.eligible(drop.nhl_player_id):
            out.append(
                f"{drop.name} has been kept {keepers.kept(drop.nhl_player_id)} times - "
                f"not a keeper next season"
            )
        elif drop.keeper_rank is not None:
            out.append(f"{drop.name} ranks {drop.keeper_rank} of your keepers next season")
        else:
            out.append(f"{drop.name}: not ranked as a keeper next season")
    return out
