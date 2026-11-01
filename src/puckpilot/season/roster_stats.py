"""The numbers a manager sets a player's drop tier by, for the page's roster.

Two, because they answer different questions (asked for on 2026-10-31,
"season rank or VORP values so I can inform the decisions"):

    season  how he has played: this season's totals so far in the league's own
            categories, valued over replacement the way the draft board values
            a season (`valuation.rank_players`) - his rank among every player
            who has played, and that value (VORP)
    rest    what letting him go costs: his value over the rest of the season
            (`week.keep_values`), the number every drop is chosen by - and his
            place on this roster by it, 1st worth most

A player can be high on one and low on the other - a hot start on a shrinking
role, or a slow October from a regular - and the tier is the manager's call
between them.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping


def season_ranks(
    conn: sqlite3.Connection, league, season: str, day: str
) -> dict[int, tuple[int, float]]:
    """NHL player id -> (rank, VORP) on this season's totals before `day`."""
    from puckpilot.engine.aggregate import season_aggregates
    from puckpilot.engine.valuation import rank_players

    skaters, goalies = season_aggregates(conn, season, before=day)
    ranked = rank_players(
        skaters,
        goalies,
        shape=league.shape,
        skater_cats=league.skater_cats,
        goalie_cats=league.goalie_cats,
    )
    if ranked.empty:
        return {}
    return {int(pid): (i + 1, float(v)) for i, (pid, v) in enumerate(ranked["vorp"].items())}


def _nth(n: int) -> str:
    if 10 <= n % 100 <= 20:
        return f"{n}th"
    return f"{n}{ {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th') }"


def describe(roster, keep: Mapping[str, float], ranks: Mapping[int, tuple[int, float]]) -> dict:
    """player key -> his numbers and their line on the page:
    "Season #109, VORP +1.4 · Rest of season 304, 15th of 18"."""
    valued = sorted(
        (p for p in roster.players if p.player_key in keep), key=lambda p: -keep[p.player_key]
    )
    place = {p.player_key: i + 1 for i, p in enumerate(valued)}
    out = {}
    for p in roster.players:
        got = ranks.get(p.nhl_player_id) if p.nhl_player_id is not None else None
        bits = [f"Season #{got[0]}, VORP {got[1]:+.1f}" if got else "No games this season"]
        if p.player_key in place:
            bits.append(
                f"Rest of season {keep[p.player_key]:.0f}, "
                f"{_nth(place[p.player_key])} of {len(valued)}"
            )
        out[p.player_key] = {
            "season_rank": got[0] if got else None,
            "season_vorp": round(got[1], 1) if got else None,
            "rest": round(float(keep[p.player_key])) if p.player_key in place else None,
            "rest_place": place.get(p.player_key),
            "rest_of": len(valued),
            "text": " · ".join(bits),
        }
    return out
