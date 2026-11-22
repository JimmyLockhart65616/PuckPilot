"""The rest of the season: what a swap does to every week still to come.

The add search prices a swap by this week's expected categories alone
(`week._price`). That buys whatever this week's opponent is close in, and over
a season it walks the roster into categories it already wins: 2026-27's first
seven weeks, every executed add was sold on SOG, HIT or BLK while assists,
PPP and the goalie categories went 11th-12th. A category out of reach this
week is worth nothing to the week's odds however many weeks are left.

`SeasonOutlook` adds the other side of the ledger. Its `leverage` is what one
more unit of each counting category is worth in a typical week - the next
full week's schedule, our roster against each rival's current roster, averaged
over every rival, so no one opponent's quirks decide it. A swap's season gain
is that leverage times the change in each category's total over the rest of
the regular season: each of the two players' starts there are his share of his
club's games in the days the swap was priced (after the swap for the one
coming in, before it for the one going), times his club's games left - so a
streamer's four-game week is not extrapolated, and a regular's every game
counts. A player with no game in the days priced gets a skater's full share,
or a goalie's mean P(start) in the typical week. `weight` (0 = this week only,
as built) is how much of it a swap's price carries.

A category won by forty earns nothing more (its leverage is near zero against
every rival); one we trail most rivals in earns what a fix pays against those
it is close to, across every week left.
"""

from __future__ import annotations

from dataclasses import dataclass, field


class _Memo:
    """A goalie source answering each date once."""

    def __init__(self, source):
        self.source = source
        self._by: dict[str, dict[int, float]] = {}

    def starts(self, date: str) -> dict[int, float]:
        if date not in self._by:
            self._by[date] = self.source.starts(date) if self.source else {}
        return self._by[date]


@dataclass
class SeasonOutlook:
    """Leverage per unit of a week's total, and each club's games left."""

    leverage: dict[str, float]
    rest_games: dict[str, int]  # club -> regular-season games after this week
    goalie_share: dict[int, float]  # goalie -> mean P(start) a club game, typical week
    rivals: int
    weeks: int
    weight: float = 0.0
    _week_games: dict = field(default_factory=dict)

    def week_games(self, conn, runtime, days: list[str]) -> dict[str, int]:
        """club -> games in `days` (the days a swap was priced over)."""
        key = tuple(days)
        if key not in self._week_games:
            self._week_games = {key: _club_games(conn, runtime, days)}
        return self._week_games[key]

    def season_starts(self, p, starts: float, week_games: dict[str, int]) -> float:
        """His starts over the rest of the season, from `starts` in the days priced."""
        rest = self.rest_games.get(p.team, 0)
        n = week_games.get(p.team, 0)
        if n > 0:
            return min(starts / n, 1.0) * rest
        share = self.goalie_share.get(p.nhl_player_id, 0.0) if p.position == "G" else 1.0
        return share * rest

    def player_value(self, p, rates, starts: float | None, week_games: dict[str, int]) -> float:
        """A player's rest-of-season expected categories, on the season term's
        scale: leverage times his line times his starts left. `starts` None:
        a skater's full share, a goalie's typical-week P(start)."""
        if p is None or p.nhl_player_id is None:
            return 0.0
        n = (
            self.season_starts(p, starts, week_games)
            if starts is not None
            else (self.goalie_share.get(p.nhl_player_id, 0.0) if p.position == "G" else 1.0)
            * self.rest_games.get(p.team, 0)
        )
        line = rates.get(p.nhl_player_id) or {}
        return sum(lev * line.get(k, 0.0) * n for k, lev in self.leverage.items())

    def season_gain(self, t, rates, week_games: dict[str, int]) -> float:
        """Expected categories won over the rest of the season, after v before."""
        if not self.leverage:
            return 0.0

        def line(p, starts: float) -> dict[str, float]:
            n = self.season_starts(p, starts, week_games)
            return {k: v * n for k, v in (rates.get(p.nhl_player_id) or {}).items()}

        incoming = line(t.player, t.starts)
        outgoing = line(t.drop, t.drop_starts) if t.drop is not None else {}
        return sum(
            lev * (incoming.get(k, 0.0) - outgoing.get(k, 0.0)) for k, lev in self.leverage.items()
        )


def _club_games(conn, runtime, days) -> dict[str, int]:
    from puckpilot.season import calendar

    out: dict[str, int] = {}
    for d in days:
        for club in calendar.teams_playing(conn, d, runtime.nhl_season):
            out[club] = out.get(club, 0) + 1
    return out


def _regular(runtime) -> list:
    playoffs = getattr(runtime, "playoff_start_week", 0) or 0
    return [w for w in runtime.weeks if not playoffs or w.number < playoffs]


def remaining_days(runtime, week) -> list[str]:
    """Every regular-season day after `week`."""
    return [d for w in _regular(runtime) if w.start > week.end for d in w.dates()]


def typical_week(runtime, week):
    """The week after this one when there is one: a full schedule to price on."""
    later = [w for w in _regular(runtime) if w.start > week.end]
    return later[0] if later else week


def build_outlook(
    conn,
    runtime,
    league,
    week,
    ours,
    rivals: list,
    rates,
    goalie_source,
    values,
    weight: float,
    model=None,
    tilt: dict[str, float] | None = None,
) -> SeasonOutlook | None:
    """The season outlook for `ours` against `rivals` (their current rosters).

    `tilt` weights a category's leverage (`category_plan.tilt`: a manager's
    pinned stance); absent = 1. None when there is nothing to judge: no week
    left, or no rival read.
    """
    from puckpilot.season import odds as odds_mod
    from puckpilot.season.week import _skaters_only, expected_starts

    days = remaining_days(runtime, week)
    rivals = [r for r in rivals if r is not None and r.team_key != ours.team_key]
    if not days or not rivals:
        return None
    model = model or odds_mod.OddsModel()
    source = _Memo(goalie_source)
    typ = typical_week(runtime, week)
    cats = league.all_cats

    def side(roster):
        goalie_games: dict[int, list[float]] = {}
        games = expected_starts(
            conn, runtime, typ, roster.players, source, values, goalie_games=goalie_games
        )
        return odds_mod.side({}, _skaters_only(roster.players, games), goalie_games, rates)

    mine = side(ours)
    total: dict[str, float] = {}
    for r in rivals:
        for k, v in odds_mod.leverage(model, cats, mine, side(r)).items():
            total[k] = total.get(k, 0.0) + v

    seen: dict[int, list[float]] = {}
    for d in typ.dates():
        for pid, p in source.starts(d).items():
            seen.setdefault(pid, []).append(float(p))
    return SeasonOutlook(
        leverage={k: v / len(rivals) * (tilt or {}).get(k, 1.0) for k, v in total.items()},
        rest_games=_club_games(conn, runtime, days),
        goalie_share={pid: sum(ps) / len(ps) for pid, ps in seen.items() if ps},
        rivals=len(rivals),
        weeks=len(days) // 7,
        weight=weight,
    )
