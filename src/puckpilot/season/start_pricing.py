"""A goalie start priced by its game: the opponent, the venue, this season.

The odds price every start a goalie makes at one flat line - his projected
wins a game and shots a game - whoever the opponent and wherever the game.
Measured on 2024-25 / 2025-26 logs: win rate a start runs from .29 (a weak
club against a strong one) to .70, home is worth +4 to +12 points, and the
opponent moves shots faced by a standard deviation of 1.6-1.9 a game.

`validate` asks whether pricing by game beats the flat line, walk-forward:
every start of a season after each club's first `WARMUP` games, as of the
morning of the game -
- win: logit(p) = logit(his flat rate) + b x (club strength - opponent
  strength) + c x (home - away), strength a club's goal difference a game so
  far shrunk toward zero by `SHRINK_GAMES`; b and c fitted on the first season
  only, judged on the rest by Brier;
- saves: his flat saves a start x the opponent's shots for a game so far
  (shrunk toward the league's) / the league's, judged by mean absolute error.
The flat line is the goalie's own season-to-date rate, shrunk toward the
league's by `SHRINK_GAMES` of the league average - a fair baseline a game
model must beat. Nothing here is used unless it wins every judged season.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

WARMUP = 10  # club games before a start is scored
SHRINK_GAMES = 20.0  # pseudo-games of the league average in every rate
B_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)
C_GRID = (0.0, 0.05, 0.1, 0.15, 0.2, 0.3)


@dataclass(frozen=True)
class TeamGame:
    date: str
    game_id: int
    team: str
    opp: str
    home: bool
    shots_against: float
    goals_against: float
    win: bool


def team_games(conn, season: str) -> list[TeamGame]:
    """Every club's games, from its goalies' logs (starter and relief summed)."""
    by: dict[tuple[int, str], dict] = {}
    for gid, date, team, opp, home, stats_json in conn.execute(
        "SELECT game_id, game_date, team_abbrev, opponent_abbrev, is_home, stats_json "
        "FROM nhl_game_logs WHERE season = ? AND game_type = 2 "
        "AND json_extract(stats_json, '$.shotsAgainst') IS NOT NULL",
        (season,),
    ):
        s = json.loads(stats_json)
        g = by.setdefault(
            (gid, team),
            {
                "date": date,
                "opp": opp,
                "home": bool(home),
                "sa": 0.0,
                "ga": 0.0,
                "win": False,
            },  # fmt: skip
        )
        g["sa"] += float(s.get("shotsAgainst") or 0)
        g["ga"] += float(s.get("goalsAgainst") or 0)
        g["win"] = g["win"] or s.get("decision") == "W"
    return sorted(
        (
            TeamGame(v["date"], gid, team, v["opp"], v["home"], v["sa"], v["ga"], v["win"])
            for (gid, team), v in by.items()
        ),
        key=lambda g: (g.date, g.game_id, g.team),
    )


@dataclass(frozen=True)
class Start:
    """One goalie start and what was known that morning."""

    goalie: int
    home: bool
    strength_gap: float  # club - opponent, goal difference a game so far, shrunk
    opp_shots: float  # opponent's shots for a game so far, shrunk, over the league's
    flat_win: float
    flat_saves: float
    win: bool
    saves: float


def starts(conn, season: str) -> list[Start]:
    """Every scorable start of `season`, as of its morning."""
    games = team_games(conn, season)
    by_game: dict[int, dict[str, TeamGame]] = {}
    for g in games:
        by_game.setdefault(g.game_id, {})[g.team] = g
    gf: dict[str, list[float]] = {}  # club -> goals for, game by game
    ga: dict[str, list[float]] = {}
    sf: dict[str, list[float]] = {}  # club -> shots for
    g_rate: dict[int, list[float]] = {}  # goalie -> [starts, wins, saves]
    lg = [0.0, 0.0, 0.0, 0.0]  # team-games, wins, saves, shots
    starter: dict[tuple[int, str], tuple[int, float, bool]] = {}
    for gid, _date, team, stats_json in conn.execute(
        "SELECT game_id, game_date, team_abbrev, stats_json FROM nhl_game_logs WHERE season = ? "
        "AND game_type = 2 AND json_extract(stats_json, '$.gamesStarted') = 1",
        (season,),
    ):
        s = json.loads(stats_json)
        sa = float(s.get("shotsAgainst") or 0)
        starter[(gid, team)] = (0, sa - float(s.get("goalsAgainst") or 0), s.get("decision") == "W")
    pid_of = {
        (gid, team): pid
        for pid, gid, team in conn.execute(
            "SELECT player_id, game_id, team_abbrev FROM nhl_game_logs WHERE season = ? "
            "AND game_type = 2 AND json_extract(stats_json, '$.gamesStarted') = 1",
            (season,),
        )
    }
    out: list[Start] = []
    seen: set[int] = set()
    for g in games:
        if g.game_id in seen:
            continue
        seen.add(g.game_id)
        both = by_game[g.game_id]
        if len(both) != 2:
            continue
        a, b = both.values()
        for me, them in ((a, b), (b, a)):
            pid = pid_of.get((g.game_id, me.team))
            st = starter.get((g.game_id, me.team))
            n_me, n_them = len(gf.get(me.team, [])), len(gf.get(them.team, []))
            if pid is None or st is None or min(n_me, n_them) < WARMUP or lg[0] == 0:
                continue

            def strength(club: str) -> float:
                d = sum(gf[club]) - sum(ga[club])
                return d / (len(gf[club]) + SHRINK_GAMES)

            league_sf = lg[3] / lg[0]
            opp_sf = (sum(sf[them.team]) + SHRINK_GAMES * league_sf) / (
                len(sf[them.team]) + SHRINK_GAMES
            )
            r = g_rate.get(pid, [0.0, 0.0, 0.0])
            flat_w = (r[1] + SHRINK_GAMES * lg[1] / lg[0]) / (r[0] + SHRINK_GAMES)
            flat_s = (r[2] + SHRINK_GAMES * lg[2] / lg[0]) / (r[0] + SHRINK_GAMES)
            out.append(
                Start(
                    pid,
                    me.home,
                    strength(me.team) - strength(them.team),
                    opp_sf / league_sf,
                    flat_w,
                    flat_s,
                    st[2],
                    st[1],
                )  # fmt: skip
            )
        # The game is played: everything above was as of its morning.
        for me, them in ((a, b), (b, a)):
            gf.setdefault(me.team, []).append(them.goals_against)
            ga.setdefault(me.team, []).append(me.goals_against)
            sf.setdefault(me.team, []).append(them.shots_against)
            st = starter.get((g.game_id, me.team))
            pid = pid_of.get((g.game_id, me.team))
            if st is not None and pid is not None:
                r = g_rate.setdefault(pid, [0.0, 0.0, 0.0])
                r[0] += 1
                r[1] += st[2]
                r[2] += st[1]
            lg[0] += 1
            lg[1] += me.win
            lg[2] += me.shots_against - me.goals_against
            lg[3] += me.shots_against
    return out


def _logit(p: float) -> float:
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


def p_win(s: Start, b: float, c: float) -> float:
    z = _logit(s.flat_win) + b * s.strength_gap + c * (1.0 if s.home else -1.0)
    return 1.0 / (1.0 + math.exp(-z))


def brier(xs: list[Start], b: float, c: float) -> float:
    return sum((p_win(s, b, c) - s.win) ** 2 for s in xs) / len(xs)


def saves_mae(xs: list[Start], by_opponent: bool) -> float:
    return sum(
        abs(s.flat_saves * (s.opp_shots if by_opponent else 1.0) - s.saves) for s in xs
    ) / len(xs)


@dataclass(frozen=True)
class SeasonResult:
    season: str
    starts: int
    brier_flat: float
    brier_game: float
    mae_flat: float
    mae_game: float


def validate(conn, seasons: list[str]) -> tuple[float, float, list[SeasonResult]]:
    """(b, c fitted on the first season, every season's result)."""
    data = {s: starts(conn, s) for s in seasons}
    first = data[seasons[0]]
    b, c = min(((b, c) for b in B_GRID for c in C_GRID), key=lambda bc: brier(first, *bc))
    out = [
        SeasonResult(
            s,
            len(xs),
            brier(xs, 0.0, 0.0),
            brier(xs, b, c),
            saves_mae(xs, False),
            saves_mae(xs, True),
        )  # fmt: skip
        for s, xs in data.items()
    ]
    return b, c, out


def text(b: float, c: float, results: list[SeasonResult]) -> str:
    out = [
        "A start priced by its game v the flat line (walk-forward, after each club's first "
        f"{WARMUP} games):",
        f"  win: b {b:g} a goal of strength gap a game, c {c:g} for home (fitted on "
        f"{results[0].season})",
    ]
    for i, r in enumerate(results):
        role = "fit" if i == 0 else "judged"
        out.append(
            f"  {r.season} ({role}, {r.starts} starts): win Brier {r.brier_flat:.4f} -> "
            f"{r.brier_game:.4f}; saves MAE {r.mae_flat:.2f} -> {r.mae_game:.2f}"
        )
    judged = results[1:]
    win_ok = all(r.brier_game < r.brier_flat for r in judged)
    sv_ok = all(r.mae_game < r.mae_flat for r in judged)
    out.append(
        f"  verdict: wins {'PASS' if win_ok else 'FAIL'}, saves {'PASS' if sv_ok else 'FAIL'}"
        " (better in every judged season)"
    )
    return "\n".join(out)


class OpponentShots:
    """Each club's shots for a game as of a date, over the league's: the
    factor a goalie's shots faced (and so his saves) is scaled by for a game
    against it. Shrunk toward the league by `SHRINK_GAMES`."""

    def __init__(self, conn, season: str):
        self._games = team_games(conn, season)
        opp: dict[tuple[str, str], str] = {}
        for g in self._games:
            opp[(g.date, g.team)] = g.opp
        for d, home, away in conn.execute(
            "SELECT game_date, home_team, away_team FROM nhl_schedule WHERE season = ? "
            "AND game_type = 2",
            (season,),
        ):
            opp.setdefault((d, home), away)
            opp.setdefault((d, away), home)
        self._opp = opp
        self._cache: dict[str, dict[str, float]] = {}

    def opponent(self, date: str, club: str) -> str:
        return self._opp.get((date, club), "")

    def ratios(self, cutoff: str) -> dict[str, float]:
        """club -> shots for a game before `cutoff`, shrunk, over the league's."""
        if cutoff not in self._cache:
            shots: dict[str, list[float]] = {}
            by_game: dict[int, list[TeamGame]] = {}
            for g in self._games:
                if g.date < cutoff:
                    by_game.setdefault(g.game_id, []).append(g)
            for pair in by_game.values():
                if len(pair) != 2:
                    continue
                a, b = pair
                shots.setdefault(a.team, []).append(b.shots_against)
                shots.setdefault(b.team, []).append(a.shots_against)
            n = sum(len(v) for v in shots.values())
            league = sum(sum(v) for v in shots.values()) / n if n else 0.0
            self._cache[cutoff] = (
                {
                    t: (sum(v) + SHRINK_GAMES * league) / (len(v) + SHRINK_GAMES) / league
                    for t, v in shots.items()
                }
                if league
                else {}
            )
        return self._cache[cutoff]

    def factor(self, club: str, dates: list[str], weights: list[float], cutoff: str) -> float:
        """A goalie's mean opponent factor over his games (`weights`: his chances)."""
        r = self.ratios(cutoff)
        total = sum(weights)
        if not total:
            return 1.0
        return (
            sum(r.get(self.opponent(d, club), 1.0) * w for d, w in zip(dates, weights, strict=True))
            / total
        )
