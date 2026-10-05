"""Who is starting in goal tonight, in descending order of how much we know.

The bench-regret replay measured what goalie information is worth: an optimizer
with *perfect* starting-goalie data captured 92.6% of the hindsight ceiling and
one with announcements 90% accurate captured 92.3%, against 80.3% for
set-and-forget. That 0.3pt gap was read as "goalie precision barely matters".

Measuring the model below says the reading was too comfortable. A schedule-and-
workload model with no announcement at all tops out near **62%**, not 90%, and
nothing tried moved it far:

    trailing workload share only                     59.0%
    strict alternation (never start twice running)   53.0%   <- tandems do not alternate
    share, demoting the previous starter             61.5%   <- adopted
    (same, demoting only on a back-to-back)          60.6%

So the honest position is that 90% is announcement territory and this is the
floor, roughly 30 points below it. It is still the right floor - it needs no
external source, it works on opening night, and workload is the most repeatable
goalie signal there is (games/starts repeat at 0.53 year over year against 0.13
for save percentage). But a real announcement feed is worth chasing, and
`ChainedGoalieSource` exists so one can be dropped in front of this without
touching anything downstream.

All sources implement `data.goalies.GoalieStartSource`: one method returning
P(start) per NHL player id for a date. The optimizer weights value by that
probability, so a source may be confident (1.0/0.0) or hedged (0.62).
"""

from __future__ import annotations

import bisect
import json
import sqlite3
from collections.abc import Iterable

# Team games of history to weigh. Swept over 2025-26 and confirmed on 2024-25
# and 2023-24; 10 was best or tied-best in all three, and the plateau across
# 10-20 games is broad enough that the exact value is not load-bearing.
# Re-swept on the current model (2026-10-05, `goalie-check`): 6 and 8 worse in
# every season and measure; 12 and 15 a little better on Brier (+0.001 to
# +0.006) and a little worse at naming tonight's starter (-0.2 to -1.0 pt),
# all inside noise. Demotion 0.4 costs 2-3 points tonight, 0.6-0.7 Brier.
TRAILING_TEAM_GAMES = 10

# Multiplicative demotion for whoever started the team's previous game. 0.5 was
# best on all three seasons; 1.0 (no demotion) costs 2-4 points of accuracy,
# and going below 0.35 costs more than it gains - starters ride, so a model
# that insists on alternating is worse than one that ignores rest entirely.
PREVIOUS_STARTER_DAMPING = 0.5

# Below this, call it "not starting" rather than a long shot: the optimizer
# would otherwise hold a slot open on a 3% chance instead of filling it.
MIN_MEANINGFUL_P = 0.05

# How a game after the team's next one is forecast. The demotion above is a
# fact about the next game - whoever started last night is less likely to
# start tonight - and "damped" applies it to every later game as well, so the
# goalie who started Monday is marked down for Saturday too. "share" uses the
# plain trailing share for every game after the next; "chain" walks the
# demotion forward a game at a time, from a mix of goalies instead of a known
# one.
AHEAD_RULES = ("damped", "share", "chain")

# The model's defaults since 2026-10-05, from `ppilot season goalie-check`
# (every team-date after the 21st game date, 2023-24 to 2025-26, frozen as of
# the date) against the model as first measured:
#
#                     tonight's favourite started   Brier, games 2-7 ahead
#     first measured  61.8% / 63.7% / 60.9%         0.572 / 0.567 / 0.592
#     these defaults  64.0% / 66.0% / 63.2%         0.543 / 0.535 / 0.561
#
# +2.2 to +2.4 points tonight (about 4 SE, clustered by club) and 0.21-0.22
# fewer starts misplaced per 7 games, in all three seasons. Downstream:
# the live lineup path is flat (+1.4 +/- 1.3 and -0.0 +/- 0.4 a roster), and
# the add gate with this model for the tested team only, 4 drafts x 2 seasons,
# pooled +0.05 +/- 0.02 categories a week (one league clear of 2 SE, none
# worse by it). Two clubs a goalie dressed
# for one after another count him only for the latter; one who has not
# dressed in two games is out; games after the next walk the demotion forward.
CURRENT_CLUB = True
DRESSED_WINDOW = 2
AHEAD = "chain"


class TrailingStartShareSource:
    """P(start) from who has been starting, demoting the previous starter.

    Indexed eagerly for the season, like `HindsightGoalieSource`, because the
    per-date form of this was a 30-day window scan per call.

    `fallback_season` covers opening weeks, when the current season has too few
    games to say anything: last season's share is a better prior than nothing.
    It is consulted only for teams the current season cannot answer for.

    The keyword knobs are judged by `season.goalie_gate`; their defaults are
    the module constants above, and the spec "old" is the model first measured:

    - `current_club`: a goalie counts only for the club he last dressed for.
      Starts are keyed by the club he made them for, so after a trade his old
      club's window still held him.
    - `dressed_window`: a goalie who has not dressed in any of his club's last
      that-many games is out - injured, sent down or traded - and his share
      goes to whoever is left, instead of leaving his partner under the floor.
    - `ahead`: how a game after the next one is forecast (`AHEAD_RULES`).
    - `unavailable`: goalies known to be out from another source, such as an
      injury tag on a roster read this run.
    - `healthy`: goalies known to be fit, exempt from `dressed_window`. Two
      games without dressing cannot tell an injury from a return: across all
      clubs the rule zeroes the night's actual starter 22-29 times a season,
      10-12 of them the favourite back from an absence (it turns 61-71 misses
      into hits). A clean tag on a roster we read says which it is.

    Sharing out what the `MIN_MEANINGFUL_P` cut removes was measured too, and
    changed nothing: with ten games and a 0.5 demotion the smallest share the
    model can give is 0.053, so the cut never fires on the next game.
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        season: str,
        fallback_season: str | None = None,
        trailing_games: int = TRAILING_TEAM_GAMES,
        previous_damping: float = PREVIOUS_STARTER_DAMPING,
        *,
        current_club: bool = CURRENT_CLUB,
        dressed_window: int | None = DRESSED_WINDOW,
        ahead: str = AHEAD,
        unavailable: Iterable[int] = (),
        healthy: Iterable[int] = (),
    ):
        if ahead not in AHEAD_RULES:
            raise ValueError(f"ahead must be one of {AHEAD_RULES}, not {ahead!r}")
        self.season = season
        self.trailing_games = trailing_games
        self.previous_damping = previous_damping
        self.current_club = current_club
        self.dressed_window = dressed_window
        self.ahead = ahead
        self.unavailable = frozenset(unavailable)
        self.healthy = frozenset(healthy)
        self._starts = _season_starts(conn, season)
        self._team_dates = _team_dates(self._starts)
        self._fallback = _season_starts(conn, fallback_season) if fallback_season else {}
        self._fallback_dates = _team_dates(self._fallback)
        self._schedule = _scheduled_teams(conn, season)
        self._games = _team_schedule(self._schedule)
        self._dressed = _dressed(conn, season) if (current_club or dressed_window) else {}
        self._dressed_dates = _team_dates(self._dressed)
        self._clubs = _clubs(self._dressed)
        self._cache: dict[str, dict[int, float]] = {}

    def starts(self, date: str) -> dict[int, float]:
        if date in self._cache:
            return self._cache[date]
        out: dict[int, float] = {}
        # Sorted: a set's order changes with the interpreter's hash seed, and a
        # goalie listed by two clubs on one date took whichever came last.
        for team in sorted(self._schedule.get(date, ())):
            out.update(self.team_starts(team, date))
        self._cache[date] = out
        return out

    def team_starts(self, team: str, date: str, cutoff: str | None = None) -> dict[int, float]:
        """P(start) for `team`'s game on `date`, from games played before `cutoff`.

        `cutoff` defaults to `date` itself: everything played before the game.
        """
        cutoff = date if cutoff is None else cutoff
        history = [d for d in self._team_dates.get(team, ()) if d < cutoff]
        starts = self._starts
        if not history and self._fallback_dates.get(team):
            history = list(self._fallback_dates[team])
            starts = self._fallback
        if not history:
            return {}
        previous = starts.get((history[-1], team))

        counts: dict[int, int] = {}
        for day in history[-self.trailing_games :]:
            pid = starts.get((day, team))
            if pid and self._available(pid, team, cutoff):
                counts[pid] = counts.get(pid, 0) + 1
        if not counts:
            return {}

        p = self._next(counts, previous)
        games = self._games_between(team, history[-1], date)
        if games > 1 and self.ahead == "share":
            p = self._next(counts, None)
        elif games > 1 and self.ahead == "chain":
            for _ in range(games - 1):
                after: dict[int, float] = {}
                for prev, q in p.items():
                    for pid, r in self._next(counts, prev).items():
                        after[pid] = after.get(pid, 0.0) + q * r
                p = after

        return {pid: round(v, 4) for pid, v in p.items() if v >= MIN_MEANINGFUL_P}

    def _team_probabilities(self, team: str, date: str) -> dict[int, float]:
        """The next game as seen from `date`; the name older callers use."""
        return self.team_starts(team, date)

    def _next(self, counts: dict[int, int], previous: int | None) -> dict[int, float]:
        """The game after one started by `previous`: the share, him demoted."""
        weights = {
            pid: n * (self.previous_damping if pid == previous else 1.0)
            for pid, n in counts.items()
        }
        total = sum(weights.values())
        if total <= 0:
            return {}
        return {pid: w / total for pid, w in weights.items()}

    def _games_between(self, team: str, last: str, date: str) -> int:
        """Which game from `last` the one on `date` is: 1 for the very next."""
        games = self._games.get(team, [])
        return bisect.bisect_right(games, date) - bisect.bisect_right(games, last)

    def _available(self, pid: int, team: str, cutoff: str) -> bool:
        if pid in self.unavailable:
            return False
        if self.current_club:
            club = _club_as_of(self._clubs, pid, cutoff)
            if club is not None and club != team:
                return False
        if self.dressed_window and pid not in self.healthy:
            dates = self._dressed_dates.get(team, [])
            i = bisect.bisect_left(dates, cutoff)
            recent = dates[max(0, i - self.dressed_window) : i]
            if recent and not any(pid in self._dressed[(d, team)] for d in recent):
                return False
        return True


def parse_spec(spec: str) -> dict:
    """'club-dw2-chain' -> `TrailingStartShareSource` keyword arguments.

    So a gate can be told which model to run from the command line. Tokens:
    club (current club only), dw<n> (out after n games not dressed),
    damped/share/chain (games after the next), t<n> (trailing team games),
    x<f> (previous-starter damping). 'old' is the model as first measured;
    '' leaves every knob at its default.
    """
    kw: dict = {}
    for tok in spec.split("-") if spec else ():
        if tok == "old":
            kw.update(current_club=False, dressed_window=None, ahead="damped")
        elif tok == "club":
            kw["current_club"] = True
        elif tok.startswith("dw") and tok[2:].isdigit():
            kw["dressed_window"] = int(tok[2:])
        elif tok in AHEAD_RULES:
            kw["ahead"] = tok
        elif tok.startswith("t") and tok[1:].isdigit():
            kw["trailing_games"] = int(tok[1:])
        elif tok.startswith("x"):
            try:
                kw["previous_damping"] = float(tok[1:])
            except ValueError:
                raise ValueError(f"unknown goalie model token {tok!r} in {spec!r}") from None
        else:
            raise ValueError(f"unknown goalie model token {tok!r} in {spec!r}")
    return kw


def trailing_model(
    conn: sqlite3.Connection,
    season: str,
    fallback_season: str | None = None,
    spec: str = "",
    **kw,
) -> TrailingStartShareSource:
    """The trailing model as `spec` describes it; '' is the default model."""
    return TrailingStartShareSource(
        conn, season, fallback_season=fallback_season, **{**parse_spec(spec), **kw}
    )


class AsOfGoalieSource:
    """P(start) for any later date, as the trailing model saw it on `cutoff`.

    Live, a later date's history cannot include games not yet played, so the
    trailing model is as-of by construction. In a replay it is not: asked about
    Saturday on Wednesday, it would count Thursday's and Friday's starts. This
    freezes each team's history at the cutoff - the only honest forecast a
    Wednesday can make of a Saturday.
    """

    def __init__(self, source: TrailingStartShareSource, cutoff: str):
        self.source = source
        self.cutoff = cutoff

    def starts(self, date: str) -> dict[int, float]:
        out: dict[int, float] = {}
        for team in sorted(self.source._schedule.get(date, ())):
            out.update(self.source.team_starts(team, date, cutoff=self.cutoff))
        return out


class ChainedGoalieSource:
    """The first source with an opinion about a goalie wins, per date.

    Lets a confirmed announcement override the workload model for the goalies
    it covers while the model still answers for everyone else. A source that
    knows nothing today returns {} and costs nothing, and one that raises is
    skipped rather than blinding the rest - a dead feed should degrade to the
    floor, not to an empty lineup.
    """

    def __init__(self, *sources):
        self.sources = [s for s in sources if s is not None]

    def starts(self, date: str) -> dict[int, float]:
        out: dict[int, float] = {}
        for src in reversed(self.sources):  # later sources are the weaker prior
            try:
                out.update(src.starts(date))
            except Exception:  # noqa: BLE001
                continue
        return out


class StaticGoalieSource:
    """A fixed answer: for tests, and for a starter confirmed by hand."""

    def __init__(self, by_date: dict[str, dict[int, float]] | None = None):
        self.by_date = by_date or {}

    def starts(self, date: str) -> dict[int, float]:
        return dict(self.by_date.get(date, {}))


# -- the rest of the season --------------------------------------------------

# Games of the preseason projection that a goalie's share so far is shrunk
# toward: after that many club games, what he has done counts as much as what
# was expected of him. Swept by `season.goalie_gate` on remaining-share error.
WORKLOAD_PRIOR_GAMES = 10.0


class GoalieWorkload:
    """Each goalie's expected share of his club's remaining starts.

    A goalie's value over the rest of the season is his value per start times
    his starts, and his starts are his club's games times his share of them -
    not the club's games. Counting every club game overstated a starter about
    1.6 times and a backup two to three times, which made a backup look like a
    keeper on any drop or upgrade measured over the season.

    The share is what he has started for his current club since he first
    dressed for it, shrunk toward his preseason projection (`priors`: projected
    games over the season's games) by `prior_games` pseudo-games. A goalie with
    no projection gets his share so far, unshrunk; one with neither, None.
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        season: str,
        priors: dict[int, float] | None = None,
        prior_games: float = WORKLOAD_PRIOR_GAMES,
    ):
        self.priors = priors or {}
        self.prior_games = prior_games
        self._starts = _season_starts(conn, season)
        self._dressed = _dressed(conn, season)
        self._clubs = _clubs(self._dressed)
        self._club_dates = _team_dates(self._dressed)

    def share(self, pid: int, cutoff: str) -> float | None:
        """His expected share of his club's games from `cutoff` on."""
        prior = self.priors.get(pid)
        club = _club_as_of(self._clubs, pid, cutoff)
        if club is None:
            return prior
        dates, clubs = self._clubs[pid]
        j = bisect.bisect_left(dates, cutoff) - 1
        while j > 0 and clubs[j - 1] == club:
            j -= 1
        played = self._club_dates.get(club, [])
        lo, hi = bisect.bisect_left(played, dates[j]), bisect.bisect_left(played, cutoff)
        games = hi - lo
        starts = sum(1 for d in played[lo:hi] if self._starts.get((d, club)) == pid)
        if prior is None:
            return starts / games if games else None
        return (starts + self.prior_games * prior) / (games + self.prior_games)

    def shares(self, cutoff: str, pids) -> dict[int, float]:
        """share() for many goalies at once, leaving out any it cannot answer."""
        out = {}
        for pid in pids:
            got = self.share(pid, cutoff)
            if got is not None:
                out[pid] = got
        return out


def projected_shares(frame, season_games: int) -> dict[int, float]:
    """goalie -> projected games over the season's games, from a projection frame."""
    if frame is None or season_games <= 0 or "proj_gp" not in getattr(frame, "columns", ()):
        return {}
    out = {}
    for pid, row in frame.iterrows():
        if str(row.get("position", "G")) != "G":
            continue
        try:
            gp = float(row["proj_gp"])
        except (TypeError, ValueError):
            continue
        if gp == gp:  # not NaN
            out[int(pid)] = min(max(gp / season_games, 0.0), 1.0)
    return out


# -- indexing ---------------------------------------------------------------


def _season_starts(conn: sqlite3.Connection, season: str | None) -> dict[tuple[str, str], int]:
    """(date, team) -> the goalie who started. Relief appearances are not starts."""
    if not season:
        return {}
    rows = conn.execute(
        "SELECT l.game_date, l.team_abbrev, l.player_id, l.stats_json FROM nhl_game_logs l "
        "JOIN nhl_players p ON p.player_id = l.player_id "
        "WHERE l.season = ? AND p.position = 'G'",
        (season,),
    ).fetchall()
    out: dict[tuple[str, str], int] = {}
    for r in rows:
        if not r["team_abbrev"]:
            continue
        try:
            if int(json.loads(r["stats_json"]).get("gamesStarted", 0) or 0) == 1:
                out[(r["game_date"], r["team_abbrev"])] = r["player_id"]
        except (ValueError, TypeError):
            continue
    return out


def _team_dates(starts: dict[tuple[str, str], int]) -> dict[str, list[str]]:
    out: dict[str, set[str]] = {}
    for date, team in starts:
        out.setdefault(team, set()).add(date)
    return {t: sorted(d) for t, d in out.items()}


def _dressed(conn: sqlite3.Connection, season: str) -> dict[tuple[str, str], frozenset[int]]:
    """(date, team) -> the goalies who dressed, starter and backup, from boxscores.

    Game logs list only goalies who played, so a backup who sat on the bench
    is invisible there; the boxscore lists both dressed goalies every game.
    """
    rows = conn.execute(
        "SELECT s.game_date, b.team_abbrev, b.player_id FROM nhl_boxscore_stats b "
        "JOIN nhl_schedule s ON s.game_id = b.game_id "
        "WHERE b.season = ? AND json_extract(b.stats_json, '$.position') = 'G'",
        (season,),
    ).fetchall()
    out: dict[tuple[str, str], set[int]] = {}
    for date, team, pid in rows:
        if team:
            out.setdefault((date, team), set()).add(int(pid))
    return {k: frozenset(v) for k, v in out.items()}


def _clubs(dressed: dict[tuple[str, str], frozenset[int]]) -> dict[int, tuple[list, list]]:
    """goalie -> (dates, clubs) he dressed for, in date order, for bisection."""
    seen: dict[int, list[tuple[str, str]]] = {}
    for (date, team), pids in dressed.items():
        for pid in pids:
            seen.setdefault(pid, []).append((date, team))
    out = {}
    for pid, rows in seen.items():
        rows.sort()
        out[pid] = ([d for d, _ in rows], [t for _, t in rows])
    return out


def _club_as_of(clubs: dict[int, tuple[list, list]], pid: int, cutoff: str) -> str | None:
    """The club a goalie last dressed for before `cutoff`; None if he has not yet."""
    got = clubs.get(pid)
    if not got:
        return None
    i = bisect.bisect_left(got[0], cutoff)
    return got[1][i - 1] if i else None


def _team_schedule(schedule: dict[str, set[str]]) -> dict[str, list[str]]:
    """team -> every date it has a game, played or not, in order."""
    out: dict[str, list[str]] = {}
    for date in sorted(schedule):
        for team in schedule[date]:
            out.setdefault(team, []).append(date)
    return out


def _scheduled_teams(conn: sqlite3.Connection, season: str) -> dict[str, set[str]]:
    """date -> teams with a regular-season game. Includes days not yet played."""
    rows = conn.execute(
        "SELECT game_date, home_team, away_team FROM nhl_schedule "
        "WHERE season = ? AND game_type = 2",
        (season,),
    ).fetchall()
    out: dict[str, set[str]] = {}
    for r in rows:
        out.setdefault(r["game_date"], set()).update({r["home_team"], r["away_team"]})
    return out


def probable_starters(
    source, date: str, among: Iterable[int] | None = None
) -> list[tuple[int, float]]:
    """(goalie, P(start)) for a date, most likely first, optionally filtered."""
    p = source.starts(date)
    if among is not None:
        keep = set(among)
        p = {k: v for k, v in p.items() if k in keep}
    return sorted(p.items(), key=lambda kv: -kv[1])
