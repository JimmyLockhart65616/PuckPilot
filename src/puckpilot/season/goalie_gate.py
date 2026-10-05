"""Gate for the starting-goalie model, scored on who actually started.

Three questions, because the lineup asks the model three different things:

- **Tonight.** Who starts the team's next game? Accuracy (the favourite
  started), Brier and log-loss over every goalie the model named plus the one
  who started. This is what decides whether a goalie is slotted.
- **The rest of the week.** The team's 2nd to 7th games from now, asked on the
  same day with nothing after it known - which is how the weekly plan, the
  odds and the add search use it. Brier per game ahead, and the starts each
  goalie was expected to make over the seven games against the starts he made.
- **The rest of the season.** Each dressed goalie's share of his club's
  remaining games, which is what a drop or an upgrade measured over the
  season needs. Error weighted by the games remaining.

The protocol is every team-date of a completed season after its first
`SKIP_DATES` game dates, with the model frozen as of that date. The model as
first measured (`old`) used 60 sampled dates instead; run with that sample it
still reproduces 61.5% / Brier 0.2174 on 2025-26 exactly, and every team-date
is simply the larger sample of the same thing.

Differences between variants are paired - the same team-dates, scored twice -
and their standard errors are clustered by club, since a club's dates are not
independent of each other.
"""

from __future__ import annotations

import bisect
import math
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, field

from puckpilot.season.goalies import (
    GoalieWorkload,
    TrailingStartShareSource,
    _team_dates,
    projected_shares,
    trailing_model,
)

SEASONS = ("20232024", "20242025", "20252026")

# The season's first game dates are skipped: with a handful of games behind
# them every model is mostly guessing, and the fallback season answers instead.
SKIP_DATES = 21

# Tonight and the team's next six games: the length of a fantasy week.
AHEAD = 7

# A start the model called impossible costs log(100), not infinity.
LOG_FLOOR = 0.01

# Goalies whose remaining share is scored: dressed in one of the club's last
# this-many games, which is the club's goaltending as it stands.
SHARE_GROUP_GAMES = 5

# Model specs, as `goalies.parse_spec` reads them.
VARIANTS = (
    "old",
    "club",
    "club-dw2",
    "club-dw2-share",
    "club-dw2-chain",
)

SHARE_PRIOR_GAMES = (5.0, 10.0, 20.0, 40.0)


# -- scoring -----------------------------------------------------------------


@dataclass
class Scores:
    """Per team-date scores for one variant on one season, in a fixed order."""

    variant: str
    season: str
    teams: list[str] = field(default_factory=list)
    hit: list[float] = field(default_factory=list)
    brier: list[float] = field(default_factory=list)
    logloss: list[float] = field(default_factory=list)
    covered: list[float] = field(default_factory=list)
    # k -> (team, Brier) for the k-th game from the cutoff, k = 2..AHEAD
    ahead: dict[int, list[tuple[str, float]]] = field(default_factory=dict)
    # starts misplaced over the next AHEAD games, scaled to AHEAD games
    week: list[tuple[str, float]] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.hit)

    def mean(self, name: str) -> float:
        xs = getattr(self, name)
        return sum(xs) / len(xs) if xs else float("nan")

    def ahead_brier(self) -> float:
        xs = [b for k in self.ahead for _, b in self.ahead[k]]
        return sum(xs) / len(xs) if xs else float("nan")

    def week_error(self) -> float:
        xs = [e for _, e in self.week]
        return sum(xs) / len(xs) if xs else float("nan")


def _brier(p: dict[int, float], actual: int) -> float:
    names = set(p) | {actual}
    return sum((p.get(g, 0.0) - (1.0 if g == actual else 0.0)) ** 2 for g in names)


def _favourite(p: dict[int, float]) -> int | None:
    if not p:
        return None
    return min(p.items(), key=lambda kv: (-kv[1], kv[0]))[0]


def _eval_dates(src: TrailingStartShareSource) -> list[str]:
    played = sorted({d for d, _ in src._starts})
    return played[SKIP_DATES:]


def score_variant(conn: sqlite3.Connection, season: str, spec: str, fallback: str | None) -> Scores:
    """Every team-date after the skip, frozen as of that date."""
    src = trailing_model(conn, season, fallback, spec)
    truth = src._starts
    out = Scores(variant=spec, season=season)
    for cutoff in _eval_dates(src):
        for team in sorted(src._schedule.get(cutoff, ())):
            actual = truth.get((cutoff, team))
            if actual is None:
                continue
            games = src._games.get(team, [])
            i = bisect.bisect_left(games, cutoff)
            week = games[i : i + AHEAD]

            p = src.team_starts(team, cutoff, cutoff=cutoff)
            out.teams.append(team)
            out.hit.append(1.0 if _favourite(p) == actual else 0.0)
            out.brier.append(_brier(p, actual))
            out.logloss.append(-math.log(max(p.get(actual, 0.0), LOG_FLOOR)))
            out.covered.append(1.0 if p else 0.0)

            expected: dict[int, float] = dict(p)
            made: dict[int, float] = {actual: 1.0}
            known = 1
            for k, day in enumerate(week[1:], start=2):
                real = truth.get((day, team))
                if real is None:
                    continue
                q = src.team_starts(team, day, cutoff=cutoff)
                out.ahead.setdefault(k, []).append((team, _brier(q, real)))
                for g, v in q.items():
                    expected[g] = expected.get(g, 0.0) + v
                made[real] = made.get(real, 0.0) + 1.0
                known += 1
            names = set(expected) | set(made)
            misplaced = sum(abs(expected.get(g, 0.0) - made.get(g, 0.0)) for g in names) / 2
            out.week.append((team, misplaced * AHEAD / known))
    return out


# -- the rest of the season ---------------------------------------------------


@dataclass
class ShareScores:
    """Remaining-share error per rule, weighted by the games remaining."""

    season: str
    rules: dict[str, list[tuple[str, float, float]]] = field(default_factory=dict)

    def add(self, rule: str, team: str, err: float, weight: float) -> None:
        self.rules.setdefault(rule, []).append((team, err, weight))

    def mae(self, rule: str) -> float:
        rows = self.rules.get(rule, [])
        w = sum(r[2] for r in rows)
        return sum(r[1] * r[2] for r in rows) / w if w else float("nan")


def score_shares(
    conn: sqlite3.Connection,
    season: str,
    priors: dict[int, float],
    prior_games: tuple[float, ...] = SHARE_PRIOR_GAMES,
) -> ShareScores:
    """Each dressed goalie's predicted share of what is left against the truth.

    Rules: `team` (every club game: the rule the season value used), `trailing`
    (his share of the last 10), `so far` (since he joined the club, unshrunk),
    `prior` (the projection alone) and `K=n` (so far, shrunk n games toward
    the projection - `GoalieWorkload`, as it runs live).

    The truth counts his starts anywhere, so a goalie traded away keeps the
    value he has: what a fantasy roster holds is the goalie, not his club.
    """
    plain = GoalieWorkload(conn, season, priors={})
    shrunk = {k: GoalieWorkload(conn, season, priors=priors, prior_games=k) for k in prior_games}
    starts = plain._starts
    dressed = plain._dressed
    club_dates = _team_dates(dressed)
    by_goalie: dict[int, list[str]] = {}
    for (d, _), g in starts.items():
        by_goalie.setdefault(g, []).append(d)
    for g in by_goalie:
        by_goalie[g].sort()

    out = ShareScores(season=season)
    played = sorted({d for d, _ in starts})
    for cutoff in played[SKIP_DATES:]:
        for team, dates in sorted(club_dates.items()):
            i = bisect.bisect_left(dates, cutoff)
            if i >= len(dates) or dates[i] != cutoff:
                continue  # score each club on its own game dates only
            left = len(dates) - i
            recent = dates[max(0, i - SHARE_GROUP_GAMES) : i]
            group = sorted({g for d in recent for g in dressed[(d, team)]})
            last10 = dates[max(0, i - 10) : i]
            for g in group:
                mine = by_goalie.get(g, [])
                actual = (len(mine) - bisect.bisect_left(mine, cutoff)) / left
                trailing = (
                    sum(1 for d in last10 if starts.get((d, team)) == g) / len(last10)
                    if last10
                    else None
                )
                so_far = plain.share(g, cutoff)
                guesses = {
                    "team": 1.0,
                    "trailing": trailing,
                    "so far": so_far,
                    "prior": priors.get(g, so_far),
                }
                for k, w in shrunk.items():
                    guesses[f"K={k:g}"] = w.share(g, cutoff)
                for rule, guess in guesses.items():
                    if guess is not None:
                        out.add(rule, team, abs(guess - actual), left)
    return out


# -- the report ----------------------------------------------------------------


def _clustered(pairs: list[tuple[str, float]]) -> tuple[float, float]:
    """Mean and club-clustered standard error of paired differences."""
    n = len(pairs)
    if n < 2:
        return float("nan"), float("nan")
    mean = sum(d for _, d in pairs) / n
    by_team: dict[str, float] = {}
    for team, d in pairs:
        by_team[team] = by_team.get(team, 0.0) + (d - mean)
    var = sum(s * s for s in by_team.values()) / (n * n)
    return mean, math.sqrt(var)


def _paired(a: Scores, b: Scores, metric: str) -> tuple[float, float]:
    """mean(a - b) for a per-team-date metric, with its clustered SE."""
    if metric in ("hit", "brier", "logloss"):
        xs, ys = getattr(a, metric), getattr(b, metric)
        return _clustered([(t, x - y) for t, x, y in zip(a.teams, xs, ys, strict=True)])
    if metric == "week":
        return _clustered([(t, x - y) for (t, x), (_, y) in zip(a.week, b.week, strict=True)])
    if metric == "ahead":
        pairs = []
        for k in a.ahead:
            pairs += [(t, x - y) for (t, x), (_, y) in zip(a.ahead[k], b.ahead[k], strict=True)]
        return _clustered(pairs)
    raise ValueError(metric)


@dataclass(frozen=True)
class GoalieGateReport:
    text: str
    scores: dict[tuple[str, str], Scores]
    shares: dict[str, ShareScores]


def goalie_gate_report(
    conn: sqlite3.Connection,
    seasons: tuple[str, ...] = SEASONS,
    variants: tuple[str, ...] = VARIANTS,
    versus: str = "old",
    shares: bool = True,
    progress: Callable[[str], None] | None = None,
) -> GoalieGateReport:
    """Every variant on every season, then each against `versus`."""
    from puckpilot.engine import projections
    from puckpilot.engine.aggregate import season_games

    say = progress or (lambda _m: None)
    if versus not in variants:
        variants = (versus, *variants)
    scores: dict[tuple[str, str], Scores] = {}
    share_scores: dict[str, ShareScores] = {}
    lines = ["Starting-goalie model, scored on who actually started", ""]
    for season in seasons:
        y = int(season[:4])
        train = [f"{y - i}{y - i + 1}" for i in range(1, 4)]
        for spec in variants:
            say(f"  {season} {spec}")
            scores[(season, spec)] = score_variant(conn, season, spec, train[0])
        if shares:
            say(f"  {season} remaining shares")
            _, proj_g = projections.project(conn, season, train)
            priors = projected_shares(proj_g, season_games(conn, season))
            share_scores[season] = score_shares(conn, season, priors)

    for season in seasons:
        base = scores[(season, versus)]
        lines += [
            f"{season}  ({base.n} team-games; games 2-{AHEAD} ahead from the same date)",
            f"  {'variant':24}{'tonight':>8}{'Brier':>8}{'logloss':>9}{'ahead':>8}{'week':>7}",
        ]
        for spec in variants:
            s = scores[(season, spec)]
            lines.append(
                f"  {spec:24}{s.mean('hit'):>8.1%}{s.mean('brier'):>8.4f}"
                f"{s.mean('logloss'):>9.4f}{s.ahead_brier():>8.4f}{s.week_error():>7.3f}"
            )
        lines.append("")
    lines += [
        "  tonight: the favourite started.  Brier/logloss: tonight's game.",
        "  ahead: Brier, games 2-7.  week: starts misplaced per 7 games.",
        "",
        f"Against {versus} (paired, club-clustered SE; + is better):",
    ]
    for spec in variants:
        if spec == versus:
            continue
        cells = []
        for season in seasons:
            a, b = scores[(season, spec)], scores[(season, versus)]
            hit, hit_se = _paired(a, b, "hit")
            br, br_se = _paired(b, a, "brier")
            ah, ah_se = _paired(b, a, "ahead")
            wk, wk_se = _paired(b, a, "week")
            cells.append(
                f"{season[2:4]}-{season[6:]}: tonight {hit:+.1%}+/-{hit_se:.1%} "
                f"Brier {br:+.4f}+/-{br_se:.4f} ahead {ah:+.4f}+/-{ah_se:.4f} "
                f"week {wk:+.3f}+/-{wk_se:.3f}"
            )
        lines.append(f"  {spec}")
        lines += [f"    {c}" for c in cells]
    if share_scores:
        rules = list(next(iter(share_scores.values())).rules)
        lines += [
            "",
            "Share of the club's remaining games (error, weighted by games left):",
            f"  {'rule':10}" + "".join(f"{s[2:4] + '-' + s[6:]:>9}" for s in seasons),
        ]
        for rule in rules:
            lines.append(
                f"  {rule:10}" + "".join(f"{share_scores[s].mae(rule):>9.3f}" for s in seasons)
            )
    return GoalieGateReport(text="\n".join(lines), scores=scores, shares=share_scores)
