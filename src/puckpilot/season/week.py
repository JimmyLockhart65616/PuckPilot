"""The week ahead: which categories are in play, and what to do about it.

`engine/waivers.py` ranks an add by how much season value it adds. That is the
right question in a roto league and the wrong one here. Head to head, a week
is won category by category, and value banked in a category you were going to
win by miles is value wasted.

So the order of business is the opposite way round. First work out where this
week is actually decided - project both rosters over the week's real games and
see which categories are close. Then look for adds that move those, and only
those. A 4-game week against a 2-game week is double the counting stats, and
the schedule says which is which, so most of the edge here is arithmetic
rather than opinion.

Two deliberate limits, stated rather than papered over:

Margins are reported, not probabilities. Saying "68% to win HIT" would need a
fitted variance model per category, which does not exist yet; saying "you
project +14 hits, which is close" is true and is enough to decide an add.

A transaction is never executed from here. This produces proposals, and
`waiver_proposals` is the only route to acting on one.
"""

from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass, field, replace
from statistics import NormalDist

from puckpilot.engine.categories import CATALOG, Category
from puckpilot.engine.lineup import optimize_lineup
from puckpilot.league import LeagueConfig
from puckpilot.season import calendar
from puckpilot.season.add_story import SECTIONS, SeasonLeft, games_left, share_of
from puckpilot.season.pool import PoolPlayer
from puckpilot.season.roster import RosterPlayer, TeamRoster
from puckpilot.season.settings import LeagueRuntime, Week

# A category inside this share of the combined projected total is close enough
# that a single add can plausibly swing it. Judgement, not a fit - and the
# reason the output says "close", never "68%".
CLOSE_BAND = 0.06

# Rate categories are not summed; they are derived from their components.
DERIVED = {"save_pct": ("saves", "shots_against"), "gaa": ("goals_against", "toi_hours")}

# A contribution smaller than this share of the gap is not why you would make
# the move, so listing it makes every candidate look identical.
MATERIAL = 0.25

# A gap smaller than this is a coin flip; treat it as this wide so a near-tie
# does not divide by nothing.
MIN_GAP = 0.5

# Stances by odds rather than by share of the total: "likely" from 85%, "long
# shot" below 15%. The relative band compared margins across categories whose
# weekly noise differs tenfold - it called a +12% lead in wins "safe" when two
# goalie starts either way make it a coin flip. These are labels, not claims:
# no percentage is shown until the odds are calibrated against real weeks.
LIKELY = 0.85
Z_BAND = NormalDist().inv_cdf(LIKELY)

# Per-game variance as a multiple of the per-game mean, measured within player
# on 2025-26 game lines: 0.96-1.11 for goals, assists, points, PPP, shots, hits
# and blocks (Poisson), 3.84 for penalty minutes (majors and misconducts come
# in lumps). Saves and shots against run about 2. Wins and shutouts are
# yes/no per start and handled as such.
DISPERSION = {"pim": 3.84, "saves": 2.0, "shots_against": 2.0, "goals_against": 1.5}
BERNOULLI = {"wins", "shutouts"}
# Save percentage varies more than shots alone explain - shot quality, and who
# is in net - so its binomial spread is widened by this much.
SV_PCT_INFLATION = 1.3


@dataclass(frozen=True)
class CategoryOutlook:
    category: Category
    ours: float
    theirs: float
    # How much the lineup and the best available add could each still move this
    # category. None means it was never measured, which is different from zero -
    # zero is the common and informative answer.
    lineup_room: float | None = None
    add_room: float | None = None
    # Spread of the final margin from what is still to play; None when not
    # modelled. It shrinks to nothing as the week is banked.
    sd: float | None = None
    # Yahoo's own totals so far, when the week has started.
    banked_ours: float | None = None
    banked_theirs: float | None = None
    # Calibrated odds (season/odds.py, gate G1), when computed.
    p_win: float | None = None
    p_tie: float | None = None

    @property
    def expected(self) -> float | None:
        """This category's expected score: P(win) + half P(tie)."""
        if self.p_win is None:
            return None
        return self.p_win + 0.5 * (self.p_tie or 0.0)

    @property
    def room(self) -> float:
        return (self.lineup_room or 0.0) + (self.add_room or 0.0)

    @property
    def measured(self) -> bool:
        return self.lineup_room is not None

    @property
    def edge(self) -> float:
        """The margin in our favour: positive is winning, whichever way the
        category is scored. `margin` alone reads a GAA lead backwards."""
        return self.margin if self.category.higher_is_better else -self.margin

    @property
    def z(self) -> float | None:
        if self.sd is None:
            return None
        if self.sd <= 1e-9:
            return float("inf") if self.edge > 0 else float("-inf") if self.edge < 0 else 0.0
        return self.edge / self.sd

    @property
    def band(self) -> str:
        """likely / in play / long shot, by the odds; by the relative margin
        only where no spread was modelled."""
        e = self.expected
        if e is not None:
            if e >= LIKELY:
                return "likely"
            if e <= 1.0 - LIKELY:
                return "long shot"
            return "in play"
        z = self.z
        if z is None:
            if abs(self.relative) <= CLOSE_BAND:
                return "in play"
            return "likely" if self.edge > 0 else "long shot"
        if z >= Z_BAND:
            return "likely"
        if z <= -Z_BAND:
            return "long shot"
        return "in play"

    @property
    def reachable(self) -> bool:
        """Whether any lever left could still close this gap.

        A category we lead is trivially reachable. One we trail is reachable
        only if re-slotting plus the best add available could make it more
        than a long shot - with no spread modelled, only if they cover the gap.
        """
        if self.edge >= 0:
            return True
        if not self.measured:
            return True
        if self.sd is not None and self.sd > 1e-9:
            return (self.edge + self.room) / self.sd > -Z_BAND
        return abs(self.margin) <= self.room

    @property
    def margin(self) -> float:
        return self.ours - self.theirs

    @property
    def relative(self) -> float:
        total = abs(self.ours) + abs(self.theirs)
        return self.edge / total if total else 0.0

    @property
    def verdict(self) -> str:
        if self.band == "in play":
            return "close"
        return "ahead" if self.edge > 0 else "behind"

    @property
    def in_play(self) -> bool:
        return self.band == "in play"


@dataclass(frozen=True)
class AddTarget:
    """One swap, priced by what it actually changes over the week."""

    player: PoolPlayer
    starts: float
    drop_starts: float
    deltas: dict[str, float]
    helps: tuple[str, ...]
    drop: RosterPlayer | None
    score: float
    timing: str

    @property
    def extra_starts(self) -> float:
        """The only games number that matters: his starts minus the ones lost."""
        return self.starts - self.drop_starts

    labels: dict[str, str] = field(default_factory=dict)
    # Change in expected categories won this week, when priced by the odds.
    gain: float | None = None
    # The reasons, section -> lines (season/add_story.py).
    detail: dict = field(default_factory=dict)

    def moved(self, limit: int = 4) -> str:
        """The biggest category changes, in the league's own units and labels."""
        best = sorted(self.deltas.items(), key=lambda kv: -abs(kv[1]))[:limit]
        return "  ".join(f"{self.labels.get(k, k)} {v:+.1f}" for k, v in best if abs(v) >= 0.05)


@dataclass(frozen=True)
class WeekPlan:
    week: int
    start: str
    end: str
    opponent: str
    outlook: tuple[CategoryOutlook, ...] = ()
    targets: tuple[AddTarget, ...] = ()
    our_games: int = 0
    their_games: int = 0
    adds_used_week: int = 0
    adds_left_week: int | None = None
    adds_left_season: int | None = None
    notes: tuple[str, ...] = field(default=())
    # Where the week stands: Yahoo's status, the days still to play, and
    # whether the totals include Yahoo's banked results.
    status: str = ""
    days_left: int = 0
    banked: bool = False
    odds: object | None = None  # season.odds.WeekOdds
    # Proposed swaps a re-check priced below the floor (build_week_plan `reprice`).
    lapsed: tuple[AddTarget, ...] = ()

    @property
    def expected(self) -> float | None:
        """Expected categories won this week, when the odds were computed."""
        return self.odds.expected if self.odds is not None else None

    def close(self) -> tuple[CategoryOutlook, ...]:
        return tuple(o for o in self.outlook if o.in_play)

    def text(self) -> str:
        left = f" over {self.days_left} day(s)" if self.banked else ""
        lines = [
            f"Week {self.week}  {self.start} -> {self.end}   vs {self.opponent or '?'}",
            f"  starts left{left}: you {self.our_games}, them {self.their_games}",
            "",
        ]
        if self.banked:
            lines.append(f"  {'cat':5}{'now':>14}{'projected':>18}{'margin':>10}   where it stands")
        else:
            lines.append(f"  {'cat':5}{'you':>10}{'them':>10}{'margin':>10}   where it stands")
        for o in self.outlook:
            fmt = ".3f" if o.category.key in DERIVED else ".1f"
            if self.banked:
                now = (
                    f"{o.banked_ours:{fmt}}-{o.banked_theirs:{fmt}}"
                    if o.banked_ours is not None and o.banked_theirs is not None
                    else "-"
                )
                final = f"{o.ours:{fmt}}-{o.theirs:{fmt}}"
                lines.append(
                    f"  {o.category.label:5}{now:>14}{final:>18}{o.margin:>+10{fmt}}   {o.band}"
                )
            else:
                lines.append(
                    f"  {o.category.label:5}{o.ours:>10{fmt}}{o.theirs:>10{fmt}}"
                    f"{o.margin:>+10{fmt}}   {o.band}"
                )
        close = self.close()
        lines.append("")
        lines.append(
            "  In play: " + (", ".join(o.category.label for o in close) if close else "none")
        )
        counted = [o for o in self.outlook if o.measured]
        if counted and all(o.lineup_room == 0.0 for o in counted):
            lines.append(
                "  The lineup has no discretion this week - on every day, everyone "
                "with a game fits in a slot. Only an add moves anything."
            )
        gone = tuple(o for o in self.outlook if not o.reachable)
        if gone:
            lines.append(
                "  Out of reach even with an add: "
                + ", ".join(f"{o.category.label} ({o.margin:+.1f})" for o in gone)
            )
        if self.targets:
            lines.append("")
            budget = (
                f"{self.adds_left_week} left this week, {self.adds_left_season} this season"
                if self.adds_left_week is not None
                else "no cap"
            )
            lines.append(f"  Adds worth proposing ({budget}):")
            lines.append(
                "  Every number below is the NET change to your week - the week "
                "re-slotted with the swap made, minus the week as it stands."
            )
            for t in self.targets:
                lines.append("")
                lines.append(
                    f"    {t.player.name} ({t.player.team} "
                    f"{'/'.join(sorted(t.player.eligible))})"
                    + (f" for {t.drop.name}" if t.drop else "")
                )
                lines.append(
                    f"      starts {t.starts:g} this week"
                    + (
                        f", {t.drop.name} would have started {t.drop_starts:g}"
                        f" - net {t.extra_starts:+g}"
                        if t.drop
                        else ""
                    )
                )
                moved = t.moved()
                if moved:
                    lines.append(f"      net change: {moved}")
                if t.helps:
                    lines.append(f"      closes: {', '.join(t.helps)}")
                lines.append(f"      {t.timing}")
                for key, title in SECTIONS:
                    if t.detail.get(key):
                        lines.append(f"      {title}:")
                        lines.extend(f"        {line}" for line in t.detail[key])
        for n in self.notes:
            lines.append(f"  ! {n}")
        return "\n".join(lines)


# -- projection -------------------------------------------------------------


def per_game_rates(frame, cats: tuple[Category, ...]) -> dict[int, dict[str, float]]:
    """player -> per-game rate for each category the league scores.

    Rate categories are carried as their components (saves and shots against,
    not save percentage) because a week's save percentage is the ratio of the
    totals, not the average of the rates.
    """
    needed: set[str] = set()
    for c in cats:
        needed.update(DERIVED.get(c.key, (c.key,)))
    out: dict[int, dict[str, float]] = {}
    for pid, row in frame.iterrows():
        gp = max(_f(row.get("proj_gp")), 1.0)
        out[int(pid)] = {k: _f(row.get(k)) / gp for k in needed if k in frame.columns}
    return out


def _f(v) -> float:
    """A projection cell as a number, treating absent as zero.

    `float(x or 0.0)` is not enough: a goalie has no goals column and a skater
    no saves column, both arrive as NaN, and NaN is truthy - so the idiom
    passes NaN straight through and one missing cell turns a whole category
    total into NaN. Measured the hard way on a real weekly plan, where all
    twelve categories came back NaN.
    """
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if f != f else f


def project_totals(
    games: dict[int, float],
    rates: dict[int, dict[str, float]],
    cats: tuple[Category, ...],
    base: dict[str, float] | None = None,
) -> dict[str, float]:
    """Category totals for a week, from per-player expected games.

    `base` is what is already banked, as components - saves and shots against
    rather than save percentage - so a rate comes out as the ratio of the whole
    week's totals, banked and projected together.
    """
    acc: dict[str, float] = dict(base or {})
    for pid, n in games.items():
        r = rates.get(pid)
        if not r or n <= 0:
            continue
        for k, v in r.items():
            acc[k] = acc.get(k, 0.0) + v * n
    out: dict[str, float] = {}
    for c in cats:
        if c.key in DERIVED:
            num, den = DERIVED[c.key]
            d = acc.get(den, 0.0)
            out[c.key] = (acc.get(num, 0.0) / d) if d else 0.0
        else:
            out[c.key] = acc.get(c.key, 0.0)
    return out


def remaining_variance(
    games: dict[int, float],
    rates: dict[int, dict[str, float]],
    cats: tuple[Category, ...],
    totals: dict[str, float],
    base: dict[str, float] | None = None,
) -> dict[str, float]:
    """Variance of each category's total from the games still to play.

    Banked results are fixed, so only the remainder is uncertain - which is why
    the spread, and with it every "in play" call, narrows as the week goes on.
    Rates use the delta method on the final ratio: the binomial spread of the
    remaining saves, over the week's shots squared.
    """
    var: dict[str, float] = {}
    for pid, n in games.items():
        r = rates.get(pid)
        if not r or n <= 0:
            continue
        for k, rate in r.items():
            if rate <= 0:
                continue
            if k in BERNOULLI:
                per = min(rate, 1.0) * (1.0 - min(rate, 1.0))
            else:
                per = DISPERSION.get(k, 1.0) * rate
            var[k] = var.get(k, 0.0) + per * n
    out: dict[str, float] = {}
    acc_rem: dict[str, float] = {}
    for pid, n in games.items():
        for k, rate in (rates.get(pid) or {}).items():
            acc_rem[k] = acc_rem.get(k, 0.0) + rate * max(n, 0.0)
    for c in cats:
        if c.key == "save_pct":
            shots = (base or {}).get("shots_against", 0.0) + acc_rem.get("shots_against", 0.0)
            p = totals.get("save_pct", 0.0)
            rem = acc_rem.get("shots_against", 0.0)
            out[c.key] = (
                (SV_PCT_INFLATION**2) * p * (1.0 - p) * rem / shots**2 if shots > 0 else 0.0
            )
        elif c.key in DERIVED:
            out[c.key] = 0.0  # unmodelled; its stance falls back to the relative band
        else:
            out[c.key] = var.get(c.key, 0.0)
    return out


# Yahoo labels that are not categories here but are components of one.
_COMPONENT_LABELS = {"GA": "goals_against"}


def banked_components(by_label: dict[str, float]) -> dict[str, float]:
    """Yahoo's week-so-far totals, as the component keys projections use.

    Rates themselves are dropped - a banked save percentage is recomputed from
    banked saves and shots against - and GAA's hours are backed out of GA and
    GAA, since Yahoo reports no time on ice.
    """
    by_key = {c.label.casefold(): c for c in CATALOG.values()}
    out: dict[str, float] = {}
    for label, v in by_label.items():
        if label in _COMPONENT_LABELS:
            out[_COMPONENT_LABELS[label]] = v
            continue
        c = by_key.get(label.casefold())
        if c is None or c.key in DERIVED:
            continue
        out[c.key] = v
    gaa = next((v for k, v in by_label.items() if k.casefold() == "gaa"), None)
    if gaa and "goals_against" in out and gaa > 0:
        out["toi_hours"] = out["goals_against"] / gaa
    return out


def expected_starts(
    conn: sqlite3.Connection,
    runtime: LeagueRuntime,
    week: Week,
    players,
    goalie_source,
    values,
    weights: dict[str, float] | None = None,
    days: list[str] | None = None,
    exclude: dict[str, set[str]] | None = None,
    goalie_games: dict[int, list[float]] | None = None,
) -> dict[int, float]:
    """Expected *starts* for each player over the week, not team games.

    `days` narrows it to part of the week - the days still to play, once some
    have been banked. Default: all of it. `exclude` removes clubs from a day:
    games already under way, whose stats are in Yahoo's banked totals and
    must not be projected a second time.

    The distinction matters. A roster carries more players than it can start -
    thirteen slots against sixteen or seventeen bodies - so counting every
    rostered player's team games inflates both sides of the comparison, and
    inflates them unevenly: the team with the deeper bench gains the most from
    a number it can never actually collect.

    So the week is walked a day at a time and the same assignment the daily
    plan uses decides who would actually be in a slot. Goalies are counted by
    probability rather than by whether they were slotted, because a start is
    not ours to choose - but only as many as there are G slots, taken in the
    order the lineup would start them. Summing every goalie's P(start) counted a
    third goalie's start on nights when he could only have sat.
    """
    season = runtime.nhl_season
    shape = runtime.shape()
    g_slots = sum(n for pos, n in shape.slots if pos == "G")
    out: dict[int, float] = {}

    for day in week.dates() if days is None else days:
        playing = calendar.teams_playing(conn, day, season) - (exclude or {}).get(day, set())
        p_starts = goalie_source.starts(day) if goalie_source else {}
        cands = []
        goalies = []
        for p in players:
            pid = p.nhl_player_id
            if pid is None or getattr(p, "is_out", False) or p.team not in playing:
                continue
            if getattr(p, "on_ir", False):
                continue
            v = values.per_game_tilted(pid, day, weights) if weights else values.per_game(pid, day)
            if p.position == "G":
                p_start = float(p_starts.get(pid, 0.0))
                goalies.append((p_start * v, p_start, pid))
                continue
            cands.append((pid, p.eligible, v))
        goalies.sort(reverse=True)
        for _, p_start, pid in goalies[:g_slots]:
            out[pid] = out.get(pid, 0.0) + p_start
            if goalie_games is not None and p_start > 0:
                goalie_games.setdefault(pid, []).append(p_start)
        for pid in optimize_lineup(cands, shape):
            out[pid] = out.get(pid, 0.0) + 1.0
    return {k: round(v, 2) for k, v in out.items()}


def team_games_in(
    conn: sqlite3.Connection, runtime: LeagueRuntime, week: Week, days: list[str] | None = None
) -> dict[str, int]:
    """team -> games in the week (or in `days` of it). The streaming lever, on its own."""
    by_team: dict[str, int] = {}
    for day in week.dates() if days is None else days:
        for t in calendar.teams_playing(conn, day, runtime.nhl_season):
            by_team[t] = by_team.get(t, 0) + 1
    return by_team


def headroom(
    conn: sqlite3.Connection,
    runtime: LeagueRuntime,
    week: Week,
    players,
    goalie_source,
    values,
    rates: dict[int, dict[str, float]],
    cats: tuple[Category, ...],
    neutral: dict[str, float],
    favour: float = 6.0,
    days: list[str] | None = None,
) -> dict[str, float]:
    """How far each category could move if the lineup chased only that one.

    This is what makes "out of reach" a fact rather than a threshold someone
    picked. A gap is only unreachable if it survives the most one-eyed lineup
    available - so each category is priced again with itself weighted heavily,
    the week re-slotted under that, and the difference against the neutral
    projection is the most a lineup decision could add.

    It deliberately measures the *lineup* lever alone. Adds can move a category
    much further, which is why a category out of reach here can still be worth
    proposing a transaction for.
    """
    out: dict[str, float | None] = {}
    for c in cats:
        if c.key in DERIVED:
            # A rate is not a total, so "how much could this move" has no
            # answer in these units. Unmeasured, which reads as "do not claim
            # it is out of reach" - the alternative concedes every rate
            # category we trail by a hair.
            out[c.key] = None
            continue
        starts = expected_starts(
            conn, runtime, week, players, goalie_source, values, weights={c.key: favour}, days=days
        )
        best = project_totals(starts, rates, cats)
        out[c.key] = max(best.get(c.key, 0.0) - neutral.get(c.key, 0.0), 0.0)
    return out


def add_headroom(
    conn: sqlite3.Connection,
    runtime: LeagueRuntime,
    week: Week,
    ours: TeamRoster,
    pool,
    rates: dict[int, dict[str, float]],
    cats: tuple[Category, ...],
    league,
    adds_left: int = 1,
    days: list[str] | None = None,
    workload=None,
) -> dict[str, float | None]:
    """The most the acquisitions left this week could add to each category.

    The lineup is often the smaller lever - a roster of seventeen into thirteen
    slots has no choice to make on a night when nine players have games - so a
    category out of reach for the lineup can be well within range of an add.

    Every part of that sentence is load-bearing. Net of the drop, because the
    player who makes room takes his own production with him. And over the
    acquisitions actually remaining rather than one, because this league allows
    three a week: costing it at one add called a 1.3-goal gap unreachable in
    week 1, which it plainly is not.

    Best-n against worst-n rather than the best add times n, since you cannot
    sign the same player three times and each further move displaces a better
    player than the last. An open roster spot is an add with nothing given up.
    """
    from puckpilot.season.today import ir_changes, open_roster_spots

    games = team_games_in(conn, runtime, week, days)
    if workload is not None:
        goalies = {
            p.nhl_player_id
            for p in (*ours.players, *pool)
            if p.position == "G" and p.nhl_player_id is not None
        }
        games = SeasonLeft(games, workload.shares((days or week.dates())[0], sorted(goalies)))
    to_ir = {m.player.player_key for m in ir_changes(runtime, ours)[0] if m.is_ir}
    droppable = [
        p
        for p in ours.players
        if not p.is_undroppable
        and p.nhl_player_id is not None
        and not p.on_ir
        and p.player_key not in to_ir
        and not getattr(p, "keeper_protected", False)
    ]
    free = open_roster_spots(runtime, ours)
    n = max(int(adds_left), 0)
    out: dict[str, float | None] = {}
    for c in cats:
        if c.key in DERIVED:
            out[c.key] = None
            continue
        if n == 0:
            out[c.key] = 0.0
            continue

        def week_total(p, key=c.key):
            return rates.get(p.nhl_player_id, {}).get(key, 0.0) * games_left(games, p)

        give_up = sorted(week_total(p) for p in droppable)[: max(n - free, 0)]
        gain = sorted(
            (week_total(p) for p in pool if p.nhl_player_id is not None and not p.is_out),
            reverse=True,
        )[:n]
        out[c.key] = max(sum(gain) - sum(give_up), 0.0)
    return out


# -- the plan ---------------------------------------------------------------


def build_week_plan(
    conn: sqlite3.Connection,
    runtime: LeagueRuntime,
    league: LeagueConfig,
    week: Week,
    opponent_name: str,
    ours: TeamRoster,
    theirs: TeamRoster,
    pool: list[PoolPlayer],
    frame,
    goalie_source,
    values,
    adds_used_week: int = 0,
    adds_used_season: int = 0,
    min_gain: float = 0.5,
    max_targets: int = 5,
    banked_ours: dict[str, float] | None = None,
    banked_theirs: dict[str, float] | None = None,
    from_day: str | None = None,
    started: set[str] | None = None,
    status: str = "",
    find_targets: bool = True,
    odds_model=None,
    add_scoring: str = "share",
    min_expected_gain: float = 0.1,
    playoff_reserve: int = 0,
    stream_spots: int = 2,
    measure_room: bool = True,
    reprice=None,
    horizon: str = "this week",
    workload=None,
) -> WeekPlan:
    """Both sides' week: what is banked, plus what the days left should add.

    `banked_*` is Yahoo's week-so-far, by Yahoo label (`LiveMatchup.banked`);
    `from_day` is the first day not yet played, and `started` the clubs whose
    game that day is already under way - banked, so not projected again. With
    neither, this is the whole week from its first day, as before.

    Everything that compares two projections - the lineup headroom, the add
    deltas - works on the remainder alone: a banked total on one side of a
    difference would be counted as something an add could change.

    `reprice` - (add, drop) pairs already proposed - prices exactly those
    instead of searching: `targets` is the ones that still pay, `lapsed` the
    ones that no longer do (season/run.py re-checks the queue this way).

    `workload` (`goalies.GoalieWorkload`) counts a goalie's games over the rest
    of the season as his club's times his share of its starts, rather than all
    of his club's; without it every number is as it was.
    """
    cats = league.all_cats
    rates = per_game_rates(frame, cats)
    if workload is None:
        workload = getattr(values, "workload", None)
    days = [d for d in week.dates() if from_day is None or d >= from_day]
    exclude = {days[0]: set(started)} if days and started else None
    base_ours = banked_components(banked_ours or {})
    base_theirs = banked_components(banked_theirs or {})
    banked = banked_ours is not None

    our_goalie_games: dict[int, list[float]] = {}
    their_goalie_games: dict[int, list[float]] = {}
    our_games = expected_starts(
        conn,
        runtime,
        week,
        ours.players,
        goalie_source,
        values,
        days=days,
        exclude=exclude,
        goalie_games=our_goalie_games,
    )
    their_games = expected_starts(
        conn,
        runtime,
        week,
        theirs.players,
        goalie_source,
        values,
        days=days,
        exclude=exclude,
        goalie_games=their_goalie_games,
    )
    our_rest = project_totals(our_games, rates, cats)
    our_totals = project_totals(our_games, rates, cats, base=base_ours)
    their_totals = project_totals(their_games, rates, cats, base=base_theirs)
    our_var = remaining_variance(our_games, rates, cats, our_totals, base_ours)
    their_var = remaining_variance(their_games, rates, cats, their_totals, base_theirs)
    ours_now = project_totals({}, {}, cats, base=base_ours) if banked else {}
    theirs_now = project_totals({}, {}, cats, base=base_theirs) if banked else {}

    adds_left_week = (
        None
        if runtime.max_weekly_adds is None
        else max(runtime.max_weekly_adds - adds_used_week, 0)
    )
    reach: dict[str, float | None] = {}
    adds: dict[str, float | None] = {}
    # How far the lineup and the adds left could move each category. Feeds only
    # "out of reach"; a replay pricing thousands of adds can skip it.
    if measure_room:
        reach = headroom(
            conn,
            runtime,
            week,
            ours.players,
            goalie_source,
            values,
            rates,
            cats,
            our_rest,
            days=days,
        )
        adds = add_headroom(
            conn,
            runtime,
            week,
            ours,
            pool,
            rates,
            cats,
            league,
            adds_left=adds_left_week if adds_left_week is not None else 1,
            days=days,
            workload=workload,
        )

    def sd(key: str) -> float | None:
        if key in DERIVED and key != "save_pct":
            return None
        return math.sqrt(max(our_var.get(key, 0.0) + their_var.get(key, 0.0), 0.0))

    odds = their_side = None
    if odds_model is not None:
        from puckpilot.season import odds as odds_mod

        def skaters_only(players, games):
            skate = {p.nhl_player_id for p in players if p.position != "G"}
            return {pid: n for pid, n in games.items() if pid in skate}

        their_side = odds_mod.side(
            base_theirs, skaters_only(theirs.players, their_games), their_goalie_games, rates
        )
        odds = odds_model.week(
            cats,
            odds_mod.side(
                base_ours, skaters_only(ours.players, our_games), our_goalie_games, rates
            ),
            their_side,
        )

    def chance(key: str) -> tuple[float | None, float | None]:
        o = odds.of(key) if odds is not None else None
        return (o.p_win, o.p_tie) if o is not None else (None, None)

    outlook = tuple(
        CategoryOutlook(
            category=c,
            ours=our_totals.get(c.key, 0.0),
            theirs=their_totals.get(c.key, 0.0),
            lineup_room=reach.get(c.key),
            add_room=adds.get(c.key),
            sd=sd(c.key),
            banked_ours=ours_now.get(c.key) if banked else None,
            banked_theirs=theirs_now.get(c.key) if banked else None,
            p_win=chance(c.key)[0],
            p_tie=chance(c.key)[1],
        )
        for c in cats
    )
    close_by_key = {o.category.key: o for o in outlook if o.in_play}

    notes: list[str] = []
    adds_left_season = (
        None if runtime.max_adds is None else max(runtime.max_adds - adds_used_season, 0)
    )
    if adds_left_week == 0:
        notes.append("No acquisitions left this week - these are for next week.")
    if not days:
        notes.append("The week is over - nothing left to play.")

    targets: tuple[AddTarget, ...] = ()
    # The season's acquisitions are finite and the playoffs need some: past
    # the reserve, the regular season stops proposing.
    holding = (
        adds_left_season is not None
        and week.number < runtime.playoff_start_week
        and adds_left_season <= playoff_reserve
    )
    if holding and find_targets:
        notes.append(
            f"Holding the last {adds_left_season} acquisition(s) for the playoffs - "
            f"nothing proposed."
        )
    ctx = None
    floor = min_gain
    if add_scoring == "odds" and odds is not None and their_side is not None:
        ctx = OddsContext(model=odds_model, banked=base_ours, theirs=their_side, cats=cats)
        floor = min_expected_gain
    lapsed: tuple[AddTarget, ...] = ()
    if reprice is not None and days:
        targets, lapsed = _reprice(
            conn,
            runtime,
            league,
            week,
            ours,
            list(reprice),
            rates,
            close_by_key,
            goalie_source,
            values,
            floor,
            base_totals=our_rest,
            days=days,
            odds_ctx=ctx,
            exclude=exclude,
            horizon=horizon,
            workload=workload,
        )
    elif find_targets and days and not holding:
        targets = _targets(
            conn,
            runtime,
            league,
            week,
            ours,
            pool,
            rates,
            close_by_key,
            goalie_source,
            values,
            floor,
            max_targets,
            base_totals=our_rest,
            days=days,
            odds_ctx=ctx,
            adds_left=_fewest(
                adds_left_week,
                None if adds_left_season is None else adds_left_season - playoff_reserve,
            ),
            stream_spots=stream_spots,
            exclude=exclude,
            horizon=horizon,
            workload=workload,
        )

    return WeekPlan(
        week=week.number,
        start=week.start,
        end=week.end,
        opponent=opponent_name,
        outlook=outlook,
        targets=targets,
        our_games=int(sum(our_games.values())),
        their_games=int(sum(their_games.values())),
        status=status,
        days_left=len(days),
        banked=banked,
        odds=odds,
        adds_used_week=adds_used_week,
        adds_left_week=adds_left_week,
        adds_left_season=adds_left_season,
        notes=tuple(notes),
        lapsed=lapsed,
    )


@dataclass
class OddsContext:
    """What pricing an add by expected categories needs besides the roster."""

    model: object  # season.odds.OddsModel
    banked: dict[str, float]  # our banked components
    theirs: object  # season.odds.Side, fixed for the week
    cats: tuple[Category, ...]


def _targets(
    conn,
    runtime,
    league,
    week,
    ours,
    pool,
    rates,
    close_by_key,
    goalie_source,
    values,
    min_gain,
    max_targets,
    base_totals=None,
    screen: int = 20,
    days: list[str] | None = None,
    odds_ctx: OddsContext | None = None,
    adds_left: int | None = None,
    stream_spots: int = 2,
    exclude: dict[str, set[str]] | None = None,
    horizon: str = "this week",
    workload=None,
) -> tuple[AddTarget, ...]:
    """Adds that move a category in play, priced by re-slotting the actual week.

    The first version of this assumed an added player starts every one of his
    team's games and the dropped one would have too. Neither is true. On a
    night when your lineup is already full he displaces somebody, so only the
    difference counts; on a night with an empty slot he is worth the whole
    thing. The only honest number is the delta, and the only way to get it is
    to slot the week both ways and subtract.

    Priced one of two ways: with `odds_ctx`, by the change in expected
    categories won this week; without, by the share of each live gap closed.

    The result is a set that can all be made, chosen greedily: the best add,
    then the best given that one, never reusing a drop or an open spot. Priced
    independently, every candidate named the same cheapest drop, and approving
    one made the rest impossible.

    Only the `stream_spots` players worth least over the rest of the season
    may be dropped for a streamer - a manager rotates a spot or two at the
    bottom, never the core - unless the player coming in is worth more over
    the rest of the season anyway, which is not streaming but an upgrade.
    Without that, the greedy second add reached for the next-cheapest player,
    and on a real roster that is soon a regular.

    Each step costs an optimizer pass per candidate, so the pool is screened
    cheaply first and only the shortlist is priced properly.
    """
    from puckpilot.season.today import ir_changes, open_roster_spots

    cats = league.all_cats
    days = days if days is not None else week.dates()
    # Going to IR is not the same as being worth dropping: an injured regular
    # is put on IR, which frees his spot, rather than cut for a streamer.
    to_ir = {m.player.player_key for m in ir_changes(runtime, ours)[0] if m.is_ir}
    # An injured or not-active player is never proposed as a drop: whether
    # "NA" or "O" is a night or a season is a person's call, not a model's.
    droppable = [
        p
        for p in ours.players
        if not p.is_undroppable
        and p.nhl_player_id is not None
        and not p.on_ir
        and not p.is_out
        and p.player_key not in to_ir
        # Next season's best keepers are not this week's to give away.
        and not getattr(p, "keeper_protected", False)
    ]
    counts: dict[str, int] = {}
    for p in ours.players:
        if not p.on_ir and p.player_key not in to_ir:
            counts[p.position] = counts.get(p.position, 0) + 1
    rules = league.draft_rules()
    spots = open_roster_spots(runtime, ours)
    evaluate = _evaluator(
        conn, runtime, week, rates, cats, goalie_source, values, days, odds_ctx, exclude
    )

    base_players = list(ours.players)
    base_starts, first_totals, base_odds = evaluate(base_players)
    if base_totals is None:
        base_totals = first_totals
    holes = open_slot_days(conn, runtime, ours.players, goalie_source, values, days)
    playing = {d: calendar.teams_playing(conn, d, runtime.nhl_season) for d in days}
    # What each player is worth to the rest of the season, not to this week: a
    # regular with one game this week is still a regular.
    season_left = _season_left(conn, runtime, days[0], [*ours.players, *pool], workload)
    keep = _season_value(droppable, values, season_left, days[0])
    cheapest = sorted(keep.values())
    stream_floor = (
        cheapest[min(stream_spots, len(cheapest)) - 1] if cheapest and stream_spots > 0 else None
    )

    def may_drop(p, cand) -> bool:
        worth = keep.get(p.player_key, 0.0)
        if stream_floor is not None and worth <= stream_floor:
            return True
        return worth <= values.per_game(cand.nhl_player_id, days[0]) * games_left(season_left, cand)

    # Cheap screen: his rate times the games he would actually fill - days his
    # team plays AND a slot he can take is empty. Ranking by team games alone
    # favoured a four-game week that lands on nights the lineup is already full.
    screened = sorted(
        (
            c
            for c in pool
            if c.nhl_player_id is not None and not c.is_out and c.nhl_player_id in rates
        ),
        key=lambda c: (
            -values.per_game(c.nhl_player_id, days[0])
            * _screen_games(c, holes, playing)
            * share_of(season_left, c)
        ),
    )[: max(screen, max_targets)]

    steps = max_targets if adds_left is None else min(max_targets, adds_left)
    chosen: list[AddTarget] = []
    used_drops: set[str] = set()
    for _ in range(max(steps, 0)):
        best: AddTarget | None = None
        best_after = None
        for cand in screened:
            if any(t.player.player_key == cand.player_key for t in chosen):
                continue
            over_cap = counts.get(cand.position, 0) + 1 > rules.caps.get(cand.position, 99)
            if spots > 0 and not over_cap:
                drop = None
            else:
                drop = _cheapest_legal_drop(
                    cand,
                    [p for p in droppable if p.player_key not in used_drops and may_drop(p, cand)],
                    keep,
                    counts,
                    rules,
                )
                if drop is None:
                    continue
            t, after_week = _price(
                runtime,
                cats,
                evaluate,
                base_players,
                base_starts,
                base_totals,
                base_odds,
                close_by_key,
                cand,
                drop,
            )
            if not _pays(t, min_gain, close_by_key):
                continue
            if best is None or t.score > best.score:
                best = t
                best_after = after_week
        if best is None:
            break
        best = replace(
            best,
            detail=_explain(
                conn,
                runtime,
                days,
                base_players,
                best,
                best_after,
                base_odds,
                goalie_source,
                values,
                rates,
                odds_ctx,
                season_left,
                chosen,
                exclude,
                horizon,
            ),
        )
        chosen.append(best)
        # Make it, and price the next one against the roster it leaves.
        if best.drop is None:
            spots -= 1
        else:
            used_drops.add(best.drop.player_key)
            counts[best.drop.position] = counts.get(best.drop.position, 0) - 1
        counts[best.player.position] = counts.get(best.player.position, 0) + 1
        base_players = [
            p for p in base_players if best.drop is None or p.player_key != best.drop.player_key
        ]
        base_players.append(best.player)
        base_starts, base_totals, base_odds = evaluate(base_players)
    return tuple(chosen)


def _reprice(
    conn,
    runtime,
    league,
    week,
    ours,
    pairs,
    rates,
    close_by_key,
    goalie_source,
    values,
    min_gain,
    base_totals=None,
    days: list[str] | None = None,
    odds_ctx: OddsContext | None = None,
    exclude: dict[str, set[str]] | None = None,
    horizon: str = "this week",
    workload=None,
) -> tuple[tuple[AddTarget, ...], tuple[AddTarget, ...]]:
    """(still pays, no longer pays) for swaps already proposed, priced as the
    search would price them today.

    In the order they were proposed, each against the roster the ones kept
    before it leave - the search's own greedy order, so a later card still
    assumes the earlier add. One that no longer pays is not made, and those
    after it are priced without it. No drop is re-chosen: the card is "add X,
    drop Y", and a different drop would be a different proposal.
    """
    cats = league.all_cats
    days = days if days is not None else week.dates()
    evaluate = _evaluator(
        conn, runtime, week, rates, cats, goalie_source, values, days, odds_ctx, exclude
    )
    base_players = list(ours.players)
    base_starts, first_totals, base_odds = evaluate(base_players)
    if base_totals is None:
        base_totals = first_totals
    season_left = _season_left(
        conn, runtime, days[0], [*ours.players, *(c for c, _ in pairs)], workload
    )
    kept: list[AddTarget] = []
    lapsed: list[AddTarget] = []
    for cand, drop in pairs:
        t, after_week = _price(
            runtime,
            cats,
            evaluate,
            base_players,
            base_starts,
            base_totals,
            base_odds,
            close_by_key,
            cand,
            drop,
        )
        if not _pays(t, min_gain, close_by_key):
            lapsed.append(t)
            continue
        t = replace(
            t,
            detail=_explain(
                conn,
                runtime,
                days,
                base_players,
                t,
                after_week,
                base_odds,
                goalie_source,
                values,
                rates,
                odds_ctx,
                season_left,
                kept,
                exclude,
                horizon,
            ),
        )
        kept.append(t)
        base_players = after_week[0]
        base_starts, base_totals, base_odds = evaluate(base_players)
    return tuple(kept), tuple(lapsed)


def _evaluator(
    conn, runtime, week, rates, cats, goalie_source, values, days, odds_ctx=None, exclude=None
):
    """players -> (expected starts, remaining totals, odds) over `days`.

    Games already under way (`exclude`) are banked, so they are no more
    available to an add than to anyone else.
    """
    from puckpilot.season import odds as odds_mod

    def evaluate(players):
        ggames: dict[int, list[float]] = {}
        starts = expected_starts(
            conn,
            runtime,
            week,
            players,
            goalie_source,
            values,
            days=days,
            exclude=exclude,
            goalie_games=ggames,
        )
        totals = project_totals(starts, rates, cats)
        odds = None
        if odds_ctx is not None:
            skate = {p.nhl_player_id for p in players if p.position != "G"}
            side = odds_mod.side(
                odds_ctx.banked,
                {pid: n for pid, n in starts.items() if pid in skate},
                ggames,
                rates,
            )
            odds = odds_ctx.model.week(odds_ctx.cats, side, odds_ctx.theirs)
        return starts, totals, odds

    return evaluate


def _price(
    runtime,
    cats,
    evaluate,
    base_players,
    base_starts,
    base_totals,
    base_odds,
    close_by_key,
    cand,
    drop,
):
    """One swap against a roster: its AddTarget, and the week it leaves."""
    after = [p for p in base_players if drop is None or p.player_key != drop.player_key]
    after.append(cand)
    after_starts, after_totals, after_odds = evaluate(after)
    deltas = {
        c.key: after_totals.get(c.key, 0.0) - base_totals.get(c.key, 0.0)
        for c in cats
        if c.key not in DERIVED
    }
    gain = None
    if after_odds is not None and base_odds is not None:
        gain = after_odds.expected - base_odds.expected
        helps = _odds_moved(base_odds, after_odds)
        score = gain
    else:
        helps = _helped_by(deltas, close_by_key)
        score = _score(deltas, close_by_key)
    t = AddTarget(
        player=cand,
        starts=after_starts.get(cand.nhl_player_id, 0.0),
        drop_starts=base_starts.get(drop.nhl_player_id, 0.0) if drop is not None else 0.0,
        deltas=deltas,
        helps=helps,
        drop=drop,
        score=score,
        labels={c.key: c.label for c in cats},
        timing=cand.timing(runtime.waiver_days),
        gain=gain,
    )
    return t, (after, after_starts, after_odds)


def _pays(t: AddTarget, min_gain: float, close_by_key) -> bool:
    """Worth proposing: over the floor, and priced by shares, moving a close category."""
    if t.gain is None and close_by_key and not t.helps:
        return False
    return t.score >= min_gain


def _explain(
    conn,
    runtime,
    days,
    base_players,
    best,
    best_after,
    base_odds,
    goalie_source,
    values,
    rates,
    odds_ctx,
    season_left,
    prior=(),
    exclude=None,
    horizon="this week",
) -> dict:
    """The chosen add's reasons, from the numbers it was priced with."""
    from puckpilot.season import add_story
    from puckpilot.season.odds import OddsModel

    after, _after_starts, after_odds = best_after
    return add_story.explain(
        conn,
        runtime,
        days,
        base_players,
        after,
        best.player,
        best.drop,
        goalie_source,
        values,
        rates,
        odds_ctx.model if odds_ctx is not None else OddsModel(),
        base_odds=base_odds,
        after_odds=after_odds,
        season_left=season_left,
        prior=prior,
        exclude=exclude,
        horizon=horizon,
    )


def _fewest(*limits: int | None) -> int | None:
    """The tightest of several caps, any of which may be absent (None)."""
    known = [max(x, 0) for x in limits if x is not None]
    return min(known) if known else None


def _odds_moved(before, after, limit: int = 3) -> tuple[str, ...]:
    """The categories an add moves most, as their chance before and after."""
    moves = []
    for b in before.cats:
        a = after.of(b.category.key)
        if a is None:
            continue
        d = a.expected - b.expected
        if d > 0.005:
            moves.append((d, f"{b.category.label} {b.expected:.0%} -> {a.expected:.0%}"))
    moves.sort(key=lambda x: -x[0])
    return tuple(label for _, label in moves[:limit])


def _season_value(droppable, values, season_left, day) -> dict[str, float]:
    """What letting each player go costs: his value over the rest of the season.

    This used to be his value over *this week* - rate times this week's games -
    so a regular whose club happened to play once was the cheapest player on
    the roster, and the add engine could propose cutting him for a four-game
    streamer.
    """
    return {
        p.player_key: values.per_game(p.nhl_player_id, day) * games_left(season_left, p)
        for p in droppable
    }


def _season_left(conn, runtime, day: str, players, workload=None) -> SeasonLeft:
    """Each club's games from `day` to the season's end, and - with a goalie
    `workload` - each of these goalies' expected share of his club's."""
    by_team = calendar.games_by_team(conn, day, runtime.end_date, runtime.nhl_season)
    if workload is None:
        return SeasonLeft(by_team)
    goalies = {p.nhl_player_id for p in players if p.position == "G" and p.nhl_player_id}
    return SeasonLeft(by_team, workload.shares(day, sorted(goalies)))


def open_slot_days(conn, runtime, players, goalie_source, values, days) -> dict[str, set[str]]:
    """day -> the engine slots nobody on this roster fills that day."""
    from puckpilot.engine.lineup import slot_instances

    shape = runtime.shape()
    out: dict[str, set[str]] = {}
    for day in days:
        playing = calendar.teams_playing(conn, day, runtime.nhl_season)
        p_starts = goalie_source.starts(day) if goalie_source else {}
        cands = []
        for p in players:
            pid = p.nhl_player_id
            if pid is None or getattr(p, "is_out", False) or getattr(p, "on_ir", False):
                continue
            if p.team not in playing:
                continue
            v = values.per_game(pid, day)
            if p.position == "G":
                v *= float(p_starts.get(pid, 0.0))
            cands.append((pid, p.eligible, v))
        filled: dict[str, int] = {}
        for slot in optimize_lineup(cands, shape).values():
            filled[slot] = filled.get(slot, 0) + 1
        empty: set[str] = set()
        for slot in slot_instances(shape):
            if filled.get(slot, 0) > 0:
                filled[slot] -= 1
            else:
                empty.add(slot)
        out[day] = empty
    return out


def _screen_games(cand, holes: dict[str, set[str]], playing: dict[str, set[str]]) -> float:
    """Games a candidate would play straight into an empty slot, for the screen.

    Plus a quarter of his other games, so a week with no holes at all still
    ranks by schedule rather than arbitrarily - on a full night he can still
    displace somebody worse, which the full re-slot then prices.
    """
    from puckpilot.engine.lineup import _slots_for

    his = _slots_for(cand.eligible)
    fills = games = 0
    for day, empty in holes.items():
        if cand.team in playing.get(day, ()):
            games += 1
            if empty & his:
                fills += 1
    return fills + 0.25 * (games - fills)


def _helped_by(deltas: dict[str, float], close: dict[str, CategoryOutlook]) -> tuple[str, ...]:
    """Close categories this swap actually moves, largest share of the gap first."""
    out: list[tuple[float, str]] = []
    for key, o in close.items():
        if key in DERIVED:
            continue
        adds = deltas.get(key, 0.0)
        if adds <= 0:
            continue
        share = adds / max(abs(o.margin), MIN_GAP)
        if share >= MATERIAL:
            pct = min(share, 1.0)
            out.append((share, f"{o.category.label} {adds:+.1f} ({pct:.0%} of the gap)"))
    out.sort(key=lambda x: -x[0])
    return tuple(label for _, label in out[:3])


def _score(deltas: dict[str, float], close: dict[str, CategoryOutlook]) -> float:
    """Rank by how much of each live gap the swap closes, not by raw value.

    A player who adds six shots into a category you trail by seven is worth
    more this week than one who adds twelve into a category already won.
    """
    if not close:
        return sum(max(v, 0.0) for v in deltas.values())
    total = 0.0
    for key, o in close.items():
        if key in DERIVED:
            continue
        total += max(deltas.get(key, 0.0), 0.0) / max(abs(o.margin), MIN_GAP)
    return total


def _categories_helped(
    rate: dict[str, float], games: float, close: dict[str, CategoryOutlook]
) -> tuple[str, ...]:
    """Close categories this player would move, and by how much.

    Not "does he register in this category at all" - nearly every forward has
    goals, penalty minutes and power-play points above zero, so that test made
    every candidate read `helps G, PIM, PPP` and told you nothing. What matters
    is the size of his week against the size of the gap.

    Rate categories are left out. A skater does not move save percentage, and a
    goalie's effect on it depends on the rest of the week's saves, which is not
    a per-player number.
    """
    out: list[tuple[float, str]] = []
    for key, o in close.items():
        if key in DERIVED:
            continue
        adds = rate.get(key, 0.0) * games
        if adds <= 0:
            continue
        # Against the gap, not in absolute terms: +2 penalty minutes into a
        # dead-level category matters more than +2 into one you lead by ten.
        # The floor keeps a near-tie from dividing by nothing, and keeps the
        # most flippable category ranked highest rather than capped.
        share = adds / max(abs(o.margin), MIN_GAP)
        if share >= MATERIAL:
            out.append((share, f"{o.category.label} +{adds:.1f}"))
    out.sort(key=lambda x: -x[0])
    return tuple(label for _, label in out[:3])


def _cheapest_legal_drop(cand, droppable, drop_values, counts, rules):
    """The least valuable player we may legally drop to make room.

    Position arithmetic only, the same check `waivers.best_move` makes: over a
    positional cap the drop must come from that position, and no drop may take
    a position below its minimum - counting the player coming in, so a centre
    for a centre is never refused for leaving the roster a centre short.
    """
    pos_add = cand.position
    over_cap = counts.get(pos_add, 0) + 1 > rules.caps.get(pos_add, 99)
    best = None
    for p in sorted(droppable, key=lambda x: drop_values.get(x.player_key, 0.0)):
        pos_drop = p.position
        if over_cap and pos_drop != pos_add:
            continue
        after = counts.get(pos_drop, 0) - 1 + (1 if pos_drop == pos_add else 0)
        if after < rules.mins.get(pos_drop, 0):
            continue
        best = p
        break
    return best
