"""The week ahead: which categories are in play, and what to do about it.

`engine/waivers.py` ranks an add by how much season value it adds. That is the
right question in a roto league and the wrong one here. This league is head to
head over twelve categories: a week is won by taking seven of them, and value
banked in a category you were going to win by miles is value wasted.

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

import sqlite3
from dataclasses import dataclass, field

from puckpilot.engine.categories import Category
from puckpilot.league import LeagueConfig
from puckpilot.season import calendar
from puckpilot.season.pool import PoolPlayer
from puckpilot.season.roster import RosterPlayer, TeamRoster
from puckpilot.season.settings import LeagueRuntime, Week

# A category inside this share of the combined projected total is close enough
# that a single add can plausibly swing it. Judgement, not a fit - and the
# reason the output says "close", never "68%".
CLOSE_BAND = 0.06

# Rate categories are not summed; they are derived from their components.
DERIVED = {"save_pct": ("saves", "shots_against"), "gaa": ("goals_against", "toi_hours")}


@dataclass(frozen=True)
class CategoryOutlook:
    category: Category
    ours: float
    theirs: float

    @property
    def margin(self) -> float:
        return self.ours - self.theirs

    @property
    def relative(self) -> float:
        total = abs(self.ours) + abs(self.theirs)
        return self.margin / total if total else 0.0

    @property
    def verdict(self) -> str:
        if abs(self.relative) <= CLOSE_BAND:
            return "close"
        return "ahead" if self.margin > 0 else "behind"

    @property
    def in_play(self) -> bool:
        return self.verdict == "close"


@dataclass(frozen=True)
class AddTarget:
    player: PoolPlayer
    games: int
    value: float
    helps: tuple[str, ...]
    drop: RosterPlayer | None
    drop_value: float
    timing: str

    @property
    def gain(self) -> float:
        return self.value - self.drop_value


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

    def close(self) -> tuple[CategoryOutlook, ...]:
        return tuple(o for o in self.outlook if o.in_play)

    def text(self) -> str:
        lines = [
            f"Week {self.week}  {self.start} -> {self.end}   vs {self.opponent or '?'}",
            f"  games this week: you {self.our_games}, them {self.their_games}",
            "",
            f"  {'cat':5}{'you':>10}{'them':>10}{'margin':>10}   where it stands",
        ]
        for o in self.outlook:
            fmt = ".3f" if o.category.key in DERIVED else ".1f"
            lines.append(
                f"  {o.category.label:5}{o.ours:>10{fmt}}{o.theirs:>10{fmt}}"
                f"{o.margin:>+10{fmt}}   {o.verdict}"
            )
        close = self.close()
        lines.append("")
        lines.append(
            "  In play: " + (", ".join(o.category.label for o in close) if close else "none")
        )
        if self.targets:
            lines.append("")
            budget = (
                f"{self.adds_left_week} left this week, {self.adds_left_season} this season"
                if self.adds_left_week is not None
                else "no cap"
            )
            lines.append(f"  Adds worth proposing ({budget}):")
            for t in self.targets:
                helps = ", ".join(t.helps) if t.helps else "general value"
                lines.append(
                    f"    {t.player.name:22} {t.player.team:4} "
                    f"{'/'.join(sorted(t.player.eligible)):7} "
                    f"{t.games}g  +{t.gain:.2f}  helps {helps}"
                )
                if t.drop:
                    lines.append(f"      drop {t.drop.name} ({t.drop_value:.2f})")
                lines.append(f"      {t.timing}")
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
    games: dict[int, float], rates: dict[int, dict[str, float]], cats: tuple[Category, ...]
) -> dict[str, float]:
    """Category totals for a week, from per-player expected games."""
    acc: dict[str, float] = {}
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


def expected_games(
    conn: sqlite3.Connection,
    runtime: LeagueRuntime,
    week: Week,
    players,
    goalie_source,
) -> dict[int, float]:
    """Expected games for each player over the week.

    Skaters get their team's game count. Goalies get the sum of their start
    probabilities, which is the honest version of the same thing - a backup on
    a four-game week is not playing four times.
    """
    season = runtime.nhl_season
    dates = week.dates()
    by_team: dict[str, int] = {}
    starts: dict[int, float] = {}
    for d in dates:
        playing = calendar.teams_playing(conn, d, season)
        for t in playing:
            by_team[t] = by_team.get(t, 0) + 1
        if goalie_source:
            for pid, p in goalie_source.starts(d).items():
                starts[pid] = starts.get(pid, 0.0) + p

    out: dict[int, float] = {}
    for p in players:
        pid = p.nhl_player_id
        if pid is None or getattr(p, "is_out", False):
            continue
        if p.position == "G":
            out[pid] = round(starts.get(pid, 0.0), 2)
        else:
            out[pid] = float(by_team.get(p.team, 0))
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
) -> WeekPlan:
    cats = league.all_cats
    rates = per_game_rates(frame, cats)

    our_games = expected_games(conn, runtime, week, ours.players, goalie_source)
    their_games = expected_games(conn, runtime, week, theirs.players, goalie_source)
    our_totals = project_totals(our_games, rates, cats)
    their_totals = project_totals(their_games, rates, cats)

    outlook = tuple(
        CategoryOutlook(
            category=c, ours=our_totals.get(c.key, 0.0), theirs=their_totals.get(c.key, 0.0)
        )
        for c in cats
    )
    close_by_key = {o.category.key: o for o in outlook if o.in_play}

    notes: list[str] = []
    adds_left_week = (
        None
        if runtime.max_weekly_adds is None
        else max(runtime.max_weekly_adds - adds_used_week, 0)
    )
    adds_left_season = (
        None if runtime.max_adds is None else max(runtime.max_adds - adds_used_season, 0)
    )
    if adds_left_week == 0:
        notes.append("No acquisitions left this week - these are for next week.")

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
        min_gain,
        max_targets,
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
        adds_used_week=adds_used_week,
        adds_left_week=adds_left_week,
        adds_left_season=adds_left_season,
        notes=tuple(notes),
    )


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
) -> tuple[AddTarget, ...]:
    """Adds that move a category actually in play, with a legal drop."""
    season = runtime.nhl_season
    dates = week.dates()
    games_by_team: dict[str, int] = {}
    for d in dates:
        for t in calendar.teams_playing(conn, d, season):
            games_by_team[t] = games_by_team.get(t, 0) + 1

    # What the weakest droppable roster player is worth over the same week.
    droppable = [
        p
        for p in ours.players
        if not p.is_undroppable and p.nhl_player_id is not None and not p.on_ir
    ]
    drop_values = {
        p.player_key: values.per_game(p.nhl_player_id, week.start) * games_by_team.get(p.team, 0)
        for p in droppable
    }

    counts: dict[str, int] = {}
    for p in ours.players:
        counts[p.position] = counts.get(p.position, 0) + 1
    rules = league.draft_rules()

    out: list[AddTarget] = []
    for cand in pool:
        pid = cand.nhl_player_id
        if pid is None or cand.is_out or pid not in rates:
            continue
        games = games_by_team.get(cand.team, 0)
        if cand.position == "G":
            games = round(sum(goalie_source.starts(d).get(pid, 0.0) for d in dates), 2)
        if games <= 0:
            continue
        value = values.per_game(pid, week.start) * games
        helps = _categories_helped(rates[pid], games, close_by_key)
        drop = _cheapest_legal_drop(cand, droppable, drop_values, counts, rules)
        dv = drop_values.get(drop.player_key, 0.0) if drop else 0.0
        if value - dv < min_gain:
            continue
        # A target that moves nothing in play is season value, not a week plan.
        if close_by_key and not helps:
            continue
        out.append(
            AddTarget(
                player=cand,
                games=int(games) if cand.position != "G" else games,
                value=value,
                helps=helps,
                drop=drop,
                drop_value=dv,
                timing=cand.timing(runtime.waiver_days),
            )
        )
    out.sort(key=lambda t: -t.gain)
    return tuple(out[:max_targets])


# A contribution smaller than this share of the gap is not why you would make
# the move, so listing it makes every candidate look identical.
MATERIAL = 0.25

# A gap smaller than this is a coin flip; treat it as this wide so a near-tie
# does not divide by nothing.
MIN_GAP = 0.5


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
    a position below its minimum.
    """
    pos_add = cand.position
    over_cap = counts.get(pos_add, 0) + 1 > rules.caps.get(pos_add, 99)
    best = None
    for p in sorted(droppable, key=lambda x: drop_values.get(x.player_key, 0.0)):
        pos_drop = p.position
        if over_cap and pos_drop != pos_add:
            continue
        if counts.get(pos_drop, 0) - 1 < rules.mins.get(pos_drop, 0):
            continue
        best = p
        break
    return best
