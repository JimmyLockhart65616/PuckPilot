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
from puckpilot.engine.lineup import optimize_lineup
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

# A contribution smaller than this share of the gap is not why you would make
# the move, so listing it makes every candidate look identical.
MATERIAL = 0.25

# A gap smaller than this is a coin flip; treat it as this wide so a near-tie
# does not divide by nothing.
MIN_GAP = 0.5


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

    @property
    def room(self) -> float:
        return (self.lineup_room or 0.0) + (self.add_room or 0.0)

    @property
    def measured(self) -> bool:
        return self.lineup_room is not None

    @property
    def reachable(self) -> bool:
        """Whether any lever left could still close this gap.

        A category we lead is trivially reachable. One we trail is reachable
        only if re-slotting plus the best add available covers the gap.
        """
        if self.margin >= 0:
            return True
        if not self.measured:
            return True
        return abs(self.margin) <= self.room

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


def expected_starts(
    conn: sqlite3.Connection,
    runtime: LeagueRuntime,
    week: Week,
    players,
    goalie_source,
    values,
    weights: dict[str, float] | None = None,
) -> dict[int, float]:
    """Expected *starts* for each player over the week, not team games.

    The distinction matters. A roster carries more players than it can start -
    thirteen slots against sixteen or seventeen bodies - so counting every
    rostered player's team games inflates both sides of the comparison, and
    inflates them unevenly: the team with the deeper bench gains the most from
    a number it can never actually collect.

    So the week is walked a day at a time and the same assignment the daily
    plan uses decides who would actually be in a slot. Goalies are counted by
    probability rather than by whether they were slotted, because a start is
    not ours to choose.
    """
    season = runtime.nhl_season
    shape = runtime.shape()
    out: dict[int, float] = {}

    for day in week.dates():
        playing = calendar.teams_playing(conn, day, season)
        p_starts = goalie_source.starts(day) if goalie_source else {}
        cands = []
        for p in players:
            pid = p.nhl_player_id
            if pid is None or getattr(p, "is_out", False) or p.team not in playing:
                continue
            if p.position == "G":
                # Counted straight from the probability: the slot is not the
                # constraint, the coach is.
                out[pid] = out.get(pid, 0.0) + float(p_starts.get(pid, 0.0))
                continue
            v = values.per_game_tilted(pid, day, weights) if weights else values.per_game(pid, day)
            cands.append((pid, p.eligible, v))
        for pid in optimize_lineup(cands, shape):
            out[pid] = out.get(pid, 0.0) + 1.0
    return {k: round(v, 2) for k, v in out.items()}


def team_games_in(conn: sqlite3.Connection, runtime: LeagueRuntime, week: Week) -> dict[str, int]:
    """team -> games in the week. The streaming lever, on its own."""
    by_team: dict[str, int] = {}
    for day in week.dates():
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
            conn, runtime, week, players, goalie_source, values, weights={c.key: favour}
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
    player than the last.
    """
    games = team_games_in(conn, runtime, week)
    droppable = [
        p
        for p in ours.players
        if not p.is_undroppable and p.nhl_player_id is not None and not p.on_ir
    ]
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
            return rates.get(p.nhl_player_id, {}).get(key, 0.0) * games.get(p.team, 0)

        give_up = sorted(week_total(p) for p in droppable)[:n]
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
) -> WeekPlan:
    cats = league.all_cats
    rates = per_game_rates(frame, cats)

    our_games = expected_starts(conn, runtime, week, ours.players, goalie_source, values)
    their_games = expected_starts(conn, runtime, week, theirs.players, goalie_source, values)
    our_totals = project_totals(our_games, rates, cats)
    their_totals = project_totals(their_games, rates, cats)

    adds_left_week = (
        None
        if runtime.max_weekly_adds is None
        else max(runtime.max_weekly_adds - adds_used_week, 0)
    )
    reach = headroom(
        conn, runtime, week, ours.players, goalie_source, values, rates, cats, our_totals
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
    )
    outlook = tuple(
        CategoryOutlook(
            category=c,
            ours=our_totals.get(c.key, 0.0),
            theirs=their_totals.get(c.key, 0.0),
            lineup_room=reach.get(c.key),
            add_room=adds.get(c.key),
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
        base_totals=our_totals,
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
    base_totals=None,
    screen: int = 20,
) -> tuple[AddTarget, ...]:
    """Adds that move a category in play, priced by re-slotting the actual week.

    The first version of this assumed an added player starts every one of his
    team's games and the dropped one would have too. Neither is true. On a
    night when your lineup is already full he displaces somebody, so only the
    difference counts; on a night with an empty slot he is worth the whole
    thing. The only honest number is the delta, and the only way to get it is
    to slot the week both ways and subtract.

    That costs an optimizer pass per candidate, so the pool is screened
    cheaply first and only the shortlist is priced properly.
    """
    cats = league.all_cats
    droppable = [
        p
        for p in ours.players
        if not p.is_undroppable and p.nhl_player_id is not None and not p.on_ir
    ]
    counts: dict[str, int] = {}
    for p in ours.players:
        counts[p.position] = counts.get(p.position, 0) + 1
    rules = league.draft_rules()

    base_starts = expected_starts(conn, runtime, week, ours.players, goalie_source, values)
    if base_totals is None:
        base_totals = project_totals(base_starts, rates, cats)
    games = team_games_in(conn, runtime, week)

    # Cheap screen: his team's games times his rate, which overstates everyone
    # equally and so orders them about right.
    screened = sorted(
        (
            c
            for c in pool
            if c.nhl_player_id is not None and not c.is_out and c.nhl_player_id in rates
        ),
        key=lambda c: -values.per_game(c.nhl_player_id, week.start) * games.get(c.team, 0),
    )[: max(screen, max_targets)]

    out: list[AddTarget] = []
    for cand in screened:
        rough = _rough(droppable, values, games, week)
        drop = _cheapest_legal_drop(cand, droppable, rough, counts, rules)
        if drop is None:
            continue
        after = [p for p in ours.players if p.player_key != drop.player_key] + [cand]
        after_starts = expected_starts(conn, runtime, week, after, goalie_source, values)
        after_totals = project_totals(after_starts, rates, cats)

        deltas = {
            c.key: after_totals.get(c.key, 0.0) - base_totals.get(c.key, 0.0)
            for c in cats
            if c.key not in DERIVED
        }
        helps = _helped_by(deltas, close_by_key)
        if close_by_key and not helps:
            continue

        starts = after_starts.get(cand.nhl_player_id, 0.0)
        lost = base_starts.get(drop.nhl_player_id, 0.0)
        score = _score(deltas, close_by_key)
        if score < min_gain:
            continue
        out.append(
            AddTarget(
                player=cand,
                starts=starts,
                drop_starts=lost,
                deltas=deltas,
                helps=helps,
                drop=drop,
                score=score,
                labels={c.key: c.label for c in cats},
                timing=cand.timing(runtime.waiver_days),
            )
        )
    out.sort(key=lambda t: -t.score)
    return tuple(out[:max_targets])


def _rough(droppable, values, games, week) -> dict[str, float]:
    """A quick ordering of who is cheapest to let go, for the legality check."""
    return {
        p.player_key: values.per_game(p.nhl_player_id, week.start) * games.get(p.team, 0)
        for p in droppable
    }


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
