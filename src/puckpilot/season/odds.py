"""The chance of winning each category this week, from what is banked and what is left.

The league scores head to head by category - every category its own win, loss
or tie in the standings - so the number a week is played for is the expected
count of categories taken: the sum of P(win) plus half P(tie). It is additive,
which is what makes a per-category model sufficient. Correlation between
categories (points are goals plus assists; saves and save percentage share
shots) changes how likely a clean sweep is, never the expectation.

Each side's final total is Yahoo's banked number plus a random remainder, and
the remainder is modelled as a distribution, not a margin:

    skater counts    negative binomial on the expected total - Poisson within a
                     player (variance/mean 0.96-1.11 measured on 2025-26 lines),
                     widened by a team-level factor `phi` for what summing
                     players misses: a goal is also a linemate's assist, and
                     the projected rates are themselves uncertain
    penalty minutes  compound Poisson over the measured sizes - 81% of a
                     player-game's minutes come as a single 2, the rest in lumps
    wins             a sum of yes/no chances, one per goalie game: P(start) x
                     P(win | start)
    saves            normal, with the start itself uncertain - whether he plays
                     at all is most of a goalie's variance
    save pct         normal on the final ratio (delta method), mostly the
                     binomial spread of goals on the shots still to come

Discrete where it matters: at three power-play points a side, a tie is common,
and a normal approximation cannot see it.

Nothing here is shown as a percentage until `season/calibration.py` says the
numbers mean what they say. The draft console once showed "lasts 42%" for picks
that lasted 69% of the time; a confident wrong number is worse than none.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import NormalDist

import numpy as np
from scipy import stats

from puckpilot.engine.categories import Category

# Share of a projected per-game line a started skater actually delivers.
# Fitted by gate G1 (season/calibration.py: fitted on 2024-25, judged on
# 2025-26) at 0.78 - lower than missed games alone explain, because it also
# carries the draft's selection effect: players are rostered *for* high
# projections, and on average fall short of them. Known absences (out on
# Yahoo; two straight missed games in the replay) are excluded separately.
P_PLAY = 0.78

# Team-level widening of each count's variance over the sum of its players',
# by category key, fitted by G1 (1.0 where not listed). Assists are widened
# because one goal can credit two linemates on the same fantasy roster.
PHI: dict[str, float] = {"assists": 1.35, "sog": 1.10, "blocks": 1.50, "saves": 1.75}

# Minutes in a player-game that had any, measured on 2024-25 and 2025-26 game
# logs (the two seasons agree to within a point everywhere).
PIM_SIZES: dict[int, float] = {
    2: 0.810,
    4: 0.084,
    5: 0.050,
    6: 0.009,
    7: 0.017,
    10: 0.010,
    12: 0.008,
    14: 0.003,
    15: 0.003,
    17: 0.003,
}

# Shots faced per start: variance / mean, measured (1.82, 1.88).
SHOTS_DISPERSION = 1.85

# Save percentage varies from game to game by more than shots alone explain -
# shot quality, and who is in net - so its binomial spread is widened. Fitted
# by G1 at 1.5.
SV_PCT_INFLATION = 1.5

# How close two save percentages must be for Yahoo to call it a tie. 0 until a
# live week's stat_winners shows whether it compares the 3-decimal display.
RATIO_TIE_WIDTH = 0.0

_NORMAL = NormalDist()


@dataclass(frozen=True)
class GoalieGame:
    """One scheduled game a goalie could start for us."""

    p_start: float
    p_win: float  # given he starts
    shots: float  # expected shots faced, given he starts
    save_pct: float
    p_shutout: float = 0.0
    hours: float = 1.0  # time in net per start, for GAA


@dataclass
class Side:
    """One team's week: what is banked and what the days left should add.

    `banked` and `skaters` are keyed by component (goals, saves, shots_against),
    never by rate; a rate is recomputed from its parts.
    """

    banked: dict[str, float] = field(default_factory=dict)
    skaters: dict[str, float] = field(default_factory=dict)  # expected remaining totals
    goalies: list[GoalieGame] = field(default_factory=list)


@dataclass(frozen=True)
class CategoryOdds:
    category: Category
    p_win: float
    p_tie: float
    ours: float  # expected final total, or rate
    theirs: float

    @property
    def expected(self) -> float:
        """This category's share of the week's expected categories won."""
        return self.p_win + 0.5 * self.p_tie

    @property
    def p_loss(self) -> float:
        return max(1.0 - self.p_win - self.p_tie, 0.0)


@dataclass(frozen=True)
class WeekOdds:
    cats: tuple[CategoryOdds, ...]

    @property
    def expected(self) -> float:
        """Expected categories won this week, ties counting half."""
        return sum(c.expected for c in self.cats)

    def of(self, key: str) -> CategoryOdds | None:
        return next((c for c in self.cats if c.category.key == key), None)


# -- distributions ------------------------------------------------------------


def _support(mean: float, var: float) -> int:
    return int(mean + 12.0 * math.sqrt(max(var, 1e-9)) + 12)


def count_pmf(mean: float, phi: float = 1.0) -> np.ndarray:
    """PMF of a remaining count: Poisson, or negative binomial when phi > 1."""
    if mean <= 1e-12:
        return np.array([1.0])
    var = mean * max(phi, 1.0)
    k = np.arange(_support(mean, var) + 1)
    if var <= mean * (1.0 + 1e-9):
        pmf = stats.poisson.pmf(k, mean)
    else:
        r = mean * mean / (var - mean)
        pmf = stats.nbinom.pmf(k, r, r / (r + mean))
    return pmf / pmf.sum()


def compound_pmf(mean: float, sizes: dict[int, float] | None = None) -> np.ndarray:
    """PMF of a total made of lumps: Poisson many events, each a measured size.

    Panjer's recursion, exact on the integer support.
    """
    sizes = sizes or PIM_SIZES
    norm = sum(sizes.values())
    s = {k: v / norm for k, v in sizes.items()}
    mean_size = sum(k * v for k, v in s.items())
    lam = mean / mean_size if mean_size > 0 else 0.0
    if lam <= 1e-12:
        return np.array([1.0])
    second = sum(k * k * v for k, v in s.items())
    n = _support(mean, lam * second)
    f = np.zeros(n + 1)
    f[0] = math.exp(-lam)
    for m in range(1, n + 1):
        acc = 0.0
        for k, p in s.items():
            if k <= m:
                acc += k * p * f[m - k]
        f[m] = lam / m * acc
    total = f.sum()
    return f / total if total > 0 else np.array([1.0])


def trials_pmf(ps: list[float]) -> np.ndarray:
    """PMF of how many of these independent chances come off - exactly."""
    dist = np.array([1.0])
    for p in ps:
        p = min(max(p, 0.0), 1.0)
        dist = np.convolve(dist, [1.0 - p, p])
    return dist


def discrete_odds(ours: np.ndarray, theirs: np.ndarray, lead: float) -> tuple[float, float]:
    """P(win), P(tie) when our final is `lead` + X and theirs is Y.

    `lead` is our banked total minus theirs. With integer counts and an integer
    lead, a tie is a real outcome, and it is kept.
    """
    diff = np.convolve(ours, theirs[::-1])  # P(X - Y = i - (len(theirs) - 1))
    offset = len(theirs) - 1
    d = np.arange(len(diff)) - offset  # X - Y
    total = d + lead
    p_win = float(diff[total > 1e-9].sum())
    p_tie = float(diff[np.abs(total) <= 1e-9].sum())
    return p_win, p_tie


def normal_odds(mean: float, var: float, integer: bool = False) -> tuple[float, float]:
    """P(win), P(tie) when the final margin is normal with this mean and variance."""
    sd = math.sqrt(max(var, 0.0))
    if sd <= 1e-12:
        if abs(mean) <= (0.5 if integer else 1e-12):
            return 0.0, 1.0
        return (1.0, 0.0) if mean > 0 else (0.0, 0.0)
    if integer:
        p_tie = _NORMAL.cdf((0.5 - mean) / sd) - _NORMAL.cdf((-0.5 - mean) / sd)
        p_win = 1.0 - _NORMAL.cdf((0.5 - mean) / sd)
        return p_win, max(p_tie, 0.0)
    return 1.0 - _NORMAL.cdf(-mean / sd), 0.0


# -- the goalie side ----------------------------------------------------------


def _saves_moments(games: list[GoalieGame]) -> tuple[float, float, float, float, float]:
    """(E saves, Var saves, E shots, Var shots, Cov) over goalie games still to come."""
    es = vs = ea = va = cov = 0.0
    for g in games:
        p, a, sv = g.p_start, g.shots, g.save_pct
        var_a = SHOTS_DISPERSION * a
        e_s = a * sv
        var_s = var_a * sv * sv + a * sv * (1.0 - sv)
        cov_sa = sv * var_a
        es += p * e_s
        ea += p * a
        vs += p * var_s + p * (1.0 - p) * e_s * e_s
        va += p * var_a + p * (1.0 - p) * a * a
        cov += p * cov_sa + p * (1.0 - p) * e_s * a
    return es, vs, ea, va, cov


def _save_pct(side: Side, inflation: float = SV_PCT_INFLATION) -> tuple[float, float]:
    """Mean and variance of the week's final save percentage (delta method)."""
    es, vs, ea, va, cov = _saves_moments(side.goalies)
    num = side.banked.get("saves", 0.0) + es
    den = side.banked.get("shots_against", 0.0) + ea
    if den <= 0:
        return float("nan"), 0.0
    g = num / den
    var = (vs - 2.0 * g * cov + g * g * va) / (den * den)
    return g, max(var, 0.0) * inflation**2


def _gaa(side: Side, inflation: float = SV_PCT_INFLATION) -> tuple[float, float]:
    """Mean and variance of the week's final goals-against average, per hour.

    The same delta method: goals against over time in net, both scaled by
    whether he starts at all. Goals on a start are near-binomial on shots.
    """
    eg = vg = eh = vh = cov = 0.0
    for g in side.goalies:
        p, h = g.p_start, g.hours
        m_g = g.shots * (1.0 - g.save_pct)
        v_g = (
            g.shots * g.save_pct * (1.0 - g.save_pct)
            + SHOTS_DISPERSION * g.shots * (1.0 - g.save_pct) ** 2
        )
        eg += p * m_g
        eh += p * h
        vg += p * v_g + p * (1.0 - p) * m_g * m_g
        vh += p * (1.0 - p) * h * h
        cov += p * (1.0 - p) * m_g * h
    num = side.banked.get("goals_against", 0.0) + eg
    den = side.banked.get("toi_hours", 0.0) + eh
    if den <= 0:
        return float("nan"), 0.0
    r = num / den
    var = (vg - 2.0 * r * cov + r * r * vh) / (den * den)
    return r, max(var, 0.0) * inflation**2


# -- the week -------------------------------------------------------------------


@dataclass
class OddsModel:
    """Per-category win probabilities for a week. Every knob is a field.

    `phi` widens each count's variance (1.0 = the sum of independent players);
    `p_play` is the share of scheduled games a healthy skater plays. Both are
    what `season/calibration.py` fits.
    """

    phi: dict[str, float] = field(default_factory=lambda: dict(PHI))
    p_play: float = P_PLAY
    pim_sizes: dict[int, float] = field(default_factory=lambda: dict(PIM_SIZES))
    ratio_tie_width: float = RATIO_TIE_WIDTH
    ratio_inflation: float = SV_PCT_INFLATION

    def category(self, c: Category, ours: Side, theirs: Side) -> CategoryOdds:
        key = c.key
        if key in ("save_pct", "gaa"):
            ratio = _save_pct if key == "save_pct" else _gaa
            mo, vo = ratio(ours, self.ratio_inflation)
            mt, vt = ratio(theirs, self.ratio_inflation)
            if math.isnan(mo) or math.isnan(mt):
                # No shots faced on one side: the other takes it, or nobody does.
                p_win = 1.0 if math.isnan(mt) and not math.isnan(mo) else 0.0
                p_tie = 1.0 if math.isnan(mo) and math.isnan(mt) else 0.0
                return CategoryOdds(c, p_win, p_tie, mo, mt)
            p_win, p_tie = normal_odds(mo - mt, vo + vt)
            if self.ratio_tie_width > 0:
                sd = math.sqrt(max(vo + vt, 1e-18))
                h = self.ratio_tie_width / 2.0
                p_tie = _NORMAL.cdf((h - (mo - mt)) / sd) - _NORMAL.cdf((-h - (mo - mt)) / sd)
                p_win = 1.0 - _NORMAL.cdf((h - (mo - mt)) / sd)
            return self._oriented(c, p_win, p_tie, mo, mt)

        if key in ("saves", "shots_against"):
            es_o, vs_o, ea_o, va_o, _ = _saves_moments(ours.goalies)
            es_t, vs_t, ea_t, va_t, _ = _saves_moments(theirs.goalies)
            if key == "saves":
                eo, vo, et, vt = es_o, vs_o, es_t, vs_t
            else:
                eo, vo, et, vt = ea_o, va_o, ea_t, va_t
            phi = self.phi.get(key, 1.0)
            fo = ours.banked.get(key, 0.0) + eo
            ft = theirs.banked.get(key, 0.0) + et
            p_win, p_tie = normal_odds(fo - ft, (vo + vt) * phi, integer=True)
            return self._oriented(c, p_win, p_tie, fo, ft)

        if key in ("wins", "shutouts"):
            attr = "p_win" if key == "wins" else "p_shutout"
            po = trials_pmf([g.p_start * getattr(g, attr) for g in ours.goalies])
            pt = trials_pmf([g.p_start * getattr(g, attr) for g in theirs.goalies])
            lead = ours.banked.get(key, 0.0) - theirs.banked.get(key, 0.0)
            p_win, p_tie = discrete_odds(po, pt, lead)
            fo = ours.banked.get(key, 0.0) + float(np.dot(np.arange(len(po)), po))
            ft = theirs.banked.get(key, 0.0) + float(np.dot(np.arange(len(pt)), pt))
            return self._oriented(c, p_win, p_tie, fo, ft)

        mo = ours.skaters.get(key, 0.0) * self.p_play
        mt = theirs.skaters.get(key, 0.0) * self.p_play
        fo = ours.banked.get(key, 0.0) + mo
        ft = theirs.banked.get(key, 0.0) + mt
        if key == "plus_minus":
            # Can go negative, so no count distribution fits; about one unit
            # of variance per game is what it runs at.
            var = abs(mo) + abs(mt) + 1.0
            p_win, p_tie = normal_odds(fo - ft, var, integer=True)
            return self._oriented(c, p_win, p_tie, fo, ft)
        if key == "pim":
            po, pt = compound_pmf(mo, self.pim_sizes), compound_pmf(mt, self.pim_sizes)
        else:
            phi = self.phi.get(key, 1.0)
            po, pt = count_pmf(mo, phi), count_pmf(mt, phi)
        lead = ours.banked.get(key, 0.0) - theirs.banked.get(key, 0.0)
        p_win, p_tie = discrete_odds(po, pt, lead)
        return self._oriented(c, p_win, p_tie, fo, ft)

    def week(self, cats: tuple[Category, ...], ours: Side, theirs: Side) -> WeekOdds:
        return WeekOdds(tuple(self.category(c, ours, theirs) for c in cats))

    @staticmethod
    def _oriented(c: Category, p_win: float, p_tie: float, ours: float, theirs: float):
        """Everything above computes "ours higher"; lower-is-better flips it."""
        if c.higher_is_better:
            return CategoryOdds(c, p_win, p_tie, ours, theirs)
        return CategoryOdds(c, max(1.0 - p_win - p_tie, 0.0), p_tie, ours, theirs)


# -- building a side from a roster --------------------------------------------


def goalie_game(rates: dict[str, float], p_start: float) -> GoalieGame:
    """One goalie game from his per-game projection line."""
    shots = rates.get("shots_against", 0.0)
    saves = rates.get("saves", 0.0)
    return GoalieGame(
        p_start=p_start,
        p_win=min(max(rates.get("wins", 0.0), 0.0), 1.0),
        shots=shots,
        save_pct=(saves / shots) if shots > 0 else 0.9,
        p_shutout=min(max(rates.get("shutouts", 0.0), 0.0), 1.0),
        hours=rates.get("toi_hours", 0.0) or 1.0,
    )


def side(
    banked: dict[str, float],
    skater_games: dict[int, float],
    goalie_games: dict[int, list[float]],
    rates: dict[int, dict[str, float]],
) -> Side:
    """A side from expected skater starts and each goalie game's P(start)."""
    skaters: dict[str, float] = {}
    for pid, n in skater_games.items():
        for k, r in (rates.get(pid) or {}).items():
            skaters[k] = skaters.get(k, 0.0) + r * n
    goalies = [
        goalie_game(rates.get(pid) or {}, p)
        for pid, ps in goalie_games.items()
        for p in ps
        if p > 0
    ]
    return Side(banked=dict(banked), skaters=skaters, goalies=goalies)


# -- the live record ------------------------------------------------------------


def log_week(conn, manager: str, league_key: str, team_key: str, plan, day: str) -> bool:
    """Record what the odds said on this run, for scoring against the result.

    Returns False when the plan carries no odds. Keyed by the moment it was
    written, so every run of a day is kept - the week's arc is the point.
    """
    import json
    from datetime import UTC, datetime

    odds = getattr(plan, "odds", None)
    if odds is None:
        return False
    conn.execute(
        "INSERT OR REPLACE INTO week_odds_log (manager, league_key, team_key, week, "
        "logged_at, day, days_left, expected, cats_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            manager,
            league_key,
            team_key,
            plan.week,
            datetime.now(UTC).isoformat(timespec="seconds"),
            day,
            plan.days_left,
            round(odds.expected, 4),
            json.dumps(
                {
                    c.category.key: [
                        round(c.p_win, 4),
                        round(c.p_tie, 4),
                        None if c.ours != c.ours else round(c.ours, 4),
                        None if c.theirs != c.theirs else round(c.theirs, 4),
                    ]
                    for c in odds.cats
                }
            ),
        ),
    )
    conn.commit()
    return True


# -- tonight's goalies ------------------------------------------------------------


@dataclass(frozen=True)
class GoalieChoice:
    """Which of tonight's goalies to start, and what each choice was worth."""

    start: frozenset[int]  # goalies to start tonight
    expected: float  # expected goalie categories won with that choice
    by_subset: dict[frozenset[int], float]  # every choice considered

    def cost_of(self, other: frozenset[int]) -> float:
        """Expected categories given up by choosing `other` instead."""
        return self.expected - self.by_subset.get(other, float("-inf"))


def choose_goalies(
    model: OddsModel,
    cats: tuple[Category, ...],
    ours: Side,
    tonight: dict[int, GoalieGame],
    theirs: Side,
    slots: int,
    at_least: int = 0,
) -> GoalieChoice:
    """The set of tonight's goalies that maximises expected categories won.

    `ours` holds everything except tonight's goalie games - banked totals,
    remaining skaters, later goalie games - so each candidate set is scored by
    adding just its own games. Only goalie categories can change, so only they
    are evaluated. `at_least` is a floor from the weekly minimum: sets smaller
    than it are not considered. Starting everyone who plays is not always best:
    with wins and saves settled and save percentage close, another start can
    only cost the one category still open.

    Measured, and NOT wired into the live lineup: gate G2's goalie-odds arm (12
    teams x 22 weeks, as-of throughout) changed 96 and 68 goalie-nights a season
    - about one in twenty - for +0.000 +/- 0.010 and +0.015 +/- 0.010
    categories a week. Ratio protection is real in a single week and invisible
    over a season. Re-measure before using it.
    """
    from itertools import combinations

    goalie_cats = tuple(c for c in cats if c.kind == "goalie")
    names = sorted(tonight)
    by_subset: dict[frozenset[int], float] = {}
    for k in range(min(max(at_least, 0), len(names), slots), min(slots, len(names)) + 1):
        for combo in combinations(names, k):
            s = frozenset(combo)
            side = Side(
                banked=ours.banked,
                skaters=ours.skaters,
                goalies=list(ours.goalies) + [tonight[pid] for pid in combo],
            )
            by_subset[s] = model.week(goalie_cats, side, theirs).expected
    if not by_subset:
        return GoalieChoice(frozenset(), 0.0, {})
    best = max(by_subset, key=lambda s: (by_subset[s], len(s)))
    return GoalieChoice(best, by_subset[best], by_subset)
