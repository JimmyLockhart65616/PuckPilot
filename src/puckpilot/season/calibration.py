"""Gate G1: do the category odds mean what they say?

A probability is a claim that can be checked: of all the times the model says
70%, about 70% should come off. This replays whole seasons day by day - a
drafted twelve-team league, each team's lineup set every morning by the same
optimizer policy - and at the start of every day of every week predicts each
category from what that matchup has banked so far plus what is left, then
scores the prediction against how the week actually ended.

Everything a prediction sees is as-of that morning:

    banked       real game lines of whoever the policy started on earlier days
    skaters      the policy's own future assignments - it decides from the
                 schedule and pre-season projections alone, so they are known
                 in the morning - times pre-season per-game rates
    goalies      P(start) from the trailing model frozen at that morning
                 (`AsOfGoalieSource`): asked on Wednesday about Saturday, the
                 unfrozen model would have counted Thursday's starts

Fitting and judging are split by season: the widening factors are fitted on
one season and the verdict comes from another. Three measures, per category
and per day of the week:

    log-loss     of the realised result (win / tie / loss), the fitting target
    Brier        of the expected score against 1 / 0.5 / 0
    reliability  predicted against realised, in deciles - the "is 70% really
                 70%" check, and the one the page depends on

And one that says whether the banked days carry information at all: the error
in expected categories won must shrink from a week's first day to its last.
"""

from __future__ import annotations

import math
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, field, replace

import numpy as np

from puckpilot.engine.categories import Category
from puckpilot.league import LeagueConfig
from puckpilot.season.odds import OddsModel, Side, goalie_game, normal_odds

# Grids the fit searches. Wide enough that an edge hit is visible in the report.
PHI_GRID = (1.0, 1.1, 1.2, 1.35, 1.5, 1.75, 2.0, 2.5, 3.0)
P_PLAY_GRID = (0.7, 0.74, 0.78, 0.8, 0.82, 0.84, 0.86, 0.88, 0.9, 0.92, 0.94, 0.96, 0.98, 1.0)
INFLATION_GRID = (1.0, 1.15, 1.3, 1.5, 1.75, 2.0, 2.5)

# A reliability bin with fewer predictions than this is shown but not judged.
MIN_BIN = 100
# The pass mark for a judged decile: realised within this of predicted.
RELIABILITY_TOLERANCE = 0.05


@dataclass
class Case:
    """One matchup on one morning: both sides as they stood, and how it ended."""

    season: str
    week: int
    day: int  # 0 = the week's first day
    days_left: int
    ours: Side
    theirs: Side
    # category key -> 1.0 won / 0.5 tied / 0.0 lost, for "ours"
    outcome: dict[str, float] = field(default_factory=dict)


# A player who has missed this many of his team's games in a row is treated as
# out until he plays again - how a manager reads an injury from the box scores,
# and the replay's stand-in for Yahoo's status flag.
MISSED_FOR_OUT = 2


# -- building the cases ---------------------------------------------------------


def drop_known_absences(
    avail: dict[int, set[int]], played: dict[int, set[int]], missed: int = MISSED_FOR_OUT
) -> dict[int, set[int]]:
    """Team-game days on which a player was already known to be out.

    The replay's lineup policy knows the schedule and nothing about injuries,
    so a player hurt for thirty games kept being started and scoring zero -
    enough that a fitted P(play) ran to the bottom of any grid. Live, Yahoo's
    status benches him. Here, after `missed` straight team games without him,
    he is out until he plays again: as-of, since it reads only games already
    gone.
    """
    out: dict[int, set[int]] = {}
    for pid, days in avail.items():
        seen = played.get(pid, set())
        keep: set[int] = set()
        streak = 0
        for i in sorted(days):
            if streak < missed:
                keep.add(i)
            streak = 0 if i in seen else streak + 1
        out[pid] = keep
    return out


def _components(sk: np.ndarray, g: np.ndarray, skater_keys: list[str]) -> dict[str, float]:
    """A team's banked totals, as the component keys the odds model reads."""
    from puckpilot.draft.replay import G_GA, G_HOURS, G_SA, G_SHO, G_WINS

    out = {k: float(sk[i]) for i, k in enumerate(skater_keys)}
    out.update(
        wins=float(g[G_WINS]),
        shutouts=float(g[G_SHO]),
        goals_against=float(g[G_GA]),
        shots_against=float(g[G_SA]),
        saves=float(g[G_SA] - g[G_GA]),
        toi_hours=float(g[G_HOURS]),
    )
    return out


def _final(c: Category, comp: dict[str, float]) -> float:
    if c.key == "save_pct":
        sa = comp.get("shots_against", 0.0)
        return comp.get("saves", 0.0) / sa if sa > 0 else float("nan")
    if c.key == "gaa":
        h = comp.get("toi_hours", 0.0)
        return comp.get("goals_against", 0.0) / h if h > 0 else float("nan")
    return comp.get(c.key, 0.0)


def _result(c: Category, ours: dict[str, float], theirs: dict[str, float]) -> float:
    a, b = _final(c, ours), _final(c, theirs)
    if math.isnan(a) and math.isnan(b):
        return 0.5
    if math.isnan(b):
        return 1.0
    if math.isnan(a):
        return 0.0
    if abs(a - b) <= 1e-12:
        return 0.5
    better = a > b if c.higher_is_better else a < b
    return 1.0 if better else 0.0


def build_cases(
    conn: sqlite3.Connection,
    season: str,
    train_seasons: tuple[str, ...],
    league: LeagueConfig,
    seed: int = 20261,
    progress: Callable[[str], None] | None = None,
    goalies: str = "",
) -> list[Case]:
    """Every matchup-morning of one replayed season, with how its week ended.

    `goalies` is the starting-goalie model, as `goalies.parse_spec` reads it.
    """
    from puckpilot.draft.h2h import round_robin_schedule
    from puckpilot.draft.replay import G_WIDTH, build_replay_data
    from puckpilot.draft.sim import _default_opponents, build_universe, keepers_for, run_draft
    from puckpilot.engine.lineup_replay import (
        GameValueModel,
        iter_daily_assignments,
        projected_pg_values,
        skater_availability,
    )
    from puckpilot.season.goalies import AsOfGoalieSource, trailing_model
    from puckpilot.season.replay import _week_day_ranges
    from puckpilot.season.week import per_game_rates

    say = progress or (lambda _m: None)
    rules = league.draft_rules()
    shape = rules.shape
    rng = np.random.default_rng(seed)
    skater_keys = [c.key for c in league.skater_cats]
    cats = league.all_cats

    u = build_universe(conn, season, train_seasons, league)
    data = build_replay_data(conn, season, skater_keys)
    vm = GameValueModel(data, set(u.ids.tolist()), league.goalie_cats)
    pg_value = projected_pg_values(u.frame, vm, skater_keys)
    positions = dict(zip(u.ids.tolist(), u.pos.tolist(), strict=True))
    rates = per_game_rates(u.frame, cats)

    from puckpilot.draft.engine import RosterValuePolicy

    opponents = _default_opponents(rng, league)
    order = rng.permutation(len(opponents))
    bots, oi = [], 0
    for seat in range(shape.n_teams):
        bots.append(RosterValuePolicy() if seat == 0 else opponents[order[oi]])
        oi += 0 if seat == 0 else 1
    keepers = keepers_for(conn, u, season, league, rng)
    rosters = [[int(u.ids[i]) for i in r] for r in run_draft(u, bots, rules, rng, keepers)]
    say(f"{season}: drafted {len(rosters)} teams")

    all_pids = {p for r in rosters for p in r}
    played = {pid: set(data.skater.get(pid, {})) for pid in all_pids}
    avail = drop_known_absences(skater_availability(conn, season, data, all_pids), played)
    policy = trailing_model(conn, season, train_seasons[0], goalies)
    g_slots = sum(n for pos, n in shape.slots if pos == "G")

    # Each team's season as the policy plays it: who started each day, and the
    # real line each of them produced.
    n_days = len(data.dates)
    started: list[dict[int, list[int]]] = []
    sk_day = np.zeros((len(rosters), n_days, len(skater_keys)))
    g_day = np.zeros((len(rosters), n_days, G_WIDTH))
    for t, roster in enumerate(rosters):
        per_day: dict[int, list[int]] = {}
        for i, assigned in iter_daily_assignments(
            lambda _d, r=roster: r,
            positions,
            pg_value,
            data,
            shape,
            avail,
            policy,
            min_goalie_appearances=league.min_goalie_appearances,
        ):
            per_day[i] = list(assigned)
            for pid in assigned:
                line = data.skater.get(pid, {}).get(i)
                if line is not None:
                    sk_day[t, i] += line
                    continue
                line = data.goalie.get(pid, {}).get(i)
                if line is not None:
                    g_day[t, i] += line
        started.append(per_day)
    say(f"{season}: lineups played")

    ranges = _week_day_ranges(data)
    weeks = sorted(ranges)[: league.regular_weeks]
    schedule = round_robin_schedule(len(rosters), len(weeks))

    frozen_cache: dict[tuple[str, str], dict[int, float]] = {}

    def remaining(t: int, days: list[int], cutoff: str) -> tuple[dict[int, float], dict]:
        """Skater starts still to come, and each goalie game's as-of P(start)."""
        skater_games: dict[int, float] = {}
        goalie_games: dict[int, list[float]] = {}
        for i in days:
            for pid in started[t].get(i, ()):
                if positions.get(pid) != "G":
                    skater_games[pid] = skater_games.get(pid, 0.0) + 1.0
            key = (cutoff, data.dates[i])
            if key not in frozen_cache:
                frozen_cache[key] = AsOfGoalieSource(policy, cutoff).starts(data.dates[i])
            p = frozen_cache[key]
            gs = sorted(
                (
                    (p.get(pid, 0.0) * pg_value.get(pid, 0.0), p.get(pid, 0.0), pid)
                    for pid in rosters[t]
                    if positions.get(pid) == "G" and p.get(pid, 0.0) > 0
                ),
                reverse=True,
            )[:g_slots]
            for _, ps, pid in gs:
                goalie_games.setdefault(pid, []).append(ps)
        return skater_games, goalie_games

    def side_of(t, banked, skater_games, goalie_games) -> Side:
        skaters: dict[str, float] = {}
        for pid, n in skater_games.items():
            for k, r in (rates.get(pid) or {}).items():
                skaters[k] = skaters.get(k, 0.0) + r * n
        goalies = [
            goalie_game(rates.get(pid) or {}, p) for pid, ps in goalie_games.items() for p in ps
        ]
        return Side(banked=banked, skaters=skaters, goalies=goalies)

    cases: list[Case] = []
    for wi, w in enumerate(weeks):
        days = list(ranges[w])
        for a, b in schedule[wi]:
            final_a = _components(sk_day[a, days].sum(0), g_day[a, days].sum(0), skater_keys)
            final_b = _components(sk_day[b, days].sum(0), g_day[b, days].sum(0), skater_keys)
            outcome = {c.key: _result(c, final_a, final_b) for c in cats}
            for k, i in enumerate(days):
                done = days[:k]
                bank_a = _components(
                    sk_day[a, done].sum(0) if done else np.zeros(len(skater_keys)),
                    g_day[a, done].sum(0) if done else np.zeros(G_WIDTH),
                    skater_keys,
                )
                bank_b = _components(
                    sk_day[b, done].sum(0) if done else np.zeros(len(skater_keys)),
                    g_day[b, done].sum(0) if done else np.zeros(G_WIDTH),
                    skater_keys,
                )
                rest = days[k:]
                cutoff = data.dates[i]
                sa, ga = remaining(a, rest, cutoff)
                sb, gb = remaining(b, rest, cutoff)
                cases.append(
                    Case(
                        season=season,
                        week=wi + 1,
                        day=k,
                        days_left=len(rest),
                        ours=side_of(a, bank_a, sa, ga),
                        theirs=side_of(b, bank_b, sb, gb),
                        outcome=outcome,
                    )
                )
        say(f"{season}: week {wi + 1}/{len(weeks)}")
    return cases


# -- scoring --------------------------------------------------------------------


class NormalBaseline(OddsModel):
    """The plain alternative: every skater count normal, Poisson variance, no
    team widening. If the full model cannot beat this, ship this."""

    def category(self, c, ours, theirs):
        if c.key in ("save_pct", "gaa", "saves", "shots_against", "wins", "shutouts"):
            return super().category(c, ours, theirs)
        mo = ours.skaters.get(c.key, 0.0) * self.p_play
        mt = theirs.skaters.get(c.key, 0.0) * self.p_play
        fo = ours.banked.get(c.key, 0.0) + mo
        ft = theirs.banked.get(c.key, 0.0) + mt
        var = abs(mo) + abs(mt)
        if c.key == "pim":
            var *= 3.84
        p_win, p_tie = normal_odds(fo - ft, var, integer=True)
        return self._oriented(c, p_win, p_tie, fo, ft)


@dataclass
class Scores:
    n: int
    logloss: float
    brier: float
    by_cat: dict[str, tuple[int, float, float]]  # key -> (n, logloss, brier)
    reliability: list[tuple[float, float, int, float, float]]  # lo, hi, n, pred, real
    ecats_error: dict[int, float]  # day of week -> mean |E[cats] - cats won|
    passed_bins: int = 0
    judged_bins: int = 0

    @property
    def reliable(self) -> bool:
        return self.judged_bins > 0 and self.passed_bins == self.judged_bins


def score(
    cases: list[Case], model: OddsModel, cats: tuple[Category, ...], only: str | None = None
) -> Scores:
    """How good the model's predictions were, over every case and category."""
    ll = br = 0.0
    n = 0
    per: dict[str, list[float]] = {}
    preds: list[tuple[float, float]] = []
    err: dict[int, list[float]] = {}
    use = tuple(c for c in cats if only is None or c.key == only)
    for case in cases:
        odds = model.week(use, case.ours, case.theirs)
        exp_total = won_total = 0.0
        for o in odds.cats:
            y = case.outcome[o.category.key]
            p = {1.0: o.p_win, 0.5: o.p_tie, 0.0: o.p_loss}[y]
            loss = -math.log(min(max(p, 1e-6), 1.0))
            sq = (o.expected - y) ** 2
            ll += loss
            br += sq
            n += 1
            bucket = per.setdefault(o.category.key, [0.0, 0.0, 0.0])
            bucket[0] += 1
            bucket[1] += loss
            bucket[2] += sq
            preds.append((o.expected, y))
            exp_total += o.expected
            won_total += y
        err.setdefault(case.day, []).append(abs(exp_total - won_total))

    bins: list[tuple[float, float, int, float, float]] = []
    passed = judged = 0
    for lo in np.arange(0.0, 1.0, 0.1):
        hi = lo + 0.1
        inside = [(p, y) for p, y in preds if lo <= p < hi or (hi >= 1.0 and p == 1.0)]
        if not inside:
            continue
        mp = sum(p for p, _ in inside) / len(inside)
        my = sum(y for _, y in inside) / len(inside)
        bins.append((float(lo), float(hi), len(inside), mp, my))
        if len(inside) >= MIN_BIN:
            judged += 1
            passed += abs(mp - my) <= RELIABILITY_TOLERANCE
    return Scores(
        n=n,
        logloss=ll / max(n, 1),
        brier=br / max(n, 1),
        by_cat={k: (int(v[0]), v[1] / v[0], v[2] / v[0]) for k, v in per.items()},
        reliability=bins,
        ecats_error={d: sum(v) / len(v) for d, v in sorted(err.items())},
        passed_bins=passed,
        judged_bins=judged,
    )


def fit(cases: list[Case], cats: tuple[Category, ...], say=None) -> OddsModel:
    """Fit P(play), each count's widening, and the save-percentage spread.

    One knob at a time, each against its own categories' log-loss: P(play) on
    the skater counts together (it moves every mean), then phi per count, then
    the ratio inflation on SV% / GAA.
    """
    say = say or (lambda _m: None)
    model = OddsModel(phi={})
    counts = [c for c in cats if c.kind == "skater" and c.key not in ("pim", "plus_minus")]

    def total(m: OddsModel, keys) -> float:
        return sum(score(cases, m, cats, only=k).logloss for k in keys)

    keys = [c.key for c in counts] + (["pim"] if any(c.key == "pim" for c in cats) else [])
    best = min(P_PLAY_GRID, key=lambda p: total(replace(model, p_play=p), keys))
    model = replace(model, p_play=best)
    say(f"  p_play {best:.2f}")

    phi: dict[str, float] = {}
    for c in counts + [c for c in cats if c.key == "saves"]:
        best_phi = min(
            PHI_GRID,
            key=lambda f, k=c.key: (
                score(cases, replace(model, phi={**phi, k: f}), cats, only=k).logloss
            ),
        )
        phi[c.key] = best_phi
        say(f"  phi {c.label} {best_phi:.2f}")
    model = replace(model, phi=phi)

    ratios = [c.key for c in cats if c.key in ("save_pct", "gaa")]
    if ratios:
        best_inf = min(
            INFLATION_GRID,
            key=lambda f: total(replace(model, ratio_inflation=f), ratios),
        )
        model = replace(model, ratio_inflation=best_inf)
        say(f"  ratio inflation {best_inf:.2f}")
    return model


# -- the report -------------------------------------------------------------------


@dataclass
class CalibrationReport:
    fitted: OddsModel
    fit: Scores
    test: Scores
    baseline: Scores
    text: str

    @property
    def passed(self) -> bool:
        """Reliable on the held-out season, and no worse than the plain model."""
        return self.test.reliable and self.test.logloss <= self.baseline.logloss + 1e-9


def calibration_report(
    conn: sqlite3.Connection,
    league: LeagueConfig,
    fit_season: str = "20242025",
    test_season: str = "20252026",
    seed: int = 20261,
    progress: Callable[[str], None] | None = None,
    goalies: str = "",
) -> CalibrationReport:
    say = progress or (lambda _m: None)
    cats = league.all_cats

    def train(season: str) -> tuple[str, ...]:
        y = int(season[:4])
        return tuple(f"{y - i}{y - i + 1}" for i in range(1, 4))

    fit_cases = build_cases(conn, fit_season, train(fit_season), league, seed, say, goalies)
    test_cases = build_cases(conn, test_season, train(test_season), league, seed, say, goalies)
    say("fitting")
    model = fit(fit_cases, cats, say)
    s_fit = score(fit_cases, model, cats)
    s_test = score(test_cases, model, cats)
    s_base = score(test_cases, NormalBaseline(p_play=model.p_play), cats)
    s_flat = score(test_cases, OddsModel(p_play=model.p_play), cats)

    lines = [
        f"G1 - calibration: fitted on {fit_season} ({len(fit_cases)} matchup-mornings), "
        f"judged on {test_season} ({len(test_cases)})",
        "",
        f"fitted: p_play {model.p_play:.2f}, ratio inflation {model.ratio_inflation:.2f}",
        "        phi " + "  ".join(f"{k} {v:.2f}" for k, v in model.phi.items()),
        "",
        f"{'':26}{'log-loss':>10}{'Brier':>9}",
        f"{'fitted model, fit season':26}{s_fit.logloss:>10.4f}{s_fit.brier:>9.4f}",
        f"{'fitted model, test season':26}{s_test.logloss:>10.4f}{s_test.brier:>9.4f}",
        f"{'unfitted model, test':26}{s_flat.logloss:>10.4f}{s_flat.brier:>9.4f}",
        f"{'plain normal, test':26}{s_base.logloss:>10.4f}{s_base.brier:>9.4f}",
        f"{'coin flip (reference)':26}{math.log(3):>10.4f}{0.25:>9.4f}",
        "",
        "by category (test):   n   log-loss  Brier",
    ]
    for c in cats:
        n, ll, br = s_test.by_cat.get(c.key, (0, float("nan"), float("nan")))
        lines.append(f"  {c.label:6}{n:>8}{ll:>10.4f}{br:>8.4f}")
    lines += ["", "reliability (test) - predicted vs realised expected score:"]
    for lo, hi, n, mp, my in s_test.reliability:
        mark = "" if n < MIN_BIN else ("  ok" if abs(mp - my) <= RELIABILITY_TOLERANCE else "  OFF")
        lines.append(f"  {lo:.1f}-{hi:.1f}  n={n:>6}  predicted {mp:.3f}  realised {my:.3f}{mark}")
    lines += ["", "error in expected categories won, by day of the week (test):"]
    for d, e in s_test.ecats_error.items():
        lines.append(f"  day {d + 1}: {e:.2f}")
    days = list(s_test.ecats_error.values())
    shrinks = len(days) >= 2 and days[-1] < days[0]
    lines += [
        "",
        f"reliable: {s_test.passed_bins}/{s_test.judged_bins} judged deciles within "
        f"{RELIABILITY_TOLERANCE:.0%}",
        f"beats the plain normal model: {'yes' if s_test.logloss <= s_base.logloss else 'NO'}",
        f"banked days carry information (error shrinks through the week): "
        f"{'yes' if shrinks else 'NO'}",
    ]
    report = CalibrationReport(fitted=model, fit=s_fit, test=s_test, baseline=s_base, text="")
    verdict = "PASS" if report.passed and shrinks else "FAIL"
    lines.append(f"G1: {verdict}")
    report.text = "\n".join(lines)
    return report
