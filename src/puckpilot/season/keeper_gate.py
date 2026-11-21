"""Gate for keeper values: standing on a date inside one season, which way of
ranking players keeps the ones who turn out best the next?

Five ways of putting one number on a player, all knowable on the date:

    preseason  this season's preseason projection - August's view
    aged       the same three seasons, projected a year older: the age curve alone
    as-of      this season so far as the most recent of three - what ships
    form       value a game on the date, the preseason number pulled toward
               recent form: what a drop was judged by before keepers existed
    season     the whole of this season - next August's view. Not knowable on
               any date inside the season; it is the ceiling the others chase.

Each is scored against next season's actual value over replacement:

- **Spearman** over the players every rule can value who played next season.
- **Top-K**: of the K best players next season (K = teams x keepers, the pool
  a keeper league actually fights over), how many the rule had in its top K.
- **Keeper regret** on drafted rosters: each roster keeps its best `n_keepers`
  by the rule; regret is what the best possible keepers produced next season
  minus what those did. A player below replacement next season counts as
  replacement, since he would be benched or cut, and so does one who did not
  play at all.

Contracts are ignored throughout - every player is taken as eligible. The
question here is only whether the ranking is right.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from puckpilot.league import LeagueConfig
from puckpilot.season.keeper_value import next_season, project_next

# Seasons whose next season is complete in the data.
SEASONS = ("20232024", "20242025")

# Dates inside the season (month-day; October to December fall in its first
# calendar year). Keeper decisions are argued all year and made at the end.
AS_OF = ("11-15", "01-01", "02-15", "04-01")

SEEDS = (7, 42, 99, 20261)

RULES = ("preseason", "aged", "as-of", "form", "season")

MIN_GP_SKATER = 10
MIN_GP_GOALIE = 5


def _date(season: str, md: str) -> str:
    y = int(season[:4])
    return f"{y if md >= '07-01' else y + 1}-{md}"


def _vorp(league, skaters, goalies) -> pd.Series:
    from puckpilot.engine.valuation import rank_players

    ranked = rank_players(
        skaters,
        goalies,
        shape=league.shape,
        skater_cats=league.skater_cats,
        goalie_cats=league.goalie_cats,
    )
    out = ranked["vorp"].astype(float)
    out.index = out.index.astype(int)
    return out


@dataclass(frozen=True)
class RuleScore:
    spearman: float
    top_k: float
    regret: float
    regret_se: float


@dataclass(frozen=True)
class KeeperGateReport:
    text: str
    scores: dict[tuple[str, str, str], RuleScore]  # (season, as_of, rule)


def _rosters(conn, league, season: str, seeds) -> list[list[int]]:
    from puckpilot.draft.engine import RosterValuePolicy
    from puckpilot.draft.sim import _default_opponents, build_universe, keepers_for, run_draft

    y = int(season[:4])
    train = tuple(f"{y - i}{y - i + 1}" for i in range(1, 4))
    u = build_universe(conn, season, train, league)
    rules = league.draft_rules()
    out = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        opponents = _default_opponents(rng, league)
        order = rng.permutation(len(opponents))
        bots, oi = [], 0
        for seat in range(rules.shape.n_teams):
            bots.append(RosterValuePolicy() if seat == 0 else opponents[order[oi]])
            oi += 0 if seat == 0 else 1
        keepers = keepers_for(conn, u, season, league, rng)
        out += [[int(u.ids[i]) for i in r] for r in run_draft(u, bots, rules, rng, keepers)]
    return out


def score_rule(
    values: pd.Series,
    actual: pd.Series,
    played: set[int],
    common: set[int],
    rosters: list[list[int]],
    k: int,
    n_keep: int,
) -> RuleScore:
    """One rule's ranking against next season's actual values."""
    both = sorted(common & played)
    rho = (
        float(spearmanr(values.reindex(both), actual.reindex(both)).statistic)
        if len(both) > 2
        else float("nan")
    )
    top_rule = set(values.sort_values(ascending=False).index[:k])
    top_real = set(actual.sort_values(ascending=False).index[:k])
    realised = actual.clip(lower=0.0)
    regrets = []
    for roster in rosters:
        got = [realised.get(pid, 0.0) for pid in roster]
        best = sum(sorted(got, reverse=True)[:n_keep])
        valued = sorted((pid for pid in roster if pid in values.index), key=lambda p: -values[p])
        chosen = sum(realised.get(pid, 0.0) for pid in valued[:n_keep])
        regrets.append(best - chosen)
    r = np.asarray(regrets, dtype=float)
    return RuleScore(
        spearman=rho,
        top_k=len(top_rule & top_real) / k,
        regret=float(r.mean()) if len(r) else float("nan"),
        regret_se=float(r.std(ddof=1) / np.sqrt(len(r))) if len(r) > 1 else float("nan"),
    )


def keeper_gate_report(
    conn: sqlite3.Connection,
    league: LeagueConfig,
    seasons: tuple[str, ...] = SEASONS,
    as_of: tuple[str, ...] = AS_OF,
    seeds: tuple[int, ...] = SEEDS,
    progress: Callable[[str], None] | None = None,
) -> KeeperGateReport:
    from puckpilot.engine import projections
    from puckpilot.engine.aggregate import season_aggregates, season_games
    from puckpilot.season.values import build_value_model

    say = progress or (lambda _m: None)
    n_keep = league.n_keepers or 3
    k = league.shape.n_teams * n_keep
    scores: dict[tuple[str, str, str], RuleScore] = {}
    lines = [
        "Keeper values: ranked on a date, scored on what the players did the next season",
        f"(top-K: K = {k}; regret: best {n_keep} keepers' value next season minus the chosen "
        f"ones', per roster)",
        "",
    ]
    for season in seasons:
        nxt = next_season(season)
        y = int(season[:4])
        train_pre = [f"{y - i}{y - i + 1}" for i in range(1, 4)]
        say(f"{season}: projections")
        act_sk, act_g = season_aggregates(conn, nxt)
        actual = _vorp(league, act_sk, act_g)
        played = {int(p) for p in act_sk.index[act_sk["gp"] >= MIN_GP_SKATER]} | {
            int(p) for p in act_g.index[act_g["gp"] >= MIN_GP_GOALIE]
        }
        pre = _vorp(league, *projections.project(conn, season, train_pre))
        aged = _vorp(
            league,
            *projections.project(conn, nxt, train_pre, target_games=season_games(conn, season)),
        )
        full = project_next(conn, league, season, None)
        say(f"{season}: value model and rosters")
        values = build_value_model(conn, season, tuple(train_pre), league)
        rosters = _rosters(conn, league, season, seeds)
        lines.append(
            f"{season} -> {nxt}  ({len(rosters)} drafted rosters; Spearman over players "
            f"valued by every rule who played next season)"
        )
        lines.append(f"  {'as of':12}{'rule':11}{'Spearman':>9}{'top-K':>7}{'regret':>9}{'+/-':>7}")
        for md in as_of:
            day = _date(season, md)
            say(f"{season}: as of {day}")
            asof = project_next(conn, league, season, day)
            pool = set(pre.index) | set(asof.index) | set(full.index)
            form = pd.Series({pid: values.per_game(pid, day) for pid in pool}, dtype=float)
            by_rule = {"preseason": pre, "aged": aged, "as-of": asof, "form": form, "season": full}
            common = set.intersection(*(set(v.index) for v in by_rule.values()))
            for rule in RULES:
                s = score_rule(by_rule[rule], actual, played, common, rosters, k, n_keep)
                scores[(season, day, rule)] = s
                lines.append(
                    f"  {day:12}{rule:11}{s.spearman:>9.3f}{s.top_k:>7.0%}"
                    f"{s.regret:>9.2f}{s.regret_se:>7.2f}"
                )
            lines.append("")
    lines += _verdict(scores, seasons)
    return KeeperGateReport(text="\n".join(lines), scores=scores)


def _verdict(scores, seasons) -> list[str]:
    """The bar: from mid-season on, as-of has less regret than preseason and
    form in every season, and ranks at least as well as preseason."""
    ok = True
    for (season, day, rule), s in scores.items():
        if rule != "as-of" or day < _date(season, "01-01"):
            continue
        pre = scores[(season, day, "preseason")]
        form = scores[(season, day, "form")]
        if not (s.regret < pre.regret and s.regret < form.regret and s.spearman >= pre.spearman):
            ok = False
    return [
        "Bar: from January on, as-of keeps better players than preseason and form in "
        f"every season, and ranks at least as well as preseason -> {'PASS' if ok else 'FAIL'}"
    ]
