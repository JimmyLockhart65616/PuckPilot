from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from puckpilot.engine.categories import (
    GOALIE_CATS_DEFAULT,
    SKATER_CATS_DEFAULT,
    Category,
)

# volume column used to weight each rate category's impact
RATE_VOLUME = {"save_pct": "shots_against", "gaa": "toi_hours"}


@dataclass(frozen=True)
class LeagueShape:
    """Roster shape driving pool sizes and replacement levels.

    Yahoo-ish 12-team default until real league settings arrive via the Yahoo API.
    Bench pool sizing assumes ~3 of 4 bench spots go to skaters.
    """

    n_teams: int = 12
    slots: tuple[tuple[str, int], ...] = (("C", 2), ("L", 2), ("R", 2), ("D", 4), ("G", 2))
    util_slots: int = 2
    bench_slots: int = 4

    @property
    def skater_pool_size(self) -> int:
        starters = sum(n for pos, n in self.slots if pos != "G") + self.util_slots
        return self.n_teams * (starters + 3)

    @property
    def goalie_pool_size(self) -> int:
        return self.n_teams * (dict(self.slots).get("G", 2) + 1)

    def starters_by_pos(self) -> dict[str, int]:
        d = {pos: self.n_teams * n for pos, n in self.slots}
        util_total = self.n_teams * self.util_slots
        for pos in ("C", "L", "R"):
            d[pos] = d.get(pos, 0) + util_total // 3
        return d


DEFAULT_SHAPE = LeagueShape()


def _rate_impact(df: pd.DataFrame, cat: Category, pool: pd.Index) -> pd.Series:
    """Volume-weighted deviation from the pool's volume-weighted average.

    e.g. SV%: (sv - pool_sv) * shots_against == saves above pool average.
    """
    vol = RATE_VOLUME[cat.key]
    p = df.loc[pool]
    pool_avg = (p[cat.key] * p[vol]).sum() / p[vol].sum()
    x = (df[cat.key] - pool_avg) * df[vol]
    return x if cat.higher_is_better else -x


def value_players(
    df: pd.DataFrame, cats: tuple[Category, ...], pool_size: int, iters: int = 3
) -> pd.DataFrame:
    """Adds z_<cat> columns and z_total, normalized against the draftable pool.

    The pool starts as everyone, then shrinks to the top pool_size by z_total and
    the stats are recomputed — a few iterations is standard and converges fast.
    """
    df = df.copy()
    if df.empty:
        df["z_total"] = pd.Series(dtype=float)
        return df
    pool = df.index
    for _ in range(iters):
        z = {}
        for cat in cats:
            if cat.rate:
                x = _rate_impact(df, cat, pool)
            else:
                x = df[cat.key].astype(float)
                if not cat.higher_is_better:
                    x = -x
            m = x.loc[pool].mean()
            s = x.loc[pool].std(ddof=0)
            z[cat.key] = (x - m) / s if s > 0 else x * 0.0
        zdf = pd.DataFrame(z)
        total = zdf.sum(axis=1)
        pool = total.nlargest(min(pool_size, len(total))).index
    for key in zdf.columns:
        df[f"z_{key}"] = zdf[key]
    df["z_total"] = total
    return df


def replacement_adjust(df: pd.DataFrame, starters_by_pos: dict[str, int]) -> pd.DataFrame:
    """vorp = z_total minus the z_total of the last starter at that position."""
    df = df.copy()
    df["vorp"] = df["z_total"]
    for pos, group in df.groupby("position"):
        k = starters_by_pos.get(pos, 0)
        if k <= 0:
            continue
        sorted_z = group["z_total"].sort_values(ascending=False)
        repl = sorted_z.iloc[min(k, len(sorted_z)) - 1]
        df.loc[group.index, "vorp"] = group["z_total"] - repl
    return df


def rank_players(
    skaters: pd.DataFrame,
    goalies: pd.DataFrame,
    shape: LeagueShape = DEFAULT_SHAPE,
    skater_cats: tuple[Category, ...] = SKATER_CATS_DEFAULT,
    goalie_cats: tuple[Category, ...] = GOALIE_CATS_DEFAULT,
) -> pd.DataFrame:
    """Combined skater+goalie ranking by VORP (descending)."""
    starters = shape.starters_by_pos()
    parts = []
    if not skaters.empty:
        sk = replacement_adjust(
            value_players(skaters, skater_cats, shape.skater_pool_size), starters
        )
        sk["kind"] = "skater"
        parts.append(sk)
    if not goalies.empty:
        g = replacement_adjust(
            value_players(goalies, goalie_cats, shape.goalie_pool_size), starters
        )
        g["kind"] = "goalie"
        parts.append(g)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts).sort_values("vorp", ascending=False)
