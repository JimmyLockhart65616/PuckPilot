from __future__ import annotations

import sqlite3

import pandas as pd
from scipy.stats import spearmanr

from puckpilot.engine import projections
from puckpilot.engine.aggregate import season_aggregates
from puckpilot.engine.categories import GOALIE_CATS_DEFAULT, SKATER_CATS_DEFAULT
from puckpilot.engine.valuation import DEFAULT_SHAPE, LeagueShape, rank_players

MIN_ACTUAL_GP_SKATER = 10
MIN_ACTUAL_GP_GOALIE = 5

SPEARMAN_BAR = 0.6  # plan's sanity bar for overall value rank correlation


def _cat_mae(proj: pd.DataFrame, actual: pd.DataFrame, keys: list[str], min_gp: int) -> dict:
    m = proj[keys + ["proj_gp"]].join(
        actual[keys + ["gp"]], how="inner", lsuffix="_p", rsuffix="_a"
    )
    m = m[m["gp"] >= min_gp]
    out = {"n": len(m), "gp": (m["proj_gp"] - m["gp"]).abs().mean()}
    for k in keys:
        out[k] = (m[f"{k}_p"] - m[f"{k}_a"]).abs().mean()
    return out


def _spearman(proj_ranked: pd.DataFrame, act_ranked: pd.DataFrame, act_gp: pd.Series) -> dict:
    m = pd.DataFrame({"p": proj_ranked["vorp"], "kind": proj_ranked["kind"]}).join(
        pd.DataFrame({"a": act_ranked["vorp"], "gp": act_gp}), how="inner"
    )
    m = m[
        ((m["kind"] == "skater") & (m["gp"] >= MIN_ACTUAL_GP_SKATER))
        | ((m["kind"] == "goalie") & (m["gp"] >= MIN_ACTUAL_GP_GOALIE))
    ]

    def rho(sub: pd.DataFrame) -> float:
        return float(spearmanr(sub["p"], sub["a"]).statistic) if len(sub) > 2 else float("nan")

    return {
        "overall": rho(m),
        "skaters": rho(m[m["kind"] == "skater"]),
        "goalies": rho(m[m["kind"] == "goalie"]),
        "n": len(m),
    }


def walk_forward(
    conn: sqlite3.Connection,
    target_season: str = "20252026",
    train_seasons: tuple[str, ...] = ("20242025", "20232024"),
    shape: LeagueShape = DEFAULT_SHAPE,
) -> tuple[dict, str]:
    """Train on train_seasons, predict target_season, score against actuals.

    Returns (metrics, printable report). Also scores a naive last-season-carry-over
    baseline so the blend has to earn its keep.
    """
    proj_sk, proj_g = projections.project(conn, target_season, list(train_seasons))
    act_sk, act_g = season_aggregates(conn, target_season)

    sk_keys = [c.key for c in SKATER_CATS_DEFAULT]
    g_keys = [c.key for c in GOALIE_CATS_DEFAULT]
    mae_sk = _cat_mae(proj_sk, act_sk, sk_keys, MIN_ACTUAL_GP_SKATER)
    mae_g = _cat_mae(proj_g, act_g, g_keys, MIN_ACTUAL_GP_GOALIE)

    act_gp = pd.concat([act_sk["gp"], act_g["gp"]])
    act_ranked = rank_players(act_sk, act_g, shape)
    sp = _spearman(rank_players(proj_sk, proj_g, shape), act_ranked, act_gp)

    naive_sk, naive_g = projections.project(conn, target_season, [train_seasons[0]], weights=(1.0,))
    sp_naive = _spearman(rank_players(naive_sk, naive_g, shape), act_ranked, act_gp)

    metrics = {
        "target": target_season,
        "train": train_seasons,
        "mae_skaters": mae_sk,
        "mae_goalies": mae_g,
        "spearman": sp,
        "spearman_naive": sp_naive,
        "bar": SPEARMAN_BAR,
        "passed": sp["overall"] > SPEARMAN_BAR,
    }

    lines = [
        f"Walk-forward validation: train {' + '.join(train_seasons)} -> predict {target_season}",
        "",
        f"Skater category MAE (n={mae_sk['n']}, actual GP >= {MIN_ACTUAL_GP_SKATER}):",
        f"  GP {mae_sk['gp']:.1f}  "
        + "  ".join(f"{c.label} {mae_sk[c.key]:.2f}" for c in SKATER_CATS_DEFAULT),
        f"Goalie category MAE (n={mae_g['n']}, actual GP >= {MIN_ACTUAL_GP_GOALIE}):",
        f"  GP {mae_g['gp']:.1f}  "
        + "  ".join(f"{c.label} {mae_g[c.key]:.2f}" for c in GOALIE_CATS_DEFAULT),
        "",
        f"Spearman rank corr, projected vs actual VORP (n={sp['n']}):",
        f"  overall {sp['overall']:.3f}   skaters {sp['skaters']:.3f}   "
        f"goalies {sp['goalies']:.3f}",
        f"  naive last-season baseline: overall {sp_naive['overall']:.3f}   "
        f"skaters {sp_naive['skaters']:.3f}   goalies {sp_naive['goalies']:.3f}",
        "",
        f"Bar: overall Spearman > {SPEARMAN_BAR} -> {'PASS' if metrics['passed'] else 'FAIL'}",
    ]
    return metrics, "\n".join(lines)
