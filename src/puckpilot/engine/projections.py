from __future__ import annotations

import sqlite3

import pandas as pd

from puckpilot.engine.aggregate import SKATER_COLS, season_aggregates, season_games

# Marcel-style season weights, most recent season first. Deliberately simple:
# the walk-forward report (validate.py) is the evidence for anything fancier.
DEFAULT_WEIGHTS = (0.5, 0.3, 0.2)

MIN_TRAIN_GP_SKATER = 10
MIN_TRAIN_GP_GOALIE = 5

GOALIE_COUNT_COLS = ["wins", "shutouts"]
GOALIE_VOLUME_COLS = ["shots_against", "goals_against", "toi_hours"]


def blend_counting(
    frames: list[tuple[pd.DataFrame, int]],
    weights: tuple[float, ...],
    cat_cols: list[str],
    target_games: int,
    min_train_gp: int,
) -> pd.DataFrame:
    """Blend per-game rates and GP fraction across seasons into projected totals.

    frames: [(per-player frame with 'gp' + cat_cols, games_in_that_season), ...],
    most recent first, aligned with weights. Per-player weights renormalize over
    the seasons the player actually appeared in.
    """
    idx = pd.Index([], name="player_id")
    for df, _ in frames:
        idx = idx.union(df.index)

    w_sum = pd.Series(0.0, index=idx)
    rate_sum = pd.DataFrame(0.0, index=idx, columns=cat_cols)
    frac_sum = pd.Series(0.0, index=idx)
    gp_total = pd.Series(0, index=idx)

    for (df, games), w in zip(frames, weights, strict=False):
        d = df.reindex(idx)
        has = d["gp"].notna() & (d["gp"] > 0)
        w_eff = has.astype(float) * w
        rates = d[cat_cols].div(d["gp"], axis=0).fillna(0.0)
        rate_sum += rates.mul(w_eff, axis=0)
        frac_sum += (d["gp"].fillna(0) / games) * w_eff
        w_sum += w_eff
        gp_total += d["gp"].fillna(0).astype(int)

    keep = (gp_total >= min_train_gp) & (w_sum > 0)
    proj_rate = rate_sum[keep].div(w_sum[keep], axis=0)
    proj_gp = ((frac_sum[keep] / w_sum[keep]).clip(upper=1.0) * target_games).round(1)
    out = proj_rate.mul(proj_gp, axis=0)
    out.insert(0, "proj_gp", proj_gp)
    return out


def project_skaters(
    frames: list[tuple[pd.DataFrame, int]],
    target_games: int,
    weights: tuple[float, ...] = DEFAULT_WEIGHTS,
) -> pd.DataFrame:
    return blend_counting(frames, weights, SKATER_COLS, target_games, MIN_TRAIN_GP_SKATER)


def project_goalies(
    frames: list[tuple[pd.DataFrame, int]],
    target_games: int,
    weights: tuple[float, ...] = DEFAULT_WEIGHTS,
) -> pd.DataFrame:
    """Counting cats blend like skaters; SV% and GAA blend volume-weighted:
    save_pct = sum(w*saves)/sum(w*shots_against), gaa = sum(w*GA)/sum(w*hours).
    """
    out = blend_counting(
        frames, weights, GOALIE_COUNT_COLS + GOALIE_VOLUME_COLS, target_games, MIN_TRAIN_GP_GOALIE
    )
    idx = out.index

    saves_w = pd.Series(0.0, index=idx)
    sa_w = pd.Series(0.0, index=idx)
    ga_w = pd.Series(0.0, index=idx)
    hours_w = pd.Series(0.0, index=idx)
    for (df, _games), w in zip(frames, weights, strict=False):
        d = df.reindex(idx).fillna(0.0)
        sa_w += w * d["shots_against"]
        saves_w += w * (d["shots_against"] - d["goals_against"])
        ga_w += w * d["goals_against"]
        hours_w += w * d["toi_hours"]

    out["save_pct"] = (saves_w / sa_w.where(sa_w > 0)).fillna(0.0)
    out["gaa"] = (ga_w / hours_w.where(hours_w > 0)).fillna(0.0)
    return out


def project(
    conn: sqlite3.Connection,
    target_season: str,
    train_seasons: list[str],
    weights: tuple[float, ...] = DEFAULT_WEIGHTS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(skaters, goalies) projected category totals for target_season.

    train_seasons most recent first. Players with no NHL history (rookies)
    get no projection — documented limitation until Phase 3 adds overlays.
    """
    target_games = season_games(conn, target_season)
    sk_frames: list[tuple[pd.DataFrame, int]] = []
    g_frames: list[tuple[pd.DataFrame, int]] = []
    for season in train_seasons:
        skaters, goalies = season_aggregates(conn, season)
        games = season_games(conn, season)
        sk_frames.append((skaters, games))
        g_frames.append((goalies, games))

    meta = pd.read_sql_query(
        "SELECT player_id, full_name AS name, team_abbrev AS team, position FROM nhl_players",
        conn,
        index_col="player_id",
    )
    skaters = project_skaters(sk_frames, target_games, weights).join(meta, how="left")
    goalies = project_goalies(g_frames, target_games, weights).join(meta, how="left")
    return skaters, goalies
