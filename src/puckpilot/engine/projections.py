from __future__ import annotations

import sqlite3

import pandas as pd

from puckpilot.engine.aggregate import SKATER_COLS, season_aggregates, season_games

# Marcel-style season weights, most recent season first. Deliberately simple:
# the walk-forward report (validate.py) is the evidence for anything fancier.
# Heavier recency (0.6/0.3/0.1) scored better on the tuning fits but WORSE on the
# held-out season, so it was rejected.
DEFAULT_WEIGHTS = (0.5, 0.3, 0.2)

MIN_TRAIN_GP_SKATER = 10
MIN_TRAIN_GP_GOALIE = 5

# Availability is only ~0.45 repeatable year over year, so a skater's own games
# played is a weak signal and gets shrunk toward the league average. Goalies are
# the opposite: workload is their single most repeatable trait (rho 0.53), so
# theirs is left alone.
GP_REGRESS_SEASONS = 1.5
GOALIE_GP_REGRESS_SEASONS = 0.0

# Multiplicative aging curve. The young-improvement slope carries most of the
# value: young=.04/old=.02 scored 0.795 while the reverse scored 0.761.
AGE_PEAK = 26.0
AGE_YOUNG_SLOPE = 0.04
AGE_OLD_SLOPE = 0.02
AGE_CLIP = (0.5, 1.5)

GOALIE_COUNT_COLS = ["wins", "shutouts"]
GOALIE_VOLUME_COLS = ["shots_against", "goals_against", "toi_hours"]

# A goalie's own win total is noisy (wins depend on the team scoring for them),
# while team strength persists year to year, so wins project best as a blend of
# the goalie's own win rate and their team's. 0.5 sits on a broad plateau
# (0.45-0.65 all beat the self-only model on both tuning and holdout goalie VORP).
GOALIE_TEAM_WIN_BLEND = 0.5


def blend_counting(
    frames: list[tuple[pd.DataFrame, int]],
    weights: tuple[float, ...],
    cat_cols: list[str],
    target_games: int,
    min_train_gp: int,
    gp_regress: float = 0.0,
) -> pd.DataFrame:
    """Blend per-game rates and a GP fraction across seasons into projected totals.

    frames: [(per-player frame with 'gp' + cat_cols, games_in_that_season), ...],
    most recent first, aligned with weights.

    Counting stats are summed with the season weight and divided by weighted
    GAMES, so a 70-game season outweighs a 12-game one — a plain average of
    per-season rates lets a hot cameo count as much as a full year.

    gp_regress shrinks the availability fraction toward the league mean by that
    many seasons' worth of evidence.
    """
    # a season with no data yet (not synced, or not started) contributes nothing
    # rather than crashing, so fewer training seasons degrades gracefully
    frames = [(df, games) for df, games in frames if not df.empty and "gp" in df.columns]
    idx = pd.Index([], name="player_id")
    for df, _ in frames:
        idx = idx.union(df.index)
    if not frames:
        return pd.DataFrame(columns=["proj_gp", *cat_cols])

    stat_sum = pd.DataFrame(0.0, index=idx, columns=cat_cols)
    game_sum = pd.Series(0.0, index=idx)
    frac_sum = pd.Series(0.0, index=idx)
    w_sum = pd.Series(0.0, index=idx)
    gp_total = pd.Series(0.0, index=idx)

    for (df, games), w in zip(frames, weights, strict=False):
        d = df.reindex(idx)
        gp = d["gp"].fillna(0.0)
        played = (gp > 0).astype(float)
        stat_sum += d[cat_cols].fillna(0.0).mul(w, axis=0)
        game_sum += w * gp
        frac_sum += (gp / games) * (w * played)
        w_sum += w * played
        gp_total += gp

    keep = (gp_total >= min_train_gp) & (w_sum > 0)
    rate = stat_sum[keep].div(game_sum[keep].where(game_sum[keep] > 0), axis=0).fillna(0.0)

    frac = (frac_sum[keep] / w_sum[keep]).clip(upper=1.0)
    if gp_regress:
        seasons_seen = w_sum[keep]
        frac = (frac * seasons_seen + frac.mean() * gp_regress) / (seasons_seen + gp_regress)
    proj_gp = (frac * target_games).round(1)

    out = rate.mul(proj_gp, axis=0)
    out.insert(0, "proj_gp", proj_gp)
    # How much NHL evidence this projection actually stands on. Computed here
    # anyway as the min-GP gate, and previously discarded - but it is the honest
    # answer to "how far should I trust this number", and it is the quantity
    # that separates a genuine disagreement with the market from a player we
    # simply cannot see. A 20-year-old with 12 games and a 26-year-old with 12
    # games are the same epistemic problem; age alone would not say so.
    out.insert(1, "train_gp", gp_total[keep].round(0))
    return out


def player_ages(conn: sqlite3.Connection, season: str) -> pd.Series:
    """Age on Feb 1 of the season's second year. Empty when bios are unsynced."""
    rows = conn.execute(
        "SELECT player_id, birth_date FROM nhl_player_bio WHERE birth_date IS NOT NULL"
    ).fetchall()
    if not rows:
        return pd.Series(dtype=float)
    born = pd.Series({r[0]: r[1] for r in rows})
    ref = pd.Timestamp(int(season[4:]), 2, 1)
    return (ref - pd.to_datetime(born)).dt.days / 365.25


def age_factor(
    ages: pd.Series,
    peak: float = AGE_PEAK,
    young: float = AGE_YOUNG_SLOPE,
    old: float = AGE_OLD_SLOPE,
) -> pd.Series:
    """Production multiplier by age; 1.0 at peak and for unknown ages."""
    f = pd.Series(1.0, index=ages.index, dtype=float)
    rising = ages < peak
    f[rising] = 1.0 + young * (peak - ages[rising])
    f[~rising] = 1.0 - old * (ages[~rising] - peak)
    return f.clip(*AGE_CLIP).fillna(1.0)


def _apply_age(df: pd.DataFrame, ages: pd.Series) -> pd.DataFrame:
    if df.empty or ages.empty:
        return df
    f = age_factor(ages.reindex(df.index)).fillna(1.0)
    cols = [c for c in df.columns if c != "proj_gp"]
    df[cols] = df[cols].mul(f, axis=0)
    return df


def project_skaters(
    frames: list[tuple[pd.DataFrame, int]],
    target_games: int,
    weights: tuple[float, ...] = DEFAULT_WEIGHTS,
) -> pd.DataFrame:
    return blend_counting(
        frames, weights, SKATER_COLS, target_games, MIN_TRAIN_GP_SKATER, GP_REGRESS_SEASONS
    )


def _team_win_rate_by_goalie(frame: pd.DataFrame) -> pd.Series:
    """Per-goalie leave-one-out team win rate: their team's wins-per-game with
    the goalie's own games removed, so a workhorse isn't blended against their
    own record. Indexed by player_id."""
    if frame.empty or "team" not in frame or "wins" not in frame:
        return pd.Series(dtype=float)
    team = frame["team"]
    team_w = frame.groupby("team")["wins"].sum()
    team_g = frame.groupby("team")["gp"].sum()
    loo_w = team.map(team_w).fillna(0.0) - frame["wins"].fillna(0.0)
    loo_g = team.map(team_g).fillna(0.0) - frame["gp"].fillna(0.0)
    return loo_w / loo_g.where(loo_g > 0)


def project_goalies(
    frames: list[tuple[pd.DataFrame, int]],
    target_games: int,
    weights: tuple[float, ...] = DEFAULT_WEIGHTS,
    ages: pd.Series | None = None,
    team_win_blend: float = GOALIE_TEAM_WIN_BLEND,
) -> pd.DataFrame:
    """Counting cats blend like skaters; SV%, GAA and SV derive from projected
    volume AFTER any age adjustment, so the rates and the totals can never
    disagree (and an age multiplier can never push SV% past 1.0).

    Wins are additionally blended toward team strength (see GOALIE_TEAM_WIN_BLEND),
    using the most recent training season's team win rates.
    """
    out = blend_counting(
        frames,
        weights,
        GOALIE_COUNT_COLS + GOALIE_VOLUME_COLS,
        target_games,
        MIN_TRAIN_GP_GOALIE,
        GOALIE_GP_REGRESS_SEASONS,
    )
    if ages is not None:
        out = _apply_age(out, ages)

    if team_win_blend and frames and not out.empty:
        proj_gp = out["proj_gp"].where(out["proj_gp"] > 0)
        own_wr = (out["wins"] / proj_gp).reindex(out.index)
        team_wr = _team_win_rate_by_goalie(frames[0][0]).reindex(out.index)
        team_wr = team_wr.fillna(own_wr.mean())
        blended = (1 - team_win_blend) * own_wr + team_win_blend * team_wr
        out["wins"] = (blended.fillna(own_wr) * out["proj_gp"]).fillna(0.0)

    sa, ga = out["shots_against"], out["goals_against"]
    out["save_pct"] = (1.0 - ga / sa.where(sa > 0)).fillna(0.0)
    out["gaa"] = (ga / out["toi_hours"].where(out["toi_hours"] > 0)).fillna(0.0)
    out["saves"] = sa - ga
    return out


def project(
    conn: sqlite3.Connection,
    target_season: str,
    train_seasons: list[str],
    weights: tuple[float, ...] = DEFAULT_WEIGHTS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(skaters, goalies) projected category totals for target_season.

    train_seasons most recent first. Players with no NHL history (rookies)
    get no projection — see docs/STATUS.md for the measured size of that gap.
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
    ages = player_ages(conn, target_season)
    skaters = _apply_age(project_skaters(sk_frames, target_games, weights), ages)
    goalies = project_goalies(g_frames, target_games, weights, ages=ages)
    # Age is consumed by `_apply_age` and then thrown away, so nothing
    # downstream can tell a fading veteran from an unproven youngster. Carry it.
    meta = meta.join(ages.rename("age"), how="left")
    return skaters.join(meta, how="left"), goalies.join(meta, how="left")
