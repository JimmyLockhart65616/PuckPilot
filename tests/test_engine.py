import pandas as pd
import pytest

from puckpilot.engine import aggregate, valuation
from puckpilot.engine.categories import GOALIE_CATS_DEFAULT, Category
from puckpilot.engine.projections import (
    AGE_CLIP,
    AGE_PEAK,
    age_factor,
    blend_counting,
    project_goalies,
)
from tests.conftest import add_goalie_game, add_player, add_skater_game


def test_toi_seconds_handles_over_an_hour():
    assert aggregate.toi_seconds("62:13") == 3733
    assert aggregate.toi_seconds("00:45") == 45


def test_season_aggregates_splits_and_sums(db):
    add_player(db, 1, "Skater One", "C")
    add_player(db, 2, "Goalie One", "G")
    add_skater_game(db, 1, "20242025", 100, goals=2, assists=1, shots=5, plusMinus=1)
    add_skater_game(db, 1, "20242025", 101, goals=0, assists=3, shots=2, plusMinus=-2)
    add_goalie_game(
        db, 2, "20242025", 100, shots_against=30, goals_against=3, toi="60:00", decision="W"
    )
    add_goalie_game(
        db, 2, "20242025", 101, shots_against=20, goals_against=0, toi="30:00", shutouts=0
    )

    skaters, goalies = aggregate.season_aggregates(db, "20242025")
    s = skaters.loc[1]
    assert s["gp"] == 2
    assert s["goals"] == 2
    assert s["assists"] == 4
    assert s["sog"] == 7
    assert s["plus_minus"] == -1
    assert s["position"] == "C"

    g = goalies.loc[2]
    assert g["gp"] == 2
    assert g["wins"] == 1
    assert g["shots_against"] == 50
    assert g["save_pct"] == pytest.approx(1 - 3 / 50)
    assert g["gaa"] == pytest.approx(3 / 1.5)  # 3 GA in 90 minutes


def test_season_aggregates_pulls_hits_blocks_from_boxscores(db):
    add_player(db, 1, "Grinder", "L")
    add_skater_game(db, 1, "20242025", 100, goals=1, hits=4, blocks=2)
    add_skater_game(db, 1, "20242025", 101, goals=0, hits=3, blocks=1)
    add_skater_game(db, 1, "20242025", 102, goals=1)  # no boxscore row at all

    skaters, _ = aggregate.season_aggregates(db, "20242025")
    s = skaters.loc[1]
    assert s["gp"] == 3
    assert s["goals"] == 2
    # the game missing its boxscore contributes 0 rather than dropping the game
    assert s["hits"] == 7
    assert s["blocks"] == 3


def test_season_aggregates_derives_goalie_saves(db):
    add_player(db, 2, "Goalie One", "G")
    add_goalie_game(db, 2, "20242025", 100, shots_against=30, goals_against=3)
    add_goalie_game(db, 2, "20242025", 101, shots_against=20, goals_against=0)

    _, goalies = aggregate.season_aggregates(db, "20242025")
    assert goalies.loc[2, "saves"] == 47  # (30-3) + (20-0)


def test_season_games_from_schedule(db):
    from puckpilot.data.store import upsert_schedule_game

    for i, (h, a) in enumerate([("AAA", "BBB"), ("BBB", "AAA"), ("AAA", "CCC")]):
        upsert_schedule_game(
            db,
            game_id=i,
            season="20242025",
            game_type=2,
            game_date="2024-10-01",
            start_time_utc=None,
            home_team=h,
            away_team=a,
        )
    assert aggregate.season_games(db, "20242025") == 3  # AAA plays 3
    assert aggregate.season_games(db, "20992100") == 82  # no schedule -> default


def _frame(rows: dict[int, dict]) -> pd.DataFrame:
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "player_id"
    return df


def test_blend_counting_weighted_rates_and_gp():
    recent = _frame({1: {"gp": 40, "goals": 20}, 3: {"gp": 5, "goals": 1}})
    older = _frame({1: {"gp": 60, "goals": 30}, 2: {"gp": 60, "goals": 15}})
    out = blend_counting(
        [(recent, 80), (older, 80)], (0.5, 0.3), ["goals"], target_games=84, min_train_gp=10
    )
    # player 1: rate 0.5 both seasons; gp frac (0.5*0.5 + 0.3*0.75)/0.8
    assert out.loc[1, "proj_gp"] == pytest.approx(49.9)
    assert out.loc[1, "goals"] == pytest.approx(0.5 * 49.9)
    # player 2 appears only in the older season: weights renormalize to it alone
    assert out.loc[2, "proj_gp"] == pytest.approx(63.0)
    assert out.loc[2, "goals"] == pytest.approx(0.25 * 63.0)
    # player 3: only 5 train GP -> filtered out
    assert 3 not in out.index


def test_project_goalies_volume_weighted_rates():
    recent = _frame(
        {
            1: {
                "gp": 10,
                "wins": 5,
                "shutouts": 1,
                "shots_against": 100,
                "goals_against": 10,
                "toi_hours": 10.0,
            }
        }
    )
    older = _frame(
        {
            1: {
                "gp": 20,
                "wins": 8,
                "shutouts": 0,
                "shots_against": 200,
                "goals_against": 30,
                "toi_hours": 20.0,
            }
        }
    )
    out = project_goalies([(recent, 80), (older, 80)], target_games=84, weights=(0.5, 0.3))
    assert out.loc[1, "save_pct"] == pytest.approx(96 / 110)  # (0.5*90+0.3*170)/(0.5*100+0.3*200)
    assert out.loc[1, "gaa"] == pytest.approx(14 / 11)  # (0.5*10+0.3*30)/(0.5*10+0.3*20)
    assert out.loc[1, "proj_gp"] == pytest.approx(14.4)
    # counting stats are sample-weighted: (0.5*5 + 0.3*8) wins over
    # (0.5*10 + 0.3*20) games, so the 20-game season outweighs the 10-game one.
    # Averaging the per-season rates instead would give 0.4625/game.
    assert out.loc[1, "wins"] == pytest.approx((4.9 / 11.0) * 14.4)
    assert out.loc[1, "saves"] == pytest.approx(
        out.loc[1, "shots_against"] - out.loc[1, "goals_against"]
    )


def test_blend_counting_weights_by_sample_size():
    """A 60-game season must outweigh a 10-game hot streak in the same weight slot."""
    recent = _frame({1: {"gp": 10, "goals": 10}})  # 1.0 g/gm over 10 games
    older = _frame({1: {"gp": 60, "goals": 12}})  # 0.2 g/gm over 60 games
    out = blend_counting([(recent, 82), (older, 82)], (0.5, 0.5), ["goals"], 82, 1)
    rate = out.loc[1, "goals"] / out.loc[1, "proj_gp"]
    # sample-weighted: (0.5*10 + 0.5*12) / (0.5*10 + 0.5*60) = 11/35
    assert rate == pytest.approx(11 / 35)
    assert rate < 0.6  # a plain average of the two rates would give 0.6


def test_gp_regression_shrinks_availability_toward_the_mean():
    """Games played is only ~0.45 repeatable, so an outlier gets pulled in."""
    frames = [
        (_frame({1: {"gp": 82, "goals": 20}, 2: {"gp": 20, "goals": 5}}), 82),
    ]
    raw = blend_counting(frames, (1.0,), ["goals"], 82, 1, gp_regress=0.0)
    shrunk = blend_counting(frames, (1.0,), ["goals"], 82, 1, gp_regress=1.5)
    # the iron man comes down, the part-timer comes up, order preserved
    assert shrunk.loc[1, "proj_gp"] < raw.loc[1, "proj_gp"]
    assert shrunk.loc[2, "proj_gp"] > raw.loc[2, "proj_gp"]
    assert shrunk.loc[1, "proj_gp"] > shrunk.loc[2, "proj_gp"]


def test_blend_counting_tolerates_unsynced_seasons():
    """Configuring 3 training seasons but syncing 2 must degrade, not crash."""
    real = _frame({1: {"gp": 40, "goals": 20}})
    out = blend_counting(
        [(real, 82), (pd.DataFrame(), 82)], (0.5, 0.3), ["goals"], 82, min_train_gp=1
    )
    assert out.loc[1, "goals"] > 0
    empty = blend_counting([(pd.DataFrame(), 82)], (1.0,), ["goals"], 82, 1)
    assert empty.empty and "proj_gp" in empty.columns


def test_age_factor_rewards_youth_and_fades_veterans():
    ages = pd.Series({1: 21.0, 2: AGE_PEAK, 3: 36.0, 4: float("nan")})
    f = age_factor(ages)
    assert f[1] > 1.0  # ascending
    assert f[2] == pytest.approx(1.0)  # at peak
    assert f[3] < 1.0  # declining
    assert f[4] == 1.0  # unknown age is never penalised
    assert f.between(*AGE_CLIP).all()


def test_goalie_wins_blend_toward_team_strength():
    """A goalie on a strong team gets a win bump; one on a weak team gets docked,
    even with identical personal win rates."""
    # two goalies per team so the leave-one-out team rate is non-trivial
    frame = _frame(
        {
            1: {"gp": 40, "wins": 20, "shutouts": 2, "shots_against": 1200,
                "goals_against": 100, "toi_hours": 40.0, "team": "STRONG"},
            2: {"gp": 20, "wins": 12, "shutouts": 1, "shots_against": 600,
                "goals_against": 50, "toi_hours": 20.0, "team": "STRONG"},
            3: {"gp": 40, "wins": 20, "shutouts": 2, "shots_against": 1200,
                "goals_against": 100, "toi_hours": 40.0, "team": "WEAK"},
            4: {"gp": 20, "wins": 4, "shutouts": 0, "shots_against": 600,
                "goals_against": 60, "toi_hours": 20.0, "team": "WEAK"},
        }
    )
    base = project_goalies([(frame, 82)], 82, (1.0,), team_win_blend=0.0)
    blended = project_goalies([(frame, 82)], 82, (1.0,), team_win_blend=0.5)
    # goalies 1 and 3 have identical personal stats (0.5 win rate) but 1's
    # teammate wins more, so 1's projected wins rise above 3's after blending
    assert base.loc[1, "wins"] == pytest.approx(base.loc[3, "wins"])
    assert blended.loc[1, "wins"] > blended.loc[3, "wins"]


def test_age_never_pushes_save_pct_above_one():
    """Age must scale goalie volume, not the derived rates."""
    young = _frame(
        {
            9: {
                "gp": 60,
                "wins": 30,
                "shutouts": 3,
                "shots_against": 1800,
                "goals_against": 150,
                "toi_hours": 60.0,
            }
        }
    )
    out = project_goalies([(young, 82)], 82, (1.0,), ages=pd.Series({9: 20.0}))
    assert 0.0 < out.loc[9, "save_pct"] < 1.0
    assert out.loc[9, "saves"] == pytest.approx(
        out.loc[9, "shots_against"] - out.loc[9, "goals_against"]
    )


def test_value_players_counting_z():
    df = _frame(
        {
            1: {"goals": 10, "position": "C"},
            2: {"goals": 20, "position": "C"},
            3: {"goals": 30, "position": "C"},
        }
    )
    cats = (Category("goals", "G", "skater"),)
    out = valuation.value_players(df, cats, pool_size=3, iters=1)
    assert out.loc[2, "z_goals"] == pytest.approx(0.0)
    assert out.loc[3, "z_goals"] == pytest.approx(1.2247, abs=1e-3)
    assert out.loc[1, "z_total"] == pytest.approx(-1.2247, abs=1e-3)


def test_value_players_gaa_lower_is_better():
    df = _frame(
        {
            1: {"gaa": 1.0, "toi_hours": 10.0, "position": "G"},
            2: {"gaa": 3.0, "toi_hours": 10.0, "position": "G"},
        }
    )
    cats = tuple(c for c in GOALIE_CATS_DEFAULT if c.key == "gaa")
    out = valuation.value_players(df, cats, pool_size=2, iters=1)
    assert out.loc[1, "z_gaa"] > 0 > out.loc[2, "z_gaa"]


def test_replacement_adjust_last_starter_is_zero():
    df = _frame(
        {
            1: {"z_total": 5.0, "position": "C"},
            2: {"z_total": 3.0, "position": "C"},
            3: {"z_total": 1.0, "position": "C"},
        }
    )
    out = valuation.replacement_adjust(df, {"C": 2})
    assert out.loc[2, "vorp"] == pytest.approx(0.0)  # 2nd starter is replacement level
    assert out.loc[1, "vorp"] == pytest.approx(2.0)
    assert out.loc[3, "vorp"] == pytest.approx(-2.0)


def test_rank_players_combines_and_sorts():
    skaters = _frame({1: {"goals": 30, "position": "C"}, 2: {"goals": 10, "position": "C"}})
    goalies = _frame(
        {
            3: {
                "wins": 40,
                "shutouts": 5,
                "gaa": 2.0,
                "save_pct": 0.92,
                "shots_against": 1500.0,
                "toi_hours": 60.0,
                "position": "G",
            },
            4: {
                "wins": 10,
                "shutouts": 0,
                "gaa": 3.5,
                "save_pct": 0.88,
                "shots_against": 800.0,
                "toi_hours": 30.0,
                "position": "G",
            },
        }
    )
    skater_cats = (Category("goals", "G", "skater"),)
    out = valuation.rank_players(skaters, goalies, skater_cats=skater_cats)
    assert set(out["kind"]) == {"skater", "goalie"}
    assert list(out["vorp"]) == sorted(out["vorp"], reverse=True)
    assert out.loc[3, "vorp"] > out.loc[4, "vorp"]  # workhorse beats backup
