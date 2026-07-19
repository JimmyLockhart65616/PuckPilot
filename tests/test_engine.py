import pandas as pd
import pytest

from puckpilot.engine import aggregate, valuation
from puckpilot.engine.categories import GOALIE_CATS_DEFAULT, Category
from puckpilot.engine.projections import blend_counting, project_goalies
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
    assert out.loc[1, "wins"] == pytest.approx(0.4625 * 14.4)


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
