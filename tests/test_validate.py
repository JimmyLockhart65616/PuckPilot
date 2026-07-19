import math

from puckpilot.engine.validate import walk_forward
from tests.conftest import add_goalie_game, add_player, add_skater_game

SEASONS = ("20232024", "20242025", "20252026")


def _seed(db):
    """4 skaters with stable ordering + 2 goalies, 12 games each season."""
    for i, (pid, goals_per_game) in enumerate([(1, 1.0), (2, 0.75), (3, 0.5), (4, 0.25)]):
        add_player(db, pid, f"Skater {i}", "C")
        for si, season in enumerate(SEASONS):
            for g in range(12):
                gid = int(f"{si + 1}{pid}{g:02d}")
                add_skater_game(
                    db,
                    pid,
                    season,
                    gid,
                    goals=round(goals_per_game * (1 if g % 2 else 0) * 2),
                    assists=1,
                    shots=3,
                )
    for pid, ga in [(10, 1), (11, 3)]:
        add_player(db, pid, f"Goalie {pid}", "G")
        for si, season in enumerate(SEASONS):
            for g in range(12):
                gid = int(f"{si + 1}{pid}{g:02d}")
                add_goalie_game(
                    db,
                    pid,
                    season,
                    gid,
                    shots_against=30,
                    goals_against=ga,
                    decision="W" if ga == 1 else "L",
                )


def test_walk_forward_smoke(db):
    _seed(db)
    metrics, text = walk_forward(db)
    assert metrics["target"] == "20252026"
    assert metrics["mae_skaters"]["n"] == 4
    assert metrics["mae_goalies"]["n"] == 2
    # stable synthetic ordering -> near-perfect skater rank correlation
    assert metrics["spearman"]["skaters"] > 0.9
    assert math.isfinite(metrics["spearman"]["overall"])
    assert "Walk-forward validation" in text
    assert "Spearman" in text


def test_walk_forward_gp_mae_reflects_projection(db):
    _seed(db)
    metrics, _ = walk_forward(db)
    # every player repeats 12 GP each season; projected GP should be close to actual
    assert metrics["mae_skaters"]["gp"] < 1.0
