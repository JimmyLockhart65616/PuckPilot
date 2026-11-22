"""A goalie start priced by its game."""

from __future__ import annotations

import json

import pytest

from puckpilot.data import store
from puckpilot.season import start_pricing as sp

S = "20252026"


def _goalie_line(db, pid, gid, date, team, opp, home, sa, ga, decision=None, started=1):
    s = {"gamesStarted": started, "shotsAgainst": sa, "goalsAgainst": ga}
    if decision:
        s["decision"] = decision
    store.upsert_game_log(
        db, player_id=pid, game_id=gid, season=S, game_type=2, game_date=date,
        team_abbrev=team, opponent_abbrev=opp, is_home=int(home), stats_json=json.dumps(s),
    )  # fmt: skip


def _game(db, gid, date, home, away, sa_home, sa_away, home_wins=True):
    _goalie_line(db, 100 + gid, gid, date, home, away, True, sa_home, 2,
                 "W" if home_wins else "L")  # fmt: skip
    _goalie_line(db, 200 + gid, gid, date, away, home, False, sa_away, 3,
                 "L" if home_wins else "W")  # fmt: skip
    store.upsert_schedule_game(
        db, game_id=gid, season=S, game_type=2, game_date=date, start_time_utc=None,
        home_team=home, away_team=away,
    )  # fmt: skip


def test_team_games_sum_a_clubs_goalies_and_carry_the_result(db):
    _game(db, 1, "2025-10-10", "AAA", "BBB", sa_home=30, sa_away=20)
    _goalie_line(db, 999, 1, "2025-10-10", "AAA", "BBB", True, 5, 1, started=0)  # relief
    got = {g.team: g for g in sp.team_games(db, S)}
    assert (got["AAA"].shots_against, got["AAA"].goals_against) == (35.0, 3.0)
    assert got["AAA"].win and not got["BBB"].win and got["AAA"].home


def test_an_opponent_who_shoots_more_raises_the_shots_a_goalie_faces(db):
    # AAA's opponents faced 40 shots a game from AAA; CCC's faced 20.
    for i, d in enumerate(("2025-10-10", "2025-10-12", "2025-10-14")):
        _game(db, 10 + i, d, "AAA", "BBB", sa_home=30, sa_away=40)
        _game(db, 20 + i, d, "CCC", "DDD", sa_home=30, sa_away=20)
    _game(db, 30, "2025-10-20", "BBB", "AAA", sa_home=0, sa_away=0)
    _game(db, 31, "2025-10-20", "DDD", "CCC", sa_home=0, sa_away=0)
    o = sp.OpponentShots(db, S)
    r = o.ratios("2025-10-15")
    assert r["AAA"] > 1.0 > r["CCC"]
    assert o.opponent("2025-10-20", "BBB") == "AAA"
    f_vs_aaa = o.factor("BBB", ["2025-10-20"], [1.0], "2025-10-15")
    f_vs_ccc = o.factor("DDD", ["2025-10-20"], [1.0], "2025-10-15")
    assert f_vs_aaa == pytest.approx(r["AAA"]) and f_vs_ccc == pytest.approx(r["CCC"])
    assert o.factor("BBB", [], [], "2025-10-15") == 1.0


def test_shot_lines_are_scaled_only_for_goalies_with_games():
    from types import SimpleNamespace

    from puckpilot.season.week import _shots_scaled

    opp = SimpleNamespace(factor=lambda club, dates, weights, cutoff: 1.2)
    rates = {1: {"shots_against": 25.0, "saves": 22.5, "wins": 0.5}, 2: {"goals": 0.4}}
    players = [SimpleNamespace(nhl_player_id=1, team="AAA", position="G"),
               SimpleNamespace(nhl_player_id=2, team="AAA", position="C")]  # fmt: skip
    got = _shots_scaled(rates, opp, players, {1: ["d"]}, {1: [0.8]}, "d")
    assert got[1]["shots_against"] == pytest.approx(30.0) and got[1]["saves"] == pytest.approx(27.0)
    assert got[1]["wins"] == 0.5 and got[2] == rates[2] and rates[1]["saves"] == 22.5
