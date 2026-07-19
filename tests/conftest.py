import json

import pytest

from puckpilot.data import store


@pytest.fixture
def db(tmp_path):
    conn = store.connect(tmp_path / "t.db")
    store.init_db(conn)
    return conn


def add_player(conn, pid, name, pos, team="AAA"):
    store.upsert_player(conn, pid, name, pos, team)


def add_skater_game(conn, pid, season, game_id, date="2024-01-01", **stats):
    entry = {
        "gameId": game_id,
        "gameDate": date,
        "teamAbbrev": "AAA",
        "opponentAbbrev": "BBB",
        "homeRoadFlag": "H",
        "goals": 0,
        "assists": 0,
        "points": 0,
        "plusMinus": 0,
        "pim": 0,
        "powerPlayPoints": 0,
        "shots": 0,
        "shorthandedPoints": 0,
        "gameWinningGoals": 0,
        **stats,
    }
    store.upsert_game_log(
        conn,
        player_id=pid,
        game_id=game_id,
        season=season,
        game_type=2,
        game_date=date,
        team_abbrev="AAA",
        opponent_abbrev="BBB",
        is_home=1,
        stats_json=json.dumps(entry),
    )


def add_goalie_game(
    conn,
    pid,
    season,
    game_id,
    date="2024-01-01",
    shots_against=30,
    goals_against=2,
    toi="60:00",
    decision=None,
    started=1,
    shutouts=0,
):
    entry = {
        "gameId": game_id,
        "gameDate": date,
        "teamAbbrev": "AAA",
        "opponentAbbrev": "BBB",
        "homeRoadFlag": "H",
        "gamesStarted": started,
        "shotsAgainst": shots_against,
        "goalsAgainst": goals_against,
        "shutouts": shutouts,
        "toi": toi,
    }
    if decision is not None:
        entry["decision"] = decision
    store.upsert_game_log(
        conn,
        player_id=pid,
        game_id=game_id,
        season=season,
        game_type=2,
        game_date=date,
        team_abbrev="AAA",
        opponent_abbrev="BBB",
        is_home=1,
        stats_json=json.dumps(entry),
    )
