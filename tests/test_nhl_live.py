"""Live read-only contract tests against the real NHL API.

Excluded by default (pytest addopts = -m 'not live'). Run with: pytest -m live
Purpose: catch schema drift in the unofficial API early.
"""

import pytest

from puckpilot.data.nhl import NhlClient

pytestmark = pytest.mark.live

MCDAVID = 8478402


@pytest.fixture(scope="module")
def nhl() -> NhlClient:
    return NhlClient()


def test_player_landing_schema(nhl):
    data = nhl.player_landing(MCDAVID)
    assert data["lastName"]["default"] == "McDavid"
    assert data["position"] == "C"


def test_club_schedule_full_season(nhl):
    data = nhl.club_schedule_season("EDM", "20252026")
    regular = [g for g in data["games"] if g["gameType"] == 2]
    assert len(regular) == 82
    sample = regular[0]
    assert {"id", "gameDate", "homeTeam", "awayTeam", "startTimeUTC"} <= set(sample.keys())


def test_player_game_log_schema(nhl):
    data = nhl.player_game_log(MCDAVID, "20252026")
    logs = data["gameLog"]
    assert len(logs) > 40  # played most of the season
    assert {"gameId", "gameDate", "goals", "assists", "points", "shots", "opponentAbbrev"} <= set(
        logs[0].keys()
    )


def test_boxscore_carries_hits_and_blocks(nhl):
    """HIT/BLK are scoring categories but appear nowhere in the game-log endpoint;
    this is the only source, so schema drift here silently zeroes two categories."""
    data = nhl.boxscore(2025020001)
    pbg = data["playerByGameStats"]
    skater = pbg["homeTeam"]["forwards"][0]
    assert {"playerId", "hits", "blockedShots", "goals", "assists", "sog"} <= set(skater.keys())
    goalie = pbg["homeTeam"]["goalies"][0]
    assert {"playerId", "saves", "shotsAgainst", "starter"} <= set(goalie.keys())


def test_standings_has_32_teams(nhl):
    data = nhl.standings_now()
    assert len(data["standings"]) == 32
