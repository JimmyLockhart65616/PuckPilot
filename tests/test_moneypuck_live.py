"""Live contract tests: MoneyPuck CSV layout and the published 2026-27 schedule.

Excluded by default (pytest addopts = -m 'not live'). Run with: pytest -m live
"""

import pytest

from puckpilot.data.moneypuck import MoneyPuckClient
from puckpilot.data.nhl import NhlClient

pytestmark = pytest.mark.live

MCDAVID = "8478402"
SKINNER = 8479973


def test_skaters_csv_layout(tmp_path):
    rows = MoneyPuckClient(cache_dir=tmp_path).season_csv("20252026", "skaters")
    assert rows is not None
    header = set(rows[0])
    assert {"playerId", "season", "name", "team", "position", "situation", "games_played"} <= header
    mcdavid_all = [r for r in rows if r["playerId"] == MCDAVID and r["situation"] == "all"]
    assert len(mcdavid_all) == 1


def test_goalies_csv_layout(tmp_path):
    rows = MoneyPuckClient(cache_dir=tmp_path).season_csv("20252026", "goalies")
    assert rows is not None
    assert {"playerId", "name", "team", "situation", "games_played"} <= set(rows[0])


def test_2026_27_schedule_is_published():
    data = NhlClient().club_schedule_season("EDM", "20262027")
    regular = [g for g in data["games"] if g["gameType"] == 2]
    assert len(regular) >= 82


def test_goalie_game_log_has_starter_fields():
    log = NhlClient().player_game_log(SKINNER, "20252026")["gameLog"]
    assert all({"gameId", "gameDate", "gamesStarted"} <= set(e) for e in log)
    # 'decision' is omitted for no-decision appearances (e.g. pulled mid-game)
    assert any("decision" in e for e in log)
