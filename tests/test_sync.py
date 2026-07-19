import json
from datetime import date

from puckpilot.data import store, sync

COMPLETED = "20232024"
FAR_FUTURE = "20992100"  # never "complete" -> exercises the current-season path


class FakeNhl:
    """Stands in for NhlClient; both teams report the same game to exercise dedupe."""

    def __init__(self):
        self.game_log_calls = []

    def standings_now(self):
        return {
            "standings": [
                {"teamAbbrev": {"default": "EDM"}},
                {"teamAbbrev": {"default": "CGY"}},
            ]
        }

    def club_schedule_season(self, team_abbrev, season):
        game = {
            "id": 2023020001,
            "season": int(season),
            "gameType": 2,
            "gameDate": "2023-10-08",
            "startTimeUTC": "2023-10-08T23:00:00Z",
            "homeTeam": {"abbrev": "EDM"},
            "awayTeam": {"abbrev": "CGY"},
        }
        return {"games": [game]}

    def player_game_log(self, player_id, season, game_type=2):
        self.game_log_calls.append((player_id, season))
        return {
            "gameLog": [
                {
                    "gameId": 2023020001,
                    "gameDate": "2023-10-08",
                    "teamAbbrev": "EDM",
                    "opponentAbbrev": "CGY",
                    "homeRoadFlag": "H",
                    "goals": 1,
                    "assists": 2,
                }
            ]
        }


class FakeMp:
    def season_csv(self, season, kind, refresh=False):
        if kind == "skaters":
            base = {"playerId": "8478402", "name": "Connor McDavid", "team": "EDM", "position": "C"}
            return [{**base, "situation": "all"}, {**base, "situation": "5on4"}]
        return [
            {
                "playerId": "8479973",
                "name": "Stuart Skinner",
                "team": "EDM",
                "position": "G",
                "situation": "all",
            }
        ]


class UnpublishedMp:
    def season_csv(self, season, kind, refresh=False):
        return None


def _db(tmp_path):
    conn = store.connect(tmp_path / "t.db")
    store.init_db(conn)
    return conn


def test_season_is_complete_boundary():
    assert sync.season_is_complete("20252026", today=date(2026, 7, 1))
    assert not sync.season_is_complete("20252026", today=date(2026, 6, 30))
    assert not sync.season_is_complete("20262027", today=date(2026, 7, 17))


def test_sync_schedules_dedupes_and_is_idempotent(tmp_path):
    conn = _db(tmp_path)
    counts = sync.sync_schedules(conn, FakeNhl(), [COMPLETED], delay=0)
    assert counts == {COMPLETED: 1}
    sync.sync_schedules(conn, FakeNhl(), [COMPLETED], delay=0)
    rows = conn.execute("SELECT * FROM nhl_schedule").fetchall()
    assert len(rows) == 1
    assert rows[0]["home_team"] == "EDM"
    assert rows[0]["away_team"] == "CGY"


def test_sync_players_and_logs_completed_season(tmp_path):
    conn = _db(tmp_path)
    nhl = FakeNhl()
    report = sync.sync_players_and_logs(conn, nhl, FakeMp(), [COMPLETED], delay=0)
    assert report[COMPLETED] == {
        "players": 2,
        "stat_rows": 3,
        "logs_synced": 2,
        "logs_skipped": 0,
    }
    names = {r["full_name"] for r in conn.execute("SELECT * FROM nhl_players")}
    assert names == {"Connor McDavid", "Stuart Skinner"}
    kinds = {
        (r["kind"], r["situation"])
        for r in conn.execute("SELECT kind, situation FROM mp_season_stats")
    }
    assert kinds == {("skater", "all"), ("skater", "5on4"), ("goalie", "all")}
    row = conn.execute("SELECT * FROM nhl_game_logs WHERE player_id=8478402").fetchone()
    assert row["is_home"] == 1
    assert row["opponent_abbrev"] == "CGY"
    assert json.loads(row["stats_json"])["assists"] == 2


def test_completed_season_logs_skipped_on_rerun(tmp_path):
    conn = _db(tmp_path)
    nhl = FakeNhl()
    sync.sync_players_and_logs(conn, nhl, FakeMp(), [COMPLETED], delay=0)
    calls_before = len(nhl.game_log_calls)
    report = sync.sync_players_and_logs(conn, nhl, FakeMp(), [COMPLETED], delay=0)
    assert report[COMPLETED]["logs_skipped"] == 2
    assert report[COMPLETED]["logs_synced"] == 0
    assert len(nhl.game_log_calls) == calls_before


def test_current_season_logs_always_refetched(tmp_path):
    conn = _db(tmp_path)
    nhl = FakeNhl()
    sync.sync_players_and_logs(conn, nhl, FakeMp(), [FAR_FUTURE], delay=0)
    report = sync.sync_players_and_logs(conn, nhl, FakeMp(), [FAR_FUTURE], delay=0)
    assert report[FAR_FUTURE]["logs_synced"] == 2
    assert len(nhl.game_log_calls) == 4
    done_keys = conn.execute("SELECT key FROM sync_meta WHERE value='done'").fetchall()
    assert done_keys == []


def test_unpublished_season_is_skipped(tmp_path):
    conn = _db(tmp_path)
    report = sync.sync_players_and_logs(conn, FakeNhl(), UnpublishedMp(), ["20262027"], delay=0)
    assert report["20262027"]["players"] == 0
    assert conn.execute("SELECT COUNT(*) c FROM nhl_players").fetchone()["c"] == 0
