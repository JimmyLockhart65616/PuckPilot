import json
from datetime import date

from puckpilot.data import store, sync

COMPLETED = "20232024"
FAR_FUTURE = "20992100"  # never "complete" -> exercises the current-season path


class FakeNhl:
    """Stands in for NhlClient; both teams report the same game to exercise dedupe."""

    def __init__(self):
        self.game_log_calls = []
        self.boxscore_calls = []

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

    def boxscore(self, game_id):
        self.boxscore_calls.append(game_id)
        return {
            "awayTeam": {"abbrev": "CGY"},
            "homeTeam": {"abbrev": "EDM"},
            "playerByGameStats": {
                "homeTeam": {
                    "forwards": [{"playerId": 8478402, "hits": 3, "blockedShots": 1}],
                    "defense": [{"playerId": 8477498, "hits": 5, "blockedShots": 4}],
                    "goalies": [{"playerId": 8479973, "saves": 28, "starter": True}],
                },
                "awayTeam": {
                    "forwards": [{"playerId": 8480012, "hits": 2, "blockedShots": 0}],
                    "defense": [],
                    "goalies": [],
                },
            },
        }

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


def test_sync_boxscores_stores_hits_and_blocks(tmp_path):
    conn = _db(tmp_path)
    nhl = FakeNhl()
    sync.sync_schedules(conn, nhl, [COMPLETED], delay=0)
    report = sync.sync_boxscores(conn, nhl, [COMPLETED], delay=0, today=date(2024, 6, 1))

    assert report[COMPLETED]["synced"] == 1
    assert report[COMPLETED]["rows"] == 4  # both teams, all three player groups
    row = conn.execute("SELECT * FROM nhl_boxscore_stats WHERE player_id = 8477498").fetchone()
    assert row["team_abbrev"] == "EDM"
    stats = json.loads(row["stats_json"])
    assert (stats["hits"], stats["blockedShots"]) == (5, 4)


def test_sync_boxscores_is_incremental_and_skips_unplayed(tmp_path):
    conn = _db(tmp_path)
    nhl = FakeNhl()
    sync.sync_schedules(conn, nhl, [COMPLETED], delay=0)

    # game_date 2023-10-08 is still in the future -> nothing to fetch yet
    report = sync.sync_boxscores(conn, nhl, [COMPLETED], delay=0, today=date(2023, 10, 1))
    assert report[COMPLETED]["games"] == 0
    assert nhl.boxscore_calls == []

    sync.sync_boxscores(conn, nhl, [COMPLETED], delay=0, today=date(2024, 6, 1))
    assert len(nhl.boxscore_calls) == 1
    report = sync.sync_boxscores(conn, nhl, [COMPLETED], delay=0, today=date(2024, 6, 1))
    assert report[COMPLETED]["skipped"] == 1
    assert len(nhl.boxscore_calls) == 1  # not refetched


def test_unpublished_season_is_skipped(tmp_path):
    conn = _db(tmp_path)
    report = sync.sync_players_and_logs(conn, FakeNhl(), UnpublishedMp(), ["20262027"], delay=0)
    assert report["20262027"]["players"] == 0
    assert conn.execute("SELECT COUNT(*) c FROM nhl_players").fetchone()["c"] == 0


# ---- current rosters -------------------------------------------------------
#
# `upsert_player` is INSERT OR REPLACE and is called once per synced season, so
# whichever season ran last decided every player's team. That left 484 of 1289
# players carrying a club they no longer play for - and it is not cosmetic,
# because projections blend goalie wins on team strength at 0.5.


class RosterNhl:
    """Two clubs, and a player who moved between them."""

    def __init__(self):
        self.roster_calls = []

    def standings_now(self):
        return {
            "standings": [{"teamAbbrev": {"default": "DAL"}}, {"teamAbbrev": {"default": "COL"}}]
        }

    def roster(self, team_abbrev, season):
        self.roster_calls.append((team_abbrev, season))
        if team_abbrev == "DAL":
            return {"forwards": [{"id": 1}], "defensemen": [], "goalies": [{"id": 3}]}
        return {"forwards": [{"id": 2}], "defensemen": [], "goalies": []}


def test_sync_current_rosters_moves_a_traded_player(db):
    store.upsert_player(db, 1, "Moved Winger", "R", "COL")  # stale: now on DAL
    store.upsert_player(db, 2, "Stayed Centre", "C", "COL")
    store.upsert_player(db, 3, "Moved Goalie", "G", "ANA")

    result = sync.sync_current_rosters(db, RosterNhl(), "20262027", delay=0)

    teams = dict(db.execute("SELECT player_id, team_abbrev FROM nhl_players"))
    assert teams[1] == "DAL", "a traded player must follow his new club"
    assert teams[2] == "COL", "a player who did not move must not churn"
    assert teams[3] == "DAL"
    assert result["changed"] == 2, "only the two who actually moved count as changed"
    assert result["players"] == 3


def test_updating_a_team_does_not_clobber_name_or_position(db):
    """The whole reason this is not `upsert_player`: a whole-row replace is what
    caused the staleness in the first place."""
    store.upsert_player(db, 1, "Moved Winger", "R", "COL")
    store.update_player_team(db, 1, "DAL")
    row = tuple(list(db.execute("SELECT full_name, position, team_abbrev FROM nhl_players"))[0])
    assert row == ("Moved Winger", "R", "DAL")


def test_a_club_with_no_roster_for_that_season_is_skipped(db):
    """Defunct franchises and expansion clubs must not abort the sync."""
    from puckpilot.data.nhl import NhlApiError

    class Missing(RosterNhl):
        def roster(self, team_abbrev, season):
            if team_abbrev == "COL":
                raise NhlApiError("404")
            return super().roster(team_abbrev, season)

    store.upsert_player(db, 1, "Moved Winger", "R", "COL")
    result = sync.sync_current_rosters(db, Missing(), "20262027", delay=0)
    assert result["teams"] == 2 and result["changed"] == 1
