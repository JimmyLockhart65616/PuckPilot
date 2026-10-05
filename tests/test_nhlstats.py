"""The NHL stats API client and the ice-time sync, against mocked transport."""

from __future__ import annotations

import httpx
import respx

from puckpilot.data import store
from puckpilot.data.nhlstats import BASE_URL, NhlStatsClient
from puckpilot.data.sync import sync_skater_toi


def _row(pid, gid=2025020100, pp=82, toi=1000):
    return {
        "playerId": pid,
        "gameId": gid,
        "timeOnIce": toi,
        "ppTimeOnIce": pp,
        "shTimeOnIce": 0,
        "evTimeOnIce": toi - pp,
    }


@respx.mock
def test_one_request_brings_a_whole_date():
    route = respx.get(f"{BASE_URL}/skater/timeonice").respond(
        json={"data": [_row(1), _row(2)], "total": 2}
    )
    rows = NhlStatsClient().skater_toi("2025-11-01")
    assert [r["playerId"] for r in rows] == [1, 2]
    assert route.call_count == 1
    params = route.calls[0].request.url.params
    assert params["limit"] == "-1"
    where = params["cayenneExp"]
    assert 'gameDate>="2025-11-01"' in where and "gameTypeId=2" in where


@respx.mock
def test_a_short_page_falls_back_to_paging():
    """If the API ever caps `limit=-1`, the rest is fetched page by page."""

    def answer(request: httpx.Request):
        start = int(request.url.params["start"])
        rows = [_row(i) for i in range(start, min(start + 100, 150))]
        return httpx.Response(200, json={"data": rows, "total": 150})

    respx.get(f"{BASE_URL}/skater/timeonice").mock(side_effect=answer)
    rows = NhlStatsClient().skater_toi("2025-11-01")
    assert [r["playerId"] for r in rows] == list(range(150))


def _date(db, gid, d, season="20252026"):
    store.upsert_schedule_game(
        db,
        game_id=gid,
        season=season,
        game_type=2,
        game_date=d,
        start_time_utc=None,
        home_team="TOR",
        away_team="MTL",
    )


class _Stats:
    def __init__(self, by_date):
        self.by_date = by_date
        self.asked: list[str] = []

    def skater_toi(self, date):
        self.asked.append(date)
        return self.by_date.get(date, [])


def test_ice_time_is_stored_once_per_date_and_empty_dates_retried(db):
    from datetime import date

    _date(db, 1, "2025-11-01")
    _date(db, 2, "2025-11-02")
    _date(db, 3, "2025-11-05")  # not played yet on the 'today' below
    db.commit()
    stats = _Stats({"2025-11-01": [_row(7, gid=1, pp=120)]})
    out = sync_skater_toi(db, stats, ["20252026"], delay=0, today=date(2025, 11, 3))
    assert out["20252026"] == {"dates": 1, "skipped": 0, "empty": 1, "rows": 1}
    row = db.execute(
        "SELECT season, game_date, pp_toi_s, toi_s FROM nhl_skater_toi WHERE player_id = 7"
    ).fetchone()
    assert tuple(row) == ("20252026", "2025-11-01", 120, 1000)
    # The 1st is done; the empty 2nd is asked again; the 5th is not played.
    stats.asked.clear()
    sync_skater_toi(db, stats, ["20252026"], delay=0, today=date(2025, 11, 3))
    assert stats.asked == ["2025-11-02"]
