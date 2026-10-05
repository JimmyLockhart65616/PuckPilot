"""The NHL stats API (api.nhle.com/stats/rest): ice time by situation, per game.

`api-web.nhle.com` - `data/nhl.py` - has each game's total time on ice but
not how much of it was on the power play, and power-play time is the earliest
sign of a role change that power-play points will follow. This endpoint has it
per player per game: `ppTimeOnIce`, `shTimeOnIce`, `evTimeOnIce`, in seconds.

One request covers a whole game date (`limit=-1` returns every row; a shorter
page than `total` falls back to paging). Unofficial, like the other one: the
`live`-marked contract test exists to catch it changing shape.
"""

from __future__ import annotations

import time
from typing import Any

import httpx

BASE_URL = "https://api.nhle.com/stats/rest/en"
RETRYABLE = {429, 500, 502, 503, 504}
PAGE = 100


class NhlStatsError(RuntimeError):
    pass


class NhlStatsClient:
    """Thin client for the NHL stats REST API. Accepts an injected httpx.Client."""

    def __init__(self, http: httpx.Client | None = None):
        self._http = http or httpx.Client(
            base_url=BASE_URL,
            timeout=30.0,
            follow_redirects=True,
            headers={"User-Agent": "puckpilot/0.1 (personal fantasy tool)"},
        )

    def _get(self, path: str, params: dict, retries: int = 3) -> Any:
        last_err = ""
        for attempt in range(retries):
            try:
                resp = self._http.get(path, params=params)
            except httpx.TransportError as e:
                last_err = str(e)
            else:
                if resp.status_code == 200:
                    return resp.json()
                last_err = f"HTTP {resp.status_code}"
                if resp.status_code not in RETRYABLE:
                    break
            if attempt < retries - 1:
                time.sleep(1.5 * (attempt + 1))
        raise NhlStatsError(f"GET {path} failed: {last_err}")

    def skater_toi(self, date: str) -> list[dict]:
        """Every skater's ice time, by situation, in each regular-season game
        played on `date` (YYYY-MM-DD)."""
        params = {
            "isAggregate": "false",
            "isGame": "true",
            "sort": '[{"property":"playerId","direction":"ASC"}]',
            "cayenneExp": f'gameDate>="{date}" and gameDate<="{date}" and gameTypeId=2',
        }
        got = self._get("/skater/timeonice", {**params, "start": 0, "limit": -1})
        rows = list(got.get("data") or [])
        total = int(got.get("total") or 0)
        start = len(rows)
        while start < total:
            page = self._get("/skater/timeonice", {**params, "start": start, "limit": PAGE})
            more = list(page.get("data") or [])
            if not more:
                break
            rows += more
            start += len(more)
        return rows
