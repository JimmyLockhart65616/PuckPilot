from __future__ import annotations

import time
from typing import Any

import httpx

BASE_URL = "https://api-web.nhle.com/v1"
RETRYABLE = {429, 500, 502, 503, 504}

# NHL API game types
PRESEASON, REGULAR_SEASON, PLAYOFFS = 1, 2, 3


class NhlApiError(RuntimeError):
    pass


class NhlClient:
    """Thin client for the NHL web API (unofficial, no auth, JSON GETs).

    Endpoint reference: https://github.com/Zmalski/NHL-API-Reference
    Accepts an injected httpx.Client so tests can mock transport with respx.
    """

    def __init__(self, http: httpx.Client | None = None):
        self._http = http or httpx.Client(
            base_url=BASE_URL,
            timeout=20.0,
            # some endpoints (e.g. /standings/now) answer with a 307 to a dated URL
            follow_redirects=True,
            headers={"User-Agent": "puckpilot/0.1 (personal fantasy tool)"},
        )

    def _get(self, path: str, retries: int = 3) -> Any:
        last_err: str = ""
        for attempt in range(retries):
            try:
                resp = self._http.get(path)
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
        raise NhlApiError(f"GET {path} failed: {last_err}")

    def player_landing(self, player_id: int) -> dict:
        return self._get(f"/player/{player_id}/landing")

    def player_game_log(self, player_id: int, season: str, game_type: int = REGULAR_SEASON) -> dict:
        """season like '20252026'."""
        return self._get(f"/player/{player_id}/game-log/{season}/{game_type}")

    def club_schedule_season(self, team_abbrev: str, season: str) -> dict:
        return self._get(f"/club-schedule-season/{team_abbrev}/{season}")

    def schedule_for_date(self, date: str) -> dict:
        """date like '2026-01-15'; returns the week starting at that date."""
        return self._get(f"/schedule/{date}")

    def boxscore(self, game_id: int) -> dict:
        return self._get(f"/gamecenter/{game_id}/boxscore")

    def standings_now(self) -> dict:
        return self._get("/standings/now")
