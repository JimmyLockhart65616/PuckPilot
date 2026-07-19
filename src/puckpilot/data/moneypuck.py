from __future__ import annotations

import csv
import io
import time
from pathlib import Path

import httpx

BASE_URL = "https://moneypuck.com/moneypuck/playerData/seasonSummary"
RETRYABLE = {429, 500, 502, 503, 504}

KINDS = ("skaters", "goalies")


class MoneyPuckError(RuntimeError):
    pass


def season_start_year(season: str) -> int:
    """'20252026' -> 2025 (MoneyPuck names files by the season's start year)."""
    if len(season) != 8 or not season.isdigit() or int(season[4:]) != int(season[:4]) + 1:
        raise ValueError(f"bad season string: {season!r}")
    return int(season[:4])


class MoneyPuckClient:
    """Downloads MoneyPuck season-summary CSVs with a local file cache.

    CSV layout (verified 2026-07-17): one row per player per situation
    ('all', '5on5', '5on4', '4on5', 'other'); playerId is the NHL player id,
    so these files double as player discovery for game-log syncing.
    """

    def __init__(self, cache_dir: Path, http: httpx.Client | None = None):
        self._cache_dir = cache_dir
        self._http = http or httpx.Client(
            timeout=60.0,
            follow_redirects=True,
            headers={"User-Agent": "puckpilot/0.1 (personal fantasy tool)"},
        )

    def _fetch(self, url: str, retries: int = 3) -> bytes | None:
        """GET a CSV; None on 404 (season not published yet)."""
        last_err = ""
        for attempt in range(retries):
            try:
                resp = self._http.get(url)
            except httpx.TransportError as e:
                last_err = str(e)
            else:
                if resp.status_code == 200:
                    return resp.content
                if resp.status_code == 404:
                    return None
                last_err = f"HTTP {resp.status_code}"
                if resp.status_code not in RETRYABLE:
                    break
            if attempt < retries - 1:
                time.sleep(1.5 * (attempt + 1))
        raise MoneyPuckError(f"GET {url} failed: {last_err}")

    def season_csv(self, season: str, kind: str, refresh: bool = False) -> list[dict] | None:
        """Parsed rows for one season+kind ('skaters'|'goalies').

        Returns None when MoneyPuck has no file for the season (not played yet).
        Rows come back as dicts keyed by the CSV header.
        """
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
        cache = self._cache_dir / f"{season}_{kind}.csv"
        if refresh or not cache.exists():
            year = season_start_year(season)
            data = self._fetch(f"{BASE_URL}/{year}/regular/{kind}.csv")
            if data is None:
                return None
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_bytes(data)
        text = cache.read_text(encoding="utf-8-sig")
        return list(csv.DictReader(io.StringIO(text)))
