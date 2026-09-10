"""DIAGNOSTIC FALLBACK: session-cookie reads of Yahoo's fantasy web API.

**This is not the supported integration and must not become one.**
`yahoo/client.py` (OAuth + `yahoo_fantasy_api`) is the real path. This module
exists only because our Fantasy API application is still pending, so the
documented endpoints answer 401 `additional_authorization_required` while the
site's own API answers 200 for the same account.

Ground rules it enforces, because "temporary" is how this kind of thing becomes
permanent:

- **Read-only.** Nothing here posts, drafts, or changes league state. There is
  no write method and there must never be one; writes go through OAuth or not
  at all.
- **Only the signed-in user's own leagues.** No enumeration of other accounts,
  no public-league crawling.
- **Throttled.** `MIN_INTERVAL_S` between requests, so a paging loop cannot turn
  into a burst.
- **Self-retiring.** `require_fallback_still_needed()` refuses to run once OAuth
  works. When the application is approved this module stops functioning by
  design rather than by anyone remembering to remove it.
- **Interactive use only.** A session cookie needs a logged-in browser alive,
  which is fine while sitting at a draft and wrong for a scheduled job. Do not
  wire this into cron.

Requests are issued from inside a page on a Yahoo origin via `fetch(...,
{credentials: 'include'})` - the same call the site itself makes - so cookies are
never extracted, decrypted, or stored anywhere by us.
"""

from __future__ import annotations

import contextlib
import json
import os
import time
from pathlib import Path
from typing import Any

BASE = "https://pub-api-ro.fantasysports.yahoo.com/fantasy/v2"
# Politeness floor between requests. `playermap.fetch_players` pages through the
# pool, and a tight loop against an undocumented endpoint is how a fallback that
# is tolerated becomes one that is blocked.
MIN_INTERVAL_S = 0.75
# Any Yahoo fantasy page works as the origin; the lobby is light and always up.
ORIGIN = "https://hockey.fantasysports.yahoo.com/hockey/mock_lobby"

# Structured fields worth keeping whole rather than flattening away.
KEEP_WHOLE = ("roster_positions", "stat_categories", "bye_weeks")

FETCH_JS = """async (u) => {
    const r = await fetch(u, {credentials: 'include'});
    return {status: r.status, body: await r.text()};
}"""


class YahooSessionError(RuntimeError):
    pass


class FallbackNoLongerNeeded(YahooSessionError):
    """Raised when OAuth works, so this module should not be used."""


def require_fallback_still_needed(settings=None) -> None:
    """Refuse to use the cookie path once the documented API is available.

    Called on entry rather than left to a comment. A network failure or a
    missing token is *not* proof that OAuth works, so anything short of a
    confirmed 200 lets the fallback proceed.
    """
    try:
        from puckpilot.config import Settings
        from puckpilot.yahoo.probe import probe

        result = probe(settings or Settings())
    except Exception:
        return  # cannot tell; the fallback is still the only option we have
    if result.scope_granted:
        raise FallbackNoLongerNeeded(
            "Yahoo OAuth now returns 200 - the Fantasy API application is approved. "
            "Use yahoo/client.py (the supported path); this cookie fallback is retired. "
            "Set PUCKPILOT_ALLOW_SESSION_FALLBACK=1 only to debug a discrepancy."
        )


class NotLoggedIn(YahooSessionError):
    def __init__(self, path: str):
        super().__init__(
            f"Yahoo returned 401 for {path}. The capture profile is not logged in "
            "(or the session expired).\n"
            "Run: ppilot draft capture --profile   and sign into Yahoo, then retry."
        )


def flatten(obj: Any, out: dict | None = None) -> dict:
    """Collapse Yahoo's dict/list-of-dicts hybrids into one mapping.

    Yahoo returns objects as lists of single-key dicts interleaved with real
    dicts, so a plain `d["draft_time"]` fails unpredictably depending on where
    in the response the field landed.
    """
    out = {} if out is None else out
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (str, int, float)) or k in KEEP_WHOLE:
                out.setdefault(k, v)
            elif k == "eligible_positions" and isinstance(v, list):
                # [{"position": "C"}, {"position": "LW"}] -> ["C", "LW"];
                # multi-position eligibility is what the lineup optimizer exists for.
                out.setdefault(k, [p.get("position") if isinstance(p, dict) else p for p in v])
            elif isinstance(v, (dict, list)):
                # Nested a level deeper: `name` as a dict, `draft_analysis` as a
                # list. The latter carries average_pick - Yahoo's real ADP.
                flatten(v, out)
    elif isinstance(obj, list):
        for item in obj:
            flatten(item, out)
    return out


class YahooSession:
    """A logged-in browser profile, used as a read-only API client.

    Use as a context manager; the browser is headless by default. Entering
    checks that the OAuth path is still unavailable, so this retires itself the
    moment the Fantasy API application is approved.
    """

    def __init__(self, user_data_dir: Path, headless: bool = True, check_oauth: bool = True):
        self.user_data_dir = Path(user_data_dir)
        self.headless = headless
        self.check_oauth = check_oauth
        self._pw = None
        self._ctx = None
        self._page = None
        self._last_request = 0.0

    def __enter__(self) -> YahooSession:
        from playwright.sync_api import sync_playwright

        if self.check_oauth and not os.environ.get("PUCKPILOT_ALLOW_SESSION_FALLBACK"):
            require_fallback_still_needed()
        self._pw = sync_playwright().start()
        self._ctx = self._pw.chromium.launch_persistent_context(
            str(self.user_data_dir), headless=self.headless, channel="chrome"
        )
        self._page = self._ctx.pages[0] if self._ctx.pages else self._ctx.new_page()
        self._page.goto(ORIGIN, wait_until="domcontentloaded")
        return self

    def __exit__(self, *exc) -> None:
        for close in (
            getattr(self._ctx, "close", None),
            getattr(self._pw, "stop", None),
        ):
            if close:
                with contextlib.suppress(Exception):
                    close()

    def get(self, path: str) -> dict:
        """GET a /fantasy/v2 path and parse the JSON body.

        Read-only by construction: there is no verb parameter and no body.
        """
        if self._page is None:
            raise YahooSessionError("YahooSession must be used as a context manager")
        wait = MIN_INTERVAL_S - (time.monotonic() - self._last_request)
        if wait > 0:
            time.sleep(wait)
        self._last_request = time.monotonic()
        sep = "&" if "?" in path else "?"
        url = f"{BASE}/{path.lstrip('/')}{sep}format=json"
        res = self._page.evaluate(FETCH_JS, url)
        if res["status"] == 401:
            raise NotLoggedIn(path)
        if res["status"] != 200:
            raise YahooSessionError(f"HTTP {res['status']} for {path}: {res['body'][:200]}")
        try:
            return json.loads(res["body"])
        except json.JSONDecodeError as e:
            raise YahooSessionError(f"non-JSON response for {path}: {res['body'][:200]}") from e

    # ---- endpoints ---------------------------------------------------------

    def league_keys(self, game_code: str = "nhl") -> list[str]:
        data = self.get(f"users;use_login=1/games;game_keys={game_code}/leagues")
        keys: list[str] = []

        def walk(o):
            if isinstance(o, dict):
                if "league_key" in o and isinstance(o["league_key"], str):
                    keys.append(o["league_key"])
                for v in o.values():
                    walk(v)
            elif isinstance(o, list):
                for v in o:
                    walk(v)

        walk(data)
        return sorted(set(keys))

    def league_meta(self, league_key: str) -> dict:
        """Name, team count, scoring type, draft time and draft status."""
        lg = self.get(f"league/{league_key}/settings")["fantasy_content"]["league"]
        merged = {**flatten(lg[1]["settings"]), **flatten(lg[0])}
        return merged

    def teams(self, league_key: str) -> list[dict]:
        node = self.get(f"league/{league_key}/teams")["fantasy_content"]["league"][1]["teams"]
        out = []
        for i in range(int(node["count"])):
            out.append(flatten(node[str(i)]["team"]))
        return out

    def draft_results(self, league_key: str) -> list[dict]:
        """Picks so far, in order. Empty before the draft starts.

        Each entry carries pick number, round, team_key and player_key. Player
        *names* are not included - resolve them through the player map.
        """
        lg = self.get(f"league/{league_key}/draftresults")["fantasy_content"]["league"]
        node = next(
            (x["draft_results"] for x in lg if isinstance(x, dict) and "draft_results" in x),
            None,
        )
        if not node:
            return []
        out = []
        for i in range(int(node.get("count", 0))):
            entry = node.get(str(i), {}).get("draft_result")
            if entry:
                out.append(flatten(entry))
        return out

    def players(
        self, league_key: str, start: int = 0, count: int = 25, extra: str = ""
    ) -> list[dict]:
        """A page of the league player pool.

        `extra` accepts Yahoo's semicolon filters, e.g. 'sort=AR' for average
        draft position order - which is the real ADP the pseudo-ADP stands in for.
        """
        path = f"league/{league_key}/players;start={start};count={count}"
        if extra:
            path += f";{extra.strip(';')}"
        lg = self.get(path)["fantasy_content"]["league"]
        node = next((x["players"] for x in lg if isinstance(x, dict) and "players" in x), None)
        if not node:
            return []
        out = []
        for i in range(int(node.get("count", 0))):
            entry = node.get(str(i), {}).get("player")
            if not entry:
                continue
            out.append(flatten(entry))
        return out
